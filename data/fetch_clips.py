"""Fetch the frames of a clip manifest from the public Bonn RGB-D Dynamic and
KITTI archives (standard library only, so it runs on a bare login node).

    python3 data/fetch_clips.py --manifest data/train_clips.json --dest data/downloads
    python3 data/fetch_clips.py --manifest eval/test/clips.json --dest eval/test/downloads

Only the members a clip needs are read, with HTTP range requests on the
original zip archives: RGB frames, depth maps, camera poses / OXTS records and
calibration files. Files keep their archive-relative paths under
``<dest>/bonn/`` and ``<dest>/kitti/{raw,depth}/``. Idempotent: members already
on disk with the right size are skipped. Every member is CRC-checked.

Sources (public):
  Bonn RGB-D Dynamic  https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/
  KITTI raw + depth   https://www.cvlibs.net/datasets/kitti/  (CC BY-NC-SA 3.0)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BONN_URL = "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_{seq}.zip"
KITTI_RAW_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{drive}/{drive}_sync.zip"
KITTI_CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{date}_calib.zip"
KITTI_DEPTH_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"

GROUP_GAP = 1 << 20  # merge members closer than this into one range request
GROUP_MAX = 64 << 20  # bytes per range request


def _request(url: str, start: int | None = None, end: int | None = None, method: str = "GET") -> bytes | dict:
    headers = {"User-Agent": "dataset-fetch/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end - 1}"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=300) as r:
                if method == "HEAD":
                    return dict(r.headers)
                data = r.read()
            if start is not None and len(data) != end - start:
                raise IOError(f"short read {len(data)} != {end - start}")
            return data
        except Exception as exc:  # retry transient network errors
            if attempt == 5:
                raise
            print(f"  retry {attempt + 1} for {url} [{start}-{end}]: {exc}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class _HTTPFile(io.RawIOBase):
    """Seekable read-only view of a remote file (used to read the zip directory)."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.size = int(_request(url, method="HEAD")["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = _request(self.url, self.pos, self.pos + n)
        b[:n] = data
        self.pos += n
        return n


_DIRS: dict[str, dict[str, zipfile.ZipInfo]] = {}


def directory(url: str) -> dict[str, zipfile.ZipInfo]:
    if url not in _DIRS:
        zf = zipfile.ZipFile(io.BufferedReader(_HTTPFile(url), buffer_size=4 << 20))
        _DIRS[url] = {zi.filename: zi for zi in zf.infolist()}
        print(f"  directory of {url}: {len(_DIRS[url])} entries", flush=True)
    return _DIRS[url]


def _extract(blob: bytes, base: int, zi: zipfile.ZipInfo) -> bytes:
    off = zi.header_offset - base
    if blob[off:off + 4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {zi.filename}")
    n, e = struct.unpack("<HH", blob[off + 26:off + 30])
    start = off + 30 + n + e
    raw = blob[start:start + zi.compress_size]
    if len(raw) != zi.compress_size:
        raise IOError(f"truncated member {zi.filename}")
    if zi.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif zi.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(raw)
    else:
        raise IOError(f"unsupported compression {zi.compress_type} for {zi.filename}")
    if zlib.crc32(data) & 0xFFFFFFFF != zi.CRC or len(data) != zi.file_size:
        raise IOError(f"CRC/size mismatch for {zi.filename}")
    return data


def fetch_members(url: str, wanted: dict[str, Path], workers: int) -> None:
    """Download archive members ``{member_name: destination}``."""
    zdir = directory(url)
    todo = []
    for name, dest in wanted.items():
        if name not in zdir:
            raise SystemExit(f"member missing from {url}: {name}")
        zi = zdir[name]
        if dest.is_file() and dest.stat().st_size == zi.file_size:
            continue
        todo.append((zi, dest))
    if not todo:
        return
    todo.sort(key=lambda t: t[0].header_offset)
    groups, cur = [], []
    for zi, dest in todo:
        end = zi.header_offset + 30 + len(zi.filename.encode()) + 1024 + zi.compress_size
        if cur:
            g_start = cur[0][0].header_offset
            g_end = cur[-1][2]
            if zi.header_offset - g_end > GROUP_GAP or end - g_start > GROUP_MAX:
                groups.append(cur)
                cur = []
        cur.append((zi, dest, end))
    if cur:
        groups.append(cur)
    print(f"  {url}: {len(todo)} members in {len(groups)} range requests", flush=True)

    def work(group):
        start = group[0][0].header_offset
        stop = min(max(g[2] for g in group), _size(url))
        blob = _request(url, start, stop)
        for zi, dest, _ in group:
            data = _extract(blob, start, zi)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        return len(group)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, g) for g in groups]
        for fut in as_completed(futures):
            done += fut.result()
    print(f"  {url}: fetched {done} members", flush=True)


_SIZES: dict[str, int] = {}


def _size(url: str) -> int:
    if url not in _SIZES:
        _SIZES[url] = int(_request(url, method="HEAD")["Content-Length"])
    return _SIZES[url]


BONN_MAX_DT = 0.02  # s; an RGB frame without a depth map this close gets no ground truth


def bonn_frame_index(names: list[str], prefix: str) -> list[tuple[str, str]]:
    """RGB frames of a sequence in time order, each with the depth map closest in
    time (``-`` when none lies within BONN_MAX_DT). The archives hold a few RGB
    frames without depth, so pairing by list position would drift."""
    rgb = sorted((float(n[len(prefix) + 4:-4]), n[len(prefix):]) for n in names
                 if n.startswith(prefix + "rgb/") and n.endswith(".png"))
    dep = sorted((float(n[len(prefix) + 6:-4]), n[len(prefix):]) for n in names
                 if n.startswith(prefix + "depth/") and n.endswith(".png"))
    out, j = [], 0
    for t, rname in rgb:
        while j + 1 < len(dep) and abs(dep[j + 1][0] - t) <= abs(dep[j][0] - t):
            j += 1
        match = dep[j][1] if dep and abs(dep[j][0] - t) <= BONN_MAX_DT else "-"
        out.append((rname, match))
    return out


def fetch_bonn(clips: list[dict], dest: Path, workers: int) -> None:
    by_seq: dict[str, list[dict]] = {}
    for c in clips:
        by_seq.setdefault(c["sequence"], []).append(c)
    for seq, cs in sorted(by_seq.items()):
        url = BONN_URL.format(seq=seq)
        prefix = f"rgbd_bonn_{seq}/"
        root = dest / "bonn"
        fetch_members(url, {prefix + f: root / prefix / f for f in ("rgb.txt", "depth.txt", "groundtruth.txt")}, workers)
        index = bonn_frame_index(list(directory(url)), prefix)
        (root / prefix / "frame_index.txt").write_text(
            "# rgb_file depth_file ('-' = no depth within %.3f s); written by data/fetch_clips.py\n" % BONN_MAX_DT
            + "".join(f"{r} {d}\n" for r, d in index))
        wanted = {}
        for c in cs:
            lo, hi = c["start"], c["start"] + c["frames"]
            if hi > len(index):
                raise SystemExit(f"bonn {seq}: clip [{lo},{hi}) exceeds {len(index)} frames")
            for i in range(lo, hi):
                for rel in index[i]:
                    if rel != "-":
                        wanted[prefix + rel] = root / prefix / rel
        fetch_members(url, wanted, workers)


def fetch_kitti(clips: list[dict], dest: Path, workers: int) -> None:
    raw_root = dest / "kitti" / "raw"
    depth_root = dest / "kitti" / "depth"
    dates = sorted({c["drive"][:10] for c in clips})
    for date in dates:
        names = [f"{date}/calib_{k}.txt" for k in ("cam_to_cam", "imu_to_velo", "velo_to_cam")]
        fetch_members(KITTI_CALIB_URL.format(date=date), {n: raw_root / n for n in names}, workers)
    depth_wanted = {}
    by_drive: dict[str, list[dict]] = {}
    for c in clips:
        by_drive.setdefault(c["drive"], []).append(c)
    for drive, cs in sorted(by_drive.items()):
        date = drive[:10]
        base = f"{date}/{drive}_sync/"
        wanted = {base + "oxts/timestamps.txt": raw_root / base / "oxts/timestamps.txt",
                  base + "image_02/timestamps.txt": raw_root / base / "image_02/timestamps.txt"}
        for c in cs:
            for i in range(c["start"], c["start"] + c["frames"]):
                for rel in (f"image_02/data/{i:010d}.png", f"oxts/data/{i:010d}.txt"):
                    wanted[base + rel] = raw_root / base / rel
                dname = f"{c['depth_split']}/{drive}_sync/proj_depth/groundtruth/image_02/{i:010d}.png"
                depth_wanted[dname] = depth_root / dname
        fetch_members(KITTI_RAW_URL.format(drive=drive), wanted, workers)
    fetch_members(KITTI_DEPTH_URL, depth_wanted, workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    dest = Path(args.dest)
    videos = manifest["videos"]
    t0 = time.time()
    fetch_bonn([v for v in videos if v["dataset"] == "bonn"], dest, args.workers)
    fetch_kitti([v for v in videos if v["dataset"] == "kitti"], dest, args.workers)
    print(f"fetched {len(videos)} clips into {dest} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
