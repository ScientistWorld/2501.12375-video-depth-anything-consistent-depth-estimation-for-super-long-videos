"""Hidden test-scene renderer (private to eval/test/).

Static indoor corridor scenes rendered by exact ray casting from a smoothly
moving pinhole camera, with z-depth, intrinsics and camera-to-world poses.
The test videos are rendered from this module at test time from seeds that
exist only in eval/test/evaluate.py; nothing is written to disk except the
RGB frames handed to each predictor.
"""

from __future__ import annotations

import numpy as np

HEIGHT = 144
WIDTH = 192
FPS = 12
FOCAL_SCALE = 0.9  # focal length in pixels = FOCAL_SCALE * WIDTH

# Surface ids for the six bounding planes; boxes start at FIRST_BOX_ID.
FLOOR, CEILING, WALL_NEG_X, WALL_POS_X, END_WALL, START_WALL = range(6)
FIRST_BOX_ID = 6
_PLANE_NORMALS = np.array(
    [
        [0.0, -1.0, 0.0],  # floor (y down, so the floor normal points to -y)
        [0.0, 1.0, 0.0],  # ceiling
        [1.0, 0.0, 0.0],  # wall at x = -hw
        [-1.0, 0.0, 0.0],  # wall at x = +hw
        [0.0, 0.0, -1.0],  # far end wall
        [0.0, 0.0, 1.0],  # start wall
    ]
)


def intrinsics(height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    f = FOCAL_SCALE * width
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _smooth(rng: np.random.Generator, frames: int, periods, amps) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64)
    out = np.zeros(frames)
    for period, amp in zip(periods, amps):
        out += amp * np.sin(2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi))
    return out


def _color(rng: np.random.Generator, lo: float = 60.0, hi: float = 210.0) -> np.ndarray:
    return rng.uniform(lo, hi, size=3)


def build_scene(seed: int, frames: int, height: int = HEIGHT, width: int = WIDTH) -> dict:
    """Sample the static scene, the camera trajectory and the appearance."""
    rng = np.random.default_rng(seed)
    hw = rng.uniform(1.5, 2.4)  # corridor half width
    floor_y = rng.uniform(1.25, 1.6)  # camera height above floor (y down)
    ceil_y = -rng.uniform(1.0, 1.6)  # ceiling above the camera
    speed = rng.uniform(0.022, 0.034)  # mean forward motion per frame

    v = speed * (1.0 + _smooth(rng, frames, [rng.uniform(90, 200), rng.uniform(250, 500)], [0.45, 0.35]))
    v = np.clip(v, 0.15 * speed, None)
    cam_z = np.concatenate([[0.0], np.cumsum(v[:-1])])
    cam_x = _smooth(rng, frames, [rng.uniform(80, 160), rng.uniform(200, 400)], [0.12, 0.12])
    cam_y = _smooth(rng, frames, [rng.uniform(40, 80)], [0.03])
    yaw_amp = np.deg2rad(rng.uniform(25.0, 45.0))
    yaw = _smooth(rng, frames, [rng.uniform(140, 280), rng.uniform(60, 110)], [yaw_amp, 0.3 * yaw_amp])
    pitch = _smooth(rng, frames, [rng.uniform(70, 150)], [np.deg2rad(rng.uniform(3.0, 10.0))])
    pitch = pitch + np.deg2rad(rng.uniform(-4.0, 6.0))

    z_start = -3.0
    z_end = float(cam_z[-1] + rng.uniform(6.0, 12.0))

    poses = np.zeros((frames, 4, 4))
    for t in range(frames):
        poses[t, :3, :3] = _rot_y(yaw[t]) @ _rot_x(pitch[t])
        poses[t, :3, 3] = [cam_x[t], cam_y[t], cam_z[t]]
        poses[t, 3, 3] = 1.0

    # Surface appearance: base colour, pattern type (0 checker, 1 stripes,
    # 2 bricks), pattern scale and contrast.
    colors = [_color(rng, 70, 170), _color(rng, 120, 200), _color(rng), _color(rng), _color(rng), _color(rng)]
    ptypes = [0, 0, 2, 2, int(rng.integers(0, 3)), 0]
    pscales = [rng.uniform(0.3, 0.7), rng.uniform(0.6, 1.2), rng.uniform(0.15, 0.35), rng.uniform(0.15, 0.35),
               rng.uniform(0.2, 0.5), 0.5]
    pamps = [rng.uniform(12, 26), rng.uniform(6, 14), rng.uniform(10, 22), rng.uniform(10, 22),
             rng.uniform(10, 22), 10.0]

    bmin, bmax = [], []

    def add_box(lo, hi, color, ptype, scale, amp):
        bmin.append(lo)
        bmax.append(hi)
        colors.append(np.asarray(color, dtype=np.float64))
        ptypes.append(int(ptype))
        pscales.append(float(scale))
        pamps.append(float(amp))

    # Partition walls with doorways.
    zp = z_start + rng.uniform(4.0, 7.0)
    while zp < z_end - 2.0:
        thick = rng.uniform(0.1, 0.25)
        door_c = rng.uniform(-0.1, 0.1)
        door_hw = rng.uniform(0.6, 0.95)
        door_top = min(floor_y - rng.uniform(2.0, 2.3), -0.35)
        col = _color(rng)
        pt, ps, pa = int(rng.integers(0, 3)), rng.uniform(0.15, 0.4), rng.uniform(10, 24)
        add_box([-hw - 0.01, ceil_y - 0.01, zp], [door_c - door_hw, floor_y + 0.01, zp + thick], col, pt, ps, pa)
        add_box([door_c + door_hw, ceil_y - 0.01, zp], [hw + 0.01, floor_y + 0.01, zp + thick], col, pt, ps, pa)
        if door_top > ceil_y + 0.05:
            add_box([door_c - door_hw, ceil_y - 0.01, zp], [door_c + door_hw, door_top, zp + thick], col, pt, ps, pa)
        zp += rng.uniform(5.0, 10.0)

    # Furniture against the side walls.
    n_side = int((z_end - z_start) * rng.uniform(0.6, 1.0))
    for _ in range(n_side):
        side = 1.0 if rng.random() < 0.5 else -1.0
        inner = max(hw - rng.uniform(0.25, 0.9), 0.7)
        zc, zl = rng.uniform(z_start + 0.5, z_end - 0.5), rng.uniform(0.3, 1.6)
        if rng.random() < 0.6:
            y0, y1 = max(floor_y - rng.uniform(0.3, 2.0), ceil_y + 0.05), floor_y + 0.01
        else:
            yc, hh = rng.uniform(ceil_y + 0.3, floor_y - 0.6), rng.uniform(0.08, 0.3)
            y0, y1 = yc - hh, yc + hh
        x0, x1 = (inner, hw + 0.01) if side > 0 else (-hw - 0.01, -inner)
        add_box([x0, y0, zc - zl / 2], [x1, y1, zc + zl / 2], _color(rng), rng.integers(0, 3),
                rng.uniform(0.08, 0.3), rng.uniform(12, 30))

    # Low objects on the floor that the camera passes over.
    n_low = int((z_end - z_start) * rng.uniform(0.1, 0.25))
    for _ in range(n_low):
        xc, zc = rng.uniform(-0.6, 0.6), rng.uniform(z_start + 2.0, z_end - 0.5)
        sx, sz, top = rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.55)
        add_box([xc - sx, floor_y - top, zc - sz], [xc + sx, floor_y + 0.01, zc + sz], _color(rng),
                rng.integers(0, 3), rng.uniform(0.05, 0.15), rng.uniform(12, 30))

    light = np.array([rng.uniform(-0.4, 0.4), 1.0, rng.uniform(0.2, 0.8)])
    return {
        "seed": int(seed),
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "hw": float(hw),
        "floor_y": float(floor_y),
        "ceil_y": float(ceil_y),
        "z_start": float(z_start),
        "z_end": float(z_end),
        "poses": poses,
        "K": intrinsics(height, width),
        "bmin": np.asarray(bmin, dtype=np.float64).reshape(-1, 3),
        "bmax": np.asarray(bmax, dtype=np.float64).reshape(-1, 3),
        "colors": np.asarray(colors, dtype=np.float64),
        "ptypes": np.asarray(ptypes, dtype=np.int64),
        "pscales": np.asarray(pscales, dtype=np.float64),
        "pamps": np.asarray(pamps, dtype=np.float64),
        "light": light / np.linalg.norm(light),
        "ambient": float(rng.uniform(0.45, 0.6)),
        "diffuse": float(rng.uniform(0.25, 0.4)),
        "fog": float(rng.uniform(14.0, 30.0)),
        "fog_color": _color(rng, 80, 150),
        "exposure_amp": float(rng.uniform(0.06, 0.14)),
        "exposure_period": float(rng.uniform(120, 300)),
        "exposure_phase": float(rng.uniform(0, 2 * np.pi)),
    }


def _hash(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    h = np.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return h - np.floor(h)


def _camera_rays(height: int, width: int, K: np.ndarray) -> np.ndarray:
    vv, uu = np.mgrid[0:height, 0:width].astype(np.float64)
    x = (uu - K[0, 2]) / K[0, 0]
    y = (vv - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)


def render_frame(scene: dict, t: int, rays_cam: np.ndarray | None = None):
    """Render frame ``t``. Returns (rgb uint8 [H,W,3], z-depth float32 [H,W])."""
    h, w = scene["height"], scene["width"]
    if rays_cam is None:
        rays_cam = _camera_rays(h, w, scene["K"])
    pose = scene["poses"][t]
    R, c = pose[:3, :3], pose[:3, 3]
    d = rays_cam @ R.T  # world directions; the camera-frame z component is 1, so t == z-depth
    n = d.shape[0]
    best_t = np.full(n, np.inf)
    best_id = np.full(n, -1, dtype=np.int64)

    def update(tt: np.ndarray, sid: int) -> None:
        m = (tt > 1e-6) & (tt < best_t)
        best_t[m] = tt[m]
        best_id[m] = sid

    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        update(np.where(dy > 1e-9, (scene["floor_y"] - c[1]) / dy, np.inf), FLOOR)
        update(np.where(dy < -1e-9, (scene["ceil_y"] - c[1]) / dy, np.inf), CEILING)
        update(np.where(dx < -1e-9, (-scene["hw"] - c[0]) / dx, np.inf), WALL_NEG_X)
        update(np.where(dx > 1e-9, (scene["hw"] - c[0]) / dx, np.inf), WALL_POS_X)
        update(np.where(dz > 1e-9, (scene["z_end"] - c[2]) / dz, np.inf), END_WALL)
        update(np.where(dz < -1e-9, (scene["z_start"] - c[2]) / dz, np.inf), START_WALL)

    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    bmin, bmax = scene["bmin"], scene["bmax"]
    active = np.flatnonzero((bmax[:, 2] > c[2] - 5.0) & (bmin[:, 2] < c[2] + 60.0))
    for k in active:
        t1 = (bmin[k] - c) * inv
        t2 = (bmax[k] - c) * inv
        tn = np.minimum(t1, t2).max(axis=1)
        tf = np.maximum(t1, t2).min(axis=1)
        hit = (tf >= tn) & (tn > 1e-6) & (tn < best_t)
        best_t[hit] = tn[hit]
        best_id[hit] = FIRST_BOX_ID + k

    p = c + d * best_t[:, None]
    normal = np.zeros((n, 3))
    is_plane = best_id < FIRST_BOX_ID
    normal[is_plane] = _PLANE_NORMALS[best_id[is_plane]]
    is_box = ~is_plane
    if np.any(is_box):
        k = best_id[is_box] - FIRST_BOX_ID
        ctr = 0.5 * (bmin[k] + bmax[k])
        half = 0.5 * (bmax[k] - bmin[k])
        q = (p[is_box] - ctr) / half
        ax = np.argmax(np.abs(q), axis=1)
        nb = np.zeros_like(q)
        nb[np.arange(q.shape[0]), ax] = np.sign(q[np.arange(q.shape[0]), ax])
        normal[is_box] = nb

    axis = np.argmax(np.abs(normal), axis=1)
    uu = np.where(axis == 0, p[:, 2], p[:, 0])
    vv = np.where(axis == 1, p[:, 2], p[:, 1])
    sid = best_id
    s = scene["pscales"][sid]
    ptype = scene["ptypes"][sid]
    checker = ((np.floor(uu / s) + np.floor(vv / s)) % 2) * 2.0 - 1.0
    stripes = (np.floor((uu + 0.5 * vv) / s) % 2) * 2.0 - 1.0
    row = np.floor(vv / s)
    fu = (uu + (row % 2) * s) / (2.0 * s)
    mortar = ((vv / s - row) < 0.1) | ((fu - np.floor(fu)) < 0.05)
    brick = np.where(mortar, -1.5, (_hash(row, np.floor(fu), sid.astype(np.float64)) * 2.0 - 1.0) * 0.6)
    pattern = np.where(ptype == 0, checker, np.where(ptype == 1, stripes, brick))
    fine = _hash(np.floor(p[:, 0] * 7.0), np.floor(p[:, 1] * 7.0), np.floor(p[:, 2] * 7.0)) * 2.0 - 1.0
    col = scene["colors"][sid] + (scene["pamps"][sid] * pattern)[:, None] + 7.0 * fine[:, None]

    dnorm = np.linalg.norm(d, axis=1)
    lam = np.clip(-(normal @ scene["light"]), 0.0, None)
    head = np.abs(np.sum(normal * d, axis=1)) / dnorm
    shade = scene["ambient"] + scene["diffuse"] * lam + 0.25 * head
    col = col * shade[:, None]
    fog = 1.0 - np.exp(-(best_t * dnorm) / scene["fog"])
    col = col * (1.0 - fog)[:, None] + scene["fog_color"][None, :] * fog[:, None]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((xx - (w - 1) / 2) / (w / 2)) ** 2 + ((yy - (h - 1) / 2) / (h / 2)) ** 2
    frng = np.random.default_rng([scene["seed"] % (2**63), t])
    gain = 1.0 + scene["exposure_amp"] * np.sin(2 * np.pi * t / scene["exposure_period"] + scene["exposure_phase"])
    gain *= 1.0 + frng.normal(0.0, 0.012)
    img = col.reshape(h, w, 3) * (1.0 - 0.3 * r2)[..., None] * gain
    img = img + frng.normal(0.0, 2.0, size=img.shape)
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    return rgb, best_t.reshape(h, w).astype(np.float32)


def render_video(seed: int, frames: int, height: int = HEIGHT, width: int = WIDTH) -> dict:
    """Render a whole video. Returns rgb, depth, K, poses (camera-to-world) and fps."""
    scene = build_scene(seed, frames, height, width)
    rays = _camera_rays(height, width, scene["K"])
    rgb = np.empty((frames, height, width, 3), dtype=np.uint8)
    depth = np.empty((frames, height, width), dtype=np.float32)
    for t in range(frames):
        rgb[t], depth[t] = render_frame(scene, t, rays)
    return {"rgb": rgb, "depth": depth, "K": scene["K"], "poses": scene["poses"], "fps": FPS}
