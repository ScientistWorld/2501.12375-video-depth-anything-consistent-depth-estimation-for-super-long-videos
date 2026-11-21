"""Flag KITTI clips whose GPS/IMU pose steps disagree with the measured speed
(GPS glitches make the relative camera motion used by TAE wrong).

    python3 redteam/hardening2/oxts_scan.py <data root> <clip manifest>

Prints, per clip, the consecutive-frame steps where |step - speed*dt| exceeds
max(0.3 m, 30%) or the frame interval is irregular. Standard library only."""
import json, math, sys
from datetime import datetime
def scan(root, manifest):
    out = {}
    for c in json.load(open(manifest))['videos']:
        if c['dataset'] != 'kitti': continue
        d = f"{root}/kitti/raw/{c['drive'][:10]}/{c['drive']}_sync"
        ts = [l.strip() for l in open(d + '/oxts/timestamps.txt')]
        def t(i):
            s = ts[i]; return datetime.strptime(s[:26], '%Y-%m-%d %H:%M:%S.%f').timestamp()
        prev = None; bad = []
        lat0 = None
        for k, i in enumerate(range(c['start'], c['start'] + c['frames'])):
            v = [float(x) for x in open(f'{d}/oxts/data/{i:010d}.txt').read().split()]
            if lat0 is None: lat0 = v[0]
            sc = math.cos(math.radians(lat0)); er = 6378137.0
            p = (sc * v[1] * math.pi * er / 180, sc * er * math.log(math.tan((90 + v[0]) * math.pi / 360)), v[2])
            if prev:
                dt = t(i) - t(i - 1)
                step = math.dist(p, prev[0])
                speed = math.sqrt(v[8] ** 2 + v[9] ** 2 + v[10] ** 2)
                exp = 0.5 * (speed + prev[1]) * dt
                if abs(step - exp) > max(0.3, 0.3 * exp) or dt > 0.15 or dt < 0.05:
                    bad.append((k, round(step, 2), round(exp, 2), round(dt, 3)))
            prev = (p, math.sqrt(v[8] ** 2 + v[9] ** 2 + v[10] ** 2))
        out[c['id']] = bad
        print(c['id'], len(bad), bad[:6])
    return out


if __name__ == "__main__":
    scan(sys.argv[1], sys.argv[2])
