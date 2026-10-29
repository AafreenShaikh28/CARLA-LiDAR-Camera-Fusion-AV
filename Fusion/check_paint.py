"""
check_paint.py - objective check of the painted channels.
For every GT 3D box, look at the painted points inside it (camera view only) and report the mean
probability of 'vehicle' and 'pedestrian' per object class, versus background points outside all boxes.

    python check_paint.py --ids 000469 003999 005999 007999 009999
    python check_paint.py --n 50          # 50 evenly spaced frames among those already painted
"""
import os, json, argparse, collections
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='../Data_collection_LIDAR/dataset/Town03')
    ap.add_argument('--paint_name', default='velodyne_paint')
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--n', type=int, default=0)
    args = ap.parse_args()

    root = args.data_root
    paint_dir = os.path.join(root, args.paint_name)
    names = json.load(open(os.path.join(paint_dir, 'paint_meta.json')))['classes']
    iv, ip = names.index('vehicle'), names.index('pedestrian')

    if args.ids:
        ids = args.ids
    else:
        ids = sorted(f[:-4] for f in os.listdir(paint_dir) if f.endswith('.npy'))
        if args.n and len(ids) > args.n:
            ids = [ids[k] for k in np.linspace(0, len(ids) - 1, args.n).astype(int)]

    acc = collections.defaultdict(lambda: dict(boxes=0, pts=0, pv=0.0, pp=0.0, av=0, ap=0))
    bg = dict(pts=0, pv=0.0, pp=0.0, av=0, ap=0)
    for fid in ids:
        pts = np.fromfile(os.path.join(root, 'velodyne', fid + '.bin'), dtype=np.float32).reshape(-1, 4)[:, :3]
        ex = np.load(os.path.join(paint_dir, fid + '.npy'))
        incam = ex[:, -1] > 0
        probs = ex[:, :-1].astype(np.float32) / 255.0
        am = probs.argmax(1)
        in_any = np.zeros(len(pts), bool)
        lp = os.path.join(root, args.label_dir, fid + '.txt')
        for line in (open(lp) if os.path.exists(lp) else []):
            f = line.split()
            if len(f) < 8:
                continue
            x, y, z, dx, dy, dz, yaw = map(float, f[1:8])
            rel = pts - np.array([x, y, z])
            c, s = np.cos(yaw), np.sin(yaw)
            lx = rel[:, 0] * c + rel[:, 1] * s
            ly = -rel[:, 0] * s + rel[:, 1] * c
            inside = (np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2) & (np.abs(rel[:, 2]) <= dz / 2) & incam
            in_any |= inside
            k = inside.sum()
            if k == 0:
                continue
            a = acc[f[0].lower()]
            a['boxes'] += 1
            a['pts'] += int(k)
            a['pv'] += float(probs[inside, iv].sum())
            a['pp'] += float(probs[inside, ip].sum())
            a['av'] += int((am[inside] == iv).sum())
            a['ap'] += int((am[inside] == ip).sum())
        b = incam & ~in_any
        bg['pts'] += int(b.sum())
        bg['pv'] += float(probs[b, iv].sum())
        bg['pp'] += float(probs[b, ip].sum())
        bg['av'] += int((am[b] == iv).sum())
        bg['ap'] += int((am[b] == ip).sum())

    print('\n%d frames. Painted points INSIDE GT boxes (camera view only):\n' % len(ids))
    print('%-12s %6s %7s %10s %10s %11s %11s' % ('object', 'boxes', 'points', 'P(vehicle)', 'P(ped)', 'argmax=veh', 'argmax=ped'))
    for k in sorted(acc, key=lambda n: -acc[n]['pts']):
        a = acc[k]
        n = max(a['pts'], 1)
        print('%-12s %6d %7d %10.2f %10.2f %10.0f%% %10.0f%%' %
              (k, a['boxes'], a['pts'], a['pv'] / n, a['pp'] / n, 100 * a['av'] / n, 100 * a['ap'] / n))
    n = max(bg['pts'], 1)
    print('%-12s %6s %7d %10.2f %10.2f %10.0f%% %10.0f%%' %
          ('BACKGROUND', '-', bg['pts'], bg['pv'] / n, bg['pp'] / n, 100 * bg['av'] / n, 100 * bg['ap'] / n))
    print('\nExpect: car/truck/bus -> high P(vehicle); bicycle/motorcycle -> a mix of vehicle and pedestrian'
          ' (the rider is a pedestrian pixel); BACKGROUND -> both low.')


if __name__ == '__main__':
    main()