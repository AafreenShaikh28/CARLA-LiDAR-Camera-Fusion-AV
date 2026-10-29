"""
check_projection.py - verify the LiDAR -> camera projection BEFORE any PointPainting.

Geometry (from data_collection.py): camera at (x=1.5, z=2.4), LiDAR at (x=0, z=2.4), both unrotated,
so a LiDAR point is shifted 1.5 m back along x to get camera coordinates. f = W/(2 tan(FOV/2)).

It does two checks:
1. NUMERIC: projects every LiDAR point and compares its distance with the saved depth image
    (depth/<id>.npy). Tries both y conventions and tells you which one is right.
2. VISUAL : saves overlays with points coloured by distance and the 3D label boxes projected.

Run from Detection_LIDAR/ :
    python check_projection.py
    python check_projection.py --ids 000469 --out_dir runs_fusion/check_projection
"""
import os, argparse
import numpy as np
import cv2


def load_cloud(root, fid):
    """returns (points [N,4], native y convention). .bin from the new collector is y-LEFT;
    a raw CARLA .ply is y-RIGHT."""
    p = os.path.join(root, 'velodyne', fid + '.bin')
    if os.path.exists(p):
        return np.fromfile(p, dtype=np.float32).reshape(-1, 4), 'left'
    for sub in ('velodyne', 'velodyne_ply'):
        p = os.path.join(root, sub, fid + '.ply')
        if os.path.exists(p):
            lines = open(p).read().splitlines()
            i = lines.index('end_header')
            arr = np.array([[float(v) for v in l.split()] for l in lines[i + 1:] if l.strip()], np.float32)
            return arr, 'right'
    raise FileNotFoundError('no velodyne/%s.bin or .ply under %s' % (fid, root))


def load_boxes(root, fid, label_dir):
    p = os.path.join(root, label_dir, fid + '.txt')
    out = []
    if os.path.exists(p):
        for l in open(p):
            f = l.split()
            if len(f) >= 8:
                out.append((f[0],) + tuple(float(v) for v in f[1:8]))
    return out


def to_cam(xyz, ysign, offset):
    """saved LiDAR frame -> standard camera frame (X right, Y down, Z forward)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    X = -y if ysign == 'left' else y
    return np.stack([X, -z, x - offset], 1)


def project(cam, f, cx, cy, w, h):
    Z = cam[:, 2]
    ok = Z > 0.5
    u = np.full(len(cam), -1.0)
    v = np.full(len(cam), -1.0)
    u[ok] = f * cam[ok, 0] / Z[ok] + cx
    v[ok] = f * cam[ok, 1] / Z[ok] + cy
    ok &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    return u, v, Z, ok


def depth_score(cam, depth, f, cx, cy):
    h, w = depth.shape
    u, v, Z, ok = project(cam, f, cx, cy, w, h)
    if ok.sum() == 0:
        return 0, 0.0, float('nan')
    d = depth[v[ok].astype(int), u[ok].astype(int)]
    err = np.abs(d - Z[ok])
    tol = np.maximum(0.5, 0.05 * Z[ok])
    return int(ok.sum()), float((err < tol).mean()), float(np.median(err))


CLS_COL = {'car': (255, 128, 0), 'truck': (0, 128, 255), 'bus': (0, 128, 255),
           'bicycle': (0, 220, 0), 'motorcycle': (0, 220, 0), 'person': (0, 0, 255)}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
FRONT = [(0, 1), (1, 5), (5, 4), (4, 0)]          # +x (heading) face is drawn thicker


def box_corners(x, y, z, dx, dy, dz, yaw):
    sx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * dx / 2
    sy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * dy / 2
    sz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * dz / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + sx * c - sy * s, y + sx * s + sy * c, z + sz], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='../Data_collection_LIDAR/dataset/Town03')
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--ysign', default='auto', choices=['auto', 'left', 'right'])
    ap.add_argument('--offset', type=float, default=1.5, help='camera is this many metres ahead of the LiDAR')
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--no_boxes', action='store_true')
    ap.add_argument('--out_dir', default='runs_fusion/check_projection')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    root = args.data_root
    if args.ids:
        ids = args.ids
    else:
        all_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'velodyne'))
                         if f.endswith('.bin') or f.endswith('.ply'))
        ids = [all_ids[k] for k in np.linspace(0, len(all_ids) - 1, args.n).astype(int)]

    frames = []
    for fid in ids:
        pts, native = load_cloud(root, fid)
        img = cv2.imread(os.path.join(root, 'image_2', fid + '.png'))
        dp = os.path.join(root, 'depth', fid + '.npy')
        depth = np.load(dp) if os.path.exists(dp) else None
        frames.append((fid, pts, native, img, depth))

    # ---- pass 1: numeric check, both y conventions ----
    stats = {'left': [], 'right': []}
    print('\nframe    ysign  pts_in_view  depth-agreement  median|err|(m)')
    for fid, pts, native, img, depth in frames:
        h, w = img.shape[:2]
        f = w / (2 * np.tan(np.deg2rad(args.fov) / 2))
        for sign in ('left', 'right'):
            if depth is None:
                continue
            n, inl, med = depth_score(to_cam(pts[:, :3], sign, args.offset), depth, f, w / 2, h / 2)
            stats[sign].append(inl)
            print('%s  %-5s  %9d  %14.1f%%  %12.2f' % (fid, sign, n, 100 * inl, med))

    if args.ysign != 'auto':
        sign = args.ysign
    elif stats['left']:
        ml, mr = np.mean(stats['left']), np.mean(stats['right'])
        sign = 'left' if ml >= mr else 'right'
        print('\nMEAN depth agreement:  y-left %.1f%%   y-right %.1f%%   ->  use ysign=%s' % (100 * ml, 100 * mr, sign))
        if max(ml, mr) < 0.6:
            print('WARNING: even the better convention agrees on <60%% of points. Something else is off '
                  '(camera offset, FOV, or frame mismatch) - do not paint yet.')
    else:
        sign = frames[0][2]
        print('\nno depth/*.npy found - using the native convention of the files: ysign=%s' % sign)

    if frames[0][4] is not None:                       # sanity: camera offset
        fid, pts, native, img, depth = frames[0]
        h, w = img.shape[:2]
        f = w / (2 * np.tan(np.deg2rad(args.fov) / 2))
        a = depth_score(to_cam(pts[:, :3], sign, args.offset), depth, f, w / 2, h / 2)[1]
        b = depth_score(to_cam(pts[:, :3], sign, 0.0), depth, f, w / 2, h / 2)[1]
        print('offset sanity on %s: %.1f%% with offset %.1f m  vs  %.1f%% with offset 0 m' % (fid, 100 * a, args.offset, 100 * b))

    # ---- pass 2: overlays ----
    cmap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
    for fid, pts, native, img, depth in frames:
        h, w = img.shape[:2]
        f = w / (2 * np.tan(np.deg2rad(args.fov) / 2))
        cx, cy = w / 2, h / 2
        cam = to_cam(pts[:, :3], sign, args.offset)
        u, v, Z, ok = project(cam, f, cx, cy, w, h)
        order = np.argsort(-Z[ok])
        uu, vv, zz = u[ok][order], v[ok][order], Z[ok][order]
        cols = cv2.applyColorMap(np.clip(zz / 60.0 * 255, 0, 255).astype(np.uint8).reshape(-1, 1), cmap).reshape(-1, 3)
        out = img.copy()
        for x_, y_, c_ in zip(uu, vv, cols):
            cv2.circle(out, (int(x_), int(y_)), 2, tuple(int(t) for t in c_), -1)
        if not args.no_boxes:
            for b in load_boxes(root, fid, args.label_dir):
                cc = to_cam(box_corners(*b[1:]), sign, args.offset)
                if (cc[:, 2] < 0.5).any():
                    continue
                px = np.stack([f * cc[:, 0] / cc[:, 2] + cx, f * cc[:, 1] / cc[:, 2] + cy], 1)
                if (np.abs(px) > 4000).any():
                    continue
                col = CLS_COL.get(b[0].lower(), (255, 255, 255))
                for i, j in EDGES:
                    th = 3 if (i, j) in FRONT or (j, i) in FRONT else 1
                    cv2.line(out, tuple(int(t) for t in px[i]), tuple(int(t) for t in px[j]), col, th)
                cv2.putText(out, b[0], (int(px[:, 0].min()), max(int(px[:, 1].min()) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(out, '%s  ysign=%s  points in view: %d' % (fid, sign, int(ok.sum())), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out_dir, 'proj_%s.png' % fid), out)

    print('\nsaved overlays to %s' % args.out_dir)
    print('Points are coloured by distance (blue = near, red = far). Boxes: blue car, orange truck/bus,')
    print('green bike, red person; the thick face is the front.')


if __name__ == '__main__':
    main()