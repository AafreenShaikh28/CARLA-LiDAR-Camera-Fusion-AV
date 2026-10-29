"""
lidar_val.py (Fusion copy) - evaluate a checkpoint (BEV AP and 3D AP, KITTI R40).
Works for both the LiDAR-only baseline and the painted fusion model (reads 'paint_name' from the checkpoint).

Filters for a fair LiDAR-only vs fusion comparison (applied to ground truth AND predictions):
    --fov_only            keep only objects inside the camera's 90 degree view (painting only exists there)
    --dist_min / --dist_max   keep only objects whose centre is within this distance band (metres)

Run from Fusion/ :
    python lidar_val.py --ckpt ..\\Detection_LIDAR\\runs_lidar\\pp_town03\\best.pt --fov_only
    python lidar_val.py --ckpt runs_lidar\\pp_painted\\best.pt --fov_only
    python lidar_val.py --ckpt runs_lidar\\pp_painted\\best.pt --fov_only --dist_min 20 --dist_max 40
"""
import os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import pp_common as C

CAM_X = 1.5          # camera is 1.5 m ahead of the LiDAR (data_collection.py)


def region_mask(boxes, fov_only, dmin, dmax):
    if len(boxes) == 0:
        return np.zeros(0, bool)
    m = np.ones(len(boxes), bool)
    if fov_only:
        x = boxes[:, 0] - CAM_X
        m &= (x > 0.5) & (np.abs(boxes[:, 1]) <= x)          # inside the 90 degree horizontal FOV
    d = np.hypot(boxes[:, 0], boxes[:, 1])
    m &= (d >= dmin) & (d < dmax)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs_lidar/pp_painted/best.pt')
    ap.add_argument('--data_root', default='../Data_collection_LIDAR/dataset/Town03')
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--score_thr', type=float, default=0.1)
    ap.add_argument('--vis', type=int, default=0)
    ap.add_argument('--save_preds', action='store_true')
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--fov_only', action='store_true')
    ap.add_argument('--dist_min', type=float, default=0.0)
    ap.add_argument('--dist_max', type=float, default=1e9)
    ap.add_argument('--paint_name', default=None, help='override; normally read from the checkpoint')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = C.build_model_from_ckpt(ckpt, dev)

    pname = args.paint_name or ckpt.get('paint_name')
    paint_dir = None
    if pname:
        paint_dir = os.path.join(args.data_root, pname)
        if not os.path.isdir(paint_dir):
            raise SystemExit('this checkpoint needs painted data in %s - run paint_points.py first.' % paint_dir)

    tag = ('_fov' if args.fov_only else '')
    if args.dist_min > 0 or args.dist_max < 1e8:
        tag += '_d%g-%g' % (args.dist_min, args.dist_max)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), 'eval_' + args.split + tag)
    os.makedirs(out_dir, exist_ok=True)

    if args.split == 'all':
        ids = C.get_frame_ids(args.data_root)
    else:
        sp = os.path.join(args.data_root, 'split_lidar.json')
        if not os.path.exists(sp):
            raise SystemExit(f'{sp} not found - run lidar_train.py first (it creates the split).')
        ids = json.load(open(sp))[args.split]
    ds = C.LidarDataset(args.data_root, ids, args.label_dir, train=False, paint_dir=paint_dir)
    loader = DataLoader(ds, args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=C.collate, persistent_workers=args.workers > 0)
    print(f'[val] {len(ds)} frames  ckpt epoch {ckpt["epoch"]+1}  painted={bool(pname)}'
          f'  fov_only={args.fov_only}  dist=[{args.dist_min:g},{args.dist_max:g})')

    preds, gts, vis_done = [], [], 0
    t_inf, n_frames = 0.0, 0
    with torch.no_grad():
        for b in loader:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            out = model(b['points'])
            res = C.decode_predictions(out, model, score_thr=args.score_thr)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t_inf += time.time() - t0; n_frames += len(b['id'])
            for k in range(len(b['id'])):
                g = {'boxes': b['boxes'][k].numpy(), 'cls': b['cls'][k].numpy()}
                preds.append(res[k]); gts.append(g)
                if vis_done < args.vis:
                    import cv2
                    img = C.draw_bev(b['points'][k].numpy(), gt=g['boxes'], pred=res[k]['boxes'])
                    cv2.imwrite(os.path.join(out_dir, f"bev_{b['id'][k]}.png"), img)
                    vis_done += 1
                if args.save_preds:
                    pdir = os.path.join(out_dir, 'pred_txt'); os.makedirs(pdir, exist_ok=True)
                    with open(os.path.join(pdir, b['id'][k] + '.txt'), 'w') as f:
                        for bx, s, c in zip(res[k]['boxes'], res[k]['scores'], res[k]['cls']):
                            f.write(f"{C.CLASS_NAMES[c]} {s:.4f} " + ' '.join(f'{v:.3f}' for v in bx) + '\n')

    # ---- region filter (same region for GT and predictions) ----
    if args.fov_only or args.dist_min > 0 or args.dist_max < 1e8:
        for p, g in zip(preds, gts):
            gm = region_mask(g['boxes'], args.fov_only, args.dist_min, args.dist_max)
            g['boxes'], g['cls'] = g['boxes'][gm], g['cls'][gm]
            pm = region_mask(p['boxes'], args.fov_only, args.dist_min, args.dist_max)
            p['boxes'], p['scores'], p['cls'] = p['boxes'][pm], p['scores'][pm], p['cls'][pm]

    n_gt = {C.CLASS_NAMES[c]: int(sum((g['cls'] == c).sum() for g in gts)) for c in range(C.NUM_CLASSES)}
    print('[val] GT boxes in the evaluated region:', n_gt, '(AP on very few boxes is noisy)')
    print('[val] computing IoUs ...')
    ious = [C.pair_ious(p['boxes'], g['boxes']) for p, g in zip(preds, gts)]

    results = {'gt_counts': n_gt}
    print('\n' + '=' * 62)
    print(f"{'class':<12}{'BEV@0.5':>10}{'BEV@0.7':>10}{'3D@0.5':>10}{'3D@0.7':>10}")
    print('-' * 62)
    for c, name in enumerate(C.CLASS_NAMES):
        row = {}
        for mode, mname in ((0, 'BEV'), (1, '3D')):
            for thr in (0.5, 0.7):
                row[f'{mname}@{thr}'] = C.average_precision(preds, gts, ious, c, thr, mode)
        results[name] = row
        print(f"{name:<12}{row['BEV@0.5']:>10.2f}{row['BEV@0.7']:>10.2f}{row['3D@0.5']:>10.2f}{row['3D@0.7']:>10.2f}")
    keys = ['BEV@0.5', 'BEV@0.7', '3D@0.5', '3D@0.7']
    mean = {k: float(np.nanmean([results[n][k] for n in C.CLASS_NAMES])) for k in keys}
    results['mAP'] = mean
    print('-' * 62)
    print(f"{'mAP':<12}" + ''.join(f'{mean[k]:>10.2f}' for k in keys))
    print('=' * 62)
    fps = n_frames / max(t_inf, 1e-9)
    results['fps'] = fps
    print(f'Inference speed: {fps:.1f} frames/s  ({1000/fps:.1f} ms/frame, incl. voxelisation + NMS)')
    json.dump(results, open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=2)
    print(f'Saved metrics to {out_dir}/metrics.json')


if __name__ == '__main__':
    main()