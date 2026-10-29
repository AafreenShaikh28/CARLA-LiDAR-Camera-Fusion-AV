"""
lidar_val.py - evaluate a trained PointPillars checkpoint (BEV AP and 3D AP, KITTI R40).

    python lidar_val.py --ckpt runs_lidar/pp_town03/best.pt
    python lidar_val.py --ckpt runs_lidar/pp_town03/best.pt --vis 20
    # evaluate on a different town (best true generalisation test), using every frame:
    python lidar_val.py --ckpt ... --data_root ../Data_collection_LIDAR/dataset/Town04 --split all
"""
import os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import pp_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs_lidar/pp_town03/best.pt')
    ap.add_argument('--data_root', default='../Data_collection_LIDAR/dataset/Town03')
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--score_thr', type=float, default=0.1)
    ap.add_argument('--vis', type=int, default=0, help='save N BEV images (GT green, pred red)')
    ap.add_argument('--save_preds', action='store_true')
    ap.add_argument('--out_dir', default='')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = C.build_model_from_ckpt(ckpt, dev)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), 'eval_' + args.split)
    os.makedirs(out_dir, exist_ok=True)

    if args.split == 'all':
        ids = C.get_frame_ids(args.data_root)
    else:
        sp = os.path.join(args.data_root, 'split_lidar.json')
        if not os.path.exists(sp):
            raise SystemExit(f'{sp} not found - run lidar_train.py first (it creates the split).')
        ids = json.load(open(sp))[args.split]
    ds = C.LidarDataset(args.data_root, ids, args.label_dir, train=False)
    loader = DataLoader(ds, args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=C.collate, persistent_workers=args.workers > 0)
    print(f'[val] {len(ds)} frames  ckpt epoch {ckpt["epoch"]+1}')

    preds, gts, frame_ids, vis_done = [], [], [], 0
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
                preds.append(res[k]); gts.append(g); frame_ids.append(b['id'][k])
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

    print('[val] computing IoUs ...')
    ious = [C.pair_ious(p['boxes'], g['boxes']) for p, g in zip(preds, gts)]

    results = {}
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