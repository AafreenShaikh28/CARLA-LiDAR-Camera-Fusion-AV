"""
lidar_train.py (Fusion copy) - train PointPillars; add --painted to use the PointPainting channels.

Run from Fusion/ :
    python lidar_train.py --painted --epochs 30 --batch 4 --workers 4 --out_dir runs_lidar\\pp_painted
Everything else (split, anchors, loss, schedule) is identical to the LiDAR-only baseline, so the two are comparable.
"""
import os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import pp_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='../Data_collection_LIDAR/dataset/Town03')
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--out_dir', default='runs_lidar/pp_painted')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--gap', type=int, default=20)
    ap.add_argument('--resplit', action='store_true')
    ap.add_argument('--resume', default='')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--painted', action='store_true', help='use <data_root>/<paint_name>/ painted channels')
    ap.add_argument('--paint_name', default='velodyne_paint')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (dev.type == 'cuda') and not args.no_amp

    split_path = os.path.join(args.data_root, 'split_lidar.json')
    if args.resplit or not os.path.exists(split_path):
        split = C.make_split(args.data_root, split_path, args.val_frac, args.gap)
        print(f'[split] created {split_path}')
    else:
        split = json.load(open(split_path))
    print(f"[split] train={len(split['train'])} frames  val={len(split['val'])} frames")

    paint_dir, n_feats = None, 4
    if args.painted:
        paint_dir, n_feats = C.paint_info(args.data_root, args.paint_name)
        missing = [i for i in split['train'] + split['val']
                   if not os.path.exists(os.path.join(paint_dir, i + '.npy'))]
        if missing:
            raise SystemExit('%d frames have no painted file (e.g. %s). Run paint_points.py on all frames first.'
                             % (len(missing), missing[:3]))
        print(f'[paint] using {paint_dir}  ->  {n_feats} features per point')

    train_ds = C.LidarDataset(args.data_root, split['train'], args.label_dir, train=True, paint_dir=paint_dir)
    val_ds = C.LidarDataset(args.data_root, split['val'], args.label_dir, train=False, paint_dir=paint_dir)

    if args.check:
        import cv2
        d = os.path.join(args.out_dir, 'check'); os.makedirs(d, exist_ok=True)
        plain = C.LidarDataset(args.data_root, split['train'], args.label_dir, train=False)
        for k in np.linspace(0, len(plain) - 1, 8).astype(int):
            s = plain[k]
            cv2.imwrite(os.path.join(d, f"{s['id']}.png"),
                        C.draw_bev(s['points'].numpy(), gt=s['boxes'].numpy(), size=1600))
        print(f'[check] wrote BEV images to {d}')
        return

    ckpt = torch.load(args.resume, map_location='cpu') if args.resume else None
    if ckpt is not None:
        anchor_sizes = ckpt['anchor_sizes']
        n_feats = ckpt.get('num_pt_feats', 4)
    else:
        anchor_sizes, counts = train_ds.anchor_stats()
        print('[anchors] [l,w,h,z] per class:', [[round(v, 2) for v in a] for a in anchor_sizes],
              ' GT counts:', dict(zip(C.CLASS_NAMES, counts)))
    model = C.PointPillars(anchor_sizes, num_pt_feats=n_feats).to(dev)

    tr_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                           collate_fn=C.collate, drop_last=True, persistent_workers=args.workers > 0)
    va_loader = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers,
                           collate_fn=C.collate, persistent_workers=args.workers > 0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(tr_loader), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_ep, best = 0, float('inf')
    if ckpt is not None:
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched']); start_ep = ckpt['epoch'] + 1; best = ckpt['best_val']
        print(f'[resume] from epoch {start_ep}')

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(args.out_dir, 'tb'))
    except Exception:
        tb = None

    def save(path, ep):
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': ep, 'best_val': best, 'anchor_sizes': anchor_sizes,
                    'num_pt_feats': n_feats,
                    'paint_name': args.paint_name if args.painted else None,
                    'args': vars(args)}, path)

    for ep in range(start_ep, args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for it, b in enumerate(tr_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(b['points'])
            loss, parts = C.compute_loss(out, model, b['boxes'], b['cls'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss)
            if it % 50 == 0:
                print(f'ep {ep+1}/{args.epochs} it {it}/{len(tr_loader)} loss {float(loss):.3f} '
                      f"(cls {parts['cls']:.3f} box {parts['box']:.3f} dir {parts['dir']:.3f}) "
                      f"pos {parts['npos']} lr {sched.get_last_lr()[0]:.2e}")
        tr_loss = run / max(len(tr_loader), 1)

        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for b in va_loader:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(b['points'])
                l, _ = C.compute_loss(out, model, b['boxes'], b['cls'])
                vl += float(l) * len(b['id']); n += len(b['id'])
        val_loss = vl / max(n, 1)
        print(f'=== epoch {ep+1}: train {tr_loss:.4f}  val {val_loss:.4f}  ({time.time()-t0:.0f}s)')
        if tb:
            tb.add_scalar('loss/train', tr_loss, ep); tb.add_scalar('loss/val', val_loss, ep)
        if val_loss < best:
            best = val_loss
            save(os.path.join(args.out_dir, 'best.pt'), ep)
            print('    saved best.pt')
        save(os.path.join(args.out_dir, 'last.pt'), ep)

    print('Done. Evaluate with: python lidar_val.py --ckpt', os.path.join(args.out_dir, 'best.pt'), '--fov_only')


if __name__ == '__main__':
    main()