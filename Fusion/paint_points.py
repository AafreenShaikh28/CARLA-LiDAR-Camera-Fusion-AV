"""
paint_points.py - PointPainting step 1: attach SegFormer class probabilities to every LiDAR point.

For each frame:  RGB -> SegFormer -> per-class probabilities;  LiDAR points -> projected into the image
(verified by check_projection.py) -> probabilities sampled at the pixel under each point.

Output (does NOT touch your .bin files):
    <data_root>/velodyne_paint/<id>.npy   uint8 [N, n_classes + 1], same point order as velodyne/<id>.bin
        columns 0..n_classes-1 : class probability * 255   (0 for points outside the camera view)
        last column            : 255 if the point projects inside the image, else 0

Needs check_projection.py in the same folder.

Examples (run from Fusion/):
    python paint_points.py --seg_model <...>/runs/seg/best --limit 20 --preview 6      # quick look first
    python paint_points.py --seg_model <...>/runs/seg/best                              # all frames (resumable)
"""
import os, json, argparse, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from check_projection import to_cam, project, load_cloud

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALETTE_RGB = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [153, 153, 153], [250, 170, 30],
                        [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                        [0, 0, 142], [255, 255, 255], [110, 190, 160]], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg_model', required=True, help='folder saved by seg_train.py, e.g. runs/seg/best')
    ap.add_argument('--data_root', default='../Data_collection_LIDAR/dataset/Town03')
    ap.add_argument('--out_name', default='velodyne_paint')
    ap.add_argument('--img_size', type=int, nargs=2, default=[384, 768], metavar=('H', 'W'))
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--offset', type=float, default=1.5)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=0, help='only the first N frames (0 = all)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--preview', type=int, default=0, help='save N preview images of painted points')
    ap.add_argument('--preview_dir', default='runs_fusion/paint_preview')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = dev.type == 'cuda'
    model = SegformerForSemanticSegmentation.from_pretrained(args.seg_model).to(dev).eval()
    n_cls = model.config.num_labels
    names = [model.config.id2label[i] for i in range(n_cls)]
    print('segmentation classes (%d): %s' % (n_cls, names))

    root = args.data_root
    all_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'velodyne')) if f.endswith('.bin'))
    ids = args.ids if args.ids else all_ids
    if args.limit:
        ids = ids[:args.limit]
    out_dir = os.path.join(root, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    if args.preview:
        os.makedirs(args.preview_dir, exist_ok=True)
    json.dump({'classes': names, 'ysign': 'left', 'cam_offset': args.offset, 'fov': args.fov,
               'seg_model': args.seg_model, 'img_size': args.img_size,
               'format': 'uint8 [N, n_classes+1]: probs*255 per class, last col = in-camera flag (255/0)'},
              open(os.path.join(out_dir, 'paint_meta.json'), 'w'), indent=2)

    H_in, W_in = args.img_size
    n_done, in_cam_frac, argmax_counts, previews = 0, [], np.zeros(n_cls, np.int64), 0
    t0 = time.time()
    for s in range(0, len(ids), args.batch):
        chunk = ids[s:s + args.batch]
        todo = [i for i in chunk if args.overwrite or not os.path.exists(os.path.join(out_dir, i + '.npy'))
                or (args.preview and previews < args.preview)]
        if not todo:
            continue
        batch_imgs, bgr_imgs = [], []
        for fid in todo:
            bgr = cv2.imread(os.path.join(root, 'image_2', fid + '.png'))
            bgr_imgs.append(bgr)
            x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W_in, H_in), interpolation=cv2.INTER_LINEAR)
            x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            batch_imgs.append(x.transpose(2, 0, 1))
        batch = torch.from_numpy(np.stack(batch_imgs)).to(dev)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=half):
            logits = model(pixel_values=batch).logits.float()            # [B, C, h, w] (1/4 resolution)

        for b, fid in enumerate(todo):
            pts, native = load_cloud(root, fid)
            if native != 'left':
                raise RuntimeError('%s: expected the y-left .bin from the new collector' % fid)
            Hh, Ww = bgr_imgs[b].shape[:2]
            f = Ww / (2 * np.tan(np.deg2rad(args.fov) / 2))
            u, v, Z, ok = project(to_cam(pts[:, :3], 'left', args.offset), f, Ww / 2, Hh / 2, Ww, Hh)
            extra = np.zeros((len(pts), n_cls + 1), np.uint8)
            probs = None
            if ok.any():
                gx = (u[ok] + 0.5) / Ww * 2 - 1
                gy = (v[ok] + 0.5) / Hh * 2 - 1
                grid = torch.from_numpy(np.stack([gx, gy], 1)).float().to(dev).view(1, 1, -1, 2)
                samp = F.grid_sample(logits[b:b + 1], grid, mode='bilinear', padding_mode='border',
                                     align_corners=False)                   # [1, C, 1, K]
                probs = torch.softmax(samp[0, :, 0, :], dim=0).T.cpu().numpy()   # [K, C]
                extra[ok, :n_cls] = np.round(probs * 255).astype(np.uint8)
                extra[ok, n_cls] = 255
                argmax_counts += np.bincount(probs.argmax(1), minlength=n_cls)
            np.save(os.path.join(out_dir, fid + '.npy'), extra)
            in_cam_frac.append(float(ok.mean()))
            n_done += 1

            if args.preview and previews < args.preview and probs is not None:
                img = bgr_imgs[b].copy()
                cls = probs.argmax(1)
                for x_, y_, c_ in zip(u[ok], v[ok], cls):
                    col = PALETTE_RGB[c_][::-1]
                    cv2.circle(img, (int(x_), int(y_)), 3, tuple(int(t) for t in col), -1)
                for k, nm in enumerate(names):
                    cv2.putText(img, nm, (8, 20 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                tuple(int(t) for t in PALETTE_RGB[k][::-1]), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(args.preview_dir, 'paint_%s.png' % fid), img)
                previews += 1
        if n_done and (s // args.batch) % 20 == 0:
            print('%d/%d frames  (%.1f frames/s)' % (n_done, len(ids), n_done / (time.time() - t0)))

    print('\npainted %d frames in %.0fs; %.1f%% of points are inside the camera view on average'
          % (n_done, time.time() - t0, 100 * np.mean(in_cam_frac) if in_cam_frac else 0))
    tot = max(argmax_counts.sum(), 1)
    print('share of in-view points by predicted class:')
    for k in np.argsort(-argmax_counts):
        print('  %-14s %5.1f%%' % (names[k], 100 * argmax_counts[k] / tot))
    print('saved to', out_dir)


if __name__ == '__main__':
    main()