"""
lane_validate.py - Evaluate the lane model from lane_train.py on held-out CARLA frames.

Metrics (aggregated over the whole set):
    * pixel IoU / precision / recall / F1 at --threshold
    * tolerant precision / recall / F1: a predicted pixel counts as correct if a GT
      lane pixel lies within --tol pixels (and vice versa). Thin lines make strict
      pixel metrics harsh, so report both.
    * lane-count error: fitted lane curves vs GT lane components per frame
    * FPS (model only)
Also writes overlays: [RGB | GT lanes | prediction + fitted curves].

Example:
    python lane_validate.py --ckpt runs/lane/best.pt --data-dir datasets/carla/val \
        --out runs/lane/eval_val

Run it once per condition folder (clear / rain / night) for the ablation table.
Python 3.8 compatible.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

ROADLINE_ID = 24
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model():
    m = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None)
    m.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    m.aux_classifier = None
    return m


class LaneEvalDataset(Dataset):
    def __init__(self, split_dir, img_size, image_sub, mask_sub, mask_channel, roadline_id):
        split_dir = Path(split_dir)
        self.img_dir, self.mask_dir = split_dir / image_sub, split_dir / mask_sub
        self.files = sorted(p.name for p in self.img_dir.glob("*.png"))
        self.files = [f for f in self.files if (self.mask_dir / f).exists()]
        if not self.files:
            raise RuntimeError("No matching image/mask pairs in %s" % split_dir)
        self.h, self.w = img_size
        self.ch, self.rid = mask_channel, roadline_id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_COLOR)[:, :, self.ch]
        lane = (raw == self.rid).astype(np.float32)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        lane = (cv2.resize(lane, (self.w, self.h), interpolation=cv2.INTER_AREA) > 0.25).astype(np.uint8)
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(lane),
                torch.from_numpy(img), i)


def fit_lane_curves(mask, min_area=150, deg=2):
    """Connected components of the lane mask -> list of (poly coeffs, y_min, y_max).
    Each lane is modelled as x = f(y), which suits near-vertical lane lines in a
    forward-facing camera. Dashed markings that are broken into pieces are merged
    by a vertical closing step before labelling."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    merged = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    curves = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.nonzero((labels == k) & (mask > 0))
        if len(ys) < min_area // 3 or ys.max() - ys.min() < 15:
            continue
        d = min(deg, 2)
        try:
            coeffs = np.polyfit(ys, xs, d)
        except Exception:
            continue
        curves.append((coeffs, int(ys.min()), int(ys.max())))
    return curves


def count_components(mask, min_area=150):
    return len(fit_lane_curves(mask, min_area))


def draw_curves(img_rgb, curves, color=(255, 255, 0)):
    out = img_rgb.copy()
    for coeffs, y0, y1 in curves:
        ys = np.arange(y0, y1 + 1)
        xs = np.polyval(coeffs, ys)
        pts = np.stack([xs, ys], 1).astype(np.int32)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < out.shape[1])]
        if len(pts) > 1:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, color, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt from lane_train.py")
    ap.add_argument("--data-dir", required=True, help="split folder with image_2/ and semantic/")
    ap.add_argument("--out", default="runs/lane/eval")
    ap.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="defaults to the size stored in the checkpoint")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tol", type=int, default=3, help="pixel tolerance for tolerant metrics")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--image-sub", default="image_2")
    ap.add_argument("--mask-sub", default="semantic")
    ap.add_argument("--mask-channel", type=int, default=2)
    ap.add_argument("--roadline-id", type=int, default=ROADLINE_ID)
    ap.add_argument("--num-overlays", type=int, default=20)
    args = ap.parse_args()

    device = get_device()
    print("device:", device)
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    img_size = args.img_size or ckpt.get("img_size", [384, 768])
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds = LaneEvalDataset(args.data_dir, img_size, args.image_sub, args.mask_sub,
                         args.mask_channel, args.roadline_id)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)
    n = len(ds)
    overlay_ids = set(np.linspace(0, n - 1, min(args.num_overlays, n)).astype(int).tolist())
    dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.tol + 1, 2 * args.tol + 1))

    tp = fp = fn = 0
    p_ok = p_tot = r_ok = r_tot = 0
    count_abs_err, count_exact, infer_time, seen = 0, 0, 0.0, 0

    with torch.no_grad():
        for img, gt, rgb, idx in dl:
            img = img.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            prob = torch.sigmoid(model(img)["out"])[:, 0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time += time.time() - t0
            seen += img.shape[0]
            pred_b = (prob > args.threshold).cpu().numpy().astype(np.uint8)
            gt_b = gt.numpy().astype(np.uint8)

            for b in range(pred_b.shape[0]):
                p, g = pred_b[b], gt_b[b]
                tp += int(((p == 1) & (g == 1)).sum())
                fp += int(((p == 1) & (g == 0)).sum())
                fn += int(((p == 0) & (g == 1)).sum())
                g_d, p_d = cv2.dilate(g, dil), cv2.dilate(p, dil)
                p_ok += int(((p == 1) & (g_d == 1)).sum())
                p_tot += int(p.sum())
                r_ok += int(((g == 1) & (p_d == 1)).sum())
                r_tot += int(g.sum())

                pc, gc = fit_lane_curves(p), fit_lane_curves(g)
                count_abs_err += abs(len(pc) - len(gc))
                count_exact += int(len(pc) == len(gc))

                if int(idx[b]) in overlay_ids:
                    img_rgb = rgb[b].numpy()
                    gt_vis = img_rgb.copy()
                    gt_vis[g == 1] = (0, 255, 0)
                    pr_vis = img_rgb.copy()
                    pr_vis[p == 1] = (255, 0, 0)
                    pr_vis = draw_curves(pr_vis, pc)
                    panel = np.concatenate([img_rgb, gt_vis, pr_vis], axis=1)
                    cv2.imwrite(str(out / "overlays" / (Path(ds.files[int(idx[b])]).stem + ".png")),
                                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    t_prec = p_ok / max(p_tot, 1)
    t_rec = r_ok / max(r_tot, 1)
    f1 = lambda a, b: 2 * a * b / max(a + b, 1e-9)
    metrics = {
        "frames": n,
        "threshold": args.threshold,
        "pixel": {"iou": round(tp / max(tp + fp + fn, 1), 4), "precision": round(prec, 4),
                  "recall": round(rec, 4), "f1": round(f1(prec, rec), 4)},
        "tolerant_%dpx" % args.tol: {"precision": round(t_prec, 4), "recall": round(t_rec, 4),
                                     "f1": round(f1(t_prec, t_rec), 4)},
        "lane_count": {"mean_abs_error": round(count_abs_err / n, 4),
                       "exact_match_rate": round(count_exact / n, 4)},
        "fps_model_only": round(seen / max(infer_time, 1e-9), 2),
        "img_size": list(img_size),
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print("overlays + metrics saved to", out)


if __name__ == "__main__":
    main()