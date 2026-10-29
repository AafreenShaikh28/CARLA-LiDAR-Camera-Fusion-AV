"""
seg_validate.py - Evaluate a fine-tuned SegFormer on held-out CARLA frames.

python seg_validate.py  --model runs/seg/best --data-dir datasets/carla/val --out runs/seg/eval

Outputs (in --out):
    metrics.json          mIoU, pixel accuracy, per-class IoU/precision/recall, FPS
    confusion_matrix.npy  raw counts [gt, pred]
    confusion_matrix.png  row-normalised plot
    overlays/*.png        [RGB | ground truth | prediction] for a sample of frames

Example:
    python seg_validate.py --model runs/seg/best --data-dir datasets/carla/val \
        --out runs/seg/eval_val --img-size 384 768

Use a different --data-dir per condition (e.g. val_rain, val_night) to get the
weather / time-of-day ablation numbers. Python 3.8 compatible.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

CLASS_NAMES = [
    "road", "sidewalk", "building", "pole", "traffic_light", "traffic_sign",
    "vegetation", "terrain", "sky", "pedestrian", "vehicle", "road_line", "other",
]
NUM_CLASSES = len(CLASS_NAMES)
CARLA_TO_TRAIN = {
    0: 255, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 26: 2, 28: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8,
    12: 9, 13: 9, 14: 10, 15: 10, 16: 10, 17: 10, 18: 10, 19: 10, 24: 11,
    20: 12, 21: 12, 22: 12, 23: 12, 25: 12, 27: 12,
}
LUT = np.full(256, 255, dtype=np.uint8)
for k, v in CARLA_TO_TRAIN.items():
    LUT[k] = v

PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [153, 153, 153], [250, 170, 30],
    [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [0, 0, 142], [255, 255, 255], [110, 190, 160],
], dtype=np.uint8)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EvalDataset(Dataset):
    def __init__(self, split_dir, img_size, image_sub, mask_sub, mask_channel):
        split_dir = Path(split_dir)
        self.img_dir, self.mask_dir = split_dir / image_sub, split_dir / mask_sub
        self.files = sorted(p.name for p in self.img_dir.glob("*.png"))
        self.files = [f for f in self.files if (self.mask_dir / f).exists()]
        if not self.files:
            raise RuntimeError("No matching image/mask pairs in %s" % split_dir)
        self.h, self.w = img_size
        self.ch = mask_channel

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_COLOR)[:, :, self.ch]
        mask = cv2.resize(LUT[raw], (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        rgb_uint8 = img.copy()
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(mask).long(),
                torch.from_numpy(rgb_uint8),
                i)


def colorize(label):
    out = np.zeros(label.shape + (3,), dtype=np.uint8)
    valid = label != 255
    out[valid] = PALETTE[label[valid]]
    return out


def save_confusion_png(conf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dir saved by seg_train.py (e.g. runs/seg/best)")
    ap.add_argument("--data-dir", required=True, help="split folder containing image_2/ and semantic/")
    ap.add_argument("--out", default="runs/seg/eval")
    ap.add_argument("--img-size", type=int, nargs=2, default=[384, 768], metavar=("H", "W"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--image-sub", default="image_2")
    ap.add_argument("--mask-sub", default="semantic")
    ap.add_argument("--mask-channel", type=int, default=2)
    ap.add_argument("--num-overlays", type=int, default=20)
    args = ap.parse_args()

    device = get_device()
    print("device:", device)
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    if model.config.num_labels != NUM_CLASSES:
        raise RuntimeError("Model has %d classes, expected %d" % (model.config.num_labels, NUM_CLASSES))

    ds = EvalDataset(args.data_dir, args.img_size, args.image_sub, args.mask_sub, args.mask_channel)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)
    n = len(ds)
    overlay_ids = set(np.linspace(0, n - 1, min(args.num_overlays, n)).astype(int).tolist())

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    infer_time, seen = 0.0, 0
    with torch.no_grad():
        for img, mask, rgb, idx in dl:
            img, mask = img.to(device), mask.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            logits = model(pixel_values=img).logits
            logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time += time.time() - t0
            seen += img.shape[0]

            valid = mask != 255
            idx_flat = mask[valid] * NUM_CLASSES + pred[valid]
            conf += torch.bincount(idx_flat, minlength=NUM_CLASSES ** 2).reshape(
                NUM_CLASSES, NUM_CLASSES).cpu().numpy().astype(np.int64)

            for b in range(img.shape[0]):
                if int(idx[b]) in overlay_ids:
                    panel = np.concatenate([
                        rgb[b].numpy(),
                        colorize(mask[b].cpu().numpy()),
                        colorize(pred[b].cpu().numpy())], axis=1)
                    name = Path(ds.files[int(idx[b])]).stem
                    cv2.imwrite(str(out / "overlays" / (name + ".png")),
                                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    tp = np.diag(conf).astype(np.float64)
    gt_cnt, pr_cnt = conf.sum(1), conf.sum(0)
    union = gt_cnt + pr_cnt - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    prec = np.where(pr_cnt > 0, tp / np.maximum(pr_cnt, 1), np.nan)
    rec = np.where(gt_cnt > 0, tp / np.maximum(gt_cnt, 1), np.nan)
    f = lambda v: None if np.isnan(v) else round(float(v), 4)

    metrics = {
        "frames": n,
        "mIoU": f(np.nanmean(iou)),
        "pixel_accuracy": f(tp.sum() / max(conf.sum(), 1)),
        "per_class": {c: {"iou": f(iou[i]), "precision": f(prec[i]), "recall": f(rec[i])}
                      for i, c in enumerate(CLASS_NAMES)},
        "fps_model_only": round(seen / max(infer_time, 1e-9), 2),
        "img_size": args.img_size,
        "model": args.model,
        "data_dir": args.data_dir,
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    np.save(out / "confusion_matrix.npy", conf)
    save_confusion_png(conf, out / "confusion_matrix.png")

    print("\nframes: %d | mIoU: %.4f | pixel acc: %.4f | %.1f FPS" %
          (n, np.nanmean(iou), metrics["pixel_accuracy"], metrics["fps_model_only"]))
    for i, c in enumerate(CLASS_NAMES):
        print("  %-14s IoU %s  P %s  R %s" % (c, f(iou[i]), f(prec[i]), f(rec[i])))
    print("saved to", out)


if __name__ == "__main__":
    main()