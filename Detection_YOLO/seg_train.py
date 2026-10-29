"""
seg_train.py - Fine-tune SegFormer on CARLA RGB + semantic-camera frames.

Expected data layout (one folder per split, e.g. split by town/run):
    <data_root>/train/image_2/000001.png     RGB frame
    <data_root>/train/semantic/000001.png    CARLA *raw* semantic image (class id in the R channel)
    <data_root>/val/image_2/...
    <data_root>/val/semantic/...

If your logger saved the CityScapes-palette (coloured) version, re-save the raw
version (ColorConverter.Raw) - class ids are read from one channel of the PNG.

Example:
    python seg_train.py --data-root datasets/carla --out runs/seg --epochs 30 \
        --batch-size 8 --img-size 384 768
    # fine-tune from an earlier checkpoint:
    python seg_train.py ... --init runs/seg_prev/best

Python 3.8 compatible.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import SegformerForSemanticSegmentation

# --------------------------------------------------------------------------
# CARLA 0.9.15 semantic ids -> compact training classes (255 = ignore)
# --------------------------------------------------------------------------
CLASS_NAMES = [
    "road", "sidewalk", "building", "pole", "traffic_light", "traffic_sign",
    "vegetation", "terrain", "sky", "pedestrian", "vehicle", "road_line", "other",
]
NUM_CLASSES = len(CLASS_NAMES)
CARLA_TO_TRAIN = {
    0: 255,  # unlabeled
    1: 0, 2: 1,
    3: 2, 4: 2, 5: 2, 26: 2, 28: 2,          # building, wall, fence, bridge, guardrail
    6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8,
    12: 9, 13: 9,                              # pedestrian, rider
    14: 10, 15: 10, 16: 10, 17: 10, 18: 10, 19: 10,  # vehicles
    24: 11,                                    # road line
    20: 12, 21: 12, 22: 12, 23: 12, 25: 12, 27: 12,  # static/dynamic/other/water/ground/rail
}
LUT = np.full(256, 255, dtype=np.uint8)
for k, v in CARLA_TO_TRAIN.items():
    LUT[k] = v

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CarlaSegDataset(Dataset):
    def __init__(self, split_dir, img_size, train, image_sub="image_2",
                 mask_sub="semantic", mask_channel=2):
        split_dir = Path(split_dir)
        self.img_dir = split_dir / image_sub
        self.mask_dir = split_dir / mask_sub
        self.files = sorted(p.name for p in self.img_dir.glob("*.png"))
        self.files = [f for f in self.files if (self.mask_dir / f).exists()]
        if not self.files:
            raise RuntimeError("No matching image/mask pairs in %s" % split_dir)
        self.h, self.w = img_size
        self.train = train
        self.mask_channel = mask_channel  # OpenCV is BGR -> R is index 2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = cv2.imread(str(self.img_dir / name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_COLOR)[:, :, self.mask_channel]
        mask = LUT[raw]

        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:  # horizontal flip
                img = img[:, ::-1]
                mask = mask[:, ::-1]
            # brightness / contrast jitter
            img = np.clip((img - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1), 0, 1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return img, mask


def confusion_from(pred, gt, nc):
    valid = gt != 255
    idx = gt[valid] * nc + pred[valid]
    return torch.bincount(idx, minlength=nc * nc).reshape(nc, nc).cpu().numpy().astype(np.int64)


def miou_from_conf(conf):
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return iou, float(np.nanmean(iou))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(pixel_values=img).logits
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        conf += confusion_from(logits.argmax(1), mask, NUM_CLASSES)
    return miou_from_conf(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder containing train/ and val/")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--out", default="runs/seg")
    ap.add_argument("--model", default="nvidia/mit-b0",
                    help="HF encoder checkpoint (nvidia/mit-b0, nvidia/mit-b1, ...) or a SegFormer fine-tuned dir")
    ap.add_argument("--init", default=None, help="continue from a previously saved SegFormer dir")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, nargs=2, default=[384, 768], metavar=("H", "W"))
    ap.add_argument("--lr", type=float, default=6e-5, help="encoder lr; decode head uses 10x")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mask-channel", type=int, default=2, help="channel (BGR order) holding the class id")
    ap.add_argument("--image-sub", default="image_2")
    ap.add_argument("--mask-sub", default="semantic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print("device:", device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    json.dump(CLASS_NAMES, open(out / "classes.json", "w"))

    root = Path(args.data_root)
    tr_ds = CarlaSegDataset(root / args.train_split, args.img_size, True, args.image_sub, args.mask_sub, args.mask_channel)
    va_ds = CarlaSegDataset(root / args.val_split, args.img_size, False, args.image_sub, args.mask_sub, args.mask_channel)
    print("train frames: %d | val frames: %d" % (len(tr_ds), len(va_ds)))
    pin = device.type == "cuda"
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                       pin_memory=pin, drop_last=True)
    va_dl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    id2label = {i: n for i, n in enumerate(CLASS_NAMES)}
    label2id = {n: i for i, n in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.init or args.model, num_labels=NUM_CLASSES, id2label=id2label,
        label2id=label2id, ignore_mismatched_sizes=True).to(device)

    head = [p for n, p in model.named_parameters() if "decode_head" in n]
    body = [p for n, p in model.named_parameters() if "decode_head" not in n]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.lr * 10}], weight_decay=args.weight_decay)
    total = args.epochs * len(tr_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.lr * 10], total_steps=total, pct_start=0.05, anneal_strategy="cos")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    tb = SummaryWriter(str(out / "tb"))

    best = -1.0
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for it, (img, mask) in enumerate(tr_dl):
            img, mask = img.to(device), mask.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(pixel_values=img).logits
                logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits.float(), mask, ignore_index=255)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            step += 1
            if step % 20 == 0:
                tb.add_scalar("train/loss", loss.item(), step)
        iou, miou = evaluate(model, va_dl, device)
        tb.add_scalar("val/mIoU", miou, ep)
        print("epoch %d/%d | loss %.4f | val mIoU %.4f | %.0fs" %
              (ep + 1, args.epochs, run / max(1, len(tr_dl)), miou, time.time() - t0))
        print("   " + " ".join("%s:%.2f" % (n, v) for n, v in zip(CLASS_NAMES, iou)))
        model.save_pretrained(out / "last")
        if miou > best:
            best = miou
            model.save_pretrained(out / "best")
            json.dump({"epoch": ep + 1, "mIoU": miou,
                       "per_class_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(CLASS_NAMES, iou)}},
                      open(out / "best_metrics.json", "w"), indent=2)
    print("done. best val mIoU = %.4f -> %s" % (best, out / "best"))


if __name__ == "__main__":
    main()