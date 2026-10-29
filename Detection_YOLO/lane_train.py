
"""
lane_train.py - Train / fine-tune a lane-marking detector on CARLA frames.

Approach (the "segment on CARLA lane GT" option from the project plan):
the ground truth is the RoadLine class (id 24) of CARLA's semantic camera, so no
manual labelling is needed. The model is DeepLabV3-MobileNetV3 (COCO-pretrained
torchvision weights) with a 1-channel head predicting a lane-marking probability
map. Lane curves are fitted from that map afterwards (see lane_validate.py).

Expected layout (same as seg_train.py):
    <data_root>/train/image_2/*.png      <data_root>/train/semantic/*.png (raw, id in R)
    <data_root>/val/image_2/*.png        <data_root>/val/semantic/*.png

Example:
    python lane_train.py --data-root datasets/carla --out runs/lane --epochs 30 \
        --batch-size 8 --img-size 384 768
    python lane_train.py ... --init runs/lane_prev/best.pt     # continue fine-tuning

Note: CARLA only labels painted lane markings as RoadLine; it does not label
which lane is "ego lane". Run the decision logic on the fitted curves.
Python 3.8 compatible; needs torchvision >= 0.13.
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import (DeepLabV3_MobileNet_V3_Large_Weights,
                                             deeplabv3_mobilenet_v3_large)

ROADLINE_ID = 24  # CARLA 0.9.15 semantic tag for lane markings
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(pretrained):
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    m = deeplabv3_mobilenet_v3_large(weights=weights, weights_backbone=None)
    m.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # 1 channel: lane logit
    m.aux_classifier = None
    return m


class CarlaLaneDataset(Dataset):
    def __init__(self, split_dir, img_size, train, image_sub="image_2",
                 mask_sub="semantic", mask_channel=2, roadline_id=ROADLINE_ID):
        split_dir = Path(split_dir)
        self.img_dir, self.mask_dir = split_dir / image_sub, split_dir / mask_sub
        self.files = sorted(p.name for p in self.img_dir.glob("*.png"))
        self.files = [f for f in self.files if (self.mask_dir / f).exists()]
        if not self.files:
            raise RuntimeError("No matching image/mask pairs in %s" % split_dir)
        self.h, self.w = img_size
        self.train, self.ch, self.rid = train, mask_channel, roadline_id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_COLOR)[:, :, self.ch]
        lane = (raw == self.rid).astype(np.float32)

        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        # area-resize then threshold so thin lines are not lost by nearest-neighbour
        lane = cv2.resize(lane, (self.w, self.h), interpolation=cv2.INTER_AREA)
        lane = (lane > 0.25).astype(np.float32)

        img = img.astype(np.float32) / 255.0
        if self.train:
            # NOTE: no horizontal flip - it would swap left/right lane semantics
            img = np.clip((img - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1), 0, 1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(lane)[None])


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    tp = fp = fn = 0
    for img, lane in loader:
        img, lane = img.to(device), lane.to(device)
        pred = torch.sigmoid(model(img)["out"]) > thr
        gt = lane > 0.5
        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--out", default="runs/lane")
    ap.add_argument("--init", default=None, help="checkpoint (.pt) from an earlier lane_train run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, nargs=2, default=[384, 768], metavar=("H", "W"))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=10.0, help="BCE weight for lane pixels (they are rare)")
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mask-channel", type=int, default=2)
    ap.add_argument("--roadline-id", type=int, default=ROADLINE_ID)
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

    root = Path(args.data_root)
    kw = dict(image_sub=args.image_sub, mask_sub=args.mask_sub,
              mask_channel=args.mask_channel, roadline_id=args.roadline_id)
    tr_ds = CarlaLaneDataset(root / args.train_split, args.img_size, True, **kw)
    va_ds = CarlaLaneDataset(root / args.val_split, args.img_size, False, **kw)
    print("train frames: %d | val frames: %d" % (len(tr_ds), len(va_ds)))
    pin = device.type == "cuda"
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                       pin_memory=pin, drop_last=True)
    va_dl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    if args.init:
        model = build_model(pretrained=False)
        ckpt = torch.load(args.init, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    else:
        model = build_model(pretrained=True)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tr_dl), pct_start=0.1, anneal_strategy="cos")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    tb = SummaryWriter(str(out / "tb"))

    best, step = -1.0, 0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for img, lane in tr_dl:
            img, lane = img.to(device), lane.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(img)["out"]
                loss = bce(logits.float(), lane) + args.dice_weight * dice_loss(logits.float(), lane)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            step += 1
            if step % 20 == 0:
                tb.add_scalar("train/loss", loss.item(), step)
        m = evaluate(model, va_dl, device)
        for k, v in m.items():
            tb.add_scalar("val/" + k, v, ep)
        print("epoch %d/%d | loss %.4f | val IoU %.4f P %.4f R %.4f F1 %.4f | %.0fs" %
              (ep + 1, args.epochs, run / max(1, len(tr_dl)), m["iou"], m["precision"],
               m["recall"], m["f1"], time.time() - t0))
        state = {"model": model.state_dict(), "img_size": args.img_size, "epoch": ep + 1, "metrics": m}
        torch.save(state, out / "last.pt")
        if m["f1"] > best:
            best = m["f1"]
            torch.save(state, out / "best.pt")
            json.dump(dict(m, epoch=ep + 1), open(out / "best_metrics.json", "w"), indent=2)
    print("done. best val F1 = %.4f -> %s" % (best, out / "best.pt"))


if __name__ == "__main__":
    main()
