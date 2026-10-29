"""
pp_common.py  -  Pure-PyTorch PointPillars (no spconv / no custom CUDA ops)
Shared by lidar_train.py and lidar_val.py

Box format everywhere (LiDAR frame):  [x, y, z_center, l, w, h, yaw]
yaw = CCW angle from +x axis (l is along the heading direction).
"""
import os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

# =========================== CONFIG (edit here) ===========================
CLASS_NAMES = ['Car', 'Truck', 'Pedestrian', 'Cyclist']
CLASS_ALIASES = {'car': 0, 'vehicle': 0, 'van': 0,
                 'truck': 1, 'bus': 1,
                 'pedestrian': 2, 'walker': 2, 'person': 2,
                 'bicycle': 3, 'motorcycle': 3, 'cyclist': 3, 'bike': 3}
MIN_PTS = 5           # drop GT boxes containing fewer LiDAR points than this (undetectable)

PC_RANGE = (-51.2, -51.2, -4.0, 51.2, 51.2, 2.0)   # x0,y0,z0,x1,y1,z1 (LiDAR frame, 360 deg)
VOXEL = (0.2, 0.2)                                 # pillar size x,y  -> 512 x 512 grid
MAX_PTS = 32                                       # points per pillar
MAX_PILLARS_TRAIN = 16000
MAX_PILLARS_TEST = 40000

# --- label convention switches (verify with:  python lidar_train.py --check) ---
Z_IS_BOTTOM = False   # True if label z is the bottom of the box (KITTI camera style)
YAW_SIGN = 1.0        # set -1.0 if boxes are rotated the wrong way in the --check images
YAW_OFFSET = 0.0      # e.g. math.pi/2 if boxes are 90 deg off in the --check images

POS_THR = (0.60, 0.60, 0.50, 0.50)   # per class anchor positive IoU  (Car, Truck, Ped, Cyclist)
NEG_THR = (0.45, 0.45, 0.35, 0.35)   # per class anchor negative IoU
DIR_OFFSET = 0.78539816
# ===========================================================================

NX = int(round((PC_RANGE[3] - PC_RANGE[0]) / VOXEL[0]))
NY = int(round((PC_RANGE[4] - PC_RANGE[1]) / VOXEL[1]))
OUT_STRIDE = 2
NUM_CLASSES = len(CLASS_NAMES)
ANCHORS_PER_LOC = NUM_CLASSES * 2          # 2 rotations (0, 90deg) per class


# ------------------------------- helpers ---------------------------------
def limit_period(v, offset=0.5, period=math.pi):
    if torch.is_tensor(v):
        return v - torch.floor(v / period + offset) * period
    return v - np.floor(v / period + offset) * period


def sort_ids(ids):
    return sorted(ids, key=lambda s: (len(s), s))


def get_frame_ids(root):
    vdir = os.path.join(root, 'velodyne')
    return sort_ids({os.path.splitext(f)[0] for f in os.listdir(vdir)
                     if f.endswith('.bin') or f.endswith('.npy')})


def make_split(root, out_json, val_frac=0.2, gap=20):
    """Temporal split: first (1-val_frac) of the run = train, last val_frac = val,
    with a `gap` of frames dropped in between (consecutive frames are near-identical)."""
    ids = get_frame_ids(root)
    n = len(ids)
    n_tr = int(n * (1 - val_frac))
    split = {'train': ids[:n_tr], 'val': ids[n_tr + gap:]}
    with open(out_json, 'w') as f:
        json.dump(split, f)
    return split


def load_points(path):
    if path.endswith('.npy'):
        pts = np.load(path)
    else:
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts.astype(np.float32)


def load_labels(path):
    """Two formats are supported (LiDAR-frame boxes):
       CARLA logger : cls x y z l w h yaw [num_points]
       KITTI        : cls trunc occ alpha x1 y1 x2 y2 h w l x y z ry"""
    boxes, cls = [], []
    if os.path.exists(path):
        x0, y0, _, x1, y1, _ = PC_RANGE
        for line in open(path):
            f = line.split()
            c = CLASS_ALIASES.get(f[0].lower()) if f else None
            if c is None:
                continue
            try:
                if len(f) >= 15:
                    h, w, l = map(float, f[8:11])
                    x, y, z = map(float, f[11:14])
                    ry, npts = float(f[14]), None
                elif len(f) >= 8:
                    x, y, z, l, w, h, ry = map(float, f[1:8])
                    npts = int(float(f[8])) if len(f) > 8 else None
                else:
                    continue
            except ValueError:
                continue
            if npts is not None and npts < MIN_PTS:
                continue
            if Z_IS_BOTTOM:
                z += h / 2
            yaw = YAW_SIGN * ry + YAW_OFFSET
            if not (x0 <= x < x1 and y0 <= y < y1):
                continue
            boxes.append([x, y, z, l, w, h, yaw])
            cls.append(c)
    return (np.array(boxes, np.float32).reshape(-1, 7), np.array(cls, np.int64))


def augment(pts, boxes):
    if np.random.rand() < 0.5:                      # mirror y
        pts[:, 1] *= -1; boxes[:, 1] *= -1; boxes[:, 6] *= -1
    a = np.random.uniform(-np.pi / 4, np.pi / 4)    # global rotation
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]], np.float32)
    pts[:, :2] = pts[:, :2] @ R.T
    boxes[:, :2] = boxes[:, :2] @ R.T
    boxes[:, 6] += a
    sc = np.random.uniform(0.95, 1.05)              # global scale
    pts[:, :3] *= sc
    boxes[:, :6] *= sc
    return pts, boxes


# ------------------------------- dataset ---------------------------------
class LidarDataset(Dataset):
    def __init__(self, root, ids, label_dir='labels_3d', train=False):
        self.root, self.ids, self.label_dir, self.train = root, ids, label_dir, train

    def __len__(self):
        return len(self.ids)

    def _pts_path(self, i):
        p = os.path.join(self.root, 'velodyne', i + '.bin')
        return p if os.path.exists(p) else os.path.join(self.root, 'velodyne', i + '.npy')

    def __getitem__(self, k):
        i = self.ids[k]
        pts = load_points(self._pts_path(i))
        boxes, cls = load_labels(os.path.join(self.root, self.label_dir, i + '.txt'))
        if self.train:
            pts, boxes = augment(pts, boxes)
            x0, y0, _, x1, y1, _ = PC_RANGE
            m = (boxes[:, 0] >= x0) & (boxes[:, 0] < x1) & (boxes[:, 1] >= y0) & (boxes[:, 1] < y1)
            boxes, cls = boxes[m], cls[m]
            pts = pts[np.random.permutation(len(pts))]
        return {'id': i, 'points': torch.from_numpy(pts),
                'boxes': torch.from_numpy(boxes), 'cls': torch.from_numpy(cls)}

    def anchor_stats(self):
        """Mean [l, w, h, z] per class from the training labels -> anchor sizes."""
        acc = [[] for _ in range(NUM_CLASSES)]
        for i in self.ids:
            b, c = load_labels(os.path.join(self.root, self.label_dir, i + '.txt'))
            for k in range(NUM_CLASSES):
                if (c == k).any():
                    acc[k].append(b[c == k][:, [3, 4, 5, 2]])
        default = [[4.5, 1.9, 1.6, -1.6], [7.0, 2.5, 2.8, -1.0], [0.8, 0.6, 1.8, -1.5], [1.7, 0.6, 1.6, -1.6]]
        sizes = []
        for k in range(NUM_CLASSES):
            sizes.append(np.concatenate(acc[k]).mean(0).tolist() if acc[k] else default[k])
        counts = [int(sum(len(a) for a in acc[k])) for k in range(NUM_CLASSES)]
        return sizes, counts


def collate(batch):
    return {'id': [b['id'] for b in batch], 'points': [b['points'] for b in batch],
            'boxes': [b['boxes'] for b in batch], 'cls': [b['cls'] for b in batch]}


# ------------------------------ voxelizer --------------------------------
def voxelize(pts, max_pts, max_pillars):
    """pts [N,C] (torch, any device) -> voxels [P,max_pts,C], num [P], coords [P,2]=(iy,ix)"""
    x0, y0, z0, x1, y1, z1 = PC_RANGE
    m = ((pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
         & (pts[:, 2] >= z0) & (pts[:, 2] < z1))
    pts = pts[m]
    ix = ((pts[:, 0] - x0) / VOXEL[0]).long().clamp(0, NX - 1)
    iy = ((pts[:, 1] - y0) / VOXEL[1]).long().clamp(0, NY - 1)
    ids = iy * NX + ix
    ids_s, order = torch.sort(ids)
    pts = pts[order]
    uniq, inv, counts = torch.unique_consecutive(ids_s, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(ids_s), device=pts.device) - starts[inv]
    P = len(uniq)
    if P > max_pillars:
        sel = torch.randperm(P, device=pts.device)[:max_pillars]
        remap = torch.full((P,), -1, dtype=torch.long, device=pts.device)
        remap[sel] = torch.arange(max_pillars, device=pts.device)
        inv = remap[inv]
        keep = (rank < max_pts) & (inv >= 0)
        uniq, counts = uniq[sel], counts[sel]
    else:
        keep = rank < max_pts
    vox = pts.new_zeros((len(uniq), max_pts, pts.shape[1]))
    vox[inv[keep], rank[keep]] = pts[keep]
    num = counts.clamp(max=max_pts)
    coords = torch.stack([uniq // NX, uniq % NX], 1)
    return vox, num, coords


# -------------------------------- model ----------------------------------
class PFN(nn.Module):
    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.lin = nn.Linear(in_ch + 5, out_ch, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, vox, num, coords):          # coords: [P,3] = (b, iy, ix)
        P, N, _ = vox.shape
        mask = (torch.arange(N, device=vox.device)[None, :] < num[:, None]).unsqueeze(-1).to(vox.dtype)
        mean = vox[:, :, :3].sum(1, keepdim=True) / num.view(-1, 1, 1).clamp(min=1).to(vox.dtype)
        f_c = vox[:, :, :3] - mean
        cx = (coords[:, 2].to(vox.dtype) + 0.5) * VOXEL[0] + PC_RANGE[0]
        cy = (coords[:, 1].to(vox.dtype) + 0.5) * VOXEL[1] + PC_RANGE[1]
        f_p = torch.stack([vox[:, :, 0] - cx[:, None], vox[:, :, 1] - cy[:, None]], -1)
        f = torch.cat([vox, f_c, f_p], -1) * mask
        f = self.lin(f)
        f = F.relu(self.bn(f.permute(0, 2, 1)).permute(0, 2, 1))
        return (f * mask).max(1)[0]


def _block(cin, cout, n, stride):
    layers = [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]
    for _ in range(n - 1):
        layers += [nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]
    return nn.Sequential(*layers)


def _up(cin, cout, k):
    if k == 1:
        return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, k, k, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class PointPillars(nn.Module):
    def __init__(self, anchor_sizes, num_pt_feats=4):
        """anchor_sizes: per class [l, w, h, z_center].
        num_pt_feats: 4 = x,y,z,intensity.  PointPainting later -> 4 + num_seg_classes."""
        super().__init__()
        self.num_pt_feats = num_pt_feats
        self.anchor_sizes = anchor_sizes
        self.pfn = PFN(num_pt_feats, 64)
        self.b1, self.b2, self.b3 = _block(64, 64, 3, 2), _block(64, 128, 5, 2), _block(128, 256, 5, 2)
        self.u1, self.u2, self.u3 = _up(64, 128, 1), _up(128, 128, 2), _up(256, 128, 4)
        A = ANCHORS_PER_LOC
        self.cls_head = nn.Conv2d(384, A, 1)
        self.box_head = nn.Conv2d(384, A * 7, 1)
        self.dir_head = nn.Conv2d(384, A * 2, 1)
        nn.init.constant_(self.cls_head.bias, -math.log(99.0))
        nn.init.normal_(self.box_head.weight, std=0.001)
        anchors, cls_a = make_anchors(anchor_sizes)
        self.register_buffer('anchors', anchors, persistent=False)
        self.register_buffer('cls_a', cls_a, persistent=False)

    def forward(self, points):
        dev = next(self.parameters()).device
        vs, ns, cs = [], [], []
        mp = MAX_PILLARS_TRAIN if self.training else MAX_PILLARS_TEST
        for b, p in enumerate(points):
            v, n, c = voxelize(p.to(dev, non_blocking=True), MAX_PTS, mp)
            vs.append(v); ns.append(n)
            cs.append(torch.cat([torch.full((len(c), 1), b, dtype=torch.long, device=dev), c], 1))
        vox, num, coords = torch.cat(vs), torch.cat(ns), torch.cat(cs)
        B = len(points)
        feats = self.pfn(vox, num, coords)
        flat = coords[:, 0] * NY * NX + coords[:, 1] * NX + coords[:, 2]
        canvas = feats.new_zeros(B * NY * NX, feats.shape[1])
        canvas[flat] = feats
        x = canvas.view(B, NY, NX, -1).permute(0, 3, 1, 2).contiguous()
        x1 = self.b1(x); x2 = self.b2(x1); x3 = self.b3(x2)
        x = torch.cat([self.u1(x1), self.u2(x2), self.u3(x3)], 1)
        cls = self.cls_head(x).permute(0, 2, 3, 1).reshape(B, -1)
        box = self.box_head(x).permute(0, 2, 3, 1).reshape(B, -1, 7)
        dr = self.dir_head(x).permute(0, 2, 3, 1).reshape(B, -1, 2)
        return cls, box, dr


def make_anchors(sizes):
    H, W = NY // OUT_STRIDE, NX // OUT_STRIDE
    ys = PC_RANGE[1] + (torch.arange(H).float() + 0.5) * VOXEL[1] * OUT_STRIDE
    xs = PC_RANGE[0] + (torch.arange(W).float() + 0.5) * VOXEL[0] * OUT_STRIDE
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    anc = []
    for (l, w, h, z) in sizes:
        for rot in (0.0, math.pi / 2):
            anc.append(torch.stack([xx, yy, torch.full_like(xx, z), torch.full_like(xx, l),
                                    torch.full_like(xx, w), torch.full_like(xx, h),
                                    torch.full_like(xx, rot)], -1))
    anchors = torch.stack(anc, 2).reshape(-1, 7)                      # (H,W,A) order
    cls_a = torch.tensor([c for c in range(NUM_CLASSES) for _ in range(2)]).repeat(H * W)
    return anchors, cls_a


# ------------------------------ box coding -------------------------------
def bev_aabb(b):
    c, s = torch.abs(torch.cos(b[:, 6])), torch.abs(torch.sin(b[:, 6]))
    ex = b[:, 3] * c + b[:, 4] * s
    ey = b[:, 3] * s + b[:, 4] * c
    return torch.stack([b[:, 0] - ex / 2, b[:, 1] - ey / 2, b[:, 0] + ex / 2, b[:, 1] + ey / 2], 1)


def encode(a, g):
    da = torch.sqrt(a[:, 3] ** 2 + a[:, 4] ** 2)
    return torch.stack([(g[:, 0] - a[:, 0]) / da, (g[:, 1] - a[:, 1]) / da, (g[:, 2] - a[:, 2]) / a[:, 5],
                        torch.log(g[:, 3] / a[:, 3]), torch.log(g[:, 4] / a[:, 4]),
                        torch.log(g[:, 5] / a[:, 5]), g[:, 6] - a[:, 6]], 1)


def decode_boxes(r, a):
    da = torch.sqrt(a[:, 3] ** 2 + a[:, 4] ** 2)
    r = r.float()
    return torch.stack([r[:, 0] * da + a[:, 0], r[:, 1] * da + a[:, 1], r[:, 2] * a[:, 5] + a[:, 2],
                        torch.exp(r[:, 3].clamp(-3, 3)) * a[:, 3], torch.exp(r[:, 4].clamp(-3, 3)) * a[:, 4],
                        torch.exp(r[:, 5].clamp(-3, 3)) * a[:, 5], r[:, 6] + a[:, 6]], 1)


def dir_label(yaw):
    return (limit_period(yaw - DIR_OFFSET, 0.0, 2 * math.pi) >= math.pi).long()


# --------------------------------- loss ----------------------------------
def focal_loss(logits, tgt, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
    pt = p * tgt + (1 - p) * (1 - tgt)
    at = alpha * tgt + (1 - alpha) * (1 - tgt)
    return (at * (1 - pt) ** gamma * ce).sum()


def compute_loss(out, model, gt_boxes, gt_cls):
    cls_p, box_p, dir_p = [o.float() for o in out]
    dev = cls_p.device
    B, N = cls_p.shape
    anchors, cls_a = model.anchors, model.cls_a
    pos_thr = torch.tensor(POS_THR, device=dev)[cls_a]
    neg_thr = torch.tensor(NEG_THR, device=dev)[cls_a]
    a_bev = bev_aabb(anchors)
    labels = torch.full((B, N), -1.0, device=dev)
    box_t = torch.zeros(B, N, 7, device=dev)
    dir_t = torch.zeros(B, N, dtype=torch.long, device=dev)
    with torch.no_grad():
        for b in range(B):
            gt, gc = gt_boxes[b].to(dev), gt_cls[b].to(dev)
            if len(gt) == 0:
                labels[b] = 0
                continue
            iou = torchvision.ops.box_iou(a_bev, bev_aabb(gt))
            iou[cls_a[:, None] != gc[None, :]] = 0
            mx, arg = iou.max(1)
            lab = torch.full((N,), -1.0, device=dev)
            lab[mx < neg_thr] = 0
            lab[mx >= pos_thr] = 1
            best = iou.argmax(0)
            gi = torch.arange(len(gt), device=dev)
            ok = iou[best, gi] > 0.05
            lab[best[ok]] = 1
            arg[best[ok]] = gi[ok]
            labels[b] = lab
            matched = gt[arg]
            box_t[b] = encode(anchors, matched)
            dir_t[b] = dir_label(matched[:, 6])
    pos = labels == 1
    npos = max(int(pos.sum()), 1)
    valid = labels >= 0
    l_cls = focal_loss(cls_p[valid], labels[valid]) / npos
    if pos.any():
        bp, bt = box_p[pos], box_t[pos]
        sp = torch.sin(bp[:, 6:7]) * torch.cos(bt[:, 6:7])
        st = torch.cos(bp[:, 6:7]) * torch.sin(bt[:, 6:7])
        l_box = F.smooth_l1_loss(torch.cat([bp[:, :6], sp], 1), torch.cat([bt[:, :6], st], 1),
                                 beta=1 / 9, reduction='sum') / npos
        l_dir = F.cross_entropy(dir_p[pos], dir_t[pos], reduction='sum') / npos
    else:
        l_box = l_dir = cls_p.sum() * 0
    total = l_cls + 2.0 * l_box + 0.2 * l_dir
    return total, {'cls': float(l_cls), 'box': float(l_box), 'dir': float(l_dir), 'npos': npos}


# ------------------------------- inference -------------------------------
@torch.no_grad()
def decode_predictions(out, model, score_thr=0.1, pre_nms=2000, nms_iou=0.25, max_det=100):
    cls_p, box_p, dir_p = out
    results = []
    for b in range(cls_p.shape[0]):
        sc = torch.sigmoid(cls_p[b].float())
        idx = (sc > score_thr).nonzero().squeeze(1)
        if len(idx) > pre_nms:
            idx = idx[sc[idx].topk(pre_nms)[1]]
        if len(idx) == 0:
            results.append({'boxes': np.zeros((0, 7), np.float32), 'scores': np.zeros(0, np.float32),
                            'cls': np.zeros(0, np.int64)})
            continue
        a = model.anchors[idx]
        boxes = decode_boxes(box_p[b][idx], a)
        dl = dir_p[b][idx].argmax(-1)
        rot = limit_period(boxes[:, 6] - DIR_OFFSET, 0.0, math.pi)
        boxes[:, 6] = rot + DIR_OFFSET + math.pi * dl.float()
        s, c = sc[idx], model.cls_a[idx]
        keep = torchvision.ops.batched_nms(bev_aabb(boxes), s, c, nms_iou)[:max_det]
        results.append({'boxes': boxes[keep].cpu().numpy(), 'scores': s[keep].cpu().numpy(),
                        'cls': c[keep].cpu().numpy()})
    return results


def build_model_from_ckpt(ckpt, device):
    model = PointPillars(ckpt['anchor_sizes'], ckpt.get('num_pt_feats', 4)).to(device)
    model.load_state_dict(ckpt['model'])
    return model.eval()


# ----------------------- rotated IoU + AP (KITTI R40) --------------------
def bev_corners(b):
    x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
    dx, dy = l / 2, w / 2
    local = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])      # CCW
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def _inside(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0


def _inter(p, q, a, b):
    x1, y1 = p; x2, y2 = q; x3, y3 = a; x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return q
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def poly_clip(subj, clip):
    out = [tuple(p) for p in subj]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if _inside(e, a, b):
                if not _inside(s, a, b):
                    out.append(_inter(s, e, a, b))
                out.append(e)
            elif _inside(s, a, b):
                out.append(_inter(s, e, a, b))
            s = e
    return out


def poly_area(p):
    if len(p) < 3:
        return 0.0
    p = np.array(p)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def pair_ious(dets, gts):
    """returns (bev_iou [nd,ng], iou3d [nd,ng])"""
    nd, ng = len(dets), len(gts)
    bev, d3 = np.zeros((nd, ng)), np.zeros((nd, ng))
    if nd == 0 or ng == 0:
        return bev, d3
    cd = np.linalg.norm(dets[:, None, :2] - gts[None, :, :2], axis=-1)
    rd = np.hypot(dets[:, 3], dets[:, 4])[:, None] / 2 + np.hypot(gts[:, 3], gts[:, 4])[None] / 2
    pd = [bev_corners(d) for d in dets]
    pg = [bev_corners(g) for g in gts]
    for i, j in np.argwhere(cd < rd):
        inter = poly_area(poly_clip(pd[i], pg[j]))
        ad, ag = dets[i, 3] * dets[i, 4], gts[j, 3] * gts[j, 4]
        bev[i, j] = inter / (ad + ag - inter + 1e-9)
        zo = max(0.0, min(dets[i, 2] + dets[i, 5] / 2, gts[j, 2] + gts[j, 5] / 2)
                 - max(dets[i, 2] - dets[i, 5] / 2, gts[j, 2] - gts[j, 5] / 2))
        i3 = inter * zo
        d3[i, j] = i3 / (ad * dets[i, 5] + ag * gts[j, 5] - i3 + 1e-9)
    return bev, d3


def average_precision(preds, gts, ious, c, thr, mode):
    """preds: list of dict(boxes,scores,cls); gts: list of dict(boxes,cls); mode 0=BEV 1=3D"""
    all_s, all_tp, npos = [], [], 0
    for f in range(len(preds)):
        gm = gts[f]['cls'] == c
        dm = preds[f]['cls'] == c
        npos += int(gm.sum())
        if dm.sum() == 0:
            continue
        sc = preds[f]['scores'][dm]
        M = ious[f][mode][np.ix_(dm, gm)]
        used = np.zeros(M.shape[1], bool)
        tp = np.zeros(len(sc))
        for k in np.argsort(-sc):
            if M.shape[1] > 0:
                j = M[k].argmax()
                if M[k, j] >= thr and not used[j]:
                    used[j] = True
                    tp[k] = 1
        all_s.append(sc); all_tp.append(tp)
    if npos == 0:
        return float('nan')
    if not all_s:
        return 0.0
    s, tp = np.concatenate(all_s), np.concatenate(all_tp)
    o = np.argsort(-s)
    tp = tp[o]
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = tpc / npos, tpc / np.maximum(tpc + fpc, 1e-9)
    ap = 0.0
    for r in np.linspace(1 / 40, 1, 40):
        m = prec[rec >= r]
        ap += m.max() if m.size else 0.0
    return ap / 40 * 100


# ------------------------------ visualisation ----------------------------
def draw_bev(pts, gt=None, pred=None, size=800):
    """x forward = up, y = right. GT green, predictions red."""
    import cv2
    x0, y0, _, x1, y1, _ = PC_RANGE
    s = size / (x1 - x0)
    img = np.zeros((size, size, 3), np.uint8)
    u = ((pts[:, 1] - y0) * s).astype(int)
    v = ((x1 - pts[:, 0]) * s).astype(int)
    m = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    vm, um = v[m], u[m]
    for dy in ((0, 1) if size >= 1200 else (0,)):
        for dx in ((0, 1) if size >= 1200 else (0,)):
            img[np.clip(vm + dy, 0, size - 1), np.clip(um + dx, 0, size - 1)] = (170, 170, 170)

    def poly(b, col):
        c = bev_corners(b)
        px = np.stack([(c[:, 1] - y0) * s, (x1 - c[:, 0]) * s], 1).astype(np.int32)
        cv2.polylines(img, [px], True, col, 2)
        reach = b[3] / 2 + 1.5                       # arrow sticks out 1.5 m beyond the front edge
        f = np.array([b[0] + reach * np.cos(b[6]), b[1] + reach * np.sin(b[6])])
        c0 = (int((b[1] - y0) * s), int((x1 - b[0]) * s))
        cv2.arrowedLine(img, c0, (int((f[1] - y0) * s), int((x1 - f[0]) * s)), col, 2, tipLength=0.35)
        cv2.circle(img, c0, 4, col, -1)              # dot = box centre (tail of the arrow)
    if gt is not None:
        for b in gt:
            poly(b, (0, 255, 0))
    if pred is not None:
        for b in pred:
            poly(b, (0, 0, 255))
    cv2.circle(img, (int((0 - y0) * s), int((x1 - 0) * s)), 4, (255, 255, 0), -1)   # ego
    return img