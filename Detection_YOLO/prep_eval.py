import os, shutil, yaml
from ultralytics import YOLO

SRC = '../Data_collection_LIDAR/dataset/Town03'
DST = 'yolo_eval/Town03'

os.makedirs(f'{DST}/images/val', exist_ok=True)
os.makedirs(f'{DST}/labels/val', exist_ok=True)

n = 0
for f in os.listdir(f'{SRC}/image_2'):
    stem = os.path.splitext(f)[0]
    pairs = [(f'{SRC}/image_2/{f}',            f'{DST}/images/val/{f}'),
             (f'{SRC}/labels_yolo/{stem}.txt', f'{DST}/labels/val/{stem}.txt')]
    for s, d in pairs:
        if os.path.exists(d) or not os.path.exists(s):
            continue
        try:
            os.link(s, d)          # hard link: no extra disk
        except OSError:
            shutil.copy2(s, d)     # fallback if on a different drive
    n += 1

# Your GT uses COCO IDs, so the yaml must list all 80 COCO names
names = YOLO('yolov8n.pt').names
yaml.safe_dump({'path': os.path.abspath(DST),
                'train': 'images/val',     # required key; unused for val
                'val':   'images/val',
                'names': names},
               open(f'{DST}/data.yaml', 'w'), sort_keys=False)
print(f'{n} images linked, data.yaml written')