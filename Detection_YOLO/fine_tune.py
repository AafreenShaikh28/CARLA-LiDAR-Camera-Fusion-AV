"""
finetune.py

python fine_tune.py --weights yolo26n.pt --data yolo_eval/Town03/data.yaml --epochs 100

Fine-tunes a pretrained YOLO model on the CARLA dataset, with settings
picked specifically because bicycle/motorcycle are small and under-
represented classes:

  - imgsz=960 (not the 640 default): small-object recall is sensitive to
    input resolution; this costs more VRAM/time per epoch but is the
    single highest-leverage knob for exactly the failure mode we diagnosed
    (motorcycles/bicycles either invisible or inconsistently localized).
  - lower lr0 + cosine schedule: we're fine-tuning from a pretrained
    checkpoint, not training from scratch -- a high initial LR risks
    forgetting what already works well (car, truck).
  - close_mosaic near the end of training: mosaic augmentation helps early
    but can distort small-object aspect ratios; disabling it for the final
    epochs lets the model settle on realistic box shapes.
  - patience for early stopping, since epoch count here is just a ceiling.

# Run this locally where ultralytics + your dataset + data.yaml live.
"""

import argparse
from ultralytics import YOLO

CFG = dict(
    base_weights='yolov8n.pt',
    data_yaml='data.yaml',
    epochs=100,
    imgsz=960,
    batch=16,
    patience=20,
    lr0=0.001,
    cos_lr=True,
    close_mosaic=10,
    project='runs/detect',
    name='carla_finetune',
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=CFG['base_weights'])
    ap.add_argument('--data', default=CFG['data_yaml'])
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    ap.add_argument('--imgsz', type=int, default=CFG['imgsz'])
    ap.add_argument('--batch', type=int, default=CFG['batch'])
    args = ap.parse_args()

    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=CFG['patience'],
        lr0=CFG['lr0'],
        cos_lr=CFG['cos_lr'],
        close_mosaic=CFG['close_mosaic'],
        project=CFG['project'],
        name=CFG['name'],
    )

    # Re-run standard validation so you get a fresh per-class mAP table
    # against the same val split, for direct before/after comparison
    # against your original eval run.
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print('\nPer-class mAP50 after fine-tuning:')
    for cls_id, ap50 in enumerate(metrics.box.ap50):
        name = metrics.names.get(cls_id, str(cls_id))
        print(f'  {name}: {ap50:.3f}')


if __name__ == '__main__':
    main()


