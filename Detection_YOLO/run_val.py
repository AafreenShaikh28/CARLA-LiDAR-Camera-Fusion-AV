from ultralytics import YOLO

if __name__ == '__main__':          
    model = YOLO('runs/detect/runs/detect/carla_finetune-10/weights/best.pt')
    m = model.val(data='yolo_eval/Town03/data.yaml',
                  imgsz=800, batch=16,
                  conf=0.001,       
                  iou=0.6,
                  plots=True,
                  project='runs/eval', name='pretrained_town03', exist_ok=True)

    print(f'\nmAP50    {m.box.map50:.3f}')
    print(f'mAP50-95 {m.box.map:.3f}\n')
    for i, c in enumerate(m.box.ap_class_index):
        print(f'{model.names[c]:15s} AP50 {m.box.ap50[i]:.3f}   AP50-95 {m.box.ap[i]:.3f}')