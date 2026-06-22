import torch
from ultralytics import YOLO

def main():
    torch.manual_seed(42)

    DATA_PATH = r'D:\suju\yolov8_garbage_new\garbage.yaml'

    train_params = {
        "data": DATA_PATH,
        "epochs": 100,
        "imgsz": 640,
        "workers": 4,
        "device": 0,
        "amp": True,
        "cache": True,
        "rect": False,
        "scale": 0.25,
        "translate": 0.1,
        "patience": 20,
        "cos_lr": True,
        "mosaic": 1.0,
        "mixup": 0.15,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "save": True,
        "plots": True,
        "verbose": True,
        "resume": False,
        "batch": 16,            # 8GB 显存建议 16
        "name": "YOLOv10s（1）"
    }

    print("开始训练 YOLOv10s...")
    model = YOLO("yolov10s.pt")
    model.train(**train_params)

if __name__ == '__main__':
    main()