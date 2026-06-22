import torch
from ultralytics import YOLO

def main():
    torch.manual_seed(42)
    DATA_PATH = r'D:\suju\yolov8_garbage_new\garbage.yaml'

    train_params = {
        "data": DATA_PATH,
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "device": 0,
        "amp": True,
        "cache": False,  # 避免内存溢出
        "scale": 0.25,
        "translate": 0.1,
        "patience": 20,
        "cos_lr": True,
        "mosaic": 1.0,
        "mixup": 0.15,
        "lr0": 0.01,
        "lrf": 0.01,
        "rect": False,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "save": True,
        "plots": True,
        "verbose": True,
        "resume": False,         # 改为 False，全新训练

        "name": "YOLO26s（1）"
    }
    print("开始全新训练 YOLO26s...")
    model = YOLO("yolo26s.pt")
    model.train(**train_params)
if __name__ == '__main__':
    main()
