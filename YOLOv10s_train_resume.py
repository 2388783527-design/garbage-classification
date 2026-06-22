from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/YOLOv8n_retrain2/weights/last.pt')
    model.train(resume=True, workers=10)   # 关闭缓存，降低 workers