import torch
import torch.nn as nn
from ultralytics import YOLO

class WeightedChannelFusion(nn.Module):
    """可学习的通道加权融合模块，替代 Concat"""
    def __init__(self, reduction=16):
        super().__init__()
        self.reduction = reduction

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        B, C, H, W = out.shape
        if not hasattr(self, 'se'):
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(C, max(1, C // self.reduction), 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, C // self.reduction), C, 1, bias=False),
                nn.Sigmoid()
            ).to(out.device)
        weight = self.se(out)
        return out * weight

def replace_concats(model, concat_indices):
    detection_model = model.model
    for idx in concat_indices:
        if idx < len(detection_model.model):
            original_layer = detection_model.model[idx]
            detection_model.model[idx] = WeightedChannelFusion()
            print(f"已替换索引 {idx} 处的 {type(original_layer).__name__} 为 WeightedChannelFusion")
        else:
            print(f"警告：索引 {idx} 超出范围")

def main():
    torch.manual_seed(42)

    # 加载预训练模型
    model = YOLO('yolo26s.pt')
    # 替换四个 Concat 层（索引需根据模型结构确认）
    concat_indices = [12, 15, 18, 21]   # 适用于 yolo26s.pt 的 Concat 索引
    replace_concats(model, concat_indices)

    # 直接训练（未冻结骨干，全参数训练）
    model.train(
        data=r'D:\suju\yolov8_garbage_new\garbage.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        amp=True,
        cache=False,
        scale=0.25,
        translate=0.1,
        patience=20,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.15,
        lr0=0.01,
        lrf=0.01,
        rect=False,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        save=True,
        plots=True,
        verbose=True,
        resume=False,
        name='YOLO26s_weighted2',   # 训练结果保存目录名
    )

if __name__ == '__main__':
    main()