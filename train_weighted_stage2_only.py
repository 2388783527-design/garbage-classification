import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

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

    # 1. 从阶段一保存的 best.pt 重新加载模型
    stage1_model_path = 'D:/suju/runs/detect/YOLO26s_weighted_stage1_again/weights/best.pt'
    model = YOLO(stage1_model_path)
    print(f"成功加载阶段一模型: {stage1_model_path}")

    # 2. 再次执行 Concat 替换，确保加载后的模型结构正确
    concat_indices = [12, 15, 18, 21]
    replace_concats(model, concat_indices)

    # 3. 解冻所有参数
    for param in model.model.parameters():
        param.requires_grad = True

    # 验证可训练参数数量
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"阶段二可训练参数: {trainable}")

    # 4. 开始全参数微调
    print("阶段二：全参数微调（200 epochs）...")
    model.train(
        data=r'D:\suju\yolov8_garbage_new\garbage.yaml',
        epochs=200,
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
        lr0=0.01,           # 更小学习率，保护预训练权重
        lrf=0.01,
        rect=False,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        close_mosaic=20,     # 最后 20 轮关闭 Mosaic，提升泛化
        save=True,
        plots=True,
        verbose=True,
        resume=False,
        name='YOLO26s_weighted_final'
    )

if __name__ == '__main__':
    main()