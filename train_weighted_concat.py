import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect  # 用于检测头部

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
    torch.manual_seed(42)  # 固定随机种子，保证可复现

    # ==================== 1. 加载预训练模型并替换 Concat ====================
    model = YOLO('yolo26s.pt')
    concat_indices = [12, 15, 18, 21]  # 根据实际模型打印确认的索引
    replace_concats(model, concat_indices)

    # 可选：保存初始修改模型（不含训练权重）
    model.save('yolo26s_weighted_init.pt')

    # ==================== 2. 冻结骨干 + 解冻新模块和检测头 ====================
    # 先记录新模块的参数（使用 id 唯一标识）
    new_param_ids = set()
    for idx in concat_indices:
        if idx < len(model.model.model):
            for p in model.model.model[idx].parameters():
                new_param_ids.add(id(p))

    # 冻结所有参数
    for param in model.model.parameters():
        param.requires_grad = False

    # 解冻新模块
    for idx in concat_indices:
        if idx < len(model.model.model):
            for p in model.model.model[idx].parameters():
                p.requires_grad = True

    # 解冻检测头（Detect 层）
    for m in model.model.modules():
        if isinstance(m, Detect):
            for p in m.parameters():
                p.requires_grad = True

    # 验证可训练参数数量
    trainable_count = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"阶段一可训练参数: {trainable_count}")
    if trainable_count == 0:
        raise RuntimeError("阶段一没有可训练参数，请检查冻结/解冻逻辑或 Concat 索引。")

    # ==================== 3. 阶段一：快速预训练新模块 ====================
    print("阶段一：快速预训练新模块（5 epochs）...")
    model.train(
        data=r'D:\suju\yolov8_garbage_new\garbage.yaml',
        epochs=5,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        amp=True,
        cache=False,
        scale=0.25,
        translate=0.1,
        lr0=0.01,            # 新模块用较大学习率快速适应
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=False,        # 短训练不用余弦退火
        mosaic=1.0,
        mixup=0.0,           # 初期关闭 mixup 减少干扰
        patience=20,
        save=False,
        plots=False,
        verbose=True,
        exist_ok=True,
        name='YOLO26s_weighted_stage1_again'
    )

    # ==================== 阶段二：重新加载模型 + 全参数微调 ====================
    print("阶段二：重新加载阶段一模型，解冻全部参数并微调...")

    # 从阶段一保存的 last.pt 重新加载模型（保留训练好的 WeightedChannelFusion 权重）
    model = YOLO('D:/suju/runs/detect/YOLO26s_weighted_stage1_again/weights/last.pt')

    # 再次执行 Concat 替换（确保加载后的模型结构正确）
    replace_concats(model, [12, 15, 18, 21])

    # 解冻所有参数
    for param in model.model.parameters():
        param.requires_grad = True

    # 验证可训练参数数量
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"阶段二可训练参数: {trainable}")

    # 开始全参数微调
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
        lr0=0.01,
        lrf=0.01,
        rect=False,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        close_mosaic=20,
        save=True,
        plots=True,
        verbose=True,
        resume=False,
        name='YOLO26s_weighted_final'
    )

if __name__ == '__main__':
    main()