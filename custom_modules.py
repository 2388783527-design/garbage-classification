import torch
import torch.nn as nn

class WeightedChannelFusion(nn.Module):
    """
    可学习的通道加权融合模块，替代 Concat。
    输入：两个特征图 x1, x2
    输出：融合后的特征图，通道数为 x1.channels + x2.channels
    通过一个轻量级的通道注意力网络对拼接后的特征进行重新加权。
    """
    def __init__(self, ch1, ch2, reduction=16):
        super().__init__()
        self.ch1 = ch1
        self.ch2 = ch2
        total_c = ch1 + ch2
        # 通道注意力（类似 SE 模块）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_c, total_c // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_c // reduction, total_c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 1. 拼接
        out = torch.cat([x1, x2], dim=1)
        # 2. 计算通道权重
        weight = self.se(out)
        # 3. 加权输出
        return out * weight