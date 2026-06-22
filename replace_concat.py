import torch
from ultralytics import YOLO
from custom_modules import WeightedChannelFusion


def replace_concats(model):
    """递归遍历模型，将所有的 Concat 替换为 WeightedChannelFusion"""
    # 注意：YOLO 的 Concat 定义在 ultralytics.nn.modules.Concat
    from ultralytics.nn.modules import Concat

    def replace_module(module):
        for name, child in module.named_children():
            if isinstance(child, Concat):
                # 需要知道该 Concat 的输入通道数。这里简化处理：从上一层的输出推断。
                # 由于动态推断复杂，我们采用一个更简单的方法：直接替换，在第一次前向传播时自动适应。
                # 因此我们创建一个包装类，延迟构建真正的 WeightedChannelFusion。
                # 但为了简单，我们只替换特定位置的 Concat（例如 Neck 中的几个）。
                # 更可靠的方法：打印模型结构，手动替换指定层。
                print(f"Found Concat: {name}, will replace manually later.")
                # 这里不自动替换，因为需要通道数信息。我们改用手动替换指定索引。
            else:
                replace_module(child)

    # 打印模型结构，以便手动定位
    print(model.model.model)  # 查看各层的索引和类型
    # 根据打印结果，找到 Neck 部分的 Concat 层的位置（通常在索引 11, 14, 17, 20 附近）
    # 手动替换示例：
    # model.model.model[11] = WeightedChannelFusion(ch1=?, ch2=?)
    # 您需要根据打印出来的上一层输出通道数填写 ch1, ch2。


def main():
    # 加载预训练模型
    model = YOLO('yolo26s.pt')
    detection_model = model.model

    # 打印模型结构，找出 Concat 层位置
    print("Model structure:")
    for i, layer in enumerate(detection_model.model):
        print(f"{i:3d}: {layer}")

    # 手动替换（以下索引和通道数为示例，请根据实际打印结果修改）
    # 通常 YOLOv8/v26 的 Neck 中有 4 个 Concat，分别位于第 11, 14, 17, 20 层附近
    # 以实际输出为准，例如：
    # replace_index = [11, 14, 17, 20]
    # for idx in replace_index:
    #     conv = detection_model.model[idx]
    #     # 需要知道上一层的输出通道数，这里无法自动获取，您需要手动填写
    #     # 假设 ch1 和 ch2 分别为 256 和 256（示例）
    #     detection_model.model[idx] = WeightedChannelFusion(256, 256)

    print("Please manually set the replacement indices and channel numbers based on the printed structure.")


if __name__ == '__main__':
    main()