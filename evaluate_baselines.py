import os
import yaml
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import multiprocessing

# 设置中文字体（Windows常用SimHei，若没有则回退）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ================= 1. 配置区域 =================
# 五个基线模型的 best.pt 路径（请根据实际文件名修改！）
MODEL_PATHS = {
    'YOLOv8n': r'D:\suju\runs\detect\YOLOv8n（1）\weights\best.pt',
    'YOLOv8m': r'D:\suju\runs\detect\YOLOv8s（1）\weights\best.pt',
    'YOLOv10s': r'D:\suju\runs\detect\YOLOv10s（1）2\weights\best.pt',
    'YOLO26s': r'D:\suju\runs\detect\YOLO26s(1)2\weights\best.pt',

}

TEST_IMG_DIR = r'D:\suju\yolov8_garbage_new\images\test'
TEST_LABEL_DIR = r'D:\suju\yolov8_garbage_new\labels\test'
DATA_YAML = r'D:\suju\yolov8_garbage_new\garbage.yaml'
CONF_THRESHOLD = 0.5                      # 分类评估时的置信度阈值
OUTPUT_TABLE_IMG = r'D:\suju\baseline_comparison.png'  # 输出表格图片
# ==============================================

def collect_samples():
    """读取测试集图片路径与对应的类别标签"""
    with open(DATA_YAML, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    samples = []
    for img_file in os.listdir(TEST_IMG_DIR):
        if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(TEST_LABEL_DIR, label_file)
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                continue
            cls_id = int(line.split()[0])   # 取第一个物体的类别
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        samples.append((img_path, cls_id))
    return samples, data_cfg

def compute_classification_metrics(model, samples):
    """
    计算图片级分类准确率 (Accuracy)
    每张图：取置信度最高的检测框，与标签比较
    """
    y_true, y_pred = [], []
    for img_path, label in tqdm(samples, desc="分类评估", leave=False):
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            max_idx = np.argmax(confs)
            pred_cls = int(boxes.cls[max_idx])
        else:
            continue              # 未检测到物体，不计入Accuracy（可讨论）
        y_true.append(label)
        y_pred.append(pred_cls)
    if len(y_true) == 0:
        return 0.0
    acc = accuracy_score(y_true, y_pred)
    return acc


def plot_comparison_table(results_dict, save_path):
    """绘制带斜线表头的美化表格（四个模型，模型名作为数据列）"""
    models = list(results_dict.keys())
    # 第一列为“模型”
    col_labels = ['模型', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 'Accuracy']

    # 每行第一列是模型名称，后面是指标值
    cell_text = []
    for model in models:
        row = [
            model,
            f"{results_dict[model].get('mAP50', 0):.4f}",
            f"{results_dict[model].get('mAP50-95', 0):.4f}",
            f"{results_dict[model].get('Precision', 0):.4f}",
            f"{results_dict[model].get('Recall', 0):.4f}",
            f"{results_dict[model].get('Accuracy', 0):.4f}",
        ]
        cell_text.append(row)

    n_rows = len(models) + 1
    n_cols = len(col_labels)
    fig, ax = plt.subplots(figsize=(12, 2 + n_rows * 0.55))
    ax.axis('tight')
    ax.axis('off')

    # 不设置 rowLabels，避免 (0,-1) 单元格问题
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )

    # ---------- 美化 ----------
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.5)

    header_color = '#2c3e50'
    header_text_color = 'white'
    row_colors = ['#ecf0f1', '#ffffff']

    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i, j]
            if i == 0:                     # 表头行
                cell.set_facecolor(header_color)
                cell.set_text_props(color=header_text_color, fontweight='bold')
                cell.set_edgecolor('white')
                cell.set_linewidth(0.5)
            else:                          # 数据行
                color = row_colors[(i - 1) % 2]
                cell.set_facecolor(color)
                cell.set_edgecolor('lightgray')
                cell.set_linewidth(0.5)

    # ========== 左上角 (0,0) 单元格斜线 ==========
    corner_cell = table[0, 0]
    corner_cell.get_text().set_text('')     # 清空原有“模型”文字
    bbox = corner_cell.get_bbox()
    pts = ax.transAxes.inverted().transform(bbox.get_points())
    x0, y0 = pts[0]   # 左下
    x1, y1 = pts[1]   # 右上

    # 画对角线
    ax.plot([x0, x1], [y1, y0], color='white', linewidth=1, transform=ax.transAxes)

    # 左下角写“模型”
    ax.text(x0 + (x1 - x0) * 0.18,
            y0 + (y1 - y0) * 0.18,
            '模型', ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', transform=ax.transAxes)
    # 右上角写“指标”
    ax.text(x1 - (x1 - x0) * 0.18,
            y1 - (y1 - y0) * 0.18,
            '指标', ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', transform=ax.transAxes)

    # 不显示标题（和 baseline_comparison_final.png 保持一致）
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n对比表格已保存至: {save_path}")

def main():
    multiprocessing.freeze_support()

    # ---- 读取测试样本 ----
    samples, data_cfg = collect_samples()
    print(f"测试集图片总数: {len(samples)}")

    results_summary = {}

    # ---- 逐模型评估 ----
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"[警告] 模型文件不存在，跳过：{model_path}")
            continue

        print(f"\n{'='*50}")
        print(f"正在评估模型: {model_name}")
        model = YOLO(model_path)

        # 1. 检测指标 (mAP, Precision, Recall)
        try:
            metrics = model.val(data=DATA_YAML, split='test', conf=CONF_THRESHOLD,
                                batch=16, workers=0, verbose=False)
            map50 = metrics.box.map50
            map50_95 = metrics.box.map
            precision = metrics.box.p.mean()
            recall = metrics.box.r.mean()
            print(f"  mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
        except Exception as e:
            print(f"  检测评估失败: {e}")
            map50 = map50_95 = precision = recall = 0.0

        # 2. 图片级分类准确率
        try:
            acc = compute_classification_metrics(model, samples)
            print(f"  Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"  分类评估失败: {e}")
            acc = 0.0

        results_summary[model_name] = {
            'mAP50': map50,
            'mAP50-95': map50_95,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': acc,
        }

    # ---- 打印控制台汇总表 ----
    print("\n" + "="*70)
    print("模型性能对比汇总")
    print("="*70)
    header = f"{'模型':<12} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10} {'Accuracy':<10}"
    print(header)
    print("-"*70)
    for model, res in results_summary.items():
        print(f"{model:<12} {res['mAP50']:<10.4f} {res['mAP50-95']:<10.4f} "
              f"{res['Precision']:<10.4f} {res['Recall']:<10.4f} {res['Accuracy']:<10.4f}")

    # ---- 生成可视化表格图片 ----
    if results_summary:
        plot_comparison_table(results_summary, OUTPUT_TABLE_IMG)
    else:
        print("没有任何模型评估成功，无法生成表格。")

if __name__ == '__main__':
    main()