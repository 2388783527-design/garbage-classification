from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import base64
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载模型
model = YOLO('D:\\suju\\runs\\detect\\YOLO26s_weighted_final\\weights\\best.pt')  # 请确保路径正确

# 类别名称（40类）—— 请确保与模型训练时的顺序完全一致
import yaml

# 加载数据集配置文件，获取类别名称
with open(r'D:\suju\yolov8_garbage_new\garbage.yaml', 'r', encoding='utf-8') as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg['names']   # 自动与训练时保持一致

category_info = {
    "厨余垃圾": {"color": "#4caf50", "desc": "易腐烂的有机废弃物，如剩饭剩菜、果皮等"},
    "可回收物": {"color": "#2196f3", "desc": "可循环利用的废弃物，如纸张、塑料、玻璃等"},
    "有害垃圾": {"color": "#f44336", "desc": "对人体或环境有害的废弃物，如电池、过期药品等"},
    "其他垃圾": {"color": "#9e9e9e", "desc": "除上述三类外的其他生活垃圾"}
}

# 加载中文字体（优先使用系统字体）
font_path = None
possible_fonts = [
    "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf", # 黑体
    "C:/Windows/Fonts/simsun.ttc", # 宋体
    "C:/Windows/Fonts/DengXian.ttf", # 等线
]
for path in possible_fonts:
    if os.path.exists(path):
        font_path = path
        break

if font_path:
    font = ImageFont.truetype(font_path, 20)
else:
    font = ImageFont.load_default()
    print("警告：未找到中文字体，使用默认字体，中文可能显示为方块。")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 获取置信度阈值，默认为 0.5
    conf_threshold = float(request.form.get('conf', 0.5))
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 模型预测（使用前端传来的置信度阈值，并降低 IoU 以避免过度抑制）
    results = model.predict(source=filepath, conf=conf_threshold, iou=0.3, save=False)

    # 用 PIL 读取图片
    pil_img = Image.open(filepath).convert('RGB')
    draw = ImageDraw.Draw(pil_img)

    predictions = []
    detected_count = 0

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            detected_count = len(boxes)
            print(f"检测到 {detected_count} 个垃圾")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"类别{cls_id}"
                category = class_name.split('/')[0] if '/' in class_name else "其他垃圾"

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color_hex = category_info.get(category, category_info["其他垃圾"])["color"]
                color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

                # 绘制矩形框
                draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

                # 准备标签文本（垃圾具体名称 + 置信度）
                label = f"{class_name.split('/')[-1]} {conf*100:.1f}%"
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    text_width, text_height = draw.textsize(label, font=font)

                # 文本背景
                draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color_rgb)
                # 文本
                draw.text((x1 + 2, y1 - text_height - 3), label, fill=(255, 255, 255), font=font)

                predictions.append({
                    'class_name': class_name,
                    'category': category,
                    'confidence': round(conf * 100, 1),
                    'color': color_hex,
                    'desc': category_info.get(category, category_info["其他垃圾"])["desc"],
                    'bbox': [x1, y1, x2, y2]
                })
                print(f"  {class_name} - 置信度 {conf:.2f}")
        else:
            print("未检测到任何垃圾")

    # 将 PIL 图片转为 base64
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format='JPEG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    return jsonify({
        'success': True,
        'predictions': predictions,
        'image': img_base64,
        'detected_count': detected_count
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)