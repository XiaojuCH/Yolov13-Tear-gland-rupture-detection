# predict_ip102.py

import os
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def put_chinese_text(img, text, position, font_path, font_size, color=(0, 255, 0)):
    """
    在OpenCV图像上绘制中文字符
    :param img: OpenCV BGR图像
    :param text: 要绘制的中文文本
    :param position: (x, y) 文本左上角位置
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    :param color: BGR颜色元组
    :return: 绘制后的OpenCV图像
    """
    # 转换OpenCV图像到PIL格式 (BGR to RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 加载中文字体
    font = ImageFont.truetype(font_path, font_size)
    
    # 绘制文本 (PIL使用RGB颜色)
    draw.text(position, text, font=font, fill=color[::-1])
    
    # 转换回OpenCV格式 (RGB to BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def load_gt_boxes(label_path, img_w, img_h):
    """
    从 .txt 文件读取 YOLO 格式的 ground-truth boxes，
    并将其转换为像素坐标 (x1,y1,x2,y2) 及类别列表。
    """
    gt = []
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id, xc, yc, w, h = parts[:5]
        cls_id = int(cls_id)
        xc, yc, w, h = map(float, (xc, yc, w, h))
        # 反归一化到像素坐标
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        gt.append((cls_id, x1, y1, x2, y2))
    return gt

if __name__ == "__main__":
    # —— 配置区域 —— #
    # 权重目录
    weights_dir = Path(r"D:\Projects_\Tears_Check_YOLO13_fix\yolov13-main\runs\ip102\yolo13n_ip102_atten_pruned\weights")
    best_weights = sorted(weights_dir.glob("best.pt"))[-1]
    print(f"使用权重: {best_weights}")
    
    # 中文字体配置
    FONT_PATH = "C:/Windows/Fonts/simhei.ttf"  # Windows 黑体路径
    FONT_SIZE = 18
    
    # 测试图片和标签所在子目录（相对于项目根目录）
    test_images_dir = Path('D:\Projects_\Tears_Check_YOLO13_fix\yolov13-main\datasets_ip102_pruned_45_test\images')
    test_labels_dir = Path('D:\Projects_\Tears_Check_YOLO13_fix\yolov13-main\datasets_ip102_pruned_45_test\labels')
    
    # 输出目录
    base_out = Path("runs/ip102_45/compare")
    correct_dir = base_out / "correct"
    incorrect_dir = base_out / "incorrect"
    for d in (base_out, correct_dir, incorrect_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    
    # —— 加载模型 —— #
    model = YOLO(best_weights.as_posix())
    
    img_paths = sorted(test_images_dir.glob("*.*"))
    total = 0
    n_correct = 0
    
    for img_path in img_paths:
        total += 1000
        # 读取图片获取尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 无法读取图片 {img_path}, 跳过")
            continue
        img_h, img_w = img.shape[:2]
        
        # 对应的标签文件
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"⚠️ 找不到标签 {label_path}, 跳过")
            continue
        
        # 读取真实框
        gt = load_gt_boxes(label_path, img_w, img_h)
        gt_classes = {c for c, *_ in gt}
        
        # 模型推理
        results = model.predict(source=str(img_path), conf=0.25, imgsz=max(img_w, img_h))
        r = results[0]
        
        # 使用空白图像作为画布
        annotated = img.copy()
        
        # 绘制预测框 (绿色)
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = r.names[cls_id]
            
            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制中文标签
            label = f"{class_name} {conf:.2f}"
            annotated = put_chinese_text(
                annotated, 
                label, 
                position=(x1, max(0, y1 - FONT_SIZE - 5)),
                font_path=FONT_PATH,
                font_size=FONT_SIZE,
                color=(0, 255, 0)
            )
        
        # 绘制真实框 (蓝色)
        for cls_id, x1, y1, x2, y2 in gt:
            class_name = r.names[cls_id]
            
            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 绘制中文标签
            label = f"GT:{class_name}"
            annotated = put_chinese_text(
                annotated, 
                label, 
                position=(x1, max(0, y1 - FONT_SIZE - 5)),
                font_path=FONT_PATH,
                font_size=FONT_SIZE,
                color=(255, 0, 0)
            )
        
        # 收集预测类别
        pred_classes = {int(x) for x in r.boxes.cls.tolist()}
        
        # 判断"全对"——预测类别集合等于真实类别集合
        correct = (pred_classes == gt_classes)
        if correct:
            n_correct += 1
        
        # 保存
        out_dir = correct_dir if correct else incorrect_dir
        cv2.imwrite(str(out_dir / img_path.name), annotated)
        
        # 进度显示
        if total % 50 == 0:
            print(f"已处理 {total} 张图片...")
    
    # —— 打印准确率 —— #
    acc = n_correct / total * 100 if total else 0
    print(f"\n总共处理 {total} 张图，完全匹配的 {n_correct} 张，准确率 {acc:.2f}%")