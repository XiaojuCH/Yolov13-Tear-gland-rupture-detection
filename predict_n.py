# predict_ip102.py

import os
from pathlib import Path
import shutil
import cv2
from ultralytics import YOLO

def load_gt_boxes(label_path, img_w, img_h):
    """
    从 .txt 文件读取 YOLO 格式的 ground-truth boxes，
    并将其转换为像素坐标 (x1,y1,x2,y2) 及类别列表。
    """
    gt = []
    for line in Path(label_path).read_text().splitlines():
        cls_id, xc, yc, w, h = line.split()
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
    weights_dir = Path(r"D:\Projects_\IP102_YOLO\pt_backup\yolo11n_ip102_rtx4060_53mAP50\weights")
    best_weights = sorted(weights_dir.glob("best*.pt"))[-1]
    print(f"使用权重: {best_weights}")

    # 测试图片和标签所在子目录（相对于项目根目录）
    test_images_dir = Path(r"D:\Projects_\IP102_YOLO\dataset\Detection\IP102/images/test1")
    test_labels_dir = Path(r"D:\Projects_\IP102_YOLO\dataset\Detection\IP102/labels/test1")

    # 输出目录
    base_out = Path("runs/ip102/compare")
    correct_dir = base_out / "correct"
    incorrect_dir = base_out / "incorrect"
    for d in (correct_dir, incorrect_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    # —— 加载模型 —— #
    model = YOLO(best_weights.as_posix())

    img_paths = sorted(test_images_dir.glob("*.*"))
    total = 0
    n_correct = 0

    for img_path in img_paths:
        total += 1
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

        # Ultralytics 自带的绘制（绿色预测框）
        annotated = results[0].plot().copy()

        # 叠加真实框（蓝色）
        for cls_id, x1, y1, x2, y2 in gt:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                annotated,
                f"GT:{r.names[cls_id]}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 收集预测类别
        pred_classes = {int(x) for x in r.boxes.cls.tolist()}

        # 判断“全对”——预测类别集合等于真实类别集合
        correct = (pred_classes == gt_classes)
        if correct:
            n_correct += 1

        # 保存
        out_dir = correct_dir if correct else incorrect_dir
        cv2.imwrite(str(out_dir / img_path.name), annotated)

    # —— 打印准确率 —— #
    acc = n_correct / total * 100 if total else 0
    print(f"\n总共处理 {total} 张图，完全匹配的 {n_correct} 张，准确率 {acc:.2f}%")
