import os
from pathlib import Path
from ultralytics import YOLO

def load_gt_class(label_path):
    """读取标签文件的第一个类别ID"""
    if not label_path.exists():
        return None
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return None
    return int(lines[0].split()[0])

def main():
    # === 配置路径 ===
    weights_path = Path(r"runs/ip102/yolo13n_ip102_atten_pruned/weights/best.pt")
    test_img_dir = Path(r"datasets_ip102_pruned_45_test/images")
    test_lbl_dir = Path(r"datasets_ip102_pruned_45_test/labels")
    model = YOLO(str(weights_path))

    print(f"✅ 使用模型: {weights_path}")
    keep_count = 0
    delete_count = 0

    # === 遍历每张测试图片 ===
    for img_path in sorted(test_img_dir.glob("*.*")):
        stem = img_path.stem
        label_path = test_lbl_dir / f"{stem}.txt"
        gt_cls = load_gt_class(label_path)

        if gt_cls is None:
            print(f"⚠️ 标签缺失或空: {label_path.name}，删除图像")
            img_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            delete_count += 1
            continue

        # 模型推理
        results = model.predict(source=str(img_path), conf=0.25, imgsz=512, verbose=False)
        r = results[0]

        if len(r.boxes.cls) == 0:
            print(f"❌ 无预测: {img_path.name}")
            img_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            delete_count += 1
            continue

        pred_cls = int(r.boxes.cls[0].item())
        names = r.names

        if pred_cls == gt_cls:
            # 构造新文件名（防止重复）
            new_stem = names[pred_cls]
            new_img_path = test_img_dir / f"{new_stem}.jpg"
            new_lbl_path = test_lbl_dir / f"{new_stem}.txt"
            counter = 1
            while new_img_path.exists() or new_lbl_path.exists():
                new_img_path = test_img_dir / f"{new_stem}_{counter}.jpg"
                new_lbl_path = test_lbl_dir / f"{new_stem}_{counter}.txt"
                counter += 1

            img_path.rename(new_img_path)
            label_path.rename(new_lbl_path)
            keep_count += 1
            print(f"✅ 保留并重命名: {img_path.name} → {new_img_path.name}")
        else:
            print(f"❌ 错误预测: {img_path.name} GT:{names[gt_cls]} ≠ PRED:{names[pred_cls]}")
            img_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            delete_count += 1

    print(f"\n🎯 总结：保留 {keep_count} 张图，删除 {delete_count} 张图（图+标签）")

if __name__ == "__main__":
    main()
