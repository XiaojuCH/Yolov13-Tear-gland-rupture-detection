import os
from pathlib import Path
import shutil
from ultralytics import YOLO
import yaml

# 加载真实类别名映射
def load_names(yaml_path):
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data["names"]

def load_gt_class(label_path):
    if not label_path.exists():
        return None
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return None
    return int(lines[0].split()[0])

def main():
    # ==== 路径设置 ====
    weights_path = Path(r"runs/ip102/yolo13n_ip102_atten_pruned/weights/best.pt")
    yaml_path    = Path(r"configs/data_ip102_pruned.yaml")
    src_img_dir  = Path(r"datasets_ip102_pruned/images/val")
    src_lbl_dir  = Path(r"datasets_ip102_pruned/labels/val")

    dst_img_dir  = Path(r"datasets_ip102_pruned_45_test/images")
    dst_lbl_dir  = Path(r"datasets_ip102_pruned_45_test/labels")

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    names = load_names(yaml_path)

    error_found = 0
    max_error = 5

    for img_path in sorted(src_img_dir.glob("*.*")):
        if error_found >= max_error:
            break

        stem = img_path.stem
        label_path = src_lbl_dir / f"{stem}.txt"
        gt_cls = load_gt_class(label_path)

        if gt_cls is None:
            continue

        results = model.predict(source=str(img_path), conf=0.25, imgsz=512, verbose=False)
        r = results[0]

        if len(r.boxes.cls) == 0:
            continue  # 无预测，跳过

        pred_cls = int(r.boxes.cls[0].item())

        if pred_cls != gt_cls:
            gt_name = names[gt_cls]
            new_img_name = f"错误_{gt_name}_{stem}.jpg"
            new_lbl_name = f"错误_{gt_name}_{stem}.txt"

            shutil.copy(img_path, dst_img_dir / new_img_name)
            shutil.copy(label_path, dst_lbl_dir / new_lbl_name)

            print(f"❌ 错误样本：{stem} → 存为 {new_img_name}")
            error_found += 1

    print(f"\n✅ 共收集 {error_found} 张预测错误的图片，并写入真实类别名。")

if __name__ == "__main__":
    main()
