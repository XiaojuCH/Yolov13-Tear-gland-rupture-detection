import os
import shutil
from collections import defaultdict
from pathlib import Path
import random

# 配置路径（请根据你的实际路径调整）
LABEL_DIR = Path("datasets_ip102_pruned/labels/train")
IMAGE_DIR = Path("datasets_ip102_pruned/images/train")
SAVE_LABEL_DIR = Path("datasets_ip102_pruned_45_test/labels")
SAVE_IMAGE_DIR = Path("datasets_ip102_pruned_45_test/images")

# 创建保存目录
SAVE_LABEL_DIR.mkdir(parents=True, exist_ok=True)
SAVE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 统计每个类别对应的文件（按第一行标签为主）
cls_to_files = defaultdict(list)
for label_file in LABEL_DIR.glob("*.txt"):
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            cls_id = int(line.split()[0])
            cls_to_files[cls_id].append(label_file)
            break

# 每类随机选一个样本
selected_files = set()
for cls_id, files in cls_to_files.items():
    chosen = random.choice(files)
    selected_files.add(chosen)

# 复制对应文件
copied = 0
for label_path in selected_files:
    img_path_jpg = IMAGE_DIR / f"{label_path.stem}.jpg"
    img_path_png = IMAGE_DIR / f"{label_path.stem}.png"
    img_path = img_path_jpg if img_path_jpg.exists() else img_path_png

    if not img_path.exists():
        print(f"⚠️ 未找到图像文件: {label_path.stem}.jpg/.png")
        continue

    shutil.copy(label_path, SAVE_LABEL_DIR / label_path.name)
    shutil.copy(img_path, SAVE_IMAGE_DIR / img_path.name)
    copied += 1

print(f"✅ 成功为 {copied} 个类别各抽取一张图像并保存至 test_45 文件夹")
