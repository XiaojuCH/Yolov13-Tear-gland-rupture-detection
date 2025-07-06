# filter_ip102_classes.py

import os
import shutil
from collections import Counter
import yaml

# ——— 配置区 ———
NUM_CLASSES     = 30
# 这是你 IP102 数据集的根目录，里面有 images/train, images/val, images/test 以及 labels/train, labels/val, labels/test
DATA_ROOT       = r"D:\Projects_\IP102_YOLO\dataset\Detection\IP102"
NEW_DATA_ROOT   = "datasets_ip102_pruned_30"
ORIG_YAML       = "configs/data_ip102.yaml"
NEW_YAML        = "configs/data_ip102_pruned_30.yaml"

def load_counts():
    cnt = Counter()
    for split in ["train", "val"]:
        lbl_dir = os.path.join(DATA_ROOT, "labels", split)
        for fn in os.listdir(lbl_dir):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(lbl_dir, fn), encoding="utf-8") as f:
                for line in f:
                    cls = int(line.split()[0])
                    cnt[cls] += 1
    return cnt

def select_top_classes(cnt):
    return [cls for cls, _ in cnt.most_common(NUM_CLASSES)]

def rewrite_data_yaml(top_classes):
    orig = yaml.safe_load(open(ORIG_YAML, encoding="utf-8"))
    names_orig = orig["names"]
    # 试着先用 int key，再用 str key
    def get_name(c):
        if isinstance(names_orig, list):
            return names_orig[c]
        try:
            return names_orig[c]
        except KeyError:
            return names_orig[str(c)]
    pruned = {
        "train": os.path.join(NEW_DATA_ROOT, "images", "train"),
        "val":   os.path.join(NEW_DATA_ROOT, "images", "val"),
        "nc":    len(top_classes),
        "names": {i: get_name(c) for i, c in enumerate(top_classes)}
    }
    os.makedirs(os.path.dirname(NEW_YAML), exist_ok=True)
    with open(NEW_YAML, "w", encoding="utf-8") as w:
        yaml.safe_dump(pruned, w, allow_unicode=True, sort_keys=False)
    print(f">>> 已写入新 config: {NEW_YAML}")

def remap_and_copy(split, top_classes):
    mapping = {old: new for new, old in enumerate(top_classes)}
    src_img = os.path.join(DATA_ROOT, "images", split)
    src_lbl = os.path.join(DATA_ROOT, "labels", split)
    dst_img = os.path.join(NEW_DATA_ROOT, "images", split)
    dst_lbl = os.path.join(NEW_DATA_ROOT, "labels", split)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    # 复制图片
    for fn in os.listdir(src_img):
        if fn.lower().endswith((".jpg", ".png")):
            shutil.copy(os.path.join(src_img, fn), dst_img)

    # 重写标签
    for fn in os.listdir(src_lbl):
        if not fn.endswith(".txt"):
            continue
        lines = []
        with open(os.path.join(src_lbl, fn), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                if cls in mapping:
                    lines.append(" ".join([str(mapping[cls])] + parts[1:]))
        if lines:
            with open(os.path.join(dst_lbl, fn), "w", encoding="utf-8") as w:
                w.write("\n".join(lines))

if __name__ == "__main__":
    cnt = load_counts()
    print("► 前 20 类样本数：", cnt.most_common(20))

    top = select_top_classes(cnt)
    print(f">>> 选取前 {NUM_CLASSES} 类原始 ID：\n{top}\n")

    rewrite_data_yaml(top)

    for split in ["train", "val"]:
        print(f">>> 处理 split={split}")
        remap_and_copy(split, top)

    print(f"\n>>> 全部完成！新数据保存在 `{NEW_DATA_ROOT}`，配置文件为 `{NEW_YAML}`")
