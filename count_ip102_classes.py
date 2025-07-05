# count_ip102_classes.py
import os
import yaml
from collections import defaultdict

# —— 配置区 —— #
DATA_ROOT = "datasets_ip102_pruned"            # pruned 数据集根目录
SPLITS    = ["train", "val", "test"]           # 你的 split 列表
YAML_PATH = "configs/data_ip102_pruned.yaml"   # pruned 的 data.yaml 路径

def count_images_per_class(data_root, splits):
    """
    遍历 data_root/labels/{split} 下的所有 .txt，
    统计每个 cls_id 在多少不同图片中出现过
    """
    img_sets = defaultdict(set)  # cls_id -> set(image_id)
    for split in splits:
        lbl_dir = os.path.join(data_root, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        for fn in os.listdir(lbl_dir):
            if not fn.endswith(".txt"):
                continue
            img_id = fn[:-4]  # e.g. '000123'，去掉 .txt
            with open(os.path.join(lbl_dir, fn), encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    img_sets[cls].add(img_id)
    # 转为 cls -> 图片数量
    return {cls: len(imgs) for cls, imgs in img_sets.items()}

if __name__ == "__main__":
    # 加载 pruned data.yaml 中的 names（可能是 list 或 dict）
    with open(YAML_PATH, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg["names"]

    # 统计
    counts = count_images_per_class(DATA_ROOT, SPLITS)

    # 打印表头
    print(f"{'ID':>3s}  {'类别名称':<20s}  {'#Images':>7s}")
    print("-" * 36)
    # 按图片数从多到少排序输出
    for cls, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        label = names[cls] if isinstance(names, list) else names.get(str(cls), "未知")
        print(f"{cls:3d}  {label:<20s}  {cnt:7d}")
