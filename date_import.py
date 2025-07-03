# -*- coding: utf-8 -*-
# @Author : Xiaoju
# Merge new data into existing YOLO train/val/test split
import os
import random
import shutil

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
LABEL_EXT = '.txt'

def is_image(file: str) -> bool:
    return os.path.splitext(file)[1].lower() in IMAGE_EXTS

def unique_path(dst_dir: str, filename: str) -> str:
    """
    如果 dst_dir/filename 已存在，则在文件名后加后缀直到唯一。
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    dst = os.path.join(dst_dir, filename)
    while os.path.exists(dst):
        dst = os.path.join(dst_dir, f"{base}_{counter}{ext}")
        counter += 1
    return dst

def add_new_data(new_images: str,
                 new_labels: str,
                 dst_root: str,
                 train_ratio: float,
                 val_ratio: float,
                 seed: int = None):
    """
    将 new_images/new_labels 新样本，按 train_ratio/val_ratio/test_ratio 随机
    分配到 dst_root/train/…, dst_root/val/… 和 dst_root/test/… 下。
    """
    imgs = [f for f in os.listdir(new_images) if is_image(f)]
    if seed is not None:
        random.seed(seed)
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    splits = {
        'train': imgs[:n_train],
        'val':   imgs[n_train:n_train + n_val],
        'test':  imgs[n_train + n_val:]
    }

    for split, files in splits.items():
        img_dst = os.path.join(dst_root, split, 'images')
        lbl_dst = os.path.join(dst_root, split, 'labels')
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for fname in files:
            name, ext = os.path.splitext(fname)
            src_img = os.path.join(new_images, fname)
            src_lbl = os.path.join(new_labels, name + LABEL_EXT)

            # 拷贝图片
            dst_img = unique_path(img_dst, fname)
            shutil.copy(src_img, dst_img)

            # 拷贝标签（如果存在）
            if os.path.exists(src_lbl):
                lbl_name = os.path.basename(src_lbl)
                dst_lbl = unique_path(lbl_dst, lbl_name)
                shutil.copy(src_lbl, dst_lbl)

    print(f"Added {n} new samples:")
    print(f"  train: {len(splits['train'])}")
    print(f"  val:   {len(splits['val'])}")
    print(f"  test:  {len(splits['test'])}")

if __name__ == '__main__':
    # ====== 在这里写死你的参数 ======
    NEW_IMAGES   = r"D:\Projects_\Tears_Check_YOLO12\dataset\all_data1_2_fixed\img1"       # 新图片文件夹
    NEW_LABELS   = r"D:\Projects_\Tears_Check_YOLO12\dataset\all_data1_2_fixed\img1label"     # 新标签文件夹
    DST_ROOT     = "dataset_all"     # 原有数据集根目录
    TRAIN_RATIO  = 0.7                          # 新数据训练集比例
    VAL_RATIO    = 0.2                          # 新数据验证集比例
    SEED         = 42                           # 随机种子（可选）

    # 检查比例合法
    if TRAIN_RATIO + VAL_RATIO >= 1.0:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must be less than 1.0")

    # 执行合并
    add_new_data(
        new_images=NEW_IMAGES,
        new_labels=NEW_LABELS,
        dst_root=DST_ROOT,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED
    )
