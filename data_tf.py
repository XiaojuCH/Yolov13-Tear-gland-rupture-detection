# -*- coding: utf-8 -*-
# @Author :Xiaoju
# Time : 2025/6/23 下午3:03

import os
import random
import shutil
import argparse

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
LABEL_EXT = '.txt'


def is_image(file):
    return os.path.splitext(file)[1].lower() in IMAGE_EXTS


def split_dataset(src_images, src_labels, dst_root, train_ratio=0.7, seed=None):
    # 读取所有图片列表
    imgs = [f for f in os.listdir(src_images) if is_image(f)]
    if seed is not None:
        random.seed(seed)
    random.shuffle(imgs)

    n_train = int(len(imgs) * train_ratio)
    splits = {'train': imgs[:n_train], 'val': imgs[n_train:]}

    for split, files in splits.items():
        img_dst = os.path.join(dst_root, split, 'images')
        lbl_dst = os.path.join(dst_root, split, 'labels')
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)
        for fname in files:
            name, _ = os.path.splitext(fname)
            src_img = os.path.join(src_images, fname)
            src_lbl = os.path.join(src_labels, name + LABEL_EXT)
            shutil.copy(src_img, os.path.join(img_dst, fname))
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(lbl_dst, name + LABEL_EXT))

    print(f"Dataset split completed. Train: {len(splits['train'])}, Val: {len(splits['val'])}")


def main():
    parser = argparse.ArgumentParser(description='Split dataset into YOLO train/val folders')
    parser.add_argument('--src_images', default='D:\Projects_\Tears_Check\image\YOLODataset\img2/img2', help='原始图片文件夹')
    parser.add_argument('--src_labels', default='D:\Projects_\Tears_Check\image\YOLODataset\img2_label', help='原始标签文件夹')
    parser.add_argument('--dst',       default='dataset1', help='输出根目录（包含 train/ val）')
    parser.add_argument('--ratio',     type=float, default=0.7, help='训练集比例')
    parser.add_argument('--seed',      type=int, default=None, help='随机种子')
    args = parser.parse_args()

    split_dataset(args.src_images, args.src_labels, args.dst, args.ratio, args.seed)

if __name__ == '__main__':
    main()