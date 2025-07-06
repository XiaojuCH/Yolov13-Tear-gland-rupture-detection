# train.py
import os
import sys
import warnings
from evaluator import monitor_features

# 1. 首先解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 3. 忽略其他不重要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

from ultralytics import YOLO
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 必须放在所有import之前
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   type=str, default='configs/tear.yaml')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch',  type=int, default=12)
    parser.add_argument('--imgsz',  type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    model = YOLO("ultralytics\cfg\models/v13\yolov13.yaml")
    model.load("yolov13n.pt")  # 加载预训练权重
    project_dir = 'runs\detect'
    exp_name    = 'tears_check_yolo13n'
    # model.add_callback("on_train_epoch_end", monitor_features)


    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        

        # —————— 优化器 & 学习率调度 ——————
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0001,
        warmup_epochs=5,
        patience=30,

        # —————— 基础几何&颜色增强 ——————
        fliplr=0.5,       # 水平翻转
        flipud=0.2,       # 垂直翻转

        # —————— 仿射 & 投影变换 ——————
        translate=0.1,    # 平移
        scale=0.5,        # 缩放
        shear=0.0,        # 剪切
        perspective=0.0,  # 透视

        # —————— 马赛克 & MixUp & CopyPaste ——————
        mosaic=1.0,        # 马赛克 (0~1 开启)
        mixup=0.2,         # MixUp
        copy_paste=0.2,    # Copy‑Paste

        # —————— 自动增强 ——————
        auto_augment='randaugment',  # or 'identity', 'v0', etc

        # —————— Late‑stage 关闭增强 ——————
        close_mosaic=5,    # 最后 N 轮不做 Mosaic
        erasing=0.4,       # 随机擦除

        # —————— 其他 ——————
        pretrained=True,
        plots=True,
        val=True,

        # —————— 输出目录 ——————
        project=project_dir,
        name=exp_name,
        exist_ok=True
    )
