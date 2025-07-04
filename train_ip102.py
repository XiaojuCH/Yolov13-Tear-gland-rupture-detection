# train_ip102_yolo13n_simplified.py

import os
import sys
import warnings
import argparse
from ultralytics import YOLO

# ================ 警告过滤设置 ================
# 1. 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 2. 自定义错误过滤器 (过滤 FlashAttention 警告)
class CustomErrorFilter:
    def __init__(self):
        self.original_stderr = sys.stderr
    
    def write(self, message):
        if "FlashAttention is not available" not in message:
            self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()

sys.stderr = CustomErrorFilter()

# 3. 忽略其他不重要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", message="Failed to create a new cache")

# ================ 主训练流程 ================
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLOv13n IP102 训练脚本')
    parser.add_argument('--data', type=str, default='configs/data_ip102.yaml', help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=70, help='总训练轮数 (20+50)')
    parser.add_argument('--batch', type=int, default=32, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='训练设备')
    parser.add_argument('--freeze', type=int, default=15, help='冻结层数 (0=不解冻)')
    parser.add_argument('--project', type=str, default='runs/ip102', help='输出目录')
    parser.add_argument('--name', type=str, default='yolo13n_ip102', help='实验名称')
    args = parser.parse_args()

    # 创建模型
    model = YOLO("ultralytics/cfg/models/v13/yolov13.yaml")
    model.load("yolov13n.pt")  # 加载预训练权重

    # 训练参数配置
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "freeze": args.freeze,  # 冻结主干网络层数
        
        # 优化器 & 学习率
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.005,
        "cos_lr": True,
        "warmup_epochs": 5,
        "weight_decay": 0.05,
        
        # 数据增强
        "augment": True,
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.25,
        "auto_augment": "randaugment",
        "close_mosaic": 10,  # 最后10轮关闭Mosaic
        "hsv_h": 0.015, 
        "hsv_s": 0.7, 
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "flipud": 0.2,
        "translate": 0.1,
        "scale": 0.5,
        
        # 模型结构
        "dropout": 0.1,
        
        # 损失权重
        "box": 8.0,
        "cls": 0.8,
        
        # 输出设置
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "plots": True,
        "val": True
    }

    # 执行训练
    results = model.train(**train_args)
    
    # 训练后导出最佳模型 (可选)
    if results:
        best_model = YOLO(os.path.join(results.save_dir, "weights", "best.pt"))
        export_path = os.path.join(results.save_dir, "weights", "final_model.onnx")
        best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"\n✅ 最终模型已导出至: {export_path}")

    print("\n✅ 训练完成!")