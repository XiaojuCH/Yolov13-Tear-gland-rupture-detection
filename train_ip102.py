# train_ip102_yolo11n_optimized.py

import os
from ultralytics import YOLO
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']          # 指定黑体
matplotlib.rcParams['axes.unicode_minus'] = False           # 正常显示负号


# ——— 环境配置 ———
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ——— 主流程 ———
if __name__ == "__main__":
    # -------- 阶段 1：冻结主干，仅训练 Head --------
    print("\n🔒 Stage 1: 冻结 backbone，仅训练 head (20 epochs) - YOLOv11n\n")
    model = YOLO("ultralytics\cfg\models/v13\yolov13.yaml")
    model.load("yolov13n.pt")  # 加载预训练权重
    
    # 阶段1训练参数 - 移除不支持的参数
    args1 = dict(
        data="configs/data_ip102.yaml",
        epochs=20,
        imgsz=640,
        batch=24,
        device=0,
        freeze=15,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.005,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=5,
        patience=10,
        half=True,
        cache="disk",
        workers=4,
        augment=True,
        auto_augment="v2",
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.15,  # 增加mixup比例 (0.1 → 0.15)
        copy_paste=0.3,  # 增加copy-paste比例 (0.2 → 0.3)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.2,
        translate=0.2, scale=0.5,
        degrees=10.0, shear=5.0,
        dropout=0.2,
        weight_decay=0.05,
        project="runs/ip102",
        name="yolo13s_stage1_opt",  # 更新实验名称
        exist_ok=True,
        plots=True,
        val=True,
        # 移除不支持的 fl_gamma 参数
        # 保留支持的损失权重调整
        box=8.0,  # 增加box损失权重 (默认7.5)
        cls=0.8   # 降低cls损失权重 (默认0.5)
    )
    
    # 执行训练
    results1 = model.train(**args1)
    stage1_save_dir = results1.save_dir

    # -------- 阶段 2：解冻全部，fine-tune --------
    print("\n\n🔓 Stage 2: 解冻所有层，fine-tuning (50 epochs) - YOLOv11s\n")
    best1 = os.path.join(stage1_save_dir, "weights", "best.pt")
    model2 = YOLO(best1)
    
    # 阶段2训练参数 - 使用支持的参数
    args2 = dict(
        data="configs/data_ip102.yaml",
        epochs=50,  # 增加训练周期 (40 → 50)
        imgsz=640,
        batch=12,
        device=0,
        freeze=0,
        pretrained=False,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.005,  # 降低最终学习率 (0.01 → 0.005)
        cos_lr=True,
        warmup_epochs=8,  # 延长预热 (3 → 8)
        patience=30,  # 增加耐心值 (30 → 40)
        half=True,
        cache="disk",
        workers=4,
        augment=True,
        auto_augment="v2",
        mosaic=0.7,
        close_mosaic=20,  # 更晚关闭mosaic (15 → 20)
        mixup=0.1,  # 适当降低mixup
        copy_paste=0.25,  # 保持较高copy-paste
        hsv_h=0.01, hsv_s=0.6, hsv_v=0.3,
        fliplr=0.4, flipud=0.1,
        translate=0.1, scale=0.4,
        degrees=5.0, shear=3.0,
        dropout=0.1,
        weight_decay=0.02,
        project="runs/ip102",
        name="yolo13s_stage2_opt",  # 更新实验名称
        exist_ok=True,
        plots=True,
        val=True,
        resume=False
    )
    
    # 执行训练
    results2 = model2.train(**args2)
    
    # 训练后优化：使用测试时增强评估最佳模型
    print("\n🔍 使用测试时增强评估最佳模型...")
    best_model = YOLO(os.path.join(results2.save_dir, "weights", "best.pt"))
    best_model.val(
        data="data.yaml",
        imgsz=640,
        split="val",
        name="best_val_tta",
        augment=True,  # 测试时增强
        conf=0.25,    # 降低置信度阈值
        iou=0.45      # 降低IoU阈值
    )
    
    # 导出最终模型
    print("\n📤 导出最终模型...")
    export_path = os.path.join(results2.save_dir, "weights", "final_model.onnx")
    best_model.export(format="onnx", imgsz=640, simplify=True)
    print(f"\n✅ 最终模型已导出至: {export_path}")

    print("\n✅ 优化训练完成，结果保存在 runs/ip102/ 下。")