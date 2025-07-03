# preprocess_data.py
import os
import glob
import numpy as np

def normalize_bbox(coords):
    """确保边界框坐标在[0,1]范围内"""
    x, y, w, h = coords
    # 确保中心点在[0,1]
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    # 确保宽高在[0,1]且不会超出边界
    w = max(0.001, min(1.0 - x, w))  # 最小宽度0.001
    h = max(0.001, min(1.0 - y, h))  # 最小高度0.001
    return [x, y, w, h]

def process_labels(label_dir, num_classes):
    """处理标签目录中的所有文件"""
    for label_path in glob.glob(os.path.join(label_dir, "*.txt")):
        new_lines = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                try:
                    # 解析并验证类别ID
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= num_classes:
                        class_id = 0  # 重置为默认类别
                    
                    # 解析并标准化坐标
                    coords = list(map(float, parts[1:]))
                    coords = normalize_bbox(coords)
                    
                    # 创建新行
                    new_line = f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n"
                    new_lines.append(new_line)
                    
                except (ValueError, IndexError):
                    # 跳过无法解析的行
                    continue
        
        # 保存处理后的标签
        with open(label_path, 'w') as f:
            f.writelines(new_lines)

if __name__ == "__main__":
    # 配置参数
    label_dirs = [
        "datasets/train/labels",
        "datasets/val/labels"
    ]
    num_classes = 1  # 您的类别数量
    
    for label_dir in label_dirs:
        print(f"处理目录: {label_dir}")
        process_labels(label_dir, num_classes)
    
    print("数据预处理完成!")