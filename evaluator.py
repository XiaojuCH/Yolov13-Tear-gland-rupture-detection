# evaluator.py — 评估 mAP/Precision/Recall
import argparse
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='训练得到的 .pt 权重')
    parser.add_argument('--data',    type=str, default='data.yaml', help='数据集配置')
    parser.add_argument('--device',  type=str, default='0',               help='CUDA 设备')
    args = parser.parse_args()

    # 加载训练好的模型权重
    model = YOLO(args.weights)

    # 在验证集上评估，save_json=True 会在 runs/val 下生成详细结果
    metrics = model.val(data=args.data, device=args.device, save_json=True)

    print(f"mAP:       {metrics['metrics/mAP50-95']:.4f}")
    print(f"Precision:{metrics['metrics/precision']:.4f}")
    print(f"Recall:   {metrics['metrics/recall']:.4f}")
