# evaluator.py — 评估 mAP/Precision/Recall
import argparse
from ultralytics import YOLO
from ultralytics.nn.modules.block import ODF


# 注册特征监控回调
def monitor_features(trainer):
    # 获取ODF模块
    for module in trainer.model.modules():
        if isinstance(module, ODF):
            # 获取特征
            input_features, output_features = module.get_features()
            
            if input_features is not None and output_features is not None:
                # 计算特征变化
                input_mean = input_features.mean().item()
                output_mean = output_features.mean().item()
                input_std = input_features.std().item()
                output_std = output_features.std().item()
                
                # 记录到TensorBoard
                trainer.tblogger.add_scalar('ODF/input_mean', input_mean, trainer.epoch)
                trainer.tblogger.add_scalar('ODF/output_mean', output_mean, trainer.epoch)
                trainer.tblogger.add_scalar('ODF/input_std', input_std, trainer.epoch)
                trainer.tblogger.add_scalar('ODF/output_std', output_std, trainer.epoch)
                
                # 重置特征
                module.input_features = None
                module.output_features = None


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

