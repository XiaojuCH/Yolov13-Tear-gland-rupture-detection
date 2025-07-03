import warnings, os
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="FlashAttention is not available on this device. Using scaled_dot_product_attention instead")
from ultralytics import YOLO

torch.cuda.is_available()
if __name__ == '__main__':
    model = YOLO('ultralytics\cfg\models/v13\yolov13.yaml')
    model.load('yolov13n.pt')
    model.train(data=r'D:\Projects_\Tears_Check_YOLO13_fix\yolov13-main\configs\data.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=16,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project=r'D:\Projects_\Tears_Check_YOLO13_fix\yolov13-main\runs',
                name='yolov13',
                )

