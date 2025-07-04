from ultralytics import YOLO

model = YOLO(r'D:\Projects_\Tears_Check_YOLO13_fix\yolov13-main\runs\detect\tears_check_yolo13n\weights/best.pt')

model.export(format='onnx')