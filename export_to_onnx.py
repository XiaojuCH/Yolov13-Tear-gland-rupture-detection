from ultralytics import YOLO

model = YOLO(r'D:/Projects_/Machine_20/runs/detect/tears_check/weights/best.pt')

model.export(format='onnx')