from ultralytics import YOLO

if __name__ == "__main__":
    # 加载最优权重
    model = YOLO('yolov13n.pt')

    # 推理单张
    results = model.predict(
        source='1.png',
        conf=0.25,
        iou=0.45,
        device=0,
        save=True,       # 保存带框图片到 runs/detect
        save_txt=True    # 保存 .txt 预测
    )
    print(results)
