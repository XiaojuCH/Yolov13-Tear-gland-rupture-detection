import cv2
from ultralytics import YOLO

# === 参数 ===
weight_path = r"D:\Projects_\Tears_Check_YOLO13_fix\yolov13-main\runs\detect\tears_check_yolo13n15\weights\best.pt"# 你的模型权重路径
input_video_path ="video_/test_kj.mp4"
   # r"C:\Users\34534\Desktop\image\WIN_20250422_15_59_14_Pro\WIN_20250422_15_59_14_Pro - Trim.mp4"  # 输入视频路径；摄像头用 0
output_video_path = "vidio_perdict_kj.mp4"# 输出保存路径
conf_threshold = 0.2                          # 置信度阈值

# === 加载模型 ===
model = YOLO(weight_path)

# === 打开输入视频 ===
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

# === 视频处理循环 ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- 1. 旋转180度 ---
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # --- 2. 转为灰度图，并转回3通道（YOLO要求3通道） ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- 3. YOLOv8 推理 ---
    results = model.predict(source=gray_3channel, conf=conf_threshold, imgsz=640, verbose=False)
    print(results)
    # --- 4. 绘制预测框 ---
    annotated_frame = results[0].plot()

    # --- 5. 显示与保存 ---
    cv2.imshow('YOLOv13 Grayscale + Rotated', annotated_frame)
    out.write(annotated_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 清理资源 ===
cap.release()
out.release()
cv2.destroyAllWindows()