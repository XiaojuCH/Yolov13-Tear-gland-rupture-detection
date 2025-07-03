# -*- coding: utf-8 -*-
# @Author :Xiaoju
# Time : 2025/6/23 下午5:10

import cv2
import os

# 输入视频路径
input_path = 'img/img2.mp4'  # 替换成你的彩色视频路径
# 输出灰度视频保存路径
output_path = 'img/gray/img2.mp4'

# 打开视频
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"无法打开视频文件: {input_path}")

# 获取视频属性
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 使用相同大小但单通道灰度图像编码保存（MJPG较通用）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可换成 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

# 转换并写入每一帧
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转灰度
    out.write(gray)

# 释放资源
cap.release()
out.release()
print(f"灰度视频已保存至：{output_path}")
