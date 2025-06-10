from ultralytics import YOLO

# 加载模型（可以换 yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt）
model = YOLO('yolov5s.pt')

# 推理：输入图片路径
results = model('screenshot_close.png')

# 可视化结果（会弹窗显示，或者直接保存到文件）
results.show()           # 弹出窗口显示结果
# results.save('output/')  # 保存到 output/ 文件夹

# 打印检测到的对象信息
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"Detected class {cls} with confidence {conf:.2f} at {xyxy}")