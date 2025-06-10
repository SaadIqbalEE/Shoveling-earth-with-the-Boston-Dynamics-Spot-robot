import cv2
import json
import os
import numpy as np
from pathlib import Path

# === 路径配置 ===
INPUT_IMG_DIR = "./synthetic_dataset_unit_test/unit_test"
INPUT_BOX_NPY_DIR = "./synthetic_dataset_unit_test/unit_test"
INPUT_LABEL_JSON_DIR = "./synthetic_dataset_unit_test/unit_test"
OUTPUT_DIR = "./unit_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TARGET_CLASS = "rock_7_solid"
TARGET_CLASS = "cube"

# === 加载 ID → class 对照表 ===
label_files = sorted(Path(INPUT_LABEL_JSON_DIR).glob("bounding_box_2d_tight_labels_*.json"))
label_maps = {}
for label_file in label_files:
    idx = label_file.stem.split("_")[-1]
    with open(label_file, 'r') as f:
        label_maps[idx] = json.load(f)

# === 遍历 bounding box npy 文件 ===
for npy_file in sorted(Path(INPUT_BOX_NPY_DIR).glob("bounding_box_2d_tight_*.npy")):
    idx = npy_file.stem.split("_")[-1]
    img_file = f"rgb_{idx}.png"
    img_path = os.path.join(INPUT_IMG_DIR, img_file)

    if not os.path.exists(img_path):
        print(f"⚠ Warning: Image {img_file} not found, skipping.")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ Warning: Failed to load {img_file}, skipping.")
        continue

    h, w, _ = img.shape

    # 加载 box npy (直接 array)
    box_data = np.load(npy_file)
    label_map = label_maps.get(idx, {})

    has_rock = False  # 标记是否有 rock

    for box in box_data:
        instance_id = str(int(box[0]))  # instanceId
        x, y, width, height = box[1], box[2], box[3], box[4]
        label = label_map.get(instance_id, {}).get("class", "object").lower()

        if label != TARGET_CLASS:
            continue  # 只保留 rock_7_solid

        # 框坐标
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + width), int(y + height)

        # 打印详细信息
        print(f"📦 Image {img_file} - Box for {label}:")
        print(f"    instance_id: {instance_id}")
        print(f"    x: {x}, y: {y}, width: {width}, height: {height}")
        print(f"    x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

        # 画矩形和标签
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        has_rock = True

    if has_rock:
        out_path = os.path.join(OUTPUT_DIR, img_file)
        cv2.imwrite(out_path, img)
        print(f"✅ Saved boxed image: {out_path}")
    else:
        print(f"ℹ Skipped {img_file}, no {TARGET_CLASS} found.")

print("✅ All images processed and saved to ./unit_test_output!")
