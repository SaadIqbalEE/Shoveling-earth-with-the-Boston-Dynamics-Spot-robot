import os
import y
from pathlib import Path

from yolov5 import train
from yolov5.utils.general import check_dataset

project_dir = Path(__file__).parent.resolve()
data_dir = project_dir / "synthetic_dataset" / "yolo"

train_images = str(data_dir / "train" / "images")
val_images = str(data_dir / "val" / "images")
train_labels = str(data_dir / "train" / "labels")
val_labels = str(data_dir / "val" / "labels")

data_yaml_path = project_dir / "yolo_dataset.yaml"

data_config = {
    "train": train_images,
    "val": val_images,
    "nc": 1,  # 类别数量（你要根据真实数据修改）
    "names": ["Rock_7_solid"],  # 类别名称（你要根据真实数据修改）
}

with open(data_yaml_path, "w") as f:
    yaml.dump(data_config, f)

check_dataset(data_config)  # 检查路径正确性

# ==== 3. 调用训练 ====
train.run(
    data=str(data_yaml_path),
    imgsz=640,
    batch_size=16,
    epochs=50,
    weights="yolov5s.pt"  # 5m/5l is also proper
    project=str(project_dir),
    name="synthetic_yolo_experiment",
    exist_ok=True
)
