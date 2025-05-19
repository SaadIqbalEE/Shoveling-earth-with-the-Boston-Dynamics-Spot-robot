# Camera Used: Camera: /World/spot/body/center_camera/center_camera
# Spot Angle: Z -40--80 X 0 Y 0
# Cube Scale: 0.15 0.15 0.15 (Two CUbes)

# Isaac Sim Synthetic Data Generation Script for Object Detection (YOLO-ready)
# Requirements: This script must be run with Isaac Sim's python.sh
# Example: /home/rllab/IsaacSim/python.sh generate_yolo_data.py
# Issac Sim Full 4.5.0

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import Gf, UsdGeom
import omni.replicator.core as rep
from omni.replicator.core import annotators
import numpy as np
import random
import os
from pathlib import Path

# === CONFIGURATION ===
CAMERA_PRIM_PATH = "/World/spot/body/center_camera/center_camera"
CUBE1_PATH = "/World/Cube"
CUBE2_PATH = "/World/Cube_2"
SPOT_PATH = "/World/spot"
OUTPUT_DIR = "./synthetic_dataset"
NUM_IMAGES = 10
IMAGE_SIZE = (640, 480)
SPLITS = [("train", 0.7), ("val", 0.1), ("test", 0.2)]

sandbox_min = np.array([0.3, 0.0, 0.3])
sandbox_max = np.array([0.7, 0.0, 0.7])
cube_radius = 0.15 / 2
min_dist = cube_radius * 3

# === SCENE LOADING ===
USD_PATH = "/home/rllab/Desktop/25P24/proj_sim1/testEnvironment.usd"
omni.usd.get_context().open_stage(USD_PATH)

# === UTILITY FUNCTIONS ===
def sample_cube_positions():
    for _ in range(100):
        pos1 = np.random.uniform(sandbox_min + cube_radius, sandbox_max - cube_radius)
        pos2 = np.random.uniform(sandbox_min + cube_radius, sandbox_max - cube_radius)
        if np.linalg.norm(pos1[[0,2]] - pos2[[0,2]]) >= min_dist:
            return pos1.tolist(), pos2.tolist()
    raise RuntimeError("Failed to sample non-colliding cube positions")

def sample_spot_rotation():
    z_rot = random.uniform(-80, -40)
    return [0.0, 0.0, z_rot]

def get_split(index):
    p = index / NUM_IMAGES
    cumulative = 0
    for name, ratio in SPLITS:
        cumulative += ratio
        if p < cumulative:
            return name
    return SPLITS[-1][0]

# === SETUP OUTPUT STRUCTURE ===
yolo_paths = {}
for split, _ in SPLITS:
    img_dir = Path(OUTPUT_DIR) / "yolo" / split / "images"
    label_dir = Path(OUTPUT_DIR) / "yolo" / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    yolo_paths[split] = (img_dir, label_dir)

# === CREATE RENDER PRODUCT ===
render_product = rep.create.render_product(CAMERA_PRIM_PATH, IMAGE_SIZE)
bbox_annot = rep.annotators.get("bounding_box_2d_loose")
bbox_annot.attach([render_product])
# === GENERATE IMAGES ===
with rep.new_layer():

    cube1 = rep.get.prim_at_path(CUBE1_PATH)
    cube2 = rep.get.prim_at_path(CUBE2_PATH)
    spot = rep.get.prim_at_path(SPOT_PATH)

    def randomize_scene():
        c1, c2 = sample_cube_positions()
        rz = sample_spot_rotation()
        
        with cube1:
            rep.modify.pose(position=c1)
        with cube2:
            rep.modify.pose(position=c2)
        with spot:
            rep.modify.pose(rotation=rz)

    rep.trigger.on_frame(randomize_scene)

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=OUTPUT_DIR + "/raw", rgb=True, bounding_box_2d_loose=True)
    writer.attach([render_product])

    rep.orchestrator.run(NUM_IMAGES)

# === EXTRACT YOLO FROM JSON ===
import json
from PIL import Image

def convert_to_yolo(json_path, image_path, label_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    im = Image.open(image_path)
    width, height = im.size

    with open(label_path, 'w') as out:
        for ann in data["bounding_box_2d_loose"]:
            x, y, w, h = ann["x"] / width, ann["y"] / height, ann["width"] / width, ann["height"] / height
            cx, cy = x + w/2, y + h/2
            out.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")  # class_id = 0 (cube)

# Parse output
raw_img_dir = Path(OUTPUT_DIR) / "raw" / "rgb"
raw_ann_dir = Path(OUTPUT_DIR) / "raw" / "bounding_box_2d_loose"
all_imgs = sorted(raw_img_dir.glob("*.png"))

for idx, img_path in enumerate(all_imgs):
    print("one image in dataset is saved")
    stem = img_path.stem
    split = get_split(idx)
    json_path = raw_ann_dir / (stem + ".json")
    label_path = yolo_paths[split][1] / (stem + ".txt")
    yolo_img_path = yolo_paths[split][0] / img_path.name
    os.rename(img_path, yolo_img_path)
    convert_to_yolo(json_path, yolo_img_path, label_path)

print("Data generation complete. Images + YOLO labels saved.")
simulation_app.close()