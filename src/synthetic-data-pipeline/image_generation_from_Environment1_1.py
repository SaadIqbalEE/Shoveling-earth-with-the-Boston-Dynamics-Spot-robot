from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import UsdGeom, Gf, Usd
import omni.replicator.core as rep
import math, random, os, json, glob, shutil
from pathlib import Path
import numpy as np
import cv2

# === CONFIG ===
NUM_IMAGES = 10
IMAGE_SIZE = (640, 480)
OUTPUT_DIR = "/home/rllab/omni.replicator_out/temp_output"
TARGET_CLASS = "Rock_7_solid"
SPLITS = [("train", 0.7), ("val", 0.2), ("test", 0.1)]

radii = [1.0, 2.0, 3.0]
height_range = (0.5, 1.5)
yaw_noise = 5.0
pitch_noise = 2.0

split_dirs = {}
for split, _ in SPLITS:
    img_dir = Path(OUTPUT_DIR) / split / "images"
    label_dir = Path(OUTPUT_DIR) / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    split_dirs[split] = (img_dir, label_dir)

# === LOAD STAGE ===
usd_path = "/home/rllab/Desktop/25P24/IsaacEnvironments/Environment1_1.usd"
omni.usd.get_context().open_stage(usd_path)
stage = omni.usd.get_context().get_stage()

target_prim_path = "/World/Objects/Rock_7_solid"
target_prim = stage.GetPrimAtPath(target_prim_path)
if not stage.GetPrimAtPath(target_prim_path).IsValid():
    print(f"❌ Target object {TARGET_CLASS} not found!")
    simulation_app.close()
    exit(1)
print(f"✅ Target object {TARGET_CLASS} found.")

# semantics_iface = omni.usd.get_context().get_semantics_interface()
# semantics = semantics_iface.get_semantics_by_prim_path(target_prim_path)

# if semantics:
#     print(f"✅ Found semantic label on {target_prim_path}: {semantics}")
# else:
#     print(f"⚠ No semantic label found on {target_prim_path}")

# if not target_prim.HasAPI(UsdGeom.SemanticsAPI):
#     UsdGeom.SemanticsAPI.Apply(target_prim)
#     semantics = UsdGeom.SemanticsAPI(target_prim)
#     semantics.CreateSemanticTypeAttr("class")
#     semantics.CreateSemanticDataAttr("Rock_7_solid")
#     print("✅ Semantic label applied via script.")
# else:
#     print("✅ Semantic label already exists.")
plane_center = Gf.Vec3f(0, -2, 0)

# === ADD LIGHT ===
from pxr import UsdLux

light_prim = UsdLux.DistantLight.Define(stage, "/World/DefaultLight")
light_prim.AddTranslateOp().Set((0, 5, 0))
light_prim.CreateIntensityAttr(20000)
light_prim.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))


# === CREATE CAMERAS ===
for i in range(NUM_IMAGES):
    r = random.choice(radii)
    angle = random.uniform(0, 2 * math.pi)
    cam_x = plane_center[0] + r * math.cos(angle)
    cam_y = plane_center[1] + random.uniform(*height_range)
    cam_z = plane_center[2] + r * math.sin(angle)

    cam_path = f"/World/MyCameras/Camera_{i}"
    camera_prim = UsdGeom.Xform.Define(stage, cam_path)
    camera_prim.AddTranslateOp().Set((cam_x, cam_y, cam_z))

    direction = plane_center - Gf.Vec3f(cam_x, cam_y, cam_z)
    direction.Normalize()

    yaw = math.atan2(direction[2], direction[0]) + math.radians(random.uniform(-yaw_noise, yaw_noise))
    pitch = math.asin(direction[1]) + math.radians(random.uniform(-pitch_noise, pitch_noise))

    rot_quat = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    rot_quat *= Gf.Quatf(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    rot_quat *= Gf.Quatf(math.cos(pitch / 2), math.sin(pitch / 2), 0.0, 0.0)
    camera_prim.AddOrientOp().Set(rot_quat)
    # print(stage.GetPrimAtPath(cam_path).IsValid())
    # print(cam_path)

# === SETUP REPLICATOR ===
bbox_annot = rep.annotators.get("bounding_box_2d_tight")
render_products = []
for i in range(NUM_IMAGES):
    cam_path = f"/World/MyCameras/Camera_{i}"
    render_product = rep.create.render_product(cam_path, IMAGE_SIZE)
    bbox_annot.attach(render_product)
    render_products.append(render_product)
print(render_products)
writer = rep.WriterRegistry.get("BasicWriter")  # most used output tool
writer.initialize(output_dir=OUTPUT_DIR, rgb=True, bounding_box_2d_tight=True)
writer.attach(render_products)

print(len(render_products))

print("✅ Forcing simulation updates...")
for _ in range(NUM_IMAGES):
    simulation_app.update()

rep.orchestrator.run(NUM_IMAGES)

# === CHECK OUTPUT ===
rgb_files = sorted(glob.glob(f"{OUTPUT_DIR}/rgb/*.png"))
json_files = sorted(glob.glob(f"{OUTPUT_DIR}/bounding_box_2d_tight/*.json"))
print(f"✅ RGB images: {len(rgb_files)}, JSONs: {len(json_files)}")

# === DRAW FIRST IMAGE WITH BOX ===
vis_dir = Path("./unit_test_output")
vis_dir.mkdir(parents=True, exist_ok=True)
if rgb_files and json_files:
    first_img = cv2.imread(rgb_files[0])
    with open(json_files[0], 'r') as f:
        ann_data = json.load(f)

    for ann in ann_data.get("bounding_box_2d_tight", []):
        if ann["semanticLabel"] != TARGET_CLASS:
            continue
        x, y, w, h = int(ann["x"]), int(ann["y"]), int(ann["width"]), int(ann["height"])
        cv2.rectangle(first_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(first_img, TARGET_CLASS, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    vis_path = vis_dir / "first_image_with_box.png"
    cv2.imwrite(str(vis_path), first_img)
    print(f"✅ Saved first image with bounding box: {vis_path}")

# === YOLO CONVERSION FUNCTION ===
def convert_to_yolo(json_path, label_path, img_w, img_h):
    with open(json_path, 'r') as f:
        data = json.load(f)
    with open(label_path, 'w') as out:
        for ann in data.get("bounding_box_2d_tight", []):
            if ann["semanticLabel"] != TARGET_CLASS:
                continue
            x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            out.write(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

# === DISTRIBUTE FILES INTO SPLITS ===
for idx, img_path in enumerate(rgb_files):
    split = np.random.choice([s for s, _ in SPLITS], p=[r for _, r in SPLITS])
    img_dest = split_dirs[split][0] / Path(img_path).name
    label_dest = split_dirs[split][1] / (Path(img_path).stem + ".txt")
    json_path = Path(OUTPUT_DIR) / "bounding_box_2d_tight" / (Path(img_path).stem + ".json")

    shutil.copy(img_path, img_dest)
    convert_to_yolo(json_path, label_dest, IMAGE_SIZE[0], IMAGE_SIZE[1])

print("✅ Synthetic dataset generation complete with YOLO labels!")
simulation_app.close()
