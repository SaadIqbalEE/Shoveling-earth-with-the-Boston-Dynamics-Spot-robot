# Generate the image including obstacles
# Scenario: 2D image (finite colors), multiple blocks (obstacles)
import cv2
import numpy as np
import os
import random

width, height = 640, 480
image = np.ones((height, width, 3), dtype=np.uint8) * 255
rect_width, rect_height = 30, 20  # rectangle

for _ in range(4):
    top_left_x = random.randint(0, width - rect_width)
    top_left_y = random.randint(0, height - rect_height)
    bottom_right_x = top_left_x + rect_width
    bottom_right_y = top_left_y + rect_height
    cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 0), -1)

output_folder = '2d_output_images'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, '2d_generated_image.png')
cv2.imwrite(output_path, image)

print(f"Saved At：{output_path}")
