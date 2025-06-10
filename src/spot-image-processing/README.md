# Spot Image Processing

This directory contains scripts for processing images using YOLO and stereo vision techniques. Below is a brief description of each script:

## Scripts

- **yolo_train.py**: This script is used to train a YOLO model using synthetic data formatted in the YOLO style. It is designed to work with datasets that have been pre-processed into the YOLO format, allowing for efficient training of object detection models.

- **yolo_unit_test.py**: This script performs unit testing on a single data entry. It is useful for verifying the correctness of the YOLO model's predictions on individual samples, ensuring that the model behaves as expected.

- **spot_perception.py**: This script is responsible for obtaining stereo images within a ROS (Robot Operating System) environment. It matches the stereo images to calculate the disparity, which is then used to construct a 3D map for improved projection. This process enhances the robot's perception capabilities by providing depth information from the stereo images.

## Subdirectories

- **spot_image_object_detection/**: Contains resources and scripts related to object detection tasks.

- **spot_image_semantic_segmentation/**: Contains resources and scripts related to semantic segmentation tasks.

This setup is designed to facilitate advanced image processing and perception tasks using the Spot robot's imaging capabilities.

## Note

This README and the associated files are intended for demonstration purposes, showcasing parts of the pipeline. For a complete demo, please refer to the files in the `simulation` folder. 