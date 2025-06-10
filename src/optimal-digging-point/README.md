# Optimal Digging Point

This directory contains scripts for testing and implementing algorithms related to obstacle avoidance and depth calculation. Below is a brief description of each script:

## Scripts

- **test_image_generation.py**: This script was initially used for testing purposes. It generates a simple 2D image with obstacles to test pixel-based obstacle avoidance algorithms.

- **max_distance_choice.py**: Implements an algorithm to find the point in a simple 2D image that is farthest from all obstacles. This is useful for testing basic obstacle avoidance strategies.

- **continous_rrt.py**: This script was intended for implementing RRT (Rapidly-exploring Random Tree) in virtual scenes. However, it was found to be ineffective in such environments and was not fully implemented.

- **coordinate_transform_raft_stereo.py, coordinate_transform_train_free.py, coordinate_transform.py**: These scripts implement different methods (traditional and deep learning-based) for calculating disparity and depth from stereo images.

- **spot_perception.py**: Integrates various perception models and incorporates them into our final demonstration environment. It includes object detection and depth mapping functionalities.

## Note

This README and the associated files are intended for demonstration purposes, showcasing parts of the pipeline. For a complete demo, please refer to the files in the `simulation` folder. 