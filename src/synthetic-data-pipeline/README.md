# Synthetic Data Pipeline

This directory contains scripts for generating and testing synthetic data using the Isaac Sim environment. Below is a brief description of each script:

## Scripts

- **print_object.py**: This script is used for pre-confirmation of the environment in Isaac Sim, ensuring that objects are present. It has been used extensively for unit testing to verify the presence and properties of objects within the simulation.

- **image_generation.py**: This script completes the scene generation task in an initial cube environment. It sets up the simulation, generates images, and processes them for object detection tasks.

- **image_generation_from_Environment1_1.py**: This script performs data generation tasks in a rock environment. While the full dataset is not replicated here, the same code can be adapted to generate corresponding datasets by modifying the paths.

## Note

This README and the associated files are intended for demonstration purposes, showcasing parts of the pipeline. For a complete demo, please refer to the files in the `simulation` folder. 