# Function: Given a 2D image, choose a point whose possess biggest Euclidean Distance

import cv2
import numpy as np

def find_farthest_point(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Create a binary mask where obstacles (black pixels) are marked
    _, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY_INV)

    # Compute the distance transform
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Find the coordinates of the farthest point
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)

    # Draw the farthest point on the image
    image_with_point = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(image_with_point, max_loc, 5, (0, 0, 255), -1)

    # Save the image with the farthest point marked
    output_path = '2d_output_images/2d_generated_image_with_point.png'
    cv2.imwrite(output_path, image_with_point)
    print(f"Image with farthest point saved at: {output_path}")

    # Return the coordinates of the farthest point
    return max_loc

def find_farthest_point_manual(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Get coordinates of all black pixels (obstacles)
    black_pixels = np.column_stack(np.where(image == 0))

    # Initialize variables to track the farthest point
    max_min_distance = -1
    farthest_point = None

    # Iterate over each white pixel
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == 255:  # White pixel
                # Calculate the distance to the nearest black pixel
                min_distance = np.min(np.sqrt((black_pixels[:, 0] - y) ** 2 + (black_pixels[:, 1] - x) ** 2))

                # Update the farthest point if this is the largest minimum distance found
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    farthest_point = (x, y)

    # Draw the farthest point on the image
    image_with_point = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(image_with_point, farthest_point, 5, (0, 0, 255), -1)

    # Save the image with the farthest point marked
    output_path = '2d_output_images/2d_generated_image_with_point_manual.png'
    cv2.imwrite(output_path, image_with_point)
    print(f"Image with farthest point saved at: {output_path}")

    # Return the coordinates of the farthest point
    return farthest_point

# Example usage
image_path = '2d_output_images/2d_generated_image.png'
farthest_point = find_farthest_point(image_path)
print(f"The farthest point from all obstacles is at: {farthest_point}")

farthest_point_manual = find_farthest_point_manual(image_path)
print(f"The farthest point from all obstacles is at: {farthest_point_manual}")