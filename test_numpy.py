import numpy as np

# Example array (replace this with your actual array)
keypoints = np.array([[[604.78, 488.72],
                       [671.38, 405.92],
                       [526.46, 420.46],
                       [767.73, 422.2],
                       [436.33, 454.42],
                       [915.23, 683.14],
                       [352, 685.22],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0]]])

# Access keypoints for the first person
person_keypoints = keypoints[0]  # Shape (17, 2)

# Loop through each keypoint and access x, y coordinates
for idx, (x, y) in enumerate(person_keypoints):
    if x != 0 and y != 0:  # Ignore keypoints with (0, 0) values
        print(f"Keypoint {idx}: x={x}, y={y}")
