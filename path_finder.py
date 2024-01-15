import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to load images from a directory
def load_images(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            images.append(image)
    return images

# Function to calculate fundamental matrix and camera poses
def calculate_fundamental_matrix_and_poses(images1, images2):
    # Feature detection and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(images1, None)
    kp2, des2 = sift.detectAndCompute(images2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 2 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate fundamental matrix
    fundamental_matrix, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    try:
        # Recover camera poses from the fundamental matrix
        _, R, t, _ = cv2.recoverPose(fundamental_matrix, pts1, pts2)

        return fundamental_matrix, R, t, 1

    except cv2.error:
        return -1, 0, 0, -1

# Function to convert rotation matrix to yaw angle
def rotation_matrix_to_yaw(R):
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return yaw

# Function to create homogeneous transformation matrix from translation and rotation
def create_homogeneous_matrix(t, R):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R
    homogeneous_matrix[:3, 3] = t.flatten()
    return homogeneous_matrix

if __name__ == "__main__":
    # Directory containing the saved images
    image_directory = "images"

    # Load images
    images = load_images(image_directory)
    print(len(images))

    # Lists to store the converted poses
    homogeneous_matrices = []

    # Identity matrix as the initial pose
    current_homogeneous_matrix = np.eye(4)

    # Kalman Filter Initialization
    kalman_filter = cv2.KalmanFilter(6, 3)
    kalman_filter.transitionMatrix = np.eye(6, dtype=np.float32)
    kalman_filter.measurementMatrix = np.eye(3, 6, dtype=np.float32)
    kalman_filter.processNoiseCov = 1e-4 * np.eye(6, dtype=np.float32)
    kalman_filter.measurementNoiseCov = 1e-3 * np.eye(3, dtype=np.float32)

    # Initial state
    state = np.zeros((6, 1), dtype=np.float32)
    kalman_filter.statePost = state

    # Kalman Filter Loop
    for i in range(len(images) - 50):
        fundamental_matrix, R, t, flag = calculate_fundamental_matrix_and_poses(images[i], images[i + 1])

        if flag == -1:
            continue

        # Convert translation vector and rotation matrix to homogeneous matrix
        homogeneous_matrix = create_homogeneous_matrix(t, R)
        current_homogeneous_matrix = np.dot(current_homogeneous_matrix, homogeneous_matrix)

        # Append the current homogeneous matrix to the list
        homogeneous_matrices.append(current_homogeneous_matrix)

    # Extract positions from homogeneous matrices
    positions = np.array([homogeneous_matrix[:3, 3] for homogeneous_matrix in homogeneous_matrices])

    # Calculate mean and variance of homogeneous matrices and positions
    mean_homogeneous_matrix = np.mean(homogeneous_matrices, axis=0)
    var_homogeneous_matrix = np.var(homogeneous_matrices, axis=0)

    mean_positions = np.mean(positions, axis=0)
    var_positions = np.var(positions, axis=0)

    print("\nMean Homogeneous Matrix:")
    print(mean_homogeneous_matrix)

    print("\nVariance Homogeneous Matrix:")
    print(var_homogeneous_matrix)

    print("\nMean Positions:")
    print(mean_positions)

    print("\nVariance Positions:")
    print(var_positions)

    # Plot the positions in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], marker='o', linestyle='-', color='b')
    plt.show()
