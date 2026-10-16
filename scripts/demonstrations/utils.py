import numpy as np
import cv2 as cv

def rotation_matrix_from_directions(z_dir: np.ndarray, y_dir: np.ndarray) -> np.ndarray:
    # Normalize Z axis
    z_norm = np.linalg.norm(z_dir)
    if z_norm < 0.0001:
        raise ValueError()
    z_dir /= z_norm

    # Make Y perpendicular to Z (Gram-Schmidt)
    y_dir = y_dir - np.dot(y_dir, z_dir) * z_dir
    y_norm = np.linalg.norm(y_dir)
    if y_norm < 0.0001:
        raise ValueError()
    y_dir /= y_norm

    # Compute X to obtain a right-handed coordinate system
    x_dir = np.cross(y_dir, z_dir)
    x_dir /= np.linalg.norm(x_dir)

    # Recompute Y for a perfectly orthonormal basis
    y_dir = np.cross(z_dir, x_dir)

    # Columns are the local axes in world coordinates
    return np.column_stack((x_dir, y_dir, z_dir))


def transform_matrix_from_vectors(rvec: np.ndarray, tvec: np.ndarray):
    R, _ = cv.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T
