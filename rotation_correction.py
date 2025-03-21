import numpy as np
import open3d as o3d
import json
import os
import logging

# Configure your logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

np.random.seed(42)
o3d.utility.random.seed(42)

# Supported Python versions for open3d:
# 3.8, 3.9, 3.10 and 3.11


def rotation_matrix(axis, theta):
    """
    Create a rotation matrix for a given axis and angle.

    :param axis: 'x', 'y', or 'z'
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")


def process_ply_file(ply_file_path):
    try:

        pcd = o3d.io.read_point_cloud(ply_file_path)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_center = pcd.get_center()
        pcd.translate(-pcd_center)

        R = rotation_matrix('x', np.pi)
        pcd.rotate(R, center=(0, 0, 0))

        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        # o3d.visualization.draw_geometries([pcd], window_name="pcd")

        filtered_pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=16, std_ratio=10)

        pcd = filtered_pcd
        plane_model, _ = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=10000)

        a, b, c, _ = plane_model
        plane_normal = np.array([a, b, c])

        normal_vector = - plane_normal

        y_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(normal_vector, y_axis)

        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.dot(
            normal_vector, y_axis) / (np.linalg.norm(normal_vector) * np.linalg.norm(y_axis)))

        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * rotation_angle)

        pcd.rotate(R, center=(0, 0, 0))
        pcd_center = pcd.get_center()
        pcd.translate(-pcd_center)

        T = np.eye(4)
        T[:3, :3] = R

        threejs_matrix = T.flatten().tolist()
        return threejs_matrix

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


def save_matrix_to_json(matrix, filename='matrix_4_4.json'):
    matrix_data = {"matrix": matrix}
    with open(filename, 'w') as json_file:
        json.dump(matrix_data, json_file)
    logger.info(f"Matrix has been saved to {filename}")


def get_rotation(ply_file_path, local_matrix_path='matrix_4_4.json'):
    """
    Process a PLY file to extract a rotation matrix and save it to JSON.
    
    Args:
        ply_file_path: Path to the PLY file
        local_matrix_path: Path to save the matrix JSON
        
    Returns:
        None
    """
    matrix = process_ply_file(ply_file_path)
    if matrix:
        save_matrix_to_json(matrix, local_matrix_path)
    else:
        placeholder_matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        save_matrix_to_json(placeholder_matrix, local_matrix_path)
        logger.error("An error occurred. Placeholder matrix has been saved.")


if __name__ == "__main__":
     ply_file = "ply/model_25032.ply"
     get_rotation(ply_file)
