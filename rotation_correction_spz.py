import numpy as np
import open3d as o3d
import json
import gzip
import struct
import logging
from spz_py.constant import SPZ_MAGIC, SPZ_VERSION, dim_for_degree

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

np.random.seed(42)
o3d.utility.random.seed(42)

def rotation_matrix(axis, theta):
    """
    Create a rotation matrix for a given axis and angle.
    
    :param axis: 'x', 'y', or 'z'
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

def load_spz(file_path: str) -> dict:
    with gzip.open(file_path, 'rb') as f:
        data = f.read()

    # Parse header
    header = data[:16]
    magic, version, numPoints, shDegree, fractionalBits, flags, reserved = struct.unpack('<I I I B B B B', header)
    if magic != SPZ_MAGIC or version != SPZ_VERSION:
        raise ValueError("Invalid SPZ file header")

    # Calculate sizes
    positions_size = numPoints * 9
    scales_size = numPoints * 3
    rotations_size = numPoints * 3
    alphas_size = numPoints * 1
    colors_size = numPoints * 3
    sh_dim = dim_for_degree(shDegree)
    sh_size = numPoints * sh_dim * 3 if shDegree > 0 else 0

    # Extract sections
    offset = 16
    positions_data = data[offset:offset + positions_size]; offset += positions_size
    scales_data = data[offset:offset + scales_size]; offset += scales_size
    rotations_data = data[offset:offset + rotations_size]; offset += rotations_size
    alphas_data = data[offset:offset + alphas_size]; offset += alphas_size
    colors_data = data[offset:offset + colors_size]; offset += colors_size
    sh_data = data[offset:offset + sh_size]

    # Convert positions
    positions_list = []
    for i in range(numPoints * 3):
        start = i * 3
        byte_arr = positions_data[start:start + 3]
        fixed_val = int.from_bytes(byte_arr, 'little', signed=True)
        scale_factor = 1 << fractionalBits
        positions_list.append(fixed_val / scale_factor)

    # Convert scales
    scales_list = []
    for byte in scales_data:
        scale = (byte / 16.0) - 10.0  # Simplified linear decoding
        scales_list.append(scale)

    # Convert rotations
    rotations_list = []
    for byte in rotations_data:
        rot_byte = byte if byte < 128 else byte - 256
        rot_float = rot_byte / 127.0
        rotations_list.append(rot_float)
    rotations_arr = np.array(rotations_list).reshape(numPoints, 3)
    w = np.sqrt(np.maximum(0, 1 - np.sum(rotations_arr**2, axis=1)))
    full_rotations = np.hstack((rotations_arr, w[:, np.newaxis]))
    rotations = full_rotations.ravel().tolist()

    # Convert alphas
    alphas_list = []
    for byte in alphas_data:
        alpha = byte / 255.0
        alphas_list.append(alpha)

    # Convert colors
    colors_list = []
    for byte in colors_data:
        color = (byte - 127.5) / 38.25
        colors_list.append(color)

    # Convert spherical harmonics
    sh_list = []
    for byte in sh_data:
        sh_signed = byte if byte < 128 else byte - 256
        sh_list.append(sh_signed)

    # Construct GaussianCloud dictionary
    gaussian_cloud = {
        "numPoints": numPoints,
        "shDegree": shDegree,
        "antialiased": bool(flags & 1),  # Corrected typo from "antialiaased"
        "positions": positions_list,
        "scales": scales_list,
        "rotations": rotations,
        "alphas": alphas_list,
        "colors": colors_list,
        "sh": sh_list
    }
    return gaussian_cloud

def process_point_cloud(pcd):
    try:
        logger.info("Estimating normals")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.translate(-pcd.get_center())

        logger.info("Applying initial rotation")
        R = rotation_matrix('x', np.pi)
        pcd.rotate(R, center=(0, 0, 0))

        logger.info("Downsampling point cloud")
        pcd = pcd.voxel_down_sample(voxel_size=0.02)

        logger.info("Removing outliers")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=10)

        logger.info("Segmenting plane")
        plane_model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=10000)
        a, b, c, _ = plane_model
        normal_vector = -np.array([a, b, c])

        y_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(normal_vector, y_axis)
        norm = np.linalg.norm(rotation_axis)
        if norm < 1e-6:
            logger.warning("Rotation axis norm too small, skipping rotation")
            R = np.eye(3)
        else:
            rotation_axis /= norm
            angle = np.arccos(np.clip(np.dot(normal_vector, y_axis), -1.0, 1.0))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

        logger.info("Applying final rotation and centering")
        pcd.rotate(R, center=(0, 0, 0))
        pcd.translate(-pcd.get_center())

        T = np.eye(4)
        T[:3, :3] = R
        return T.flatten().tolist()

    except Exception as e:
        logger.error(f"Error: {e}")
        return None

def save_matrix_to_json(matrix, filename='matrix_4_4_spz.json'):
    with open(filename, 'w') as json_file:
        json.dump({"matrix": matrix}, json_file)
    logger.info(f"Matrix has been saved to {filename}")

def get_rotation_from_spz(spz_file_path, local_matrix_path='matrix_4_4_spz.json'):
    try:
        logger.info(f"Loading SPZ file: {spz_file_path}")
        gaussian_cloud = load_spz(spz_file_path)
        if not gaussian_cloud or gaussian_cloud["numPoints"] == 0:
            logger.error("Failed to load SPZ file or empty point cloud")
            save_matrix_to_json([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], local_matrix_path)
            return

        logger.info("Creating point cloud from positions")
        points = np.array(gaussian_cloud["positions"]).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        matrix = process_point_cloud(pcd)
        if matrix:
            save_matrix_to_json(matrix, local_matrix_path)
        else:
            logger.error("Error occurred. Placeholder matrix saved.")
            save_matrix_to_json([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], local_matrix_path)

    except Exception as e:
        logger.error(f"Error: {e}")
        save_matrix_to_json([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], local_matrix_path)

if __name__ == "__main__":
    print(f"Open3D version: {o3d.__version__}")
    spz_file = "model_18567.spz"
    get_rotation_from_spz(spz_file)