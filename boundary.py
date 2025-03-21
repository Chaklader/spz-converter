import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import alphashape
from scipy import spatial
from sklearn.cluster import DBSCAN
import open3d as o3d


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
np.random.seed(42)

rotation_file_path = 'matrix_4_4.json'
ply_file_path = 'ply/model_25032.ply'

def load_file(file_path: str) -> dict:
    """
    Load data from a file based on extension
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.ply':
        from spz_py.ply_loader import load_ply
        with open(file_path, 'rb') as f:
            return load_ply(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def calculate_rough_area(points_2d):
    """Calculate approximate area of the point cloud footprint"""
    try:
        hull = spatial.ConvexHull(points_2d)
        return hull.volume  # In 2D, volume is area
    except Exception as e:
        min_x, min_y = np.min(points_2d, axis=0)
        max_x, max_y = np.max(points_2d, axis=0)
        return (max_x - min_x) * (max_y - min_y)  # Bounding box area


def estimate_ground_level(y_values):
    """Estimate ground level from height distribution"""
    hist, bin_edges = np.histogram(y_values, bins=100)
    for i in range(len(hist)):
        if hist[i] > len(y_values) * 0.01: 
            return bin_edges[i]
    return np.min(y_values)


def select_filtering_parameters(points_2d, positions):
    """
    Select appropriate filtering parameters based on point cloud characteristics.
    Automatically detects building type and recommends filtering parameters.
    """
    logger.info("Analyzing point cloud to determine building characteristics...")
    
    point_count = len(points_2d)
    area = calculate_rough_area(points_2d)
    
    y_values = positions[:, 1]
    height_range = np.max(y_values) - np.min(y_values)
    
    tree = spatial.KDTree(points_2d)
    sample_size = min(5000, len(points_2d))
    sample_indices = np.random.choice(len(points_2d), sample_size, replace=False)
    sample_points = points_2d[sample_indices]
    
    height_hist, _ = np.histogram(y_values, bins=50)
    height_peaks = np.where(height_hist > np.percentile(height_hist, 90))[0]
    has_distinct_levels = len(height_peaks) >= 2
    
    try:
        eps = np.sqrt(area / point_count) * 5  # Adaptive epsilon based on average point spacing
        clustering = DBSCAN(eps=eps, min_samples=10).fit(points_2d)
        cluster_count = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        has_internal_gaps = cluster_count > 1
    except Exception:
        has_internal_gaps = False
    
    regular_geometry = False
    try:
        if len(sample_points) > 100:
            _, indices = tree.query(sample_points, k=5)
            
            angles = []
            for i, idx_array in enumerate(indices):
                if len(idx_array) >= 3:  # Need at least 3 points for angles
                    p0 = sample_points[i]
                    for j in range(1, len(idx_array)-1):
                        p1 = points_2d[idx_array[j]]
                        p2 = points_2d[idx_array[j+1]]
                        v1 = p1 - p0
                        v2 = p2 - p0
                        dot = np.dot(v1, v2)
                        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                        if norm > 0:
                            angle = np.arccos(max(-1, min(1, dot/norm)))
                            angles.append(angle)
            
            if angles:
                angle_std = np.std(angles)
                regular_geometry = angle_std < 0.3  # Low std indicates regular patterns
    except Exception:
        regular_geometry = False
    
    params = {
        "building_type": "general",
        "sor_k": 20,             # Statistical outlier removal K neighbors
        "sor_std_ratio": 2.0,    # Statistical outlier removal std dev ratio
        "height_tolerance": 1.5,  # Meters above detected maximum to allow
        "ground_tolerance": 0.5,  # Meters below ground level to allow
        "preserve_internal_gaps": False,
        "density_threshold": 0.5
    }
    
    if height_range > 15 and area > 500:
        params["building_type"] = "apartment" 
        params["sor_k"] = 30
        params["height_tolerance"] = 2.0
        logger.info(f"Detected building type: Apartment/Tall Building (height: {height_range:.1f}m)")
    
    elif area > 1000 and height_range < 10:
        params["building_type"] = "industrial"
        params["sor_k"] = 25
        params["sor_std_ratio"] = 2.5
        params["density_threshold"] = 0.3
        logger.info(f"Detected building type: Industrial/Commercial (area: {area:.1f}m²)")
    
    elif regular_geometry and height_range < 8:
        params["building_type"] = "greenhouse"
        params["sor_k"] = 15
        params["sor_std_ratio"] = 1.8
        logger.info("Detected building type: Greenhouse/Regular Structure")
    
    elif has_internal_gaps and area > 300:
        params["building_type"] = "complex_residential"
        params["preserve_internal_gaps"] = True
        params["sor_k"] = 20
        logger.info("Detected building type: Complex Residential (with courtyards/pools)")
    
    elif has_distinct_levels and height_range < 12:
        params["building_type"] = "multi_level_house"
        params["sor_k"] = 20
        logger.info("Detected building type: Multi-level House")
    
    elif area < 300 and height_range < 10:
        params["building_type"] = "house"
        params["sor_k"] = 15
        logger.info("Detected building type: House/Small Building")

    else:
        params["building_type"] = "general"
        logger.info("Detected building type: General Structure")

    logger.info(f"Point cloud statistics: {point_count} points, {area:.1f}m² area, {height_range:.1f}m height")
    logger.info(f"Selected parameters: k={params['sor_k']}, std_ratio={params['sor_std_ratio']}")
    
    return params


def apply_rotation_matrix(positions, matrix):
    """
    Apply a 4x4 rotation matrix to 3D positions
    
    Args:
        positions: Nx3 array of positions
        matrix: Flattened 4x4 matrix as a list
    
    Returns:
        Nx3 array of rotated positions
    """
    logger.info(f"Applying rotation correction to {len(positions)} points")
    
    matrix_4x4 = np.array(matrix).reshape(4, 4)
    rotation_matrix = matrix_4x4[:3, :3]
    
    rotated_positions = np.zeros_like(positions)
    for i in range(len(positions)):
        rotated_positions[i] = rotation_matrix.dot(positions[i])
    
    logger.info("Rotation correction applied")
    return rotated_positions


def remove_statistical_outliers(points_2d, positions, k=20, std_ratio=2.0):
    """
    First pass: Remove outliers using statistical analysis of neighbor distances.
    - points_2d: Nx2 array of points in XZ plane
    - positions: Nx3 array of original 3D positions
    - k: Number of neighbors to analyze
    - std_ratio: Standard deviation threshold
    """
    logger.info(f"First pass: Removing statistical outliers (k={k}, std_ratio={std_ratio})...")
    
    start_count = len(points_2d)
    
    tree = spatial.KDTree(points_2d)
    
    distances, _ = tree.query(points_2d, k=min(k+1, len(points_2d)))
    
    if distances.shape[1] < k+1:
        logger.warning(f"Not enough points for k={k}, using k={distances.shape[1]-1} instead")
        k = distances.shape[1] - 1
    
    distances = distances[:, 1:]
    
    mean_distances = np.mean(distances, axis=1)
    
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances < threshold
    
    filtered_points_2d = points_2d[inlier_mask]
    filtered_positions = positions[inlier_mask]
    
    removed_count = start_count - len(filtered_points_2d)
    removal_percentage = (removed_count / start_count) * 100 if start_count > 0 else 0
    
    logger.info(f"Removed {removed_count} statistical outliers ({removal_percentage:.2f}%)")
    
    return filtered_points_2d, filtered_positions


def apply_conditional_filters(points_2d, positions, params):
    """
    Second pass: Apply conditional filters based on building type and point cloud characteristics.
    - points_2d: Nx2 array of points in XZ plane from first pass
    - positions: Nx3 array of original 3D positions from first pass
    - params: Parameters dict from select_filtering_parameters
    """
    logger.info(f"Second pass: Applying conditional filters for {params['building_type']} type...")
    
    start_count = len(points_2d)
    building_type = params['building_type']
    
    y_values = positions[:, 1]
    
    ground_level = estimate_ground_level(y_values)
    logger.info(f"Estimated ground level: {ground_level:.2f}m")
    
    if building_type == "apartment" or building_type == "multi_level_house":
        max_height = np.percentile(y_values, 98) + params['height_tolerance']
    elif building_type == "industrial":
        max_height = np.percentile(y_values, 95) + params['height_tolerance'] + 1.0
    elif building_type == "greenhouse":
        max_height = np.percentile(y_values, 90) + params['height_tolerance']
    else:
        max_height = np.percentile(y_values, 95) + params['height_tolerance']
    
    logger.info(f"Maximum valid height: {max_height:.2f}m")
    
    height_mask = (y_values >= (ground_level - params['ground_tolerance'])) & (y_values <= max_height)
    
    if building_type == "greenhouse":
        tree = spatial.KDTree(points_2d)
        density = np.array([len(tree.query_ball_point(point, r=0.5)) for point in points_2d])
        density_threshold = np.percentile(density, 30)  # Keep top 70% density points
        density_mask = density >= density_threshold
        combined_mask = height_mask & density_mask
    
    elif building_type == "complex_residential" and params["preserve_internal_gaps"]:
        combined_mask = height_mask
    
    elif building_type == "industrial":
        combined_mask = height_mask
    
    else:
        combined_mask = height_mask
    
    filtered_points_2d = points_2d[combined_mask]
    filtered_positions = positions[combined_mask]
    
    removed_count = start_count - len(filtered_points_2d)
    removal_percentage = (removed_count / start_count) * 100 if start_count > 0 else 0
    
    logger.info(f"Removed {removed_count} points with conditional filters ({removal_percentage:.2f}%)")
    
    return filtered_points_2d, filtered_positions


def filter_point_cloud_with_open3d(positions, nb_neighbors=20, std_ratio=2.0):
    """
    Use Open3D's efficient implementation for point cloud filtering 
    
    Args:
        positions: Nx3 array of positions
        nb_neighbors: Number of neighbors to use for statistical outlier removal
        std_ratio: Standard deviation ratio threshold
        
    Returns:
        Nx3 array of filtered positions and corresponding 2D points (X,Z)
    """
    logger.info(f"Starting Open3D point cloud filtering with {len(positions)} points...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    
    logger.info(f"Applying statistical outlier removal with nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    filtered_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, 
        std_ratio=std_ratio
    )
    
    filtered_positions = np.asarray(filtered_pcd.points)
    logger.info(f"Filtered to {len(filtered_positions)} points")
    
    filtered_points_2d = filtered_positions[:, [0, 2]]
    
    return filtered_points_2d, filtered_positions


def read_rotation_matrix(matrix_file_path=rotation_file_path):
    """
    Read rotation matrix from a JSON file.
    
    Args:
        matrix_file_path: Path to the JSON file containing the rotation matrix
        
    Returns:
        list: The 4x4 rotation matrix as a flattened list
    """
    logger.info(f"Reading rotation matrix from {matrix_file_path}")
    try:
        with open(matrix_file_path, 'r') as json_file:
            matrix_data = json.load(json_file)
            return matrix_data["matrix"]
    except Exception as e:
        logger.error(f"Error reading matrix file: {e}")
        logger.info("Using identity matrix as fallback")
        return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def generate_boundary_points_from_ply(ply_file_path, boundary_file, spacing_meters=0.2, alpha=0.1):
    try:
        logger.info(f"Loading PLY file: {ply_file_path}")
        ply_data = load_file(ply_file_path)
        positions = np.array(ply_data["positions"]).reshape(-1, 3)
        
        rotation_matrix = read_rotation_matrix()
        rotated_positions = apply_rotation_matrix(positions, rotation_matrix)
        
        point_count = len(rotated_positions)
        logger.info(f"Loaded {point_count} points from PLY file")
        
        if point_count < 10:
            logger.error(f"Too few points in PLY file: {point_count}")
            return False
        
        points_2d = rotated_positions[:, [0, 2]]
        
        x_range = np.max(points_2d[:, 0]) - np.min(points_2d[:, 0])
        z_range = np.max(points_2d[:, 1]) - np.min(points_2d[:, 1])
        area = x_range * z_range
        height_range = np.max(rotated_positions[:, 1]) - np.min(rotated_positions[:, 1])
        
        logger.info(f"Building statistics: {point_count} points, {area:.1f}m² area, {height_range:.1f}m height")
        
        nb_neighbors = 20 
        std_ratio = 2.0   
        
        if area > 1000 and height_range > 15:
            nb_neighbors = 30
            std_ratio = 2.2
            logger.info("Detected building type: Apartment/Large Building")
        elif area > 800 and height_range < 15:
            nb_neighbors = 25
            std_ratio = 2.0
            logger.info("Detected building type: Industrial/Commercial")
            nb_neighbors = 25
            std_ratio = 2.0
            logger.info("Detected building type: Industrial/Commercial")
        elif area < 300 and height_range < 10:
            nb_neighbors = 15
            std_ratio = 1.8
            logger.info("Detected building type: House/Small Building")
        else:
            logger.info("Detected building type: General Structure")
        
        logger.info(f"Selected filtering parameters: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
        
        try:
            filtered_points_2d, _ = filter_point_cloud_with_open3d(
                rotated_positions, 
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
        except Exception as e:
            logger.error(f"Error in point cloud filtering: {str(e)}")
            logger.info("Using original points without filtering")
            filtered_points_2d = points_2d
        
        if len(filtered_points_2d) < 3:
            logger.error("Not enough points for boundary generation after filtering")
            logger.info("Using original points without filtering")
            filtered_points_2d = points_2d
            
            if len(filtered_points_2d) < 3:
                logger.error("Not enough points for boundary generation even in original data")
                return False
        
        logger.info(f"Using {len(filtered_points_2d)} filtered points for boundary generation")

        logger.info(f"Generating alpha shape boundary")
        alpha_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        alpha_shape = None

        for alpha in alpha_values:
            try:
                logger.info(f"Trying alpha value: {alpha}")
                alpha_shape = alphashape.alphashape(filtered_points_2d, alpha)
                if isinstance(alpha_shape, (Polygon, MultiPolygon)):
                    logger.info(f"Successfully created boundary with alpha: {alpha}")
                    break
            except Exception as e:
                logger.warning(f"Alpha value {alpha} failed: {str(e)}")
                continue

        if alpha_shape is None:
            logger.error("Could not generate valid boundary with any alpha value")
            return False

        try:
            if isinstance(alpha_shape, Polygon):
                boundary_line = alpha_shape.exterior
                logger.info(f"Created single polygon boundary")
            elif isinstance(alpha_shape, MultiPolygon):
                largest_polygon = max(alpha_shape.geoms, key=lambda p: p.area)
                boundary_line = largest_polygon.exterior
                logger.info(f"Created multi-polygon boundary, selected largest polygon")
            else:
                logger.error(f"Unexpected geometry type: {type(alpha_shape)}")
                return False

            bounds = alpha_shape.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            logger.info(f"Boundary dimensions - Width: {width:.2f}m, Height: {height:.2f}m")

            logger.info(f"Using actual boundary without inward offset")

            boundary_length = boundary_line.length
            n_points = max(int(boundary_length / spacing_meters), 4)
            logger.info(f"Boundary perimeter: {boundary_length:.2f}m, generating {n_points} boundary points")

            boundary_points = [boundary_line.interpolate(i / n_points, normalized=True)
                            for i in range(n_points)]

            boundary_points_coords = []
            for point in boundary_points:
                boundary_points_coords.append([point.x, point.y])

            boundary_points_coords = np.array(boundary_points_coords)
            logger.info(f"Generated {len(boundary_points_coords)} original boundary points")
        except Exception as e:
            logger.error(f"Error processing boundary geometry: {str(e)}")
            return False
            
        try:
            logger.info(f"Creating visualization plot")
            
            min_x = min(np.min(filtered_points_2d[:, 0]), np.min(boundary_points_coords[:, 0]))
            max_x = max(np.max(filtered_points_2d[:, 0]), np.max(boundary_points_coords[:, 0]))
            min_z = min(np.min(filtered_points_2d[:, 1]), np.min(boundary_points_coords[:, 1]))
            max_z = max(np.max(filtered_points_2d[:, 1]), np.max(boundary_points_coords[:, 1]))
            
            center_x = (min_x + max_x) / 2
            center_z = (min_z + max_z) / 2
            span_x = max_x - min_x
            span_z = max_z - min_z
            
            buffer_factor = 0.3
            min_x = center_x - span_x/2 * (1 + buffer_factor)
            max_x = center_x + span_x/2 * (1 + buffer_factor)
            min_z = center_z - span_z/2 * (1 + buffer_factor)
            max_z = center_z + span_z/2 * (1 + buffer_factor)
            
            plt.figure(figsize=(12, 10), dpi=150)
            
            sample_indices = np.random.choice(len(rotated_positions), size=min(len(rotated_positions) // 10, 5000), replace=False)
            plt.scatter(rotated_positions[sample_indices, 0], rotated_positions[sample_indices, 2],
                        color='yellow', s=1.5, alpha=0.15, label='Original Points')
            
            plt.scatter(filtered_points_2d[:, 0], filtered_points_2d[:, 1],
                        color='#0BDA47', s=6, alpha=0.6, label='Filtered Points')

            x = boundary_points_coords[:, 0]
            y = boundary_points_coords[:, 1]

            plt.plot(np.append(x, x[0]), np.append(y, y[0]), 
                    color='#57B9FF', linewidth=3, linestyle='-', label='Boundary')
            
            plt.scatter(x, y, color='darkblue', s=25, alpha=0.8, zorder=5)
            
            plt.title('Building Boundary from Point Cloud (Top View)', fontsize=16, fontweight='bold')
            plt.xlabel('X (meters)', fontsize=14)
            plt.ylabel('Z (meters)', fontsize=14)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(loc='upper right', fontsize=12)
            
            plt.axis('equal')
            
            plt.xlim(min_x, max_x)
            plt.ylim(min_z, max_z)

            image_filename = boundary_file.replace('.json', '.png')
            plt.savefig(image_filename, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved visualization to {image_filename}")
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            # Continue even if visualization fails

        try:
            points_to_save = boundary_points_coords
            output_data = [
                {
                    "x": format(point[0], '.4f'),
                    "z": format(point[1], '.4f')
                }
                for point in points_to_save
            ]

            with open(boundary_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Saved {len(output_data)} boundary points to {boundary_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving boundary file: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in boundary generation: {str(e)}")
        return False


if __name__ == "__main__":
    """
    Boundary Generation for 3D Architectural Models
    
    This script processes PLY model points and generates boundary files for various 
    architectural structures including houses, apartments, greenhouses, and other buildings.
    
    The process includes:
    1. Loading PLY file data
    2. Applying rotation correction from matrix_4_4.json
    3. Filtering points using Open3D statistical outlier removal
    4. Generating alpha shape boundary with adaptive parameters
    5. Creating visualization and saving boundary coordinates
    
    Input:
    - PLY file: Contains 3D point cloud data of the building
    - matrix_4_4.json: Contains rotation matrix (created by rotation_correction.py)
    
    Output:
    - boundary.json: Contains the calculated boundary points in XZ plane
    - boundary.png: Visualization of original points, filtered points, and boundary
    
    Note: This script is designed to be fault-tolerant and will attempt multiple
    alpha values if boundary generation fails with the initial values.
    """

    logger.info(f"Processing PLY file: {ply_file_path}...")
    try:
        generate_boundary_points_from_ply(
            ply_file_path,
            boundary_file='boundary.json',
            spacing_meters=0.2
        )
        logger.info("✓ Processing completed")
        logger.info("Generated files:")
        logger.info("- boundary.json: Contains boundary coordinates")
        logger.info("- boundary.png: Visualization plot")
    except Exception as e:
        logger.error(f"Error processing PLY file: {str(e)}")