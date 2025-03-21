import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import alphashape
from scipy import spatial
from sklearn.cluster import DBSCAN
from rotation_correction import get_rotation
import open3d as o3d

# Configure your logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
np.random.seed(42)

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
    # Use convex hull for a rough area estimation
    try:
        hull = spatial.ConvexHull(points_2d)
        return hull.volume  # In 2D, volume is area
    except Exception as e:
        # Fallback if hull calculation fails
        min_x, min_y = np.min(points_2d, axis=0)
        max_x, max_y = np.max(points_2d, axis=0)
        return (max_x - min_x) * (max_y - min_y)  # Bounding box area


def estimate_ground_level(y_values):
    """Estimate ground level from height distribution"""
    # Use histogram to find the lowest significant peak
    hist, bin_edges = np.histogram(y_values, bins=100)
    # Find lowest significant cluster
    for i in range(len(hist)):
        if hist[i] > len(y_values) * 0.01:  # At least 1% of points
            return bin_edges[i]
    return np.min(y_values)


def select_filtering_parameters(points_2d, positions):
    """
    Select appropriate filtering parameters based on point cloud characteristics.
    Automatically detects building type and recommends filtering parameters.
    """
    logger.info("Analyzing point cloud to determine building characteristics...")
    
    # Calculate basic statistics
    point_count = len(points_2d)
    area = calculate_rough_area(points_2d)
    density = point_count / area if area > 0 else 0
    
    # Height analysis
    y_values = positions[:, 1]
    height_range = np.max(y_values) - np.min(y_values)
    ground_level = np.percentile(y_values, 5)  # Estimate ground as 5th percentile
    
    # Density distribution analysis
    # Calculate point density variations
    tree = spatial.KDTree(points_2d)
    sample_size = min(5000, len(points_2d))  # Limit computation for large clouds
    sample_indices = np.random.choice(len(points_2d), sample_size, replace=False)
    sample_points = points_2d[sample_indices]
    
    neighbor_counts = [len(tree.query_ball_point(point, r=1.0)) for point in sample_points]
    density_std = np.std(neighbor_counts) / np.mean(neighbor_counts) if np.mean(neighbor_counts) > 0 else 0
    
    # Try to detect flat surfaces (potential pools, rooftops, etc)
    # Group points by height bands
    height_hist, _ = np.histogram(y_values, bins=50)
    height_peaks = np.where(height_hist > np.percentile(height_hist, 90))[0]
    has_distinct_levels = len(height_peaks) >= 2  # Multiple distinct height levels
    
    # Try to detect open areas inside boundary (potential courtyards, pools)
    # Use DBSCAN to identify clusters and gaps
    try:
        eps = np.sqrt(area / point_count) * 5  # Adaptive epsilon based on average point spacing
        clustering = DBSCAN(eps=eps, min_samples=10).fit(points_2d)
        cluster_count = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        has_internal_gaps = cluster_count > 1
    except Exception:
        has_internal_gaps = False
    
    # Detect regular geometric patterns (like greenhouse structures)
    # We use the standard deviation of angles between nearest neighbors
    regular_geometry = False
    try:
        # Sample points for efficiency
        if len(sample_points) > 100:
            # Find nearest neighbors
            distances, indices = tree.query(sample_points, k=5)
            
            # Calculate angles between consecutive neighbors
            angles = []
            for i, idx_array in enumerate(indices):
                if len(idx_array) >= 3:  # Need at least 3 points for angles
                    p0 = sample_points[i]
                    for j in range(1, len(idx_array)-1):
                        p1 = points_2d[idx_array[j]]
                        p2 = points_2d[idx_array[j+1]]
                        v1 = p1 - p0
                        v2 = p2 - p0
                        # Calculate angle
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
    
    # Determine building type and parameters
    params = {
        "building_type": "general",
        "sor_k": 20,             # Statistical outlier removal K neighbors
        "sor_std_ratio": 2.0,    # Statistical outlier removal std dev ratio
        "height_tolerance": 1.5,  # Meters above detected maximum to allow
        "ground_tolerance": 0.5,  # Meters below ground level to allow
        "preserve_internal_gaps": False,
        "density_threshold": 0.5
    }
    
    # Decision logic for building type
    if height_range > 15 and area > 500:
        # Tall building with substantial footprint - likely apartment or commercial
        params["building_type"] = "apartment" 
        params["sor_k"] = 30
        params["height_tolerance"] = 2.0
        logger.info(f"Detected building type: Apartment/Tall Building (height: {height_range:.1f}m)")
    
    elif area > 1000 and height_range < 10:
        # Large footprint but not very tall - likely industrial, warehouse or commercial
        params["building_type"] = "industrial"
        params["sor_k"] = 25
        params["sor_std_ratio"] = 2.5  # More tolerant of outliers
        params["density_threshold"] = 0.3
        logger.info(f"Detected building type: Industrial/Commercial (area: {area:.1f}m²)")
    
    elif regular_geometry and height_range < 8:
        # Regular patterns and moderate height - potential greenhouse or similar
        params["building_type"] = "greenhouse"
        params["sor_k"] = 15
        params["sor_std_ratio"] = 1.8  # More aggressive outlier removal
        logger.info("Detected building type: Greenhouse/Regular Structure")
    
    elif has_internal_gaps and area > 300:
        # Building with internal gaps - potential courtyard or pool
        params["building_type"] = "complex_residential"
        params["preserve_internal_gaps"] = True
        params["sor_k"] = 20
        logger.info("Detected building type: Complex Residential (with courtyards/pools)")
    
    elif has_distinct_levels and height_range < 12:
        # Multiple distinct levels - likely multi-level house or small apartment
        params["building_type"] = "multi_level_house"
        params["sor_k"] = 20
        logger.info("Detected building type: Multi-level House")
    
    elif area < 300 and height_range < 10:
        # Small footprint - likely a house or small structure
        params["building_type"] = "house"
        params["sor_k"] = 15
        logger.info("Detected building type: House/Small Building")
    
    else:
        # Default case - general building
        logger.info("Detected building type: General Structure")
        
    # Additional statistics for logging
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
    
    # Reshape matrix from flat list to 4x4
    matrix_4x4 = np.array(matrix).reshape(4, 4)
    
    # Extract rotation component (3x3)
    rotation_matrix = matrix_4x4[:3, :3]
    
    # Create homogeneous coordinates (add a row of ones)
    # positions_homog = np.hstack((positions, np.ones((len(positions), 1))))
    
    # Apply rotation only (no translation needed as we're getting a top-down view)
    rotated_positions = np.zeros_like(positions)
    for i in range(len(positions)):
        rotated_positions[i] = rotation_matrix.dot(positions[i])
        print(f"Point {i}: Before rotation: {positions[i]} | After rotation: {rotated_positions[i]}")
    
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
    
    # Build KDTree
    tree = spatial.KDTree(points_2d)
    
    # Find distances to k nearest neighbors
    distances, _ = tree.query(points_2d, k=min(k+1, len(points_2d)))
    
    # If we have fewer points than k+1, adjust the distances array
    if distances.shape[1] < k+1:
        logger.warning(f"Not enough points for k={k}, using k={distances.shape[1]-1} instead")
        k = distances.shape[1] - 1
    
    # Remove self-distance (first column)
    distances = distances[:, 1:]
    
    # Calculate mean distances for each point
    mean_distances = np.mean(distances, axis=1)
    
    # Calculate global stats
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # Filter points
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
    
    # Extract height values
    y_values = positions[:, 1]
    
    # Determine ground level
    ground_level = estimate_ground_level(y_values)
    logger.info(f"Estimated ground level: {ground_level:.2f}m")
    
    # Determine maximum valid height
    if building_type == "apartment" or building_type == "multi_level_house":
        # For tall buildings, use a high percentile with additional tolerance
        max_height = np.percentile(y_values, 98) + params['height_tolerance']
    elif building_type == "industrial":
        # Industrial buildings may have overhead equipment
        max_height = np.percentile(y_values, 95) + params['height_tolerance'] + 1.0
    elif building_type == "greenhouse":
        # Greenhouses typically have uniform height
        max_height = np.percentile(y_values, 90) + params['height_tolerance']
    else:
        # For general buildings
        max_height = np.percentile(y_values, 95) + params['height_tolerance']
    
    logger.info(f"Maximum valid height: {max_height:.2f}m")
    
    # Apply height filters
    height_mask = (y_values >= (ground_level - params['ground_tolerance'])) & (y_values <= max_height)
    
    # Apply additional filters based on building type
    if building_type == "greenhouse":
        # For greenhouses, focus on structural elements (higher density)
        tree = spatial.KDTree(points_2d)
        density = np.array([len(tree.query_ball_point(point, r=0.5)) for point in points_2d])
        density_threshold = np.percentile(density, 30)  # Keep top 70% density points
        density_mask = density >= density_threshold
        combined_mask = height_mask & density_mask
    
    elif building_type == "complex_residential" and params["preserve_internal_gaps"]:
        # For buildings with courtyards/pools, preserve the structure
        # but still apply height filtering
        combined_mask = height_mask
    
    elif building_type == "industrial":
        # For industrial buildings, be more permissive with isolated points (machinery)
        combined_mask = height_mask
    
    else:
        # For general buildings, apply only height filtering
        combined_mask = height_mask
    
    # Apply the combined mask
    filtered_points_2d = points_2d[combined_mask]
    filtered_positions = positions[combined_mask]
    
    removed_count = start_count - len(filtered_points_2d)
    removal_percentage = (removed_count / start_count) * 100 if start_count > 0 else 0
    
    logger.info(f"Removed {removed_count} points with conditional filters ({removal_percentage:.2f}%)")
    
    return filtered_points_2d, filtered_positions


def filter_point_cloud(points_2d, positions):
    """
    Apply multi-pass filtering to the point cloud:
    1. Analyze point cloud to determine building type and parameters
    2. Apply statistical outlier removal (first pass)
    3. Apply conditional filters based on building type (second pass)
    
    Returns filtered points_2d and positions arrays
    """
    logger.info("Starting point cloud filtering...")
    
    # Get original point count
    original_count = len(points_2d)
    
    # Analyze point cloud and select parameters
    params = select_filtering_parameters(points_2d, positions)
    
    # First pass: Statistical Outlier Removal (SOR)
    points_2d_sor, positions_sor = remove_statistical_outliers(
        points_2d, 
        positions,
        k=params['sor_k'],
        std_ratio=params['sor_std_ratio']
    )
    
    # Second pass: Conditional Filtering
    points_2d_filtered, positions_filtered = apply_conditional_filters(
        points_2d_sor,
        positions_sor,
        params
    )
    
    # Calculate total points removed
    total_removed = original_count - len(points_2d_filtered)
    total_percentage = (total_removed / original_count) * 100 if original_count > 0 else 0
    
    logger.info(f"Filtering complete. Total points removed: {total_removed} ({total_percentage:.2f}%)")
    logger.info(f"Filtered point cloud size: {len(points_2d_filtered)} points")
    
    return points_2d_filtered, positions_filtered


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
    
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    
    # Apply statistical outlier removal
    logger.info(f"Applying statistical outlier removal with nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    filtered_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, 
        std_ratio=std_ratio
    )
    
    # Get filtered positions as numpy array
    filtered_positions = np.asarray(filtered_pcd.points)
    logger.info(f"Filtered to {len(filtered_positions)} points")
    
    # Project to X-Z plane
    filtered_points_2d = filtered_positions[:, [0, 2]]
    
    return filtered_points_2d, filtered_positions


def generate_boundary_points_from_ply(ply_file_path, boundary_file, spacing_meters=0.2, alpha=0.1):
    # Load PLY file and extract positions
    logger.info(f"Loading PLY file: {ply_file_path}")
    ply_data = load_file(ply_file_path)
    
    # Reshape positions from flat array to (n_points, 3)
    positions = np.array(ply_data["positions"]).reshape(-1, 3)
    logger.info(f"Loaded {len(positions)} points from PLY file")
    
    # Get rotation correction matrix
    logger.info("Getting rotation correction matrix")
    rotation_matrix = get_rotation(ply_file_path)
    
    # Apply rotation correction to align model with XZ plane
    rotated_positions = apply_rotation_matrix(positions, rotation_matrix)
    logger.info(f"Applied rotation correction to {len(rotated_positions)} points")
    
    # Filter point cloud using Open3D (more memory efficient)
    logger.info("Applying point cloud filtering with Open3D...")
    
    # Determine building type to select appropriate parameters
    # (simplified analysis since we're using Open3D's efficient implementation)
    point_count = len(rotated_positions)
    x_range = np.max(rotated_positions[:, 0]) - np.min(rotated_positions[:, 0])
    z_range = np.max(rotated_positions[:, 2]) - np.min(rotated_positions[:, 2])
    area = x_range * z_range
    height_range = np.max(rotated_positions[:, 1]) - np.min(rotated_positions[:, 1])
    
    # Log basic building statistics
    logger.info(f"Building statistics: {point_count} points, {area:.1f}m² area, {height_range:.1f}m height")
    
    # Select filtering parameters based on building characteristics
    nb_neighbors = 20  # default
    std_ratio = 2.0    # default
    
    if area > 1000 and height_range > 15:
        # Large footprint and tall - likely an apartment complex
        nb_neighbors = 30
        std_ratio = 2.2
        logger.info("Detected building type: Apartment/Large Building")
    elif area > 800 and height_range < 15:
        # Large footprint but not tall - likely industrial or commercial
        nb_neighbors = 25
        std_ratio = 2.0
        logger.info("Detected building type: Industrial/Commercial")
    elif area < 300 and height_range < 10:
        # Small footprint - likely a house or small structure
        nb_neighbors = 15
        std_ratio = 1.8
        logger.info("Detected building type: House/Small Building")
    else:
        logger.info("Detected building type: General Structure")
    
    logger.info(f"Selected filtering parameters: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    
    # Apply Open3D filtering
    filtered_points_2d, filtered_positions = filter_point_cloud_with_open3d(
        rotated_positions, 
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    logger.info(f"Using {len(filtered_points_2d)} filtered points for boundary generation")
    
    # Generate alpha shape
    logger.info(f"Generating alpha shape boundary")
    alpha_values = [0.01, 0.05, 0.1, 0.15, 0.2]  # Try tighter boundary first
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
        raise ValueError("Could not generate valid boundary with any alpha value")

    if isinstance(alpha_shape, Polygon):
        boundary_line = alpha_shape.exterior
        logger.info(f"Created single polygon boundary")
    elif isinstance(alpha_shape, MultiPolygon):
        largest_polygon = max(alpha_shape.geoms, key=lambda p: p.area)
        boundary_line = largest_polygon.exterior
        logger.info(f"Created multi-polygon boundary, selected largest polygon")
    else:
        raise ValueError("Unexpected geometry type")

    # Calculate the characteristic size
    bounds = alpha_shape.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    characteristic_size = (width + height) / 2
    logger.info(f"Boundary dimensions - Width: {width:.2f}m, Height: {height:.2f}m")

    # We'll use no offset to get the actual boundary
    offset_meters = 0
    logger.info(f"Using actual boundary without inward offset")

    boundary_length = boundary_line.length
    n_points = max(int(boundary_length / spacing_meters), 4)
    logger.info(f"Boundary perimeter: {boundary_length:.2f}m, generating {n_points} boundary points")

    boundary_points = [boundary_line.interpolate(i / n_points, normalized=True)
                       for i in range(n_points)]

    # Calculate boundary points
    boundary_points_coords = []
    for point in boundary_points:
        boundary_points_coords.append([point.x, point.y])

    boundary_points_coords = np.array(boundary_points_coords)
    logger.info(f"Generated {len(boundary_points_coords)} original boundary points")

    # Calculate offset points
    offset_coords = None
    # Since we're not using an offset, we'll just use the original boundary points
    
    # Create visualization
    logger.info(f"Creating visualization plot")
    
    # Calculate the bounds with a buffer for better plotting
    min_x = min(np.min(filtered_points_2d[:, 0]), np.min(boundary_points_coords[:, 0]))
    max_x = max(np.max(filtered_points_2d[:, 0]), np.max(boundary_points_coords[:, 0]))
    min_z = min(np.min(filtered_points_2d[:, 1]), np.min(boundary_points_coords[:, 1]))
    max_z = max(np.max(filtered_points_2d[:, 1]), np.max(boundary_points_coords[:, 1]))
    
    # Calculate center and span
    center_x = (min_x + max_x) / 2
    center_z = (min_z + max_z) / 2
    span_x = max_x - min_x
    span_z = max_z - min_z
    
    # Add a buffer around the building (30% padding)
    buffer_factor = 0.3
    min_x = center_x - span_x/2 * (1 + buffer_factor)
    max_x = center_x + span_x/2 * (1 + buffer_factor)
    min_z = center_z - span_z/2 * (1 + buffer_factor)
    max_z = center_z + span_z/2 * (1 + buffer_factor)
    
    # Create a high-quality figure
    plt.figure(figsize=(12, 10), dpi=150)
    
    # Plot only a subset of original points to reduce clutter (sample 10%)
    sample_indices = np.random.choice(len(rotated_positions), size=min(len(rotated_positions) // 10, 5000), replace=False)
    plt.scatter(rotated_positions[sample_indices, 0], rotated_positions[sample_indices, 2],
                color='yellow', s=1.5, alpha=0.15, label='Original Points')
    
    # Plot the filtered points in green
    plt.scatter(filtered_points_2d[:, 0], filtered_points_2d[:, 1],
                color='#0BDA47', s=6, alpha=0.6, label='Filtered Points')

    x = boundary_points_coords[:, 0]
    y = boundary_points_coords[:, 1]

    # Plot the boundary as a bold line
    plt.plot(np.append(x, x[0]), np.append(y, y[0]), 
             color='#57B9FF', linewidth=3, linestyle='-', label='Boundary')
    
    # Add reference point markers at corners for clarity
    plt.scatter(x, y, color='darkblue', s=25, alpha=0.8, zorder=5)
    
    # Improve plot aesthetics
    plt.title('Building Boundary from Point Cloud (Top View)', fontsize=16, fontweight='bold')
    plt.xlabel('X (meters)', fontsize=14)
    plt.ylabel('Z (meters)', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=12)
    
    # Set fixed aspect ratio (1:1)
    plt.axis('equal')
    
    # Focus on the building by setting plot limits
    plt.xlim(min_x, max_x)
    plt.ylim(min_z, max_z)

    # Save the plot with high resolution
    image_filename = boundary_file.replace('.json', '.png')
    plt.savefig(image_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visualization to {image_filename}")

    # Save the coordinates with exactly 4 decimal places - use original boundary points
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


if __name__ == "__main__":
    """
    This script processes PLY model points and generates boundary files:
    
    Input:
    - PLY file: Contains 3D point data
    
    Output:
    - boundary.json: Contains the calculated boundary points
    - boundary.png: Visualization of PLY points and boundary
    """
    #  24998
    # Input PLY file
    ply_file_path = 'ply/model_25032.ply'

    # Process the PLY file
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