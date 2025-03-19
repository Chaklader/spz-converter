import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import alphashape
import matplotlib.pyplot as plt
import logging
from spz_py.ply_loader import load_ply

# Configure your logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
np.random.seed(42)

# Remove the offset percentage since we want the actual boundary
# BOUNDARY_OFFSET_PERCENTAGE = 5  # 5% offset


def load_file(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
        return load_ply(f)


def generate_boundary_points_from_ply(ply_file_path, boundary_file, spacing_meters=0.2, alpha=0.1):
    # Load PLY file and extract positions
    logger.info(f"Loading PLY file: {ply_file_path}")
    ply_data = load_file(ply_file_path)
    
    # Reshape positions from flat array to (n_points, 3)
    positions = np.array(ply_data["positions"]).reshape(-1, 3)
    logger.info(f"Loaded {len(positions)} points from PLY file")
    
    # Project points to X-Z plane (setting Y=0)
    points_2d = positions[:, [0, 2]]  # Extract X and Z coordinates
    logger.info(f"Projected points to X-Z plane")
    
    # Generate alpha shape
    logger.info(f"Generating alpha shape boundary")
    alpha_values = [0.01, 0.05, 0.1, 0.15, 0.2]  # Try tighter boundary first
    alpha_shape = None

    for alpha in alpha_values:
        try:
            logger.info(f"Trying alpha value: {alpha}")
            alpha_shape = alphashape.alphashape(points_2d, alpha)
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
    plt.figure(figsize=(10, 10))
    plt.scatter(points_2d[:, 0], points_2d[:, 1],
                color='green', s=5, alpha=0.5, label='PLY Points')

    plt.scatter(boundary_points_coords[:, 0], boundary_points_coords[:, 1],
                color='purple', s=50, label='Boundary Points')

    plt.title('PLY Points and Boundary (Top View)')
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.legend()
    plt.axis('equal')

    # Save the plot
    image_filename = boundary_file.replace('.json', '.png')
    plt.savefig(image_filename)
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
    
    # Input PLY file
    ply_file_path = 'ply/model_20869.ply'

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
        logger.error(f"✗ Error processing: {str(e)}")