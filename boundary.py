import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import alphashape
import matplotlib.pyplot as plt
import logging

# Configure your logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
np.random.seed(42)

BOUNDARY_OFFSET_PERCENTAGE = 5  # 5% offset


def generate_boundary_points_from_transforms(transform_file, boundary_file, camera_file, spacing_meters=0.2, alpha=0.1):
    with open(transform_file, 'r') as f:
        transforms = json.load(f)

    # Save camera positions and rotations with exactly 4 decimal places
    camera_data = []
    for frame in transforms['frames']:
        matrix = frame['transform_matrix']
        camera_data.append({
            "position": {
                "x": format(matrix[0][3], '.4f'),
                "y": format(matrix[1][3], '.4f'),
                "z": format(matrix[2][3], '.4f')
            },
            "rotation": [
                [format(matrix[0][0], '.4f'), format(matrix[0][1], '.4f'), format(matrix[0][2], '.4f')],
                [format(matrix[1][0], '.4f'), format(matrix[1][1], '.4f'), format(matrix[1][2], '.4f')],
                [format(matrix[2][0], '.4f'), format(matrix[2][1], '.4f'), format(matrix[2][2], '.4f')]
            ]
        })

    with open(camera_file, 'w') as f:
        json.dump(camera_data, f, indent=2)

    # Extract positions for boundary calculation
    points = np.array([
        [frame['transform_matrix'][0][3], frame['transform_matrix'][1][3], frame['transform_matrix'][2][3]]
        for frame in transforms['frames']
    ])

    # Generate alpha shape
    alpha_values = [0.2, 0.15, 0.1, 0.05]
    alpha_shape = None

    for alpha in alpha_values:
        try:
            alpha_shape = alphashape.alphashape(points[:, [0, 2]], alpha)
            if isinstance(alpha_shape, (Polygon, MultiPolygon)):
                break
        except Exception:
            continue

    if alpha_shape is None:
        raise ValueError("Could not generate valid boundary with any alpha value")

    if isinstance(alpha_shape, Polygon):
        boundary_line = alpha_shape.exterior
    elif isinstance(alpha_shape, MultiPolygon):
        largest_polygon = max(alpha_shape.geoms, key=lambda p: p.area)
        boundary_line = largest_polygon.exterior
    else:
        raise ValueError("Unexpected geometry type")

    # Calculate the characteristic size
    bounds = alpha_shape.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    characteristic_size = (width + height) / 2

    # Convert percentage to actual offset distance (negative for inward offset)
    offset_meters = characteristic_size * -(BOUNDARY_OFFSET_PERCENTAGE / 100.0)

    boundary_length = boundary_line.length
    n_points = max(int(boundary_length / spacing_meters), 4)

    boundary_points = [boundary_line.interpolate(i / n_points, normalized=True)
                       for i in range(n_points)]

    # Calculate boundary points
    boundary_points_coords = []
    for point in boundary_points:
        boundary_points_coords.append([point.x, point.y])

    boundary_points_coords = np.array(boundary_points_coords)

    # Calculate offset points
    offset_coords = None
    if offset_meters != 0:
        boundary_polygon = Polygon(boundary_points_coords)
        offset_polygon = boundary_polygon.buffer(offset_meters)

        if isinstance(offset_polygon, Polygon):
            offset_coords = np.array(offset_polygon.exterior.coords[:-1])
        else:
            logger.warn("Warning: Offset resulted in invalid polygon")
            offset_coords = None

    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 2],
                color='green', s=50, label='Camera Positions')

    plt.scatter(boundary_points_coords[:, 0], boundary_points_coords[:, 1],
                color='red', s=50, label='Boundary Points')

    if offset_coords is not None:
        plt.scatter(offset_coords[:, 0], offset_coords[:, 1],
                    color='purple', s=50, label='Offset Boundary')

    plt.title('Camera and User Positions (Top View)')
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.legend()
    plt.axis('equal')

    # Save the plot
    image_filename = boundary_file.replace('.json', '.png')
    plt.savefig(image_filename)
    plt.close()

    # Save the offset coordinates with exactly 4 decimal places
    points_to_save = offset_coords if offset_coords is not None else boundary_points_coords
    output_data = [
        {
            "x": format(point[0], '.4f'),
            "z": format(point[1], '.4f')
        }
        for point in points_to_save
    ]

    with open(boundary_file, 'w') as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    """
    This script processes camera transforms and generates several output files:
    
    Input:
    - transforms.json: Contains camera transformation matrices
    
    Output:
    - boundary.json: Contains the calculated boundary points
    - cameras.json: Contains processed camera positions and rotations
    - boundary.png: Visualization of camera positions and boundary points
    """
    
    # Input transforms file
    transform_file = 'transforms.json'

    # Process the transforms file
    logger.info("Processing transforms file...")
    try:
        generate_boundary_points_from_transforms(
            transform_file,
            boundary_file='boundary.json',
            camera_file='cameras.json',
            spacing_meters=0.2
        )
        logger.info("✓ Processing completed")
        logger.info("Generated files:")
        logger.info("- boundary.json: Contains boundary coordinates")
        logger.info("- cameras.json: Contains camera positions and rotations")
        logger.info("- boundary.png: Visualization plot")
    except Exception as e:
        logger.error(f"✗ Error processing: {str(e)}")