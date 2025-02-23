import numpy as np
from spz_py.constant import degree_for_dim

def load_ply(stream) -> dict:
    """
    Load a binary little‐endian PLY file from a file‐like object using NumPy.
    Returns a dictionary representing a GaussianCloud.
    """
    # Read header lines until the "end_header" marker.
    header_lines = []
    while True:
        line = stream.readline()
        if not line:
            break
        decoded_line = line.decode("utf-8").rstrip("\n")
        if decoded_line.strip() == "end_header":
            break
        header_lines.append(decoded_line)
    
    if not header_lines or header_lines[0] != "ply":
        raise Exception("[PLY ERROR] not a .ply file")
    
    if "format binary_little_endian 1.0" not in header_lines:
        raise Exception("[PLY ERROR] unsupported .ply format")
    
    # Find the vertex element count.
    num_points = None
    for line in header_lines:
        if line.startswith("element vertex"):
            parts = line.split()
            if len(parts) >= 3:
                num_points = int(parts[2])
    if num_points is None:
        raise Exception("[PLY ERROR] missing vertex count")
    if num_points <= 0 or num_points > 10 * 1024 * 1024:
        raise Exception(f"[PLY ERROR] invalid vertex count: {num_points}")
    
    # Parse property names.
    fields = []
    for line in header_lines:
        if line.startswith("property float "):
            name = line[len("property float "):].strip()
            fields.append(name)
    field_map = {name: i for i, name in enumerate(fields)}
    
    def get_field_index(name: str) -> int:
        if name not in field_map:
            print(f"[PLY ERROR] Missing field: {name}")
            return -1
        return field_map[name]
    
    pos_idx = [get_field_index(n) for n in ['x', 'y', 'z']]
    scale_idx = [get_field_index(n) for n in ['scale_0', 'scale_1', 'scale_2']]
    rot_idx = [get_field_index(n) for n in ['rot_1', 'rot_2', 'rot_3', 'rot_0']]
    alpha_idx = [get_field_index('opacity')]
    color_idx = [get_field_index(n) for n in ['f_dc_0', 'f_dc_1', 'f_dc_2']]
    
    if any(i < 0 for i in (pos_idx + scale_idx + rot_idx + alpha_idx + color_idx)):
        raise Exception("[PLY ERROR] Missing required fields")
    
    # Check for spherical harmonics properties.
    sh_indices = []
    for i in range(45):
        name = f"f_rest_{i}"
        if name in field_map:
            sh_indices.append(field_map[name])
        else:
            break
    sh_dim = (len(sh_indices) // 3) if len(sh_indices) >= 3 else 0
    
    # Read the binary blob of vertex data.
    n_fields = len(fields)
    float_size = 4
    stride = n_fields * float_size
    total_size = num_points * stride
    buffer = stream.read(total_size)
    if len(buffer) != total_size:
        raise Exception("[PLY ERROR] Could not read all binary data")
    
    # Use NumPy to convert the buffer to an array of floats.
    all_floats = np.frombuffer(buffer, dtype='<f4').reshape(num_points, n_fields)
    
    result = {
        "numPoints": num_points,
        "shDegree": degree_for_dim(sh_dim),
        "antialiased": False,
        "positions": all_floats[:, pos_idx].flatten().tolist(),
        "scales": all_floats[:, scale_idx].flatten().tolist(),
        "rotations": all_floats[:, rot_idx].flatten().tolist(),
        "alphas": all_floats[:, alpha_idx[0]].flatten().tolist(),
        "colors": all_floats[:, color_idx].flatten().tolist(),
        "sh": np.zeros((num_points, sh_dim, 3), dtype='<f4').flatten().tolist()
    }
    
    # If you need to fill SH coefficients, you can do so with slicing/reordering
    if sh_dim > 0:
        # Create a (num_points, sh_dim, 3) array for sh and fill accordingly.
        sh_array = np.empty((num_points, sh_dim, 3), dtype='<f4')
        for c in range(3):
            # Extract the SH coefficients for color channel c.
            sh_array[:, :, c] = all_floats[:, np.array(sh_indices[c*sh_dim:(c+1)*sh_dim])]
        result["sh"] = sh_array.flatten().tolist()
    
    return result