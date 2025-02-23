import math
import numpy as np
from spz_py.constant import dim_for_degree

color_scale = 0.15

def pack_gaussians(g: dict) -> dict:
    """
    Optimized packing of a GaussianCloud (dict) into a PackedGaussians structure using NumPy
    vectorized operations.
    """
    num_points = g["numPoints"]
    sh_degree = g["shDegree"]
    sh_dim = dim_for_degree(sh_degree)
    # Validate the provided clouds
    if not (len(g["positions"]) == num_points * 3 and
            len(g["scales"]) == num_points * 3 and
            len(g["rotations"]) == num_points * 4 and
            len(g["alphas"]) == num_points and
            len(g["colors"]) == num_points * 3 and
            len(g["sh"]) == num_points * sh_dim * 3):
        raise Exception("Invalid GaussianCloud")
    
    fractional_bits = 12
    scale_factor = 1 << fractional_bits

    # --- Positions ---
    # Convert positions to fixed-point values and pack each float into 3 bytes.
    positions = np.asarray(g["positions"], dtype=np.float32)  # shape: (num_points*3,)
    fixed_positions = np.rint(positions * scale_factor).astype(np.int32)
    # Create an array where each fixed32 generates 3 bytes.
    pos_out = np.empty((fixed_positions.size, 3), dtype=np.uint8)
    pos_out[:, 0] = fixed_positions & 0xff
    pos_out[:, 1] = (fixed_positions >> 8) & 0xff
    pos_out[:, 2] = (fixed_positions >> 16) & 0xff
    pos_bytes = pos_out.flatten()

    # --- Scales ---
    scales = np.asarray(g["scales"], dtype=np.float32)
    # The code uses: (scale + 10.0) * 16.0 then clamps to [0,255]
    scales_uint8 = to_uint8_np((scales + 10.0) * 16.0)

    # --- Rotations ---
    # Process quaternions per point; data is in xyzw order.
    rotations = np.asarray(g["rotations"], dtype=np.float32).reshape(num_points, 4)
    # Normalize each quaternion.
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    # Avoid divide by zero (if ever needed).
    norms[norms == 0] = 1.0
    q = rotations / norms
    # If w < 0 then multiply by -127.5, else 127.5 (scalar applied to whole quaternion).
    scale_factors = np.where(q[:, 3] < 0, -127.5, 127.5).reshape(num_points, 1)
    scaled_q = q * scale_factors
    offset_q = scaled_q + 127.5
    # Take only xyz channels, then round and clamp to 0-255.
    rot_uint8 = to_uint8_np(offset_q[:, :3])
    rot_bytes = rot_uint8.flatten()

    # --- Alphas ---
    alphas = np.asarray(g["alphas"], dtype=np.float32)
    alpha_vals = vectorized_sigmoid(alphas) * 255.0
    alphas_uint8 = to_uint8_np(alpha_vals)

    # --- Colors ---
    colors = np.asarray(g["colors"], dtype=np.float32)
    colors_scaled = colors * (color_scale * 255.0) + (0.5 * 255.0)
    colors_uint8 = to_uint8_np(colors_scaled)

    # Prepare the packed dict by creating bytearrays from the computed bytes.
    packed = {
        "numPoints": num_points,
        "shDegree": sh_degree,
        "fractionalBits": fractional_bits,
        "antialiased": g["antialiased"],
        "positions": bytearray(pos_bytes.tobytes()),
        "scales": bytearray(scales_uint8.tobytes()),
        "rotations": bytearray(rot_bytes.tobytes()),
        "alphas": bytearray(alphas_uint8.tobytes()),
        "colors": bytearray(colors_uint8.tobytes()),
        "sh": bytearray()
    }

    # --- Spherical Harmonics (SH) ---
    if sh_degree > 0:
        sh1_bits = 5
        sh_rest_bits = 4
        sh_per_point = sh_dim * 3  # total coefficients per point
        sh_array = np.asarray(g["sh"], dtype=np.float32).reshape(num_points, sh_per_point)

        # Quantization bucket sizes.
        bucket1 = 1 << (8 - sh1_bits)
        bucket_rest = 1 << (8 - sh_rest_bits)

        # For the first few (degree 1) coefficients use one bucket size,
        # and the remaining coefficients use another.
        if sh_per_point >= 9:
            first_part = sh_array[:, :9]
            rest_part = sh_array[:, 9:] if sh_per_point > 9 else None

            q_first = np.rint(first_part * 128.0 + 128.0)
            q_first = np.floor((q_first + bucket1 / 2) / bucket1) * bucket1
            q_first = np.clip(q_first, 0, 255).astype(np.uint8)

            if rest_part is not None and rest_part.size:
                q_rest = np.rint(rest_part * 128.0 + 128.0)
                q_rest = np.floor((q_rest + bucket_rest / 2) / bucket_rest) * bucket_rest
                q_rest = np.clip(q_rest, 0, 255).astype(np.uint8)
                sh_quantized = np.concatenate((q_first, q_rest), axis=1)
            else:
                sh_quantized = q_first
        else:
            # If the number of coefficients is less than 9, use the rest bucket.
            sh_quantized = np.rint(sh_array * 128.0 + 128.0)
            sh_quantized = np.floor((sh_quantized + bucket_rest / 2) / bucket_rest) * bucket_rest
            sh_quantized = np.clip(sh_quantized, 0, 255).astype(np.uint8)

        packed["sh"] = bytearray(sh_quantized.flatten().tobytes())

    return packed

def to_uint8_np(array):
    """Round and clip a NumPy array to the [0, 255] range and cast to np.uint8."""
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)

def vectorized_sigmoid(x):
    """Compute sigmoid activation on a NumPy array."""
    return 1.0 / (1.0 + np.exp(-x))




