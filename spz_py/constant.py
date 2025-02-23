# Constants and helper functions

SPZ_MAGIC = 0x5053474e  # "NGSP" = Niantic Gaussian Splat.
SPZ_VERSION = 2
FLAG_ANTIALIASED = 0x1
COLOR_SCALE = 0.15
SH_C0 = 0.28209479177387814

def degree_for_dim(dim: int) -> int:
    if dim < 3:
        return 0
    if dim < 8:
        return 1
    if dim < 15:
        return 2
    return 3

def dim_for_degree(degree: int) -> int:
    if degree == 0:
        return 0
    elif degree == 1:
        return 3
    elif degree == 2:
        return 8
    elif degree == 3:
        return 15
    else:
        print(f"[SPZ: ERROR] Unsupported SH degree: {degree}")
        return 0 