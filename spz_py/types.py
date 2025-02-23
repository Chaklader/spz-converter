from dataclasses import dataclass
from typing import List

@dataclass
class GaussianCloud:
    numPoints: int
    shDegree: int
    antialiased: bool
    positions: List[float]  # Flattened list with 3 floats per point.
    scales: List[float]     # 3 floats per point.
    rotations: List[float]  # 4 floats per point.
    alphas: List[float]     # 1 float per point.
    colors: List[float]     # 3 floats per point.
    sh: List[float]         # (dim_for_degree(shDegree)*3) floats per point.

@dataclass
class PackedGaussians:
    numPoints: int
    shDegree: int
    fractionalBits: int
    antialiased: bool
    positions: bytes
    scales: bytes
    rotations: bytes
    alphas: bytes
    colors: bytes
    sh: bytes 