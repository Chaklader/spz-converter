import struct
from spz_py.constant import SPZ_MAGIC, SPZ_VERSION, FLAG_ANTIALIASED
from spz_py.compression import compress_gzipped
from spz_py.spz_loader import pack_gaussians

def serialize_packed_gaussians(packed: dict) -> bytes:
    header_size = 16  # 4+4+4+1+1+1+1 bytes.
    total_size = (
        header_size +
        len(packed["positions"]) +
        len(packed["alphas"]) +
        len(packed["colors"]) +
        len(packed["scales"]) +
        len(packed["rotations"]) +
        len(packed["sh"])
    )
    buffer = bytearray(total_size)
    view = memoryview(buffer)
    # Write header.
    struct.pack_into("<I", buffer, 0, SPZ_MAGIC)
    struct.pack_into("<I", buffer, 4, SPZ_VERSION)
    struct.pack_into("<I", buffer, 8, packed["numPoints"])
    buffer[12] = packed["shDegree"]
    buffer[13] = packed["fractionalBits"]
    buffer[14] = FLAG_ANTIALIASED if packed["antialiased"] else 0
    buffer[15] = 0  # reserved
    offset = header_size
    buffer[offset : offset + len(packed["positions"])] = packed["positions"]
    offset += len(packed["positions"])
    buffer[offset : offset + len(packed["alphas"])] = packed["alphas"]
    offset += len(packed["alphas"])
    buffer[offset : offset + len(packed["colors"])] = packed["colors"]
    offset += len(packed["colors"])
    buffer[offset : offset + len(packed["scales"])] = packed["scales"]
    offset += len(packed["scales"])
    buffer[offset : offset + len(packed["rotations"])] = packed["rotations"]
    offset += len(packed["rotations"])
    buffer[offset : offset + len(packed["sh"])] = packed["sh"]
    return bytes(buffer)

def serialize_spz(g: dict) -> bytes:
    packed = pack_gaussians(g)
    serialized = serialize_packed_gaussians(packed)
    return compress_gzipped(serialized) 