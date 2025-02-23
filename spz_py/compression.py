import gzip

def decompress_gzipped(data: bytes) -> bytes:
    try:
        return gzip.decompress(data)
    except Exception as error:
        print("Error decompressing gzipped data:", error)
        raise

def compress_gzipped(data: bytes) -> bytes:
    try:
        return gzip.compress(data)
    except Exception as error:
        print("Error compressing gzipped data:", error)
        raise 