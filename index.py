from spz_py.ply_loader import load_ply
from spz_py.spz_serializer import serialize_spz

def load_file(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
        return load_ply(f)

def main():
    """
    Main function to convert a PLY 3D model to SPZ format.
    
    This function:
    1. Defines a model ID (20991)
    2. Constructs the path to the PLY file based on the model ID
    3. Loads the PLY file data using load_file()
    4. Converts the data to SPZ format using serialize_spz()
    5. Writes the SPZ data to a file named after the model ID
    
    The output will be a file named 'model_{model_id}.spz' in the current directory.
    """
    model_id = 20991
    ply_file_path = f"ply/model_{model_id}.ply"
    gs = load_file(ply_file_path)  

    spz_data = serialize_spz(gs)
    with open(f"model_{model_id}.spz", "wb") as f:
        f.write(spz_data)

if __name__ == '__main__':
    main()