
from spz_py.ply_loader import load_ply
from spz_py.spz_serializer import serialize_spz

def load_file(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
            return load_ply(f)

def main():
    model_id = 20991
    ply_file_path = f"ply/model_{model_id}.ply"
    gs = load_file(ply_file_path)  

    spz_data = serialize_spz(gs)
    with open(f"model_{model_id}.spz", "wb") as f:
        f.write(spz_data)

if __name__ == '__main__':
    main()