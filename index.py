
from spz_py.ply_loader import load_ply
from spz_py.spz_serializer import serialize_spz

def load_file(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
            return load_ply(f)

def main():
    # Change the file path to your target model.
    gs = load_file("ply/model_20991.ply")  

    spz_data = serialize_spz(gs)
    with open("gs1.spz", "wb") as f:
        f.write(spz_data)

if __name__ == '__main__':
    main()