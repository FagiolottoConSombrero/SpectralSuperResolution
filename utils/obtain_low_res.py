import os
import h5py
import numpy as np
from skimage.transform import resize

# === Funzione per salvataggio in HDF5 ===
def save_h5(filepath, data, dataset_name='data'):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset(dataset_name, data=data, dtype=data.dtype)

# === Funzione per leggere un file .h5 ===
def read_h5(path, key="cube"):
    """
    Legge un file .h5 contenente un cubo hyperspettrale con chiave 'data'.
    Restituisce il cubo (HxWxB) e, se presente, le bande spettrali.
    """
    with h5py.File(path, "r") as f:
        cube = np.array(f[key])               # Hypercube (HxBxW)
        cube = cube.transpose(0, 2, 1)        # → HxWxB
        wavelengths = np.array(f["bands"]) if "bands" in f else None
    return cube, wavelengths

# === Directory sorgente e destinazioni ===
src_dir = '/home/acp/Scrivania/datasets/SSD1/Arad-1K/Valid_spectral'

dst_dir_original = '/home/acp/Scrivania/projects/matteo/h5/val/val_arad1k_original'
dst_dir_x2 = '/home/acp/Scrivania/projects/matteo/h5/val/val_arad1k_x4'
dst_dir_x3 = '/home/acp/Scrivania/projects/matteo/h5/val/val_rad1k_x6'
dst_dir_x4 = '/home/acp/Scrivania/projects/matteo/h5/val/val_arad1k_x8'

# === Crea le cartelle se non esistono ===
os.makedirs(dst_dir_original, exist_ok=True)
os.makedirs(dst_dir_x2, exist_ok=True)
os.makedirs(dst_dir_x3, exist_ok=True)
os.makedirs(dst_dir_x4, exist_ok=True)

# === Lista file .mat, escludendo quelli che iniziano con '._' ===
file_list = sorted([f for f in os.listdir(src_dir) if f.endswith('.mat') and not f.startswith('._')])

# === Elaborazione dei file ===
for idx, file_name in enumerate(file_list):
    file_path = os.path.join(src_dir, file_name)
    img, _ = read_h5(file_path, key="cube")
    img = img[:, 1:481, 4:508]
    C, H, W = img.shape
    out_name = file_name.replace('.mat', '.h5')
    save_h5(os.path.join(dst_dir_original, out_name), img)

    for scale, dst_dir in zip([4, 6, 8], [dst_dir_x2, dst_dir_x3, dst_dir_x4]):
        img_ds = resize(img, (C, H // scale, W // scale), order=0, preserve_range=True, anti_aliasing=False)
        #img_up = resize(img_ds, (H, W, B), order=0, preserve_range=True, anti_aliasing=False)
        img_up = img_ds.astype(img.dtype)

        save_h5(os.path.join(dst_dir, out_name), img_up)

    print(f'Processato {idx + 1}/{len(file_list)}: {file_name}')