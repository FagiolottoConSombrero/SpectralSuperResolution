import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# === Nome base del file (senza estensione) ===
base_name = 'ARAD_1K_0912'

# === Cartelle dei file ===
root_dir = '/home/matteo/Documents/arad1k/h5/val'
src_mat_dir = os.path.join(root_dir, 'Valid_spectral')

paths = {
    'Original': os.path.join(root_dir, 'val_arad1k_original', base_name + '.h5'),
    'x4': os.path.join(root_dir, 'val_arad1k_x4', base_name + '.h5'),
    'x6': os.path.join(root_dir, 'val_rad1k_x6', base_name + '.h5'),
    'x8': os.path.join(root_dir, 'val_arad1k_x8', base_name + '.h5'),
}

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

# === Bande da visualizzare (Python indexing)
bands = [0, 5, 13]  # Bande 1, 6, 14
titles = ['Banda 1', 'Banda 6', 'Banda 14']

# === Visualizzazione ===
for label, path in paths.items():
    if label == 'MAT':
        # Carica da .mat
        cube, _ = read_h5(path)
    else:
        # Carica da .h5
        with h5py.File(path, 'r') as f:
            cube = f['data'][()]  # [H, W, B]

    print(f"{label} → shape: {cube.shape}")

    # Se necessario: trasponi da (B, H, W) a (H, W, B)
    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
        cube = np.transpose(cube, (1, 2, 0))  # (H, W, B)

    # Visualizza le 3 bande
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, b in enumerate(bands):
        axs[i].imshow(cube[:, :, b], cmap='gray')
        axs[i].set_title(f'{titles[i]}')
        axs[i].axis('off')
    fig.suptitle(f'{label} — shape: {cube.shape}', fontsize=14)
    plt.tight_layout()
    plt.show()

