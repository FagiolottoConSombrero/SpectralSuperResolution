import os
import h5py
import torch
from torch.utils.data import Dataset

class AradDataset(Dataset):
    def __init__(self, x_dir, train=True):
        """
        Dataset per coppie HSI (x = low-res, y = high-res)

        Args:
            x_dir (str): directory contenente immagini x4 (input)
            y_dir (str): directory contenente immagini originali (target)
            key (str): nome del dataset nel file .h5 (default: 'data')
            transform (callable, optional): trasformazione da applicare a x e y
        """
        self.train = train
        self.x_dir = x_dir
        if self.train:
            self.y_dir = '/home/matteo/Documents/arad1k/h5/train/train_arad1k_original'
        else:
            self.y_dir = '/home/matteo/Documents/arad1k/h5/val/val_arad1k_original'
        self.file_list = sorted([
            f for f in os.listdir(x_dir)
            if f.endswith('.h5') and not f.startswith('._')
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        x_path = os.path.join(self.x_dir, file_name)
        y_path = os.path.join(self.y_dir, file_name)

        x = self._load_h5(x_path)
        y = self._load_h5(y_path)

        return x, y  # x = low-res, y = original

    def _load_h5(self, path):
        with h5py.File(path, 'r') as f:
            data = f['data'][()]  # [C, H, W]
        return torch.tensor(data, dtype=torch.float32)       # convert to float32 for PyTorch
