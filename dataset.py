import os
import h5py
import torch
from torch.utils.data import Dataset

class AradDataset(Dataset):
    def __init__(self, x_dir, y_dir):
        self.x_dir = x_dir
        self.y_dir = y_dir
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

        return x, y

    def _load_h5(self, path):
        with h5py.File(path, 'r') as f:
            data = f['data'][()]
        return torch.tensor(data, dtype=torch.float32)

