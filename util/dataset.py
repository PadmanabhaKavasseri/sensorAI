import torch
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # shape: (N, T, C)
        self.labels = torch.tensor(labels, dtype=torch.long) # shape: (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
