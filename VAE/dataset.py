import torch
from torch.utils.data import Dataset

class DepthDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        tensor = torch.load(self.image_path[idx])
        if self.transform:
            tensor = self.transform(tensor)
        return tensor