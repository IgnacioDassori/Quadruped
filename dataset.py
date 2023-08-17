import torch
import glob
from torch.utils.data import Dataset

class DepthDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        tensor = torch.load(self.image_path[idx])
        return tensor
    
image_path = glob.glob('images/images_pt/*.pt')
dataset = DepthDataset(image_path)
train_set, val_set = torch.utils.data.random_split(dataset, [230, 48])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
print(len(train_loader))