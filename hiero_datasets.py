import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

class dnn_dataset(Dataset):
    def __init__(self, root_dir):
        self.classes = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        self.class_map = {c: i for i, c in enumerate(self.classes)}
        self.root_dir = root_dir
        self.transforms = transforms.ToTensor()
        self.png_files = list(Path(root_dir).rglob("*.png"))
    def __len__(self):
        return len(self.png_files)
    def __getitem__(self, index):
        path = self.png_files[index]
        folder_name = os.path.basename(os.path.dirname(path))
        character = folder_name[-1]
        x = self.transforms(Image.open(path).convert("L"))
        x = (x < 0.9).float()
        y_label = self.class_map[character]
        return x, y_label
    
class gbdt_dataset(dnn_dataset):
    def __getitem__(self, index):
        x_base, y_label = super().__getitem__(index)
        f1 = F.avg_pool2d(x_base, kernel_size=3, stride=3) * 9
        f2 = F.avg_pool2d(x_base, kernel_size=7, stride=7) * 49
        f3 = x_base.view(1, 3, 7, 21).sum(dim=(2,3))
        f4 = x_base.view(1, 21, 3, 7).sum(dim=(1,3))
        x = torch.cat([f1.reshape(-1), f2.reshape(-1), f3.reshape(-1), f4.reshape(-1)], dim=0)
        return x, y_label
