import os
from PIL import Image
from torchvision import transforms
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
        x = (x > 0.5).float()
        y_label = self.class_map[character]
        return x, y_label