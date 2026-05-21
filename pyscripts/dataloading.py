import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from pyscripts.mapping import get_character_id, get_directory_name
from typing import Tuple, cast
from tqdm import tqdm

class MNIST_Dataset(Dataset):
    def __init__(self, root_dir:Path, characters:list[str], augmentation:transforms.Compose = transforms.Compose([]), iterations:int = 1):
        self.root_dir = root_dir
        self.iterations = iterations
        self.characters = characters
        self.transforms = transforms.Compose([transforms.ToTensor(), augmentation, transforms.Lambda(lambda x: 1 - (x > 0.95).float())])
        self.character_map = {c: get_character_id(c) for c in characters}
        self.class_map = {id: i for i, id in enumerate(list(dict.fromkeys(self.character_map.values()))) } 
        self.ids = self.class_map.values
        self.files = []
        for character in characters:
            directory = root_dir / get_directory_name(character)
            self.files.extend(list(Path(directory).rglob("*.png")))
    def __len__(self) -> int:
        return len(self.files) * self.iterations
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        index = index % len(self.files)
        path = self.files[index]
        folder_name = os.path.basename(os.path.dirname(path))
        character = folder_name[-1]
        X = cast(torch.Tensor, self.transforms(Image.open(path).convert("L")))
        character_id = self.character_map[character]
        y = self.class_map[character_id]
        return X, y

def GetGradientBoostingDecisionTreeData(dataloader:DataLoader, title:str) -> Tuple[np.ndarray, np.ndarray]:
    X_data, y_data = [], []
    for X_batch, y_batch in tqdm(dataloader, ncols=100, bar_format=f"Loading {title} data {{bar}} {{percentage:3.0f}}%" , ascii="|█", position=0, leave=False):
        assert X_batch.shape[-2:] == (21, 21), f"Unexpected shape: {X_batch.shape}"
        for index in range(len(y_batch)):
            X = X_batch[index]
            y = y_batch[index].item()
            kernel_3x3 = F.avg_pool2d(X, kernel_size=3, stride=3).flatten() * 9
            kernel_7x7 = F.avg_pool2d(X, kernel_size=7, stride=7).flatten() * 49
            stripe_sum_ver = X.view(1, 3, 7, 21).sum(dim=(2,3)).flatten()
            stripe_sum_hor = X.view(1, 3, 21, 7).sum(dim=(1,3)).flatten()
            X_data.append(torch.cat([kernel_3x3, kernel_7x7, stripe_sum_hor, stripe_sum_ver], dim=0))
            y_data.append(y)
    X_data = torch.stack(X_data).cpu().numpy().astype("float32")
    y_data = np.array(y_data, dtype="int32")
    return X_data, y_data
    