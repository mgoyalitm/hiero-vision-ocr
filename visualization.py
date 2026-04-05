import torch
import matplotlib.pyplot as plt
from hiero_datasets import dnn_dataset

def display_random_mnist_samples(dataset:dnn_dataset, seed:int = 42, rows:int=3, coloums:int=10):
    total = rows * coloums
    fig, axes = plt.subplots(rows, coloums, figsize=(coloums*0.45, rows*0.75))
    torch.manual_seed(seed)
    for i, idx in enumerate(torch.randint(0, dataset.__len__(), (total,))):
        image_tensor, y = dataset.__getitem__(idx)
        character = dataset.classes[y]
        image_tensor = 1 - image_tensor.permute(1, 2, 0)

        row_idx = i // coloums
        col_idx = i % coloums

        ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]

        ax.imshow(image_tensor.squeeze(0), cmap='gray')
        ax.set_title(character, fontsize=12, color="#A50060")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()






