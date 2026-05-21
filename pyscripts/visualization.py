import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import cast
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.patches import Rectangle

def ShowRandonPreview(dataloader:DataLoader, rows:int, columns:int):
    count = rows * columns
    X_data = []
    fig, axes = plt.subplots(rows, columns, figsize=(columns*0.4, rows*0.4))
    # fig.patch.set_facecolor('black') 
    for X_batch, y_batch in dataloader:
        for index in range(len(y_batch)):
            X_data.append(X_batch[index])
            if len(X_data) == count: break
        if len(X_data) == count: break
    for index, image_tensor in enumerate(X_data):
        image_tensor = cast(torch.Tensor, image_tensor)
        # image_tensor = F.interpolate(image_tensor.unsqueeze(0), scale_factor=2, mode='bilinear').squeeze(0)
        row_index = index // columns
        column_index = index % columns
        ax = axes[row_index, column_index] if rows > 1 else axes[column_index]
        ax.imshow(1 - image_tensor.squeeze(0), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def PlotResults(train_loss:list, test_loss:list, train_acc:list, test_acc:list, steps:int = 1, y_name:str='Trees'):
    best_idx = int(np.argmin(test_loss))
    print("Best Iteration:", best_idx * steps + 1)
    print("Train Accuracy:", train_acc[best_idx] * 100)
    print("Test Accuracy :", test_acc[best_idx] * 100)
    print("Train LogLoss:", train_loss[best_idx])
    print("Test LogLoss :", test_loss[best_idx])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(np.arange(len(train_loss)) * steps, train_loss, label="Train Loss")
    axes[0].plot(np.arange(len(test_loss)) * steps, test_loss, label="Test Loss")
    axes[0].set_title("Log Loss vs Trees")
    axes[0].set_xlabel("Trees")
    axes[0].set_ylabel("Log Loss")
    axes[0].legend()
    axes[1].plot(np.arange(len(train_acc)) * steps, train_acc, label="Train Acc")
    axes[1].plot(np.arange(len(test_acc)) * steps, test_acc, label="Test Acc")
    axes[1].set_title(f"Accuracy vs {y_name}")
    axes[1].set_xlabel(y_name)
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def ShowDataPresentation():
    fig, axs = plt.subplots(2,2,figsize=(8,8))
    margin=.08
    colors=["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"]
    def draw_grid(ax):
        for x in range(22):
            ax.plot(
                [x,x],[0,21],
                linestyle=":",
                linewidth=.5,
                color="#555555"
            )
        for y in range(22):
            ax.plot(
                [0,21],[y,y],
                linestyle=":",
                linewidth=.5,
                color="#555555"
            )
        ax.set_xlim(0,21)
        ax.set_ylim(21,0)
        ax.set_aspect("equal")
        ax.axis("off")
    ax=axs[0,0]
    draw_grid(ax)
    idx=0
    for r in range(7):
        for c in range(7):
            x=c*3+margin
            y=r*3+margin
            ax.add_patch(
                Rectangle(
                    (x,y),
                    3-2*margin,
                    3-2*margin,
                    alpha=.28,
                    facecolor=colors[idx%5]
                )
            )
            ax.text(
                x+1.4,
                y+1.4,
                str(idx),
                fontsize=6,
                ha='center',
                va='center',
                weight='bold'
            )
            idx+=1
    ax.set_title("49 Features\n3×3 regions")
    ax=axs[0,1]
    draw_grid(ax)
    idx=49
    for r in range(3):
        for c in range(3):
            x=c*7+margin
            y=r*7+margin
            ax.add_patch(
                Rectangle(
                    (x,y),
                    7-2*margin,
                    7-2*margin,
                    alpha=.28,
                    facecolor=colors[idx%5]
                )
            )
            ax.text(
                x+3.5,
                y+3.5,
                str(idx),
                fontsize=10,
                ha='center',
                va='center',
                weight='bold'
            )
            idx+=1
    ax.set_title("9 Features\n7×7 regions")
    ax=axs[1,0]
    draw_grid(ax)
    idx=58
    for r in range(3):
        y=r*7+margin
        ax.add_patch(
            Rectangle(
                (margin,y),
                21-2*margin,
                7-2*margin,
                alpha=.28,
                facecolor=colors[r]
            )
        )
        ax.text(
            10.5,
            y+3.5,
            str(idx),
            fontsize=11,
            ha='center',
            va='center',
            weight='bold'
        )
        idx+=1
    ax.set_title("3 Horizontal Features")
    ax=axs[1,1]
    draw_grid(ax)
    idx=61
    for c in range(3):
        x=c*7+margin
        ax.add_patch(
            Rectangle(
                (x,margin),
                7-2*margin,
                21-2*margin,
                alpha=.28,
                facecolor=colors[c]
            )
        )
        ax.text(
            x+3.5,
            10.5,
            str(idx),
            fontsize=11,
            ha='center',
            va='center',
            weight='bold'
        )
        idx+=1
    ax.set_title("3 Vertical Features")
    plt.tight_layout()
    plt.show()