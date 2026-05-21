from torch import nn
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from pyscripts.visualization import PlotResults
from tqdm import tqdm

def TrainModel(model:nn.Module, train_loader:DataLoader, test_loader:DataLoader, learning_rate:float, epochs:int, device:str):
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss, test_loss, train_acc, test_acc = [], [], [], []
    batches_total = epochs * (len(train_loader) + len(test_loader))
    progress_bar = tqdm(total=batches_total, ncols=100, bar_format="{bar}| {n_fmt}/{total_fmt} {postfix}", ascii="|█", position=0, leave=False)
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0 , 0 
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_func(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            _, pridicted = y_pred.max(1)
            total += y.size(0)
            correct += pridicted.eq(y).sum().item()
            progress_bar.update(1)

        train_loss.append(running_loss / total)
        train_acc.append(correct / total)
        running_loss = 0 
        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for X , y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_func(y_pred, y)
                running_loss += loss.item() * y.size(0)
                _, pridicted = y_pred.max(1)
                total += y.size(0)
                correct += pridicted.eq(y).sum().item()
                progress_bar.update(1)

            test_loss.append(running_loss / total)
            test_acc.append(correct / total)
            total = 0
            progress_bar.set_postfix({"Accuracy": f"{test_acc[-1]:.4f}"})

    progress_bar.close()
    PlotResults(train_loss=train_loss, test_loss=test_loss, train_acc=train_acc, test_acc=test_acc)
    