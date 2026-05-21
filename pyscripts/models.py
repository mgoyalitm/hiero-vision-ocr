import torch.nn as nn
import torch.nn.functional as F

class WideBasic(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet21(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=26):
        super().__init__()
        n = (depth - 4) // 6
        k = widen_factor
        channels = [16, 16*k, 32*k, 64*k]
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        self.layer1 = self._make_layer(channels[0], channels[1], n, stride=1)
        self.layer2 = self._make_layer(channels[1], channels[2], n, stride=2)
        self.layer3 = self._make_layer(channels[2], channels[3], n, stride=2)
        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)
    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = []
        layers.append(WideBasic(in_c, out_c, stride))
        for _ in range(1, num_blocks):
            layers.append(WideBasic(out_c, out_c, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)          
        out = self.layer1(out)       
        out = self.layer2(out)       
        out = self.layer3(out)       
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)
    
    import torch

class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # (1,21,21) -> (16,21,21)
        self.block1 = WideBasic(16, 32, stride=2)  # -> (32,11,11)
        self.block2 = WideBasic(32, 32, stride=1)  # -> (32,11,11)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32, 26)  # 26 classes
    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)  # -> (32,1,1)
        x = x.view(x.size(0), -1)        # -> (32)
        return self.fc(x)