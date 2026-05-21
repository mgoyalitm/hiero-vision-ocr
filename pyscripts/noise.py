import torch
import torch.nn.functional as F

class RealisticNoise:
    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, x):
        # x is [1, H, W] after ToTensor

        # binary mask (foreground)
        mask = (x < 0.5).float()

        # expand mask (like dilation)
        kernel = torch.ones((1, 1, 3, 3), device=x.device)
        expanded = F.conv2d(mask.unsqueeze(0), kernel, padding=1)
        expanded = (expanded > 0).float().squeeze(0)

        # generate noise mask
        noise_mask = (torch.rand_like(x) < self.prob).float()

        # apply noise only near strokes
        x = x.clone()
        x[(expanded > 0) & (noise_mask > 0)] = torch.rand_like(x)[(expanded > 0) & (noise_mask > 0)]

        return x