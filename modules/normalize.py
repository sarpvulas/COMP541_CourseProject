import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetNormalize(nn.Module):
    """(x - mean) / std per channel, with the torchvision ImageNet statistics.

    Expects RGB in [0, 1], shape (N, 3, H, W). The statistics are non-persistent buffers, so
    they follow the module across devices and do not change checkpoint keys.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std
