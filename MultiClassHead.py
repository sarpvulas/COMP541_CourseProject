import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class MultiClassHead(nn.Module):
    """
    A simple multi-class classification head for feature maps.
    """
    def __init__(self, in_channels, num_classes):
        super(MultiClassHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        #self.softmax = nn.Softmax(dim=1)  # Apply along channel dimension

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): Feature maps from FPN, each with shape [B, C, H, W].

        Returns:
            class_scores (list[Tensor]): Multi-class classification scores for each feature map.
        """
        class_scores = []
        for feature in features:
            x = self.conv1(feature)
            x = self.relu(x)
            x = self.conv2(x)
            #x = self.softmax(x)  # Apply softmax to get probabilities
            class_scores.append(x)
        return class_scores