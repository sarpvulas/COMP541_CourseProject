# modules/dual_backbone.py

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class DualResNet50(nn.Module):
    """
    Creates two separate ResNet-50 streams with unshared weights:
      - branch_rgb (for original images)
      - branch_gray (for edge-enhanced grayscale)
    You extract the intermediate layers the same way for both.
    """

    def __init__(self):
        super(DualResNet50, self).__init__()

        # Branch 1: original RGB
        resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        #print("resnet1.layer1 output channels:", resnet1.layer1[-1].conv3.out_channels)
        self.rgb_conv1 = nn.Sequential(
            resnet1.conv1,
            resnet1.bn1,
            resnet1.relu,
            resnet1.maxpool
        )
        self.rgb_layer1 = resnet1.layer1
        self.rgb_layer2 = resnet1.layer2
        self.rgb_layer3 = resnet1.layer3
        self.rgb_layer4 = resnet1.layer4

        # Branch 2: edge-enhanced grayscale
        resnet2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.gray_conv1 = nn.Sequential(
            resnet2.conv1,
            resnet2.bn1,
            resnet2.relu,
            resnet2.maxpool
        )
        self.gray_layer1 = resnet2.layer1
        self.gray_layer2 = resnet2.layer2
        self.gray_layer3 = resnet2.layer3
        self.gray_layer4 = resnet2.layer4

    def forward_rgb(self, x):
        # For the original RGB images
        x = self.rgb_conv1(x)
        #print("[debug] after rgb_conv1:", x.shape)
        f1 = self.rgb_layer1(x)
        #print("[debug] after layer1:", f1.shape)
        f2 = self.rgb_layer2(f1)
        #print("[debug] after layer2:", f2.shape)
        f3 = self.rgb_layer3(f2)
        #print("[debug] after layer3:", f3.shape)
        f4 = self.rgb_layer4(f3)
        #print("[debug] after layer4:", f4.shape)
        return [f1, f2, f3, f4]

    def forward_gray(self, x):
        # For the edge-enhanced grayscale images
        x = self.gray_conv1(x)
        f1 = self.gray_layer1(x)
        f2 = self.gray_layer2(f1)
        f3 = self.gray_layer3(f2)
        f4 = self.gray_layer4(f3)
        return [f1, f2, f3, f4]

    def forward(self, rgb_input, gray_input):
        # Return features from both branches
        feats_rgb = self.forward_rgb(rgb_input)    # list of 4 features
        feats_gray = self.forward_gray(gray_input) # list of 4 features
        return feats_rgb, feats_gray
