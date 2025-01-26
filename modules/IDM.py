import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Debug import Debug

class IDM(nn.Module):
    def __init__(self):
        super(IDM, self).__init__()
        # Convolutional Block for feature extraction and dimensionality reduction
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)  # Output retains 3 channels
        )

        # Depthwise separable convolution
        self.depthwise_conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        self.pointwise_conv = nn.Conv2d(3, 3, kernel_size=1)

        # Multi-scale average pooling
        self.avg_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.AdaptiveAvgPool2d((32, 32))
        ])

        # Per-scale fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(8 * 8 * 3, 3),
            nn.Linear(16 * 16 * 3, 3),
            nn.Linear(32 * 32 * 3, 3)
        ])

        # Debug helper
        self.debugger = Debug()

    def _check_submodule_params(self, module: nn.Module, module_name: str, debug: bool):
        """
        Inspect all named parameters of a submodule, printing shape, min, max,
        and checking for NaNs if debug=True.
        """
        if debug:
            for name, param in module.named_parameters():
                self.debugger.debug_tensor(param, desc=f"[IDM] {module_name} param '{name}'")

    def forward(self, x, debug=True):
        """
        Forward pass for IDM.
        Args:
            x: Input tensor of shape (N, 3, H, W), or a list of images.
            debug (bool): Whether to print debug info and check for NaNs at each step.
        Returns:
            A list of three tensors, each shape (N, 1, H, W).
        """
        # If input is a list, stack into a single 4D tensor
        if isinstance(x, list):
            x = torch.stack(x, dim=0)

        # Debug the input
        if debug:
            self.debugger.debug_tensor(x, desc="[IDM] input x")

        # (A) Check all submodule parameters before using them
        self._check_submodule_params(self.conv_block, "conv_block", debug)
        self._check_submodule_params(self.depthwise_conv, "depthwise_conv", debug)
        self._check_submodule_params(self.pointwise_conv, "pointwise_conv", debug)
        for i, fc_layer in enumerate(self.fc_layers):
            self._check_submodule_params(fc_layer, f"fc_layers[{i}]", debug)

        # 1) Convolutional Block
        fa = self.conv_block(x)
        if debug:
            self.debugger.debug_tensor(fa, desc="[IDM] fa (after conv_block)")

        # 2) Depthwise + ReLU
        fb = F.relu(self.depthwise_conv(fa))
        if debug:
            self.debugger.debug_tensor(fb, desc="[IDM] fb (after depthwise+ReLU)")

        #    Then pointwise conv
        fb = self.pointwise_conv(fb)
        if debug:
            self.debugger.debug_tensor(fb, desc="[IDM] fb (after pointwise)")

        # 3) Multi-scale average pooling & parameter prediction
        gray_images = []
        for i, pool in enumerate(self.avg_pools):
            fc_input = pool(fb).view(fb.size(0), -1)
            if debug:
                self.debugger.debug_tensor(fc_input, desc=f"[IDM] fc_input scale={i}")

            transformation_params = F.softmax(self.fc_layers[i](fc_input), dim=1)
            if debug:
                self.debugger.debug_tensor(transformation_params,
                                           desc=f"[IDM] transform_params scale={i}")

            phi_r = transformation_params[:, 0]
            phi_g = transformation_params[:, 1]
            phi_b = transformation_params[:, 2]

            # Compute grayscale
            ir, ig, ib = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
            gray_image = (phi_r.view(-1, 1, 1) * ir
                          + phi_g.view(-1, 1, 1) * ig
                          + phi_b.view(-1, 1, 1) * ib)
            if debug:
                self.debugger.debug_tensor(gray_image, desc=f"[IDM] gray_image scale={i}")

            # shape: (N, 1, H, W)
            gray_images.append(gray_image.unsqueeze(1))

        return gray_images
