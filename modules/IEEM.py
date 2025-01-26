import torch
import torch.nn as nn
import torch.nn.functional as F

class IEEM(nn.Module):
    def __init__(self):
        super(IEEM, self).__init__()
        # Laplacian kernel for edge enhancement
        #self.laplacian_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        #self.laplacian_filter.weight = nn.Parameter(torch.tensor(
        #    [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype=torch.float32).unsqueeze(0), requires_grad=False)
        
        self.laplacian_weight = torch.tensor(
            [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype=torch.float32).unsqueeze(0)

        # Group convolution block
        self.group_conv_block = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1, groups=3),
            nn.ReLU()
        )

        # Residual Dense Block (RRDB)
        self.rrdb = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Mask and HR Conv Blocks
        self.mask_conv_block = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.hr_conv_block = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        # Upsample Conv Block
        self.upsample_block = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        )

    def laplacian_filter(self, img):
        self.laplacian_weight = self.laplacian_weight.to(img.device)
        return F.conv2d(img, self.laplacian_weight, padding=1)

    def forward(self, images):


        if images[0].shape[1] != 1:
            raise ValueError("Input images should have 1 channel for Laplacian filtering.")

        # Laplacian edge enhancement for each grayscale image
        laplacian_outputs = [self.laplacian_filter(img) for img in images]
        # Concatenate Laplacian outputs
        e_cat = torch.cat(laplacian_outputs, dim=1)  # Concatenate (N, 3, H, W)


        # Group convolution
        f_cat = self.group_conv_block(e_cat)


        # RRDB
        f_d = self.rrdb(f_cat)


        # Mask and HR Conv
        f_m = self.sigmoid(self.mask_conv_block(f_d))

        f_h = self.hr_conv_block(f_d)


        # Feature fusion and upsampling
        f_l = self.upsample_block(f_h * f_m)


        # Fuse features
        e_cat_final = torch.cat([f_l, e_cat], dim=1)


        # Reshape to ensure each output has 1 channel
        enhanced_images = tuple([e_cat_final[:, i:i + 1, ...] for i in range(3)])


        return enhanced_images





