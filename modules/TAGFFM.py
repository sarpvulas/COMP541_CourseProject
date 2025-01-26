import torch
import torch.nn as nn
import torch.nn.functional as F

class TAGFFM(nn.Module):
    """
    Tridimensional Adaptive Gated Feature Fusion Module (TAGFFM)

    The module adapts to different ResNet stages based on the stage number.
    """

    def __init__(self, stage_number):
        """
        Initializes TAGFFM for a specific ResNet stage.

        Args:
          stage_number (int): ResNet stage number (1, 2, 3, or 4).
        """
        super(TAGFFM, self).__init__()

        # Map the stage number to input dimensions
        self.in_channels, self.height, self.width = self.get_dimensions(stage_number)

        # 1×1 convolutions for height-wise gating
        self.convH_r = nn.Conv2d(in_channels=self.height, out_channels=1, kernel_size=1)
        self.convH_g = nn.Conv2d(in_channels=self.height, out_channels=1, kernel_size=1)

        # 1×1 convolutions for width-wise gating
        self.convW_r = nn.Conv2d(in_channels=self.width, out_channels=1, kernel_size=1)
        self.convW_g = nn.Conv2d(in_channels=self.width, out_channels=1, kernel_size=1)

        # 1×1 convolutions for channel-wise gating
        self.convC_H = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.convC_W = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)

    def get_dimensions(self, stage_number):
        """
        Maps the ResNet stage number to input dimensions.

        Args:
          stage_number (int): ResNet stage number (1, 2, 3, or 4).

        Returns:
          Tuple: (in_channels, height, width)
        """
        if stage_number == 1:
            in_channels, height, width = 256, 128, 128  # Stage 1 output dimensions
        elif stage_number == 2:
            in_channels, height, width = 512, 64, 64   # Stage 2 output dimensions
        elif stage_number == 3:
            in_channels, height, width = 1024, 32, 32  # Stage 3 output dimensions
        elif stage_number == 4:
            in_channels, height, width = 2048, 16, 16  # Stage 4 output dimensions
        else:
            raise ValueError("Invalid stage number. Must be 1, 2, 3, or 4.")
        return in_channels, height, width

    def forward(self, F_r, F_g):
        """
        Forward pass of TAGFFM.

        Inputs:
          F_r: Tensor of shape (B, C, H, W)
          F_g: Tensor of shape (B, C, H, W)

        Returns:
          F_out: fused tensor of shape (B, C, H, W)
        """
        B, C, H, W = F_r.shape

        # -------------------------------------------------------------
        # 1) Height-wise gating  (paper Eqs. 26–27, top half of Fig. 7)
        # -------------------------------------------------------------
        # Gate for F_r (call it G^H_r):
        #   1) permute to (B, H, C, W)
        #   2) 1×1 conv along the 'C' dimension (since it is last in that layout)
        #   3) sigmoid
        #   4) permute back to something broadcastable over (B, C, H, W)

        G_H_r = F_r.permute(0, 2, 1, 3)          # (B, H, C, W)

        G_H_r = self.convH_r(G_H_r)             # (B, 1, C, W)
        G_H_r = torch.sigmoid(G_H_r)            # (B, 1, C, W)
        G_H_r = G_H_r.permute(0, 2, 1, 3)       # (B, C, 1, W)


        # Gate for F_g (call it G^H_g), same steps:
        G_H_g = F_g.permute(0, 2, 1, 3)
        G_H_g = self.convH_g(G_H_g)
        G_H_g = torch.sigmoid(G_H_g)
        G_H_g = G_H_g.permute(0, 2, 1, 3)

        # Combine them (Eq. 27: F^H = (G^H_r * F_r) + (G^H_g * F_g)):
        F_H = F_r * G_H_r + F_g * G_H_g

        # -------------------------------------------------------------
        # 2) Width-wise gating  (paper Eqs. 22–25, middle of Fig. 7)
        # -------------------------------------------------------------
        # Gate for F_r (call it G^W_r):
        #   1) permute to (B, W, C, H)
        #   2) 1×1 conv
        #   3) sigmoid
        #   4) permute back and expand
        G_W_r = F_r.permute(0, 3, 2, 1)         # (B, W, H, C)

        G_W_r = self.convW_r(G_W_r)            # (B, 1, H, C)

        G_W_r = torch.sigmoid(G_W_r)           # (B, 1, H, C)
        # Permute back to (B, C, H, W) shape but we need to place the 'W' dimension last:
        G_W_r = G_W_r.permute(0, 3, 2, 1)      # (B, C, H, W) -> W = 1

        
        # Now it can be broadcast over (B, C, H, W) directly:
        # (because it already is (B, C, H, W)),

        ###################TO CHANGE
        
        # Gate for F_g (call it G^W_g):
        G_W_g = F_g.permute(0, 3, 2, 1)
        G_W_g = self.convW_g(G_W_g)
        G_W_g = torch.sigmoid(G_W_g)
        G_W_g = G_W_g.permute(0, 3, 2, 1)      # (B, C, H, W)

        # Combine them (similar to Eq. 27 but width‐wise):
        F_W = F_r * G_W_r + F_g * G_W_g


        # -------------------------------------------------------------
        # 3) Channel‐wise gating  (paper Eqs. 28–29, right side of Fig. 7)
        # -------------------------------------------------------------
        # We first learn gating maps for F^H and F^W across the channel dimension.
        # The figure suggests separate 1×1 convolutions → sigmoid for each path.
        # Let G^C_H = sigmoid( Conv1×1(F^H) )
        #     G^C_W = sigmoid( Conv1×1(F^W) )
        # They each become shape (B, 1, H, W), then broadcast over channel dimension.
        G_C_H = torch.sigmoid(self.convC_H(F_H))  # (B, 1, H, W)
        G_C_W = torch.sigmoid(self.convC_W(F_W))  # (B, 1, H, W)

        # Expand to (B, C, H, W) if necessary:
        G_C_H = G_C_H.expand_as(F_H)             # (B, C, H, W)
        G_C_W = G_C_W.expand_as(F_W)             # (B, C, H, W)

        # Final fused output, Eq. (29): F = (G^C_H ⊗ F^H) + (G^C_W ⊗ F^W)
        F_out = G_C_H * F_H + G_C_W * F_W



        return F_out
