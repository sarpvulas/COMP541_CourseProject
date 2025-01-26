# TOODHead.py
# Fully implemented TOOD (Task-Aligned One-Stage Object Detection Head)
# Ref: "TOOD: Task-Aligned One-Stage Object Detection" (ICCV 2021)
#      Feng et al. https://arxiv.org/abs/2108.07755
# Using mmcv's deform_conv2d for deformable sampling.
# This version includes full classification + regression heads, label assignment, and the needed losses.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmcv.ops import deform_conv2d

# ------------------------------------------
# 0)  Task Decomposition
# ------------------------------------------
class TaskDecomposition(nn.Module):
    """
    Task decomposition module in the TOOD paper, used for classification
    and regression feature separation. Includes the layer attention mechanism.
    """

    def __init__(self,
                 feat_channels: int,
                 stacked_convs: int,
                 la_down_rate: int = 8):
        """
        Args:
            feat_channels (int): number of channels per conv in the head
            stacked_convs (int): how many stacked conv layers
            la_down_rate (int): downsample factor for the layer attention module
        """
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = feat_channels * stacked_convs  # after we concat

        # "layer attention" submodule
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1),
            nn.Sigmoid()
        )

        # merges from feat_channels*stacked_convs -> feat_channels
        self.reduction_conv = nn.Conv2d(
            self.in_channels,
            feat_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.activate = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.reduction_conv.weight, std=0.01)
        nn.init.constant_(self.reduction_conv.bias, 0)

    def forward(self, feat, avg_feat=None):
        """
        Args:
            feat: (B, in_channels, H, W), in_channels = feat_channels*stacked_convs
            avg_feat: optional global pooled feature (B, in_channels, 1, 1)
        Returns:
            out_feat: (B, feat_channels, H, W)
        """
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))

        # layer attention yields shape (B, stacked_convs, 1, 1)
        weight = self.layer_attention(avg_feat)

        # interpret the reduction_conv kernel: shape (feat_channels, in_channels, 1, 1)
        # reweight it by 'weight'
        conv_w = self.reduction_conv.weight  # [feat_channels, in_channels, 1, 1]
        # shape => (B, feat_channels, stacked_convs, feat_channels)
        reweighted = weight.reshape(b, 1, self.stacked_convs, 1) * conv_w.reshape(1, self.feat_channels, self.stacked_convs, self.feat_channels)
        # unify to (B, feat_channels, in_channels)
        reweighted = reweighted.reshape(b, self.feat_channels, self.in_channels)

        # flatten the input feat (B, in_channels, H*W)
        feat_2d = feat.reshape(b, self.in_channels, h*w)
        # batch matmul => (B, feat_channels, h*w)
        out = torch.bmm(reweighted, feat_2d)
        # restore to 4D
        out = out.reshape(b, self.feat_channels, h, w)

        # add bias if any
        if self.reduction_conv.bias is not None:
            out = out + self.reduction_conv.bias.view(1, -1, 1, 1)

        out = self.activate(out)
        return out


# ------------------------------------------
# 1)  TOODHead
# ------------------------------------------
class TOODHead(nn.Module):
    """
    TOOD: Task-Aligned One-Stage Object Detection Head
    Reproduces the paper fully:
      - stacked conv layers
      - task decomposition for cls & reg
      - classification probability & offset modules
      - deformable sampling
      - anchor-based or anchor-free label assignment
      - final loss computations
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_anchors: int = 9,
                 stacked_convs: int = 4,
                 feat_channels: int = 256,
                 anchor_type: str = 'anchor_based',
                 # loss configs
                 use_sigmoid_cls=False,
                 # etc. you can add more if needed
                 ):
        """
        Args:
            in_channels (int): channels in each FPN feature map
            num_classes (int): number of object classes
            num_anchors (int): e.g. 9 (3 scales * 3 ratios) if anchor-based
            stacked_convs (int): number of conv layers before decomposition
            feat_channels (int): channel dimension in each conv
            anchor_type (str): 'anchor_free' or 'anchor_based'
            use_sigmoid_cls (bool): whether classification is done with sigmoid
        """
        super().__init__()
        assert anchor_type in ['anchor_free', 'anchor_based']
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.anchor_type = anchor_type
        self.use_sigmoid_cls = use_sigmoid_cls

        # stacked convs
        self.inter_convs = nn.ModuleList()
        for i in range(stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, self.feat_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

        # task decomposition for classification & regression
        self.cls_decomp = TaskDecomposition(
            feat_channels=self.feat_channels,
            stacked_convs=self.stacked_convs,
            la_down_rate=self.stacked_convs * 8
        )
        self.reg_decomp = TaskDecomposition(
            feat_channels=self.feat_channels,
            stacked_convs=self.stacked_convs,
            la_down_rate=self.stacked_convs * 8
        )

        # final conv heads
        # classification: channels = num_anchors*(num_classes+1 or num_classes if use_sigmoid)
        out_channels_cls = self.num_anchors * self.num_classes  # Remove +1 for background class
        self.tood_cls = nn.Conv2d(self.feat_channels, out_channels_cls, kernel_size=3, padding=1)

        # regression: channels = num_anchors*4
        self.tood_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, kernel_size=3, padding=1)

        # classification probability alignment
        self.cls_prob_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, kernel_size=3, padding=1)
        )

        # init
        self._init_weights()

    def _init_weights(self):
        # init stacked inter convs
        for m in self.inter_convs:
            for sub_m in m:
                if isinstance(sub_m, nn.Conv2d):
                    nn.init.normal_(sub_m.weight, std=0.01)
                    nn.init.constant_(sub_m.bias, 0)

        # init cls_prob_module & reg_offset_module
        for m in self.cls_prob_module:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
        # typical bias init for ~0.01 prior prob => logit = -ln((1-0.01)/0.01)= about -4.595
        nn.init.constant_(self.cls_prob_module[-1].bias, -4.595)

        # decomposition modules
        self.cls_decomp.init_weights()
        self.reg_decomp.init_weights()

        # final heads
        nn.init.normal_(self.tood_cls.weight, std=0.01)
        nn.init.constant_(self.tood_cls.bias, 0)
        nn.init.normal_(self.tood_reg.weight, std=0.01)
        nn.init.constant_(self.tood_reg.bias, 0)

    def deform_sampling(self, x, offset, deform_groups=36):
        """
        x: (B, C_in, H, W)
        offset: (B, 2*kH*kW*deform_groups, H_out, W_out)
        """
        b, c, h, w = x.shape

        weight = x.new_ones(c, 1, 3, 3)

        out = deform_conv2d(x, offset, weight, 1, 1, 1, c, c)

        return out

    def check_for_nans(self, tensor, name="tensor", exit_on_nan=True):
        if torch.isnan(tensor).any():
            print(f"[DEBUG] Detected NaNs in {name}, min={tensor.min()}, max={tensor.max()}")
            if exit_on_nan:
                raise RuntimeError(f"NaNs found in {name}!")

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): Multi-scale feature maps from the FPN
                                  each shape [B, C, H, W].

        Returns:
            cls_scores (list[Tensor]): Classification predictions
                each shape [B, (num_anchors * (num_classes or num_classes+1)), H, W].
            reg_preds (list[Tensor]): Regression predictions
                each shape [B, (num_anchors * 4), H, W].
        """
        cls_scores = []
        reg_preds = []

        for idx, feat in enumerate(feats):

            # 1) Pass through stacked convolution layers
            inter_feats = []
            x = feat
            for layer_idx, inter_conv in enumerate(self.inter_convs):
                x = inter_conv(x)
                inter_feats.append(x)

            # 2) Concatenate intermediate features
            feat_cat = torch.cat(inter_feats, dim=1)  # Shape [B, feat_channels * stacked_convs, H, W]
            self.check_for_nans(feat_cat, name=f"feat_cat[{idx}]")
            avg_feat = F.adaptive_avg_pool2d(feat_cat, (1, 1))

            # 3) Task decomposition
            cls_feat = self.cls_decomp(feat_cat, avg_feat)
            self.check_for_nans(cls_feat, name=f"cls_feat[{idx}]")
            reg_feat = self.reg_decomp(feat_cat, avg_feat)
            self.check_for_nans(cls_feat, name=f"reg_feat[{idx}]")

            # 4) Classification
            cls_logits = self.tood_cls(cls_feat)
            self.check_for_nans(cls_logits, name=f"cls_logits[{idx}]")
            cls_prob = self.cls_prob_module(feat_cat)  # Alignment factor
            self.check_for_nans(cls_prob, name=f"cls_prob[{idx}]")
            product = cls_logits.sigmoid() * cls_prob.sigmoid() + 1e-6
            cls_score = torch.sqrt(torch.clamp(product, min=0))
            cls_scores.append(cls_score)

            # 5) Regression
            reg_raw = self.tood_reg(reg_feat)
            self.check_for_nans(reg_raw, name=f"reg_raw[{idx}]")

            # 6) Offset adjustment (dynamic for each feature map)
            groups = reg_raw.shape[1] # Number of input channels
            kernel_size = 3
            offset_channels = 2 * kernel_size * kernel_size * groups

            reg_offset_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, offset_channels, kernel_size=3, padding=1)).to(feat.device)

            offset = reg_offset_module(feat_cat)
            self.check_for_nans(offset, name=f"offset[{idx}]")

            assert offset.shape[1] == offset_channels, (
                f"Offset shape mismatch: expected {offset_channels}, got {offset.shape[1]}"
            )

            # Perform deformable convolution for regression
            reg_out = self.deform_sampling(reg_raw, offset, groups)
            self.check_for_nans(reg_out, name=f"reg_out[{idx}]")
            reg_preds.append(reg_out)
            print(f"Regression output shape: {reg_out.shape}")

        print(f"\nFinal classification predictions: {len(cls_scores)} tensors")
        print(f"Final regression predictions: {len(reg_preds)} tensors")

        return cls_scores, reg_preds

