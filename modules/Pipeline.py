import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
from collections import OrderedDict

from modules.Debug import Debug
from modules.IDM import IDM
from modules.IEEM import IEEM
from modules.TAGFFM import TAGFFM
from modules.dual_backbone import DualResNet50
from TOODHead import TOODHead
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

class FullPipeline(nn.Module):
    def __init__(self, num_classes=10, backbone_out_channels=256):
        super(FullPipeline, self).__init__()
        self.num_classes = num_classes

        self.debugger = Debug(exit_on_nan=True)

        # Modules for edge enhancement branch
        self.idm = IDM()
        self.ieem = IEEM()

        # Dual backbone
        self.dual_backbone = DualResNet50()
        
        self.tagffm1 = TAGFFM(1)
        self.tagffm2 = TAGFFM(2)
        self.tagffm3 = TAGFFM(3)
        self.tagffm4 = TAGFFM(4)

        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=backbone_out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

        # Detection head (TOOD)
        self.tood_head = TOODHead(
            in_channels=backbone_out_channels,
            num_classes=num_classes,  # will produce num_classes + 1 channels
            num_anchors=9
        )

    def forward(self, x):
        x_orig = x

        self.debugger.debug_tensor(x_orig, desc="[1] Input to pipeline")

        # 1) IDM -> IEEM
        gray_images = self.idm(x_orig)
        self.debugger.debug_tensor(gray_images, desc="[2] IDM output")

        enhanced_images = self.ieem(gray_images)
        self.debugger.debug_tensor(enhanced_images, desc="[3] IEEM outputs")

        x_enhanced = torch.cat(enhanced_images, dim=1)
        self.debugger.debug_tensor(x_enhanced, desc="[4] x_enhanced")

        # 2) dual backbone
        feats_rgb, feats_gray = self.dual_backbone(x_orig, x_enhanced)
        self.debugger.debug_tensor(feats_rgb, desc="[5] feats_rgb")
        self.debugger.debug_tensor(feats_gray, desc="[6] feats_gray")

        # 3) fuse
        f1_fused = self.tagffm1(feats_rgb[0], feats_gray[0])
        f2_fused = self.tagffm2(feats_rgb[1], feats_gray[1])
        f3_fused = self.tagffm3(feats_rgb[2], feats_gray[2])
        f4_fused = self.tagffm4(feats_rgb[3], feats_gray[3])

        self.debugger.debug_tensor(f1_fused, desc="[7a] f1_fused")
        self.debugger.debug_tensor(f2_fused, desc="[7b] f2_fused")
        self.debugger.debug_tensor(f3_fused, desc="[7c] f3_fused")
        self.debugger.debug_tensor(f4_fused, desc="[7d] f4_fused")

        feature_dict = OrderedDict([
            ('feat1', f1_fused),
            ('feat2', f2_fused),
            ('feat3', f3_fused),
            ('feat4', f4_fused),
        ])
        fpn_features = self.fpn(feature_dict)
        self.debugger.debug_tensor(list(fpn_features.values()), desc="[8] FPN outputs")

        cls_preds, reg_preds = self.tood_head(list(fpn_features.values()))
        self.debugger.debug_tensor(cls_preds, desc="[9a] TOODHead cls_preds")
        self.debugger.debug_tensor(reg_preds, desc="[9b] TOODHead reg_preds")

        # reshape
        cls_scores = torch.cat([
            p.permute(0, 2, 3, 1).contiguous().view(
                p.size(0), -1, self.tood_head.num_classes
            )
            for p in cls_preds
        ], dim=1)
        bbox_preds = torch.cat([
            p.permute(0, 2, 3, 1).contiguous().view(p.size(0), -1, 4)
            for p in reg_preds
        ], dim=1)


        return (cls_scores, bbox_preds)
