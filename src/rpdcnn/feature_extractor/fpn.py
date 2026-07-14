import torch 
import torch.nn as nn 
from torchvision.ops import FeaturePyramidNetwork 
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from collections import OrderedDict
from typing import List

class FPN(nn.Module):
    """
    Wrapper around torchvision.ops.FeaturePyramidNetwork adapted for FCOS/CondInst.
    
    Takes a list of feature maps (e.g., C3, C4, C5) and outputs a list of 
    feature maps with additional downsampled levels (e.g., P3, P4, P5, P6, P7).
    """

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.num_input_levels = len(in_channels_list)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(in_channels_list),
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels, out_channels),
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != self.num_input_levels:
            raise ValueError(
                f"Expected {self.num_input_levels} input feature maps, got {len(features)}"
            )
        x = OrderedDict((str(i), feat) for i, feat in enumerate(features))
        out = self.fpn(x)

        return list(out.values())
