import torch 
import torch.nn as nn 
import math

class Scale(nn.Module):
    """
    Learnable scaler used by FCOS/CondInst to ccalibrate bbox regression 
    independently for each FPN level 
    """
    def __init__(self, init_value = 1.0):
        super().__init__() 
        self.scale = nn.Parameter(torch.tensor([float(init_value)], dtype=torch.float32))

    def forward(self, x): 
        return x * self.scale 
    
class DynamicDetHead(nn.Module): 
    """
    FOCS/CondInst detection head. 

    Returns per-FPN-level predictions: 
        bbox_preds         : (B, 4, H, W) (l, t, r, b distances) 
        cls_logits         : (B, num_classes, H, w)
        centerness_logits  : (B, 1, H, W) 
        controller         : (B, controller_dim, H, W)
    """
    def __init__(
            self, 
            in_ch, 
            num_classes, 
            controller_dim, 
            in_strides = (8, 16, 32), 
            num_convs = 4, 
            norm_groups = 8,
    ):
        super().__init__() 
        if in_ch % norm_groups != 0: 
            raise ValueError(
                f"in_ch ({in_ch}) must be divisible by norm_groups ({norm_groups})"
            )
        self.in_strides = list(in_strides) 

        self.cls_tower = self._build_tower(in_ch, num_convs, norm_groups)
        self.bbox_tower = self._build_tower(in_ch, num_convs, norm_groups)
        self.cls_logits = nn.Conv2d(
            in_ch, 
            num_classes, 
            kernel_size=3, 
            padding=1,
        )

        self.bbox_pred = nn.Conv2d(
            in_ch, 
            4, 
            kernel_size=3, 
            padding=1
        )

        self.centerness = nn.Conv2d(
            in_ch, 
            1, 
            kernel_size=3, 
            padding=1, 
        )

        self.controller = nn.Conv2d(
            in_ch,
            controller_dim, 
            kernel_size=3, 
            padding=1,
        )

        self.scales = nn.ModuleList(
            [Scale() for _ in self.in_strides]
        )
        self._init_weights() 

    def _build_tower(self, in_ch, num_convs, norm_groups):
        layers = [] 
        for _ in range(num_convs): 
            layers.extend([
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(norm_groups, in_ch), 
                nn.ReLU(inplace=True) 
            ])
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward_single(self, feat, level_idx, stride):
        cls_feat = self.cls_tower(feat)
        bbox_feat = self.bbox_tower(feat)
        cls_logits = self.cls_logits(cls_feat)
        bbox_preds = torch.exp(
            self.scales[level_idx](self.bbox_pred(bbox_feat))
        ) * stride
        centerness_logits = self.centerness(bbox_feat)
        controller = self.controller(bbox_feat)

        return (
            bbox_preds,
            cls_logits,
            centerness_logits,
            controller,
        )

    def forward(self, features): 
        if len(features) != len(self.in_strides): 
            raise ValueError(
                f"Expected {len(self.in_strides)} feature maps." 
                f"got {len(features)}" 
            )
        outputs = [] 
        for level_idx, (feat, stride) in enumerate(
            zip(features, self.in_strides)
        ):
            outputs.append(
                self.forward_single(
                    feat, 
                    level_idx, 
                    stride,
                )
            )
        return outputs
