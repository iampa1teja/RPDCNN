import dataclasses
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class RPDCFG: 
    num_classes: int = 80 
    in_strides: List[int] = field(default_factory=lambda:[4, 8, 16, 32])
    freeze_backbone: bool = False 
    freeze_backbone_bn: bool = False
    freeze_backbone_backbone_bn: bool = False
    backbone_freeze_stages: int = -1 

    img_h: int = 512 
    img_w: int = 512 

    resume_training: bool = False 
    resume_checkpoint_path: str = "" 

    backbone_name: str = "convnext_small" 
    backbone_pretrained: bool = True 

    neck_type: str = "bifpn" 
    bifpn_out_channels: int  = 256
    bifpn_num_layers: int = 3 

    use_cbam: bool = True 
    cbam_reduction: int = 16 

    det_num_convs: int = 4
    det_norm_groups: int = 8

    mask_branch_in_features: List[str] = field(default_factory=lambda: ["p3", "p4", "p5"])
    mask_branch_channels: int = 128
    mask_branch_out_channels: int = 8
    mask_branch_num_convs: int = 4
    mask_branch_norm: str = "GN"
    mask_branch_sem_loss_on: bool = False
    mask_branch_out_stride: int = 8

    mask_head_num_layers: int = 3
    mask_head_channels: int = 8
    mask_head_out_stride: int = 4
    mask_head_disable_rel_coords: bool = False
    mask_head_sizes_of_interest: List[int] = field(default_factory=lambda: [64, 128, 256])

    assigner_center_sampling_radius: float = 1.5
    assigner_regress_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(-1, 64), (64, 128), (128, 100000)]
    )

    loss_focal_alpha: float = 0.25
    loss_focal_gamma: float = 2.0
    loss_iou_type: str = "giou"


_NANO_PRESET = dict(
    backbone_name="mobilenetv3_large_100",
    backbone_pretrained=True,
    freeze_backbone=True,
    freeze_backbone_bn=True,
    in_strides=[8, 16, 32],
    neck_type="fpn",  
    bifpn_out_channels=128,
    bifpn_num_layers=3,
    use_cbam=False,
    det_num_convs=2,
    mask_branch_channels=64,
    mask_branch_out_channels=8,
    mask_branch_num_convs=2,
    mask_head_num_layers=2,
    mask_head_channels=8,
    mask_branch_in_features=["p3", "p4", "p5"],
    mask_head_sizes_of_interest=[64, 128, 256, 512],
    assigner_regress_ranges=[(-1, 64), (64, 128), (128, 256), (256, 512), (512, 100000)],
)

_FAST_PRESET = dict(
    backbone_name="resnet50",
    backbone_pretrained=True,
    freeze_backbone=False,
    freeze_backbone_bn=False,
    in_strides=[8, 16, 32],
    neck_type="bifpn",
    bifpn_out_channels=192,
    bifpn_num_layers=3,
    use_cbam=True,
    det_num_convs=3,
    mask_branch_channels=96,
    mask_branch_out_channels=8,
    mask_branch_num_convs=3,
    mask_head_num_layers=3,
    mask_head_channels=8,
    mask_branch_in_features=["p3", "p4", "p5"],
    mask_head_sizes_of_interest=[64, 128, 256],
    assigner_regress_ranges=[(-1, 64), (64, 128), (128, 100000)],
)

_MAIN_PRESET = dict(
    backbone_name="convnext_small",
    backbone_pretrained=True,
    freeze_backbone=False,
    freeze_backbone_bn=False,
    in_strides=[4, 8, 16, 32, 64],
    neck_type="bifpn",
    bifpn_out_channels=256,
    bifpn_num_layers=3,
    use_cbam=True,
    det_num_convs=4,
    mask_branch_channels=128,
    mask_branch_out_channels=8,
    mask_branch_num_convs=4,
    mask_head_num_layers=3,
    mask_head_channels=8,
    mask_branch_in_features=["p3", "p4", "p5", "p6"],
    mask_head_sizes_of_interest=[64, 128, 256, 512],
    assigner_regress_ranges=[(-1, 64), (64, 128), (128, 256), (256, 512), (512, 100000)],
)

_PRESETS = {
    "nano": _NANO_PRESET,
    "fast": _FAST_PRESET,
    "main": _MAIN_PRESET,
}


def get_preset_cfg(name: str, **overrides) -> RPDCFG:
    """
    Build an RPDCFG from a named preset, then apply any explicit user
    overrides on top.

    Precedence: user overrides > preset values > RPDCFG dataclass defaults.

    IMPORTANT: `overrides` must only contain keys the caller explicitly
    passed in. We build the base config directly from the preset dict
    (not from a default-constructed RPDCFG), so any field the preset
    doesn't mention still falls back correctly to RPDCFG's own default
    rather than being silently reset by a round-trip through RPDCFG().
    """
    if name not in _PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from {list(_PRESETS)}")

    base_cfg = RPDCFG(**_PRESETS[name])

    unknown_keys = set(overrides) - {f.name for f in dataclasses.fields(RPDCFG)}
    if unknown_keys:
        raise ValueError(f"Unknown RPDCFG override field(s): {unknown_keys}")

    return dataclasses.replace(base_cfg, **overrides)


def get_default_cfg() -> RPDCFG:
    return RPDCFG()
