import timm 
import torch.nn as nn 
class Backbone(nn.Module):
    SUPPORTED = [
        'convnext_small',
        'resnet50',
        'efficientnet_b3',
        'mobilenetv3_large_100',
        'regnety_008',
    ]
    
    def __init__(self, name: str = 'convnext_small', pretrained: bool = True, in_strides: list = [8, 16, 32]): 
        super().__init__()
        assert name in self.SUPPORTED, f"Unsupported backbone \"{name}\". Choose from {self.SUPPORTED}"
        self.net = timm.create_model(name, features_only=True, pretrained=pretrained) 
        all_channels = self.net.feature_info.channels() 
        all_strides = self.net.feature_info.reduction() 
        self.level_idx = [i for i, s in enumerate(all_strides) if s in in_strides]
        
        if len(self.level_idx) != len(in_strides):
            raise ValueError(f"Backbone {name} does not natively support all requested strides {in_strides}. Available: {all_strides}")
        self.out_channels = [all_channels[i] for i in self.level_idx] 
    
    def forward(self, x):
        all_feats = self.net(x) 
        return [all_feats[i] for i in self.level_idx]
