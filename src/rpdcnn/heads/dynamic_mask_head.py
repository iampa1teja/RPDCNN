import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from ..utils.mask_utils import (
    aligned_bilinear,
    compute_locations,
    dice_coefficient,
    parse_dynamic_params,
)



class DynamicMaskHead(nn.Module):
    def __init__(
        self, 
        in_channels=8, 
        channels=8, 
        num_layers=3, 
        mask_out_stride=4, 
        disable_rel_coords=False, 
        sizes_of_interest=[64, 128, 256, 512, 1024]
    ):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.in_channels = in_channels
        self.mask_out_stride = mask_out_stride
        self.disable_rel_coords = disable_rel_coords

        self.register_buffer("sizes_of_interest", torch.tensor(sizes_of_interest + [sizes_of_interest[-1] * 2]))

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    def mask_heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, mask_feats, mask_feat_stride, im_inds, locations, fpn_levels, mask_head_params):
        """
        im_inds: (N,) tensor mapping instances to batch images.
        locations: (N, 2) tensor of instance center locations (pixel coords).
        fpn_levels: (N,) tensor of FPN levels for sizes_of_interest mapping.
        mask_head_params: (N, controller_dim) dynamic weights.
        """
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        
        upsample_factor = int(mask_feat_stride / self.mask_out_stride)
        n_inst = len(im_inds)
        
        if n_inst == 0:
            return torch.empty(
                (0, 1, mask_feats.size(2) * upsample_factor, mask_feats.size(3) * upsample_factor), 
                device=mask_feats.device
            )

        grid_locations = compute_locations(mask_feats.size(2), mask_feats.size(3), stride=mask_feat_stride, device=mask_feats.device)
        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            relative_coords = locations.reshape(-1, 1, 2) - grid_locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)
            mask_head_inputs = torch.cat([relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        weights, biases = parse_dynamic_params(mask_head_params, self.channels, self.weight_nums, self.bias_nums)
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)
        mask_logits = mask_logits.reshape(-1, 1, H, W)
        
        mask_logits = aligned_bilinear(mask_logits, upsample_factor)

        return mask_logits

    def compute_loss(self, mask_logits, gt_bitmasks, mask_feats, mask_head_params):
        """
        gt_bitmasks should be sliced to match positive instances: gt_bitmasks[gt_inds].
        """
        if mask_logits.size(0) == 0:
            return (mask_feats.sum() * 0 + mask_head_params.sum() * 0).mean()

        mask_scores = mask_logits.sigmoid()
        mask_losses = dice_coefficient(mask_scores, gt_bitmasks.to(dtype=mask_logits.dtype))
        return mask_losses.mean()
