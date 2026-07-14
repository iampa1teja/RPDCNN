import torch 
import torch.nn as nn 
import math 

from ..utils.mask_utils import aligned_bilinear, conv_with_kaiming_uniform, sigmoid_focal_loss

INF = 100000000

class MaskBranch(nn.Module):
    def __init__(
        self, 
        in_channels_dict, 
        in_features=["p3", "p4", "p5"], 
        channels=128, 
        out_channels=8, 
        num_convs=4, 
        norm="GN", 
        sem_loss_on=False, 
        num_classes=80, 
        out_stride=8, 
        prior_prob=0.01
    ):
        super().__init__()
        self.in_features = in_features
        self.sem_loss_on = sem_loss_on
        self.num_outputs = out_channels
        self.out_stride = out_stride

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(in_channels_dict[in_feature], channels, 3, stride=1, padding=1))

        tower = []
        for i in range(num_convs):
            tower.append(conv_block(channels, channels, 3, stride=1, padding=1))
        tower.append(nn.Conv2d(channels, max(self.num_outputs, 1), 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.sem_loss_on:
            self.focal_loss_alpha = 0.25
            self.focal_loss_gamma = 2.0
            in_channels = in_channels_dict[self.in_features[0]]
            self.seg_head = nn.Sequential(
                conv_block(in_channels, channels, kernel_size=3, stride=1, padding=1),
                conv_block(channels, channels, kernel_size=3, stride=1, padding=1)
            )
            self.logits = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, features, gt_bitmasks_full_list=None, gt_classes_list=None):
        """
        features: dict of tensors, e.g. {"p3": p3_tensor, ...}
        gt_bitmasks_full_list: list of (N, H, W) tensors for semantic target generation
        gt_classes_list: list of (N,) tensors
        """
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)

        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        losses = {}
        if self.training and self.sem_loss_on and gt_bitmasks_full_list is not None:
            logits_pred = self.logits(self.seg_head(features[self.in_features[0]]))

            semantic_targets = []
            for gt_bitmasks_full, gt_classes in zip(gt_bitmasks_full_list, gt_classes_list):
                h, w = gt_bitmasks_full.size()[-2:]
                areas = gt_bitmasks_full.sum(dim=-1).sum(dim=-1)
                areas = areas[:, None, None].repeat(1, h, w)
                areas[gt_bitmasks_full == 0] = INF
                areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                min_areas, inds = areas.min(dim=1)
                per_im_sematic_targets = gt_classes[inds] + 1
                per_im_sematic_targets[min_areas == INF] = 0
                per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)
                semantic_targets.append(per_im_sematic_targets)

            semantic_targets = torch.stack(semantic_targets, dim=0)
            semantic_targets = semantic_targets[:, None, self.out_stride // 2::self.out_stride, self.out_stride // 2::self.out_stride]

            num_classes = logits_pred.size(1)
            class_range = torch.arange(num_classes, dtype=logits_pred.dtype, device=logits_pred.device)[:, None, None]
            class_range = class_range + 1
            one_hot = (semantic_targets == class_range).float()
            num_pos = (one_hot > 0).sum().float().clamp(min=1.0)

            loss_sem = sigmoid_focal_loss(
                logits_pred, one_hot,
                alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="sum"
            ) / num_pos
            losses['loss_sem'] = loss_sem

        return mask_feats, losses
