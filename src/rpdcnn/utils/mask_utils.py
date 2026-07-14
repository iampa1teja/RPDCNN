import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode="bilinear",
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device) + stride / 2.0
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device) + stride / 2.0
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    return torch.stack((shift_x, shift_y), dim=-1)

def conv_with_kaiming_uniform(norm="GN", activation=True):
    def conv_block(in_ch, out_ch, kernel_size, stride=1, padding=0):
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=(norm is None))]
        if norm == "GN": layers.append(nn.GroupNorm(8, out_ch))
        if activation: layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    return conv_block

def sigmoid_focal_loss(logits, targets, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"):
    """Replaces fvcore.nn.sigmoid_focal_loss_jit"""
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean": return loss.mean()
    elif reduction == "sum": return loss.sum()
    return loss

def unfold_wo_center(x, kernel_size, dilation):
    """Replaces adet.modeling.condinst.condinst.unfold_wo_center for BoxInst"""
    assert x.dim() == 4
    N, C, H, W = x.size()
    padding = (kernel_size - 1) * dilation // 2
    x_unfold = F.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding)
    center = kernel_size * kernel_size // 2
    x_unfold = torch.cat([x_unfold[:, :, :center], x_unfold[:, :, center + 1:]], dim=2)
    return x_unfold.view(N, C, -1, H, W)

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    num_insts = params.size(0)
    num_layers = len(weight_nums)
    params_splits = list(torch.split(params, weight_nums + bias_nums, dim=1))
    weight_splits, bias_splits = params_splits[:num_layers], params_splits[num_layers:]
    
    for l in range(num_layers):
        if l < num_layers - 1:
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)
    return weight_splits, bias_splits


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    return 1. - (2 * intersection / union)

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(mask_scores.max(dim=2, keepdim=True)[0], gt_bitmasks.max(dim=2, keepdim=True)[0])
    mask_losses_x = dice_coefficient(mask_scores.max(dim=3, keepdim=True)[0], gt_bitmasks.max(dim=3, keepdim=True)[0])
    return (mask_losses_x + mask_losses_y).mean()

def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4
    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)
    log_fg_prob_unfold = unfold_wo_center(log_fg_prob, kernel_size=pairwise_size, dilation=pairwise_dilation)
    log_bg_prob_unfold = unfold_wo_center(log_bg_prob, kernel_size=pairwise_size, dilation=pairwise_dilation)
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(torch.exp(log_same_fg_prob - max_) + torch.exp(log_same_bg_prob - max_)) + max_
    return -log_same_prob[:, 0]
