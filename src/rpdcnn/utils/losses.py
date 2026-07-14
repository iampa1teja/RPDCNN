import torch 
import torch.nn as nn 
import torch.nn.functional as F

class IOULoss(nn.Module):
    """
    Exact replica of AdelaiDet's adet.layers.iou_loss.IOULoss.
    Used for FCOS bounding box regression.
    
    Call Site Requirements:
    1. Instantiate with `loss_type="giou"` to avoid vanishing gradients on non-overlapping boxes.
    2. This returns a SUM. The caller must externally divide by `num_pos`.
    """
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        # The AdelaiDet +1.0 smoothing constant
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight.squeeze()).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Exact pure-PyTorch replica of fvcore.nn.sigmoid_focal_loss.
    Centralized here to prevent duplication across the mask and detection heads.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def compute_fcos_losses(
    pred_logits: torch.Tensor,
    pred_centerness: torch.Tensor,
    pred_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    box_loss_func: nn.Module,
    focal_loss_alpha: float = 0.25,
    focal_loss_gamma: float = 2.0
):
    """
    Aggregates FCOS detection head losses (Classification, Centerness, and Box Regression).
    
    Args:
        pred_logits: (N, num_classes) predicted class logits across all FPN levels.
        pred_centerness: (N, 1) predicted centerness logits.
        pred_boxes: (N, 4) predicted box distances (l, t, r, b).
        target_labels: (N,) target class IDs. Assumes background class is represented by `num_classes`.
        target_boxes: (N, 4) ground truth box distances.
        box_loss_func: Instantiated IOULoss module.
    
    Returns:
        dict: containing "loss_cls", "loss_centerness", "loss_box".
    """
    num_classes = pred_logits.size(1)
    
    # 1. Classification Loss (Focal Loss over ALL spatial locations)
    # Background instances (target == num_classes) will have an all-zero one-hot vector
    labels_one_hot = F.one_hot(target_labels, num_classes=num_classes + 1)[:, :-1].to(pred_logits.dtype)
    
    cls_loss = sigmoid_focal_loss(
        pred_logits, labels_one_hot, 
        alpha=focal_loss_alpha, gamma=focal_loss_gamma, reduction="sum"
    )
    
    # Identify positive locations (foreground objects)
    pos_inds = torch.nonzero(target_labels != num_classes).squeeze(1)
    num_pos = max(pos_inds.numel(), 1.0)
    
    # 2. Box and Centerness Losses (Only computed on POSITIVE spatial locations)
    if pos_inds.numel() > 0:
        pos_pred_centerness = pred_centerness[pos_inds]
        pos_pred_boxes = pred_boxes[pos_inds]
        pos_target_boxes = target_boxes[pos_inds]
        
        # Calculate ground-truth centerness targets
        left_right = pos_target_boxes[:, [0, 2]]
        top_bottom = pos_target_boxes[:, [1, 3]]
        
        # Centerness formula: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
        centerness_targets = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        ).unsqueeze(1)
        
        # Centerness Loss (BCE)
        centerness_loss = F.binary_cross_entropy_with_logits(
            pos_pred_centerness, centerness_targets, reduction="sum"
        )
        
        # Box Regression Loss (IOU)
        # FCOS weights the bounding box loss by the centerness targets
        box_loss = box_loss_func(pos_pred_boxes, pos_target_boxes, weight=centerness_targets)
        
    else:
        # Dummy losses to keep the computation graph alive and avoid DDP sync issues
        centerness_loss = pred_centerness.sum() * 0
        box_loss = pred_boxes.sum() * 0
        
    return {
        "loss_cls": cls_loss / num_pos,
        "loss_centerness": centerness_loss / num_pos,
        "loss_box": box_loss / num_pos
    }
