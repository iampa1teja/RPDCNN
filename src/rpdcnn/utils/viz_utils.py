import cv2 
import numpy as np 
import torch 
import os 

def get_class_color(class_id: int) -> tuple:
    """
    Deterministic, high-contrast BGR-ish RGB color for a given class id.
    Uses evenly-spaced hues so any number of classes gets visually distinct,
    stable colors across runs/epochs (same class_id -> same color always).
    """
    golden_ratio_conjugate = 0.618033988749895
    hue = (class_id * golden_ratio_conjugate) % 1.0
    hsv = np.array([[[hue * 255, 255, 255]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(c) for c in rgb)


def _draw_masks_on_image(
    image: np.ndarray,
    masks: np.ndarray,
    class_ids: np.ndarray = None,
    color=(0, 255, 0),
    alpha=0.45,
) -> np.ndarray:
    """
    image: (H, W, 3) uint8 RGB
    masks: (M, H, W) float/bool
    class_ids: optional (M,) int array. If provided, each mask is drawn in
        its class's color via get_class_color(); `color` is ignored.
        If omitted, every mask uses the single fallback `color` (old behavior).
    """
    out = image.copy()
    for i, m in enumerate(masks):
        m = m.astype(bool)
        if not m.any():
            continue
        mask_color = get_class_color(int(class_ids[i])) if class_ids is not None else color
        overlay = out.copy()
        overlay[m] = mask_color
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, mask_color, 1)
    return out


def _draw_class_legend(image: np.ndarray, class_ids, class_names=None) -> np.ndarray:
    """
    Draws a small color-key legend (swatch + label) in the top-right corner
    for every class id present in `class_ids`.
    """
    out = image.copy()
    unique_ids = sorted(set(int(c) for c in class_ids))
    if not unique_ids:
        return out

    swatch = 14
    pad = 6
    row_h = swatch + 6
    x0 = out.shape[1] - 140
    y0 = 10

    for row, cid in enumerate(unique_ids):
        y = y0 + row * row_h
        color = get_class_color(cid)
        cv2.rectangle(out, (x0, y), (x0 + swatch, y + swatch), color, -1)
        cv2.rectangle(out, (x0, y), (x0 + swatch, y + swatch), (255, 255, 255), 1)
        label = class_names[cid] if class_names is not None and cid < len(class_names) else f"class {cid}"
        cv2.putText(
            out, label, (x0 + swatch + pad, y + swatch - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


@torch.no_grad()
def save_epoch_visualization(
    model,
    images: torch.Tensor,         
    gt_bitmasks_full_batch,          
    gt_labels_batch=None,           
    output_dir: str = "./",
    epoch: int = 0,
    score_thresh: float = 0.3,
    max_samples: int = 4,
    class_names=None,               
):
    """
    Saves original-vs-prediction side-by-side panels for up to `max_samples`
    images from the current batch into output_dir/viz/epoch_{epoch:04d}.png.

    Each instance (GT or predicted) is colored by its class id via
    get_class_color(), with a legend in the corner of each panel mapping
    color -> class. Falls back to a single flat color if class ids for a
    given side aren't available.
    """
    os.makedirs(os.path.join(output_dir, "viz"), exist_ok=True)

    was_training = model.training
    model.eval()

    B = images.shape[0]
    n = min(B, max_samples)
    panels = []

    preds = model(images[:n], score_thresh=score_thresh)  # inference-mode forward

    for i in range(n):
        img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        gt_masks = gt_bitmasks_full_batch[i].cpu().numpy() if gt_bitmasks_full_batch[i].shape[0] > 0 else np.zeros((0, *img_np.shape[:2]))
        gt_classes = (
            gt_labels_batch[i].cpu().numpy()
            if gt_labels_batch is not None and gt_labels_batch[i].shape[0] > 0
            else np.zeros((0,), dtype=np.int64)
        )
        gt_panel = _draw_masks_on_image(img_np, gt_masks, class_ids=gt_classes if gt_classes.shape[0] else None)
        if gt_classes.shape[0] > 0:
            gt_panel = _draw_class_legend(gt_panel, gt_classes, class_names)

        # Prediction side: use the model's own decoded, NMS-filtered instance
        # list (preds["polygons"]) so the viz reflects real inference output
        # (correct per-instance class + score) rather than re-deriving a
        # separate ad-hoc foreground mask here.
        pred_masks = np.zeros((0, *img_np.shape[:2]))
        pred_classes = np.zeros((0,), dtype=np.int64)

        # Re-run the mask head restricted to this image's kept instances so
        # we can render dense masks (polygons alone aren't enough for the
        # overlay). Mirrors the selection RPDCNN.forward() already did.
        cls_scores = preds["cls_logits"][i].sigmoid()
        best_scores, best_classes = cls_scores.max(dim=-1)
        keep_idx = (best_scores > score_thresh).nonzero(as_tuple=False).squeeze(1)

        if keep_idx.numel() > 0 and "mask_feats" in preds:
            controller = preds["controller"][i][keep_idx]
            locations = preds["points_flat"][keep_idx]
            fpn_levels = _level_ids_for_points(preds["feature_shapes"], keep_idx, images.device)
            im_inds_local = torch.zeros(keep_idx.numel(), dtype=torch.long, device=images.device)

            mask_logits = model.mask_head(
                mask_feats=preds["mask_feats"][i:i + 1],
                mask_feat_stride=model.mask_branch_out_stride,
                im_inds=im_inds_local,
                locations=locations,
                fpn_levels=fpn_levels,
                mask_head_params=controller,
            )
            pred_masks = (mask_logits.sigmoid() > 0.5).squeeze(1).cpu().numpy()
            pred_classes = best_classes[keep_idx].cpu().numpy()

        pred_panel = _draw_masks_on_image(img_np, pred_masks, class_ids=pred_classes if pred_classes.shape[0] else None)
        if pred_classes.shape[0] > 0:
            pred_panel = _draw_class_legend(pred_panel, pred_classes, class_names)

        combined = np.concatenate([gt_panel, pred_panel], axis=1)
        cv2.putText(combined, "GT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Pred", (img_np.shape[1] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panels.append(combined)

    grid = np.concatenate(panels, axis=0) if panels else np.zeros((10, 10, 3), dtype=np.uint8)
    out_path = os.path.join(output_dir, "viz", f"epoch_{epoch:04d}.png")
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    if was_training:
        model.train()

    return out_path


def _level_ids_for_points(feature_shapes, keep_idx, device) -> torch.Tensor:
    """
    Maps flattened point indices back to their FPN level id, given the
    per-level (H, W) shapes used to build the flattened point/feature
    tensors. Needed because the previous viz implementation hardcoded
    fpn_levels to 0 for every point, which is only correct for points
    that happen to land on the first level and silently mis-selects the
    dynamic mask head's per-level size-of-interest for every other point.
    """
    counts = [h * w for h, w in feature_shapes]
    boundaries = torch.tensor(counts, device=device).cumsum(0)
    # searchsorted with right=True: index of the first boundary strictly
    # greater than the point index gives that point's level id.
    level_ids_all = torch.searchsorted(boundaries, keep_idx, right=True)
    return level_ids_all.clamp(max=len(counts) - 1)
