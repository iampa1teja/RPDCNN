
import cv2
import math
from typing import Dict, List, Optional, Tuple
import os
import glob
import time 
from tqdm import tqdm 
import json 

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from .cfg import RPDCFG
from .feature_extractor.backbone import Backbone
from .feature_extractor.bifpn import BiFPN
from .feature_extractor.cbam import CBAM
from .feature_extractor.fpn import FPN
from .heads.dynamic_det_head import DynamicDetHead
from .heads.dynamic_mask_head import DynamicMaskHead
from .heads.mask_branch import MaskBranch
from .utils.fcos_assigner import FCOSAssigner
from .utils.losses import IOULoss, compute_fcos_losses
from .utils.viz_utils import save_epoch_visualization
from .utils.yolo_poly_dataset import YoloPolyDataset
from .utils.yolo_poly_dataset import yolo_poly_collate_fn
from .utils.viz_utils import get_class_color, _draw_class_legend






def _level_name(stride: int) -> str:
    return f"p{int(math.log2(stride))}"


class RPDCNN(nn.Module):
    """
        RPDCNN — top-level multi-task perception model.

        Architecture (mirrors AdelaiDet's CondInst wiring):

            image
            -> Backbone (timm, features_only)
            -> Neck (BiFPN or torchvision FPN)          -> list[P_i], i = 0..L-1
            -> CBAM (optional, applied per-level on neck output)
            -> DynamicDetHead (per level)                -> bbox, cls, centerness, controller
            -> MaskBranch (fused subset of neck levels)  -> mask_feats (shared, stride = mask_branch_out_stride)
            -> (training) FCOSAssigner assigns GT to point-locations
            -> positive locations are gathered across all levels/images
            -> DynamicMaskHead consumes gathered controller params + mask_feats
                -> per-instance mask logits (dense, H x W)

        Level naming: neck outputs are named f"p{log2(stride)}" so mask_branch_in_features
        in CFG (e.g. ["p3","p4","p5"]) lines up with strides [8,16,32].

        Label format (input): RPDCNN.train_model() consumes YOLO-segmentation-style
        polygon labels by default (format="yolo_poly"). Following AdelaiDet's
        CondInst convention (adet.modeling.condinst.condinst.CondInst.add_bitmasks),
        polygons are rasterized to bitmasks ONCE per image at full input resolution
        (gt_bitmasks_full). Stride-specific targets are then derived by strided
        SUBSAMPLING of gt_bitmasks_full -- e.g. bitmasks_full[:, start::stride,
        start::stride] -- not a separate resize/interpolate call. This mirrors
        AdelaiDet exactly. fcos_assigner.py, dynamic_mask_head.py, and mask_branch.py
        are unmodified: they only ever see dense bitmask tensors, precisely as they
        did before polygon support was added.

        Output format (inference): DynamicMaskHead's raw output is unavoidably a
        dense per-pixel mask -- conditional convolutions applied over a spatial
        feature grid cannot natively emit a vertex sequence, this is a structural
        property of the operation, not a missing feature. To still hand back YOLO
        polygons from RPDCNN's public inference interface, RPDCNN converts each
        predicted dense mask to a polygon (cv2.findContours + approxPolyDP) as a
        post-processing step in the inference branch of forward(), via
        _mask_to_yolo_polygon(). Training never touches this path -- loss
        supervision remains fully mask-based (dice/BCE on dense grids), which is
        the only thing dynamic_mask_head.py can be supervised with. Mask quality
        itself is unaffected; only the final polygon representation carries the
        usual, unavoidable rasterize-then-trace simplification cost.
    """
    def __init__(self, cfg: RPDCFG):
        super().__init__()
        self.cfg = cfg
        self.in_strides = list(cfg.in_strides)
        self.num_classes = cfg.num_classes
        self.mask_out_stride = cfg.mask_head_out_stride
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w

        # ---------------- Backbone ----------------
        self.backbone = Backbone(
            name=cfg.backbone_name,
            pretrained=cfg.backbone_pretrained,
            in_strides=self.in_strides,
        )
        backbone_channels = self.backbone.out_channels

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if cfg.freeze_backbone_bn:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        # ---------------- Neck ----------------
        if cfg.neck_type == "bifpn":
            self.neck = BiFPN(
                in_channels=backbone_channels,
                out_ch=cfg.bifpn_out_channels,
                num_layers=cfg.bifpn_num_layers,
            )
            neck_out_channels = cfg.bifpn_out_channels
            self.neck_strides = self.in_strides
        elif cfg.neck_type == "fpn":
            self.neck = FPN(
                in_channels_list=backbone_channels,
                out_channels=cfg.bifpn_out_channels,
            )
            neck_out_channels = cfg.bifpn_out_channels
            self.neck_strides = self.in_strides + [
                self.in_strides[-1] * 2,
                self.in_strides[-1] * 4,
            ]
        else:
            raise ValueError(f"Unknown neck_type '{cfg.neck_type}'")

        self.neck_out_channels = neck_out_channels
        self.neck_level_names = [_level_name(s) for s in self.neck_strides]

        # ---------------- CBAM ----------------
        self.use_cbam = cfg.use_cbam
        if self.use_cbam:
            self.cbam_blocks = nn.ModuleList([
                CBAM(neck_out_channels, reduction=cfg.cbam_reduction)
                for _ in self.neck_strides
            ])

        # ---------------- Mask branch ----------------
        in_channels_dict = {name: neck_out_channels for name in self.neck_level_names}
        self.mask_branch = MaskBranch(
            in_channels_dict=in_channels_dict,
            in_features=cfg.mask_branch_in_features,
            channels=cfg.mask_branch_channels,
            out_channels=cfg.mask_branch_out_channels,
            num_convs=cfg.mask_branch_num_convs,
            norm=cfg.mask_branch_norm,
            sem_loss_on=cfg.mask_branch_sem_loss_on,
            num_classes=cfg.num_classes,
            out_stride=cfg.mask_branch_out_stride,
        )
        self.mask_branch_out_stride = cfg.mask_branch_out_stride

        # ---------------- Dynamic mask head ----------------
        self.mask_head = DynamicMaskHead(
            in_channels=cfg.mask_branch_out_channels,
            channels=cfg.mask_head_channels,
            num_layers=cfg.mask_head_num_layers,
            mask_out_stride=cfg.mask_head_out_stride,
            disable_rel_coords=cfg.mask_head_disable_rel_coords,
            sizes_of_interest=cfg.mask_head_sizes_of_interest,
        )
        controller_dim = self.mask_head.num_gen_params

        # ---------------- Detection head ----------------
        self.det_head = DynamicDetHead(
            in_ch=neck_out_channels,
            num_classes=cfg.num_classes,
            controller_dim=controller_dim,
            in_strides=self.neck_strides,
            num_convs=cfg.det_num_convs,
            norm_groups=cfg.det_norm_groups,
        )

        # ---------------- Assigner + losses ----------------
        regress_ranges = cfg.assigner_regress_ranges
        if len(regress_ranges) != len(self.neck_strides):
            raise ValueError(
                f"assigner_regress_ranges has {len(regress_ranges)} entries but "
                f"neck produces {len(self.neck_strides)} levels {self.neck_strides}. "
                f"Update CFG.assigner_regress_ranges to match."
            )
        self.assigner = FCOSAssigner(
            in_strides=self.neck_strides,
            regress_ranges=regress_ranges,
            center_sampling_radius=cfg.assigner_center_sampling_radius,
            num_classes=cfg.num_classes,
            mask_branch_out_stride=cfg.mask_branch_out_stride,
        )
        self.box_loss_func = IOULoss(loss_type=cfg.loss_iou_type)
        self.focal_loss_alpha = cfg.loss_focal_alpha
        self.focal_loss_gamma = cfg.loss_focal_gamma

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _extract_neck_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        backbone_feats = self.backbone(images)
        neck_feats = self.neck(backbone_feats)
        if self.use_cbam:
            neck_feats = [cbam(feat) for cbam, feat in zip(self.cbam_blocks, neck_feats)]
        return neck_feats

    def _flatten_det_outputs(
        self, det_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        bbox_list, cls_list, ctr_list, ctrl_list = [], [], [], []
        feature_shapes = []
        for bbox_preds, cls_logits, centerness_logits, controller in det_outputs:
            B, _, H, W = bbox_preds.shape
            feature_shapes.append((H, W))
            bbox_list.append(bbox_preds.permute(0, 2, 3, 1).reshape(B, H * W, 4))
            cls_list.append(cls_logits.permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes))
            ctr_list.append(centerness_logits.permute(0, 2, 3, 1).reshape(B, H * W, 1))
            ctrl_list.append(controller.permute(0, 2, 3, 1).reshape(B, H * W, -1))

        bbox_flat = torch.cat(bbox_list, dim=1)
        cls_flat = torch.cat(cls_list, dim=1)
        ctr_flat = torch.cat(ctr_list, dim=1)
        ctrl_flat = torch.cat(ctrl_list, dim=1)
        return bbox_flat, cls_flat, ctr_flat, ctrl_flat, feature_shapes

    def _build_mask_branch_input(self, neck_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: feat for name, feat in zip(self.neck_level_names, neck_feats)}

    def _gather_positive_instances(
        self,
        matched_gt_inds: torch.Tensor,
        controller_flat: torch.Tensor,
        points_flat: torch.Tensor,
        level_ids_flat: torch.Tensor,
    ):
        B, N = matched_gt_inds.shape
        pos_mask = matched_gt_inds >= 0

        im_inds, gt_inds, locations, fpn_levels, mask_head_params = [], [], [], [], []
        for b in range(B):
            idx = pos_mask[b].nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            im_inds.append(torch.full((idx.numel(),), b, dtype=torch.long, device=matched_gt_inds.device))
            gt_inds.append(matched_gt_inds[b, idx])
            locations.append(points_flat[idx])
            fpn_levels.append(level_ids_flat[idx])
            mask_head_params.append(controller_flat[b, idx])

        if len(im_inds) == 0:
            device = matched_gt_inds.device
            controller_dim = controller_flat.shape[-1]
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0, 2), dtype=torch.float32, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0, controller_dim), dtype=torch.float32, device=device),
            )

        return (
            torch.cat(im_inds, dim=0),
            torch.cat(gt_inds, dim=0),
            torch.cat(locations, dim=0),
            torch.cat(fpn_levels, dim=0),
            torch.cat(mask_head_params, dim=0),
        )

    # ------------------------------------------------------------------
    # Polygon -> bitmask target construction (INPUT side, training only)
    # Mirrors adet.modeling.condinst.condinst.CondInst.add_bitmasks
    # ------------------------------------------------------------------

    @staticmethod
    def _polygon_to_bitmask(polygon_rings: List[np.ndarray], h: int, w: int) -> np.ndarray:
        """
        Rasterizes one instance's polygon ring(s) into a single (h, w) uint8
        mask, functionally equivalent to detectron2's polygons_to_bitmask.
        polygon_rings: list of (K, 2) arrays in ABSOLUTE pixel coords (x, y).
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        for ring in polygon_rings:
            pts = ring.round().astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def _add_bitmasks(
        self,
        yolo_labels_batch: List[np.ndarray],
        yolo_polygons_batch: List[List[List[np.ndarray]]],
        im_h: int,
        im_w: int,
        device: torch.device,
    ):
        """
        AdelaiDet equivalent: CondInst.add_bitmasks.

        Rasterizes each instance's polygon ONCE at full input resolution
        (im_h, im_w) -> gt_bitmasks_full. Stride-specific targets are then
        obtained purely by strided subsampling of gt_bitmasks_full (see
        _subsample_bitmasks) -- NOT a resize/interpolate call, matching
        AdelaiDet exactly.

        gt_boxes are derived as the tight pixel bbox of each rasterized
        instance directly.

        Degenerate polygons (rasterize to zero pixels at this resolution)
        are dropped -- keeping them would otherwise produce a zero-area box
        that the assigner could match as a phantom object.

        Returns:
            gt_boxes_batch: list[len B] of (M_i, 4) float32 tensors [x1,y1,x2,y2]
            gt_labels_batch: list[len B] of (M_i,) long tensors
            gt_bitmasks_full_batch: list[len B] of (M_i, im_h, im_w) float32 tensors
        """
        gt_boxes_batch, gt_labels_batch, gt_bitmasks_full_batch = [], [], []

        for labels, polys in zip(yolo_labels_batch, yolo_polygons_batch):
            boxes, kept_labels, bitmasks_full = [], [], []

            for inst_idx, rings in enumerate(polys):
                # rings are normalized [0,1] (x, y); clip then scale to pixels
                abs_rings = []
                for ring in rings:
                    r = np.clip(ring, 0.0, 1.0).astype(np.float32).copy()
                    r[:, 0] *= im_w
                    r[:, 1] *= im_h
                    abs_rings.append(r)

                mask = self._polygon_to_bitmask(abs_rings, im_h, im_w)
                if mask.sum() == 0:
                    continue  # degenerate polygon at this resolution -- drop

                ys, xs = np.nonzero(mask)
                x1, x2 = float(xs.min()), float(xs.max() + 1)
                y1, y2 = float(ys.min()), float(ys.max() + 1)

                boxes.append([x1, y1, x2, y2])
                kept_labels.append(int(labels[inst_idx]))
                bitmasks_full.append(mask)

            if len(bitmasks_full) == 0:
                gt_boxes_batch.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                gt_labels_batch.append(torch.zeros((0,), dtype=torch.long, device=device))
                gt_bitmasks_full_batch.append(torch.zeros((0, im_h, im_w), dtype=torch.float32, device=device))
                continue

            bitmasks_full_np = np.stack(bitmasks_full, axis=0)  # (M, im_h, im_w)
            bitmasks_full_t = torch.from_numpy(bitmasks_full_np).to(device=device, dtype=torch.float32)

            gt_boxes_batch.append(torch.tensor(boxes, dtype=torch.float32, device=device))
            gt_labels_batch.append(torch.tensor(kept_labels, dtype=torch.long, device=device))
            gt_bitmasks_full_batch.append(bitmasks_full_t)

        return gt_boxes_batch, gt_labels_batch, gt_bitmasks_full_batch

    @staticmethod
    def _subsample_bitmasks(gt_bitmasks_full_batch: List[torch.Tensor], stride: int) -> List[torch.Tensor]:
        """
        AdelaiDet convention: bitmasks_full[:, start::stride, start::stride]
        where start = stride // 2. Pure strided indexing, no interpolation.
        """
        start = stride // 2
        return [bm_full[:, start::stride, start::stride] for bm_full in gt_bitmasks_full_batch]

    # ------------------------------------------------------------------
    # Mask -> YOLO polygon conversion (OUTPUT side, inference only)
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_to_yolo_polygon(
        mask: np.ndarray,
        img_h: int,
        img_w: int,
        approx_epsilon_frac: float = 0.005,
    ) -> Optional[np.ndarray]:
        """
        Converts a single binary (H, W) mask into a normalized YOLO-format
        polygon: an (K, 2) array of (x, y) in [0, 1], taken from the largest
        external contour.

        approx_epsilon_frac: cv2.approxPolyDP epsilon as a fraction of the
            contour perimeter. Higher -> fewer vertices, coarser polygon.
            Set to 0 to disable simplification and keep the raw contour.

        Returns None if the mask has no foreground pixels (nothing to trace).
        """
        mask_u8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        contour = max(contours, key=cv2.contourArea)
        if approx_epsilon_frac > 0:
            epsilon = approx_epsilon_frac * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)

        pts = contour.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 3:
            return None  # degenerate polygon, not usable

        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        return pts

    def _decode_instances_to_polygons(
        self,
        mask_logits: torch.Tensor,      # (N, 1, H, W) dense predicted masks
        pred_classes: torch.Tensor,     # (N,) class id per instance
        pred_scores: torch.Tensor,      # (N,) confidence per instance
        im_inds: torch.Tensor,          # (N,) which image each instance belongs to
        img_h: int,
        img_w: int,
        mask_threshold: float = 0.5,
        approx_epsilon_frac: float = 0.005,
    ) -> List[List[Dict]]:
        """
        Converts a batch of dense instance mask predictions into per-image
        lists of YOLO-format polygon detections. This is the only place a
        dense mask is ever turned into a polygon -- purely a post-processing
        step over dynamic_mask_head's unmodified output.

        Returns:
            list[len B] of list[dict(class_id, score, polygon)] where polygon
            is an (K, 2) float32 array of normalized (x, y) coords, or an
            empty list per image if there are no instances / no valid contours.
        """
        num_images = int(im_inds.max().item()) + 1 if im_inds.numel() > 0 else 0
        results: List[List[Dict]] = [[] for _ in range(num_images)]

        if mask_logits.shape[0] == 0:
            return results

        masks_np = (mask_logits.sigmoid() > mask_threshold).squeeze(1).detach().cpu().numpy()
        classes_np = pred_classes.detach().cpu().numpy()
        scores_np = pred_scores.detach().cpu().numpy()
        im_inds_np = im_inds.detach().cpu().numpy()

        for i in range(masks_np.shape[0]):
            polygon = self._mask_to_yolo_polygon(
                masks_np[i], img_h, img_w, approx_epsilon_frac=approx_epsilon_frac
            )
            if polygon is None:
                continue
            results[im_inds_np[i]].append({
                "class_id": int(classes_np[i]),
                "score": float(scores_np[i]),
                "polygon": polygon,
            })

        return results

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        yolo_labels_batch: Optional[List] = None,
        yolo_polygons_batch: Optional[List] = None,
        score_thresh: float = 0.3,
        nms_iou_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        approx_epsilon_frac: float = 0.005,
    ):
        """
        Args:
            images: (B, 3, H, W), already resized to (cfg.img_h, cfg.img_w).
            yolo_labels_batch: training-only. list[len B] of (M_i,) class-id arrays.
            yolo_polygons_batch: training-only. list[len B] of per-image polygon
                lists: each instance is a list of rings, each ring a (K, 2)
                array of normalized [0,1] (x, y) coords (YOLO-seg format).
            score_thresh / nms_iou_threshold / mask_threshold / approx_epsilon_frac:
                inference-only, control candidate selection, per-class NMS,
                and mask->polygon conversion.

        Returns:
            Training: dict of losses. Mask supervision is fully dense/mask-based
                (dynamic_mask_head.py and mask_branch.py are unmodified).
            Inference: dict with:
                "polygons": list[len B] of list[dict(class_id, score, polygon)]
                    -- normalized YOLO-format polygons, model's public output.
                Plus raw per-level tensors ("bbox_preds", "cls_logits", etc.)
                for callers that want to build a custom decode/NMS pipeline
                instead of the built-in one.
        """
        gt_boxes_batch = gt_labels_batch = gt_bitmasks_full_batch = None
        gt_bitmasks_assigner_batch = gt_bitmasks_maskloss_batch = None
        if self.training:
            if yolo_labels_batch is None or yolo_polygons_batch is None:
                raise ValueError("Training forward requires yolo_labels_batch and yolo_polygons_batch.")

            img_h, img_w = images.shape[-2:]

            # Rasterize once at full resolution (AdelaiDet: add_bitmasks).
            gt_boxes_batch, gt_labels_batch, gt_bitmasks_full_batch = self._add_bitmasks(
                yolo_labels_batch, yolo_polygons_batch, img_h, img_w, images.device
            )

            # Strided-subsample gt_bitmasks_full at each stride needed
            # downstream -- no separate rasterization or interpolation.
            # mask_branch_out_stride: used by the assigner's center-sampling.
            # mask_out_stride: used by the mask head's dice loss target.
            gt_bitmasks_assigner_batch = self._subsample_bitmasks(
                gt_bitmasks_full_batch, self.mask_branch_out_stride
            )
            gt_bitmasks_maskloss_batch = self._subsample_bitmasks(
                gt_bitmasks_full_batch, self.mask_out_stride
            )

        neck_feats = self._extract_neck_features(images)
        det_outputs = self.det_head(neck_feats)
        bbox_flat, cls_flat, ctr_flat, ctrl_flat, feature_shapes = self._flatten_det_outputs(det_outputs)

        mask_branch_input = self._build_mask_branch_input(neck_feats)
        mask_feats, mask_branch_losses = self.mask_branch(
            mask_branch_input,
            gt_bitmasks_full_list=gt_bitmasks_full_batch,
            gt_classes_list=gt_labels_batch,
        )

        points_all_levels = self.assigner.get_points(feature_shapes, images.device)
        level_ids_flat = torch.cat([
            torch.full((p.shape[0],), lvl, dtype=torch.long, device=images.device)
            for lvl, p in enumerate(points_all_levels)
        ], dim=0)
        points_flat = torch.cat(points_all_levels, dim=0)

        if self.training:
            target_labels, target_boxes, _, matched_gt_inds = self.assigner(
                gt_boxes_batch, gt_labels_batch, feature_shapes, gt_bitmasks_assigner_batch
            )

            target_labels_flat = target_labels.reshape(-1)
            target_boxes_flat = target_boxes.reshape(-1, 4)
            cls_flat_r = cls_flat.reshape(-1, self.num_classes)
            ctr_flat_r = ctr_flat.reshape(-1, 1)
            bbox_flat_r = bbox_flat.reshape(-1, 4)

            det_losses = compute_fcos_losses(
                pred_logits=cls_flat_r,
                pred_centerness=ctr_flat_r,
                pred_boxes=bbox_flat_r,
                target_labels=target_labels_flat,
                target_boxes=target_boxes_flat,
                box_loss_func=self.box_loss_func,
                focal_loss_alpha=self.focal_loss_alpha,
                focal_loss_gamma=self.focal_loss_gamma,
            )

            im_inds, gt_inds, locations, fpn_levels, mask_head_params = self._gather_positive_instances(
                matched_gt_inds, ctrl_flat, points_flat, level_ids_flat
            )

            mask_logits = self.mask_head(
                mask_feats=mask_feats,
                mask_feat_stride=self.mask_branch_out_stride,
                im_inds=im_inds,
                locations=locations,
                fpn_levels=fpn_levels,
                mask_head_params=mask_head_params,
            )

            if im_inds.numel() > 0:
                gt_bitmasks_pos = torch.cat([
                    gt_bitmasks_maskloss_batch[b.item()][g.item()].unsqueeze(0)
                    for b, g in zip(im_inds, gt_inds)
                ], dim=0).unsqueeze(1)
                loss_mask = self.mask_head.compute_loss(
                    mask_logits, gt_bitmasks_pos, mask_feats, mask_head_params
                )
            else:
                loss_mask = self.mask_head.compute_loss(
                    mask_logits, torch.zeros_like(mask_logits), mask_feats, mask_head_params
                )

            losses = dict(det_losses)
            losses["loss_mask"] = loss_mask
            losses.update(mask_branch_losses)
            return losses

        else:
            # --- Inference: candidate selection, dense mask decode, then
            # convert each predicted mask to a YOLO-format polygon. ---
            B = images.shape[0]
            img_h, img_w = images.shape[-2:]

            cls_scores = cls_flat.sigmoid()                     # (B, N, num_classes)
            best_scores, best_classes = cls_scores.max(dim=-1)  # (B, N), (B, N)
            keep_mask = best_scores > score_thresh              # (B, N)

            # Decode (l, t, r, b) distances + point locations into absolute
            # (x1, y1, x2, y2) boxes for NMS. bbox_flat is now stride-scaled
            # pixel distances (see DynamicDetHead.forward_single).
            x1y1 = points_flat.unsqueeze(0) - bbox_flat[..., :2]   # (B, N, 2)
            x2y2 = points_flat.unsqueeze(0) + bbox_flat[..., 2:]   # (B, N, 2)
            decoded_boxes = torch.cat([x1y1, x2y2], dim=-1)        # (B, N, 4)

            im_inds, locations, fpn_levels, mask_head_params = [], [], [], []
            pred_classes, pred_scores = [], []
            for b in range(B):
                idx = keep_mask[b].nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue

                # Per-class NMS: batched_nms uses `best_classes` as the
                # per-box "category" so boxes of different classes never
                # suppress each other.
                keep_nms = torchvision.ops.batched_nms(
                    decoded_boxes[b, idx],
                    best_scores[b, idx],
                    best_classes[b, idx],
                    iou_threshold=nms_iou_threshold,
                )
                idx = idx[keep_nms]
                if idx.numel() == 0:
                    continue

                im_inds.append(torch.full((idx.numel(),), b, dtype=torch.long, device=images.device))
                locations.append(points_flat[idx])
                fpn_levels.append(level_ids_flat[idx])
                mask_head_params.append(ctrl_flat[b, idx])
                pred_classes.append(best_classes[b, idx])
                pred_scores.append(best_scores[b, idx])

            if len(im_inds) > 0:
                im_inds = torch.cat(im_inds, dim=0)
                locations = torch.cat(locations, dim=0)
                fpn_levels = torch.cat(fpn_levels, dim=0)
                mask_head_params = torch.cat(mask_head_params, dim=0)
                pred_classes = torch.cat(pred_classes, dim=0)
                pred_scores = torch.cat(pred_scores, dim=0)

                mask_logits = self.mask_head(
                    mask_feats=mask_feats,
                    mask_feat_stride=self.mask_branch_out_stride,
                    im_inds=im_inds,
                    locations=locations,
                    fpn_levels=fpn_levels,
                    mask_head_params=mask_head_params,
                )
                polygons_batch = self._decode_instances_to_polygons(
                    mask_logits, pred_classes, pred_scores, im_inds,
                    img_h, img_w,
                    mask_threshold=mask_threshold,
                    approx_epsilon_frac=approx_epsilon_frac,
                )
            else:
                polygons_batch = [[] for _ in range(B)]

            return {
                "polygons": polygons_batch,   # model's native public output
                "bbox_preds": bbox_flat,
                "cls_logits": cls_flat,
                "centerness_logits": ctr_flat,
                "controller": ctrl_flat,
                "mask_feats": mask_feats,
                "points_flat": points_flat,
                "feature_shapes": feature_shapes,
            }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, output_dir: str, epoch: int, optimizer: torch.optim.Optimizer):
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        ckpt_path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch:04d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else self.cfg,
        }, ckpt_path)
        return ckpt_path

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return None
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        return ckpts[-1] if ckpts else None

    def _load_checkpoint(self, ckpt_path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("cfg")
        if isinstance(cfg, dict):
            ckpt["cfg"] = RPDCFG.from_dict(cfg)
        elif cfg is not None and not isinstance(cfg, RPDCFG):
            ckpt["cfg"] = cfg
        self.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt.get("epoch", 0)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train_model(
        self,
        img_dir: str,
        label_dir: str,
        format: str = "yolo_poly",
        output_dir: str = "./rpdcnn_output",
        num_epochs: int = 50,
        batch_size: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_workers: int = 4,
        save_freq: int = 5,
        class_names: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Trains RPDCNN end-to-end.

        format: only "yolo_poly" is currently supported. Kept as an explicit
            arg (rather than hardcoded) so other label formats can be added
            later without changing the call signature.
        class_names: optional list of display names indexed by class id,
            used only for the legend text in the per-epoch viz panel
            (e.g. ["beaker", "test_tube", "other"]). Falls back to
            "class {id}" when omitted.

        Every epoch:
            - runs one pass over the dataset, backprop on total loss
            - saves a GT-vs-prediction visualization panel to output_dir/viz/
        Every `save_freq` epochs (and at the final epoch):
            - saves a checkpoint to output_dir/checkpoints/

        Resume: controlled by self.cfg.resume_training. If True, loads
            self.cfg.resume_checkpoint_path if set, otherwise the latest
            checkpoint found under output_dir/checkpoints/.
        """
        if format != "yolo_poly":
            raise NotImplementedError(
                f"format='{format}' not supported yet. Only 'yolo_poly' is implemented."
            )

        os.makedirs(output_dir, exist_ok=True)
        self.to(device)

        dataset = YoloPolyDataset(img_dir, label_dir, img_h=self.img_h, img_w=self.img_w)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=yolo_poly_collate_fn, drop_last=True,
        )

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        start_epoch = 0
        if self.cfg.resume_training:
            ckpt_path = self.cfg.resume_checkpoint_path or self._find_latest_checkpoint(output_dir)
            if ckpt_path and os.path.exists(ckpt_path):
                start_epoch = self._load_checkpoint(ckpt_path, optimizer)
                print(f"[RPDCNN] Resumed from {ckpt_path} at epoch {start_epoch}")
            else:
                print("[RPDCNN] resume_training=True but no checkpoint found — starting fresh.")

        self.train()
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            running_losses = {}
            num_batches = 0

            last_batch_images = None
            last_batch_gt_bitmasks_full = None

            for batch in tqdm(loader):
                images = batch["images"].to(device)
                labels_batch = batch["labels_batch"]
                polygons_batch = batch["polygons_batch"]

                optimizer.zero_grad()
                losses = self.forward(
                    images,
                    yolo_labels_batch=labels_batch,
                    yolo_polygons_batch=polygons_batch,
                )
                total_loss = sum(losses.values())
                total_loss.backward()
                optimizer.step()

                for k, v in losses.items():
                    running_losses[k] = running_losses.get(k, 0.0) + v.item()
                running_losses["total"] = running_losses.get("total", 0.0) + total_loss.item()
                num_batches += 1

                last_batch_images = images
                # Reuse the same rasterizer as forward() for the viz panel,
                # rather than threading state out of forward().
                _, last_batch_gt_labels, last_batch_gt_bitmasks_full = self._add_bitmasks(
                    labels_batch, polygons_batch, images.shape[-2], images.shape[-1], device
                )

            avg_losses = {k: v / num_batches for k, v in running_losses.items()}
            elapsed = time.time() - epoch_start
            loss_str = " ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
            print(f"[RPDCNN] epoch {epoch + 1}/{num_epochs} ({elapsed:.1f}s) {loss_str}")

            if last_batch_images is not None:
                viz_path = save_epoch_visualization(
                    self, last_batch_images, last_batch_gt_bitmasks_full,
                    gt_labels_batch=last_batch_gt_labels,
                    output_dir=output_dir, epoch=epoch + 1,
                    class_names=class_names,
                )
                print(f"[RPDCNN] saved viz -> {viz_path}")

            if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
                ckpt_path = self._save_checkpoint(output_dir, epoch + 1, optimizer)
                print(f"[RPDCNN] saved checkpoint -> {ckpt_path}")

        print("[RPDCNN] training complete.")

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


    def _load_and_preprocess(self, img_path, img_h, img_w):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor, orig_h, orig_w


    def _draw_polygons_on_image(self, image, instances, alpha=0.45):
        """
        image: (H, W, 3) uint8 RGB at ORIGINAL resolution.
        instances: list of dict(class_id, score, polygon) with polygon (K,2)
            normalized [0,1] (x, y).
        """
        out = image.copy()
        h, w = image.shape[:2]
        class_ids = [inst["class_id"] for inst in instances]

        for inst in instances:
            pts = inst["polygon"].copy()
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts_i = pts.round().astype(np.int32).reshape(-1, 1, 2)

            color = get_class_color(inst["class_id"])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_i], 1)
            overlay = out.copy()
            overlay[mask.astype(bool)] = color
            out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
            cv2.polylines(out, [pts_i], isClosed=True, color=color, thickness=2)

            label = f"{inst['class_id']}:{inst['score']:.2f}"
            x0, y0 = pts_i[:, 0, 0].min(), pts_i[:, 0, 1].min()
            cv2.putText(out, label, (int(x0), max(int(y0) - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if class_ids:
            out = _draw_class_legend(out, class_ids)
        return out


    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        score_thresh: float = 0.3,
        nms_iou_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        approx_epsilon_frac: float = 0.005,
        device=None,
    ):
        """
        Runs RPDCNN on a single image.

        Returns:
            result: JSON-serializable dict —
                {"image_path", "width", "height",
                "instances": [{"class_id", "score", "polygon": [[x,y],...]}]}
                polygon coords are ABSOLUTE pixels at the image's ORIGINAL resolution.
            viz: (H, W, 3) uint8 RGB numpy array (original resolution) with
                overlaid masks/outlines/labels.
        """
        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()

        tensor, orig_h, orig_w = self._load_and_preprocess(image_path, self.img_h, self.img_w)
        images = tensor.unsqueeze(0).to(device)

        out = self(
            images,
            score_thresh=score_thresh,
            nms_iou_threshold=nms_iou_threshold,
            mask_threshold=mask_threshold,
            approx_epsilon_frac=approx_epsilon_frac,
        )
        preds = out["polygons"][0] if len(out["polygons"]) > 0 else []

        instances, norm_instances = [], []
        for p in preds:
            poly_abs = p["polygon"].copy()
            poly_abs[:, 0] *= orig_w
            poly_abs[:, 1] *= orig_h
            instances.append({
                "class_id": int(p["class_id"]),
                "score": float(p["score"]),
                "polygon": poly_abs.round(2).tolist(),
            })
            norm_instances.append({"class_id": int(p["class_id"]), "score": float(p["score"]), "polygon": p["polygon"]})

        result = {"image_path": image_path, "width": orig_w, "height": orig_h, "instances": instances}

        orig_img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        viz = self._draw_polygons_on_image(orig_img, norm_instances)

        if was_training:
            self.train()
        return result, viz


    def infer(
        self,
        model,
        img: str = None,
        img_dir: str = None,
        output_dir: str = "./rpdcnn_infer_output",
        score_thresh: float = 0.3,
        nms_iou_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        approx_epsilon_frac: float = 0.005,
        device=None,
    ):
        """
        Provide EITHER `img` (single path) OR `img_dir` (folder). If both are
        given, `img` takes priority. Saves per image:
            output_dir/json/<stem>.json
            output_dir/viz/<stem>.png
        Returns: list of result dicts.
        """
        if img is None and img_dir is None:
            raise ValueError("Provide either `img` or `img_dir`.")

        if img is not None:
            image_paths = [img]
        else:
            image_paths = sorted([
                p for p in glob.glob(os.path.join(img_dir, "*"))
                if os.path.splitext(p)[1].lower() in IMG_EXTS
            ])
            if len(image_paths) == 0:
                raise FileNotFoundError(f"No images found in {img_dir}")

        json_dir = os.path.join(output_dir, "json")
        viz_dir = os.path.join(output_dir, "viz")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        all_results = []
        for path in image_paths:
            result, viz = self.predict(
                path,
                score_thresh=score_thresh,
                nms_iou_threshold=nms_iou_threshold,
                mask_threshold=mask_threshold,
                approx_epsilon_frac=approx_epsilon_frac,
                device=device,
            )
            stem = os.path.splitext(os.path.basename(path))[0]

            json_path = os.path.join(json_dir, f"{stem}.json")
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

            viz_path = os.path.join(viz_dir, f"{stem}.png")
            cv2.imwrite(viz_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

            result["json_path"], result["viz_path"] = json_path, viz_path
            all_results.append(result)
            print(f"[RPDCNN] {stem}: {len(result['instances'])} instances -> {json_path}, {viz_path}")

        print(f"[RPDCNN] done. {len(all_results)} image(s) -> {output_dir}")
        return all_results