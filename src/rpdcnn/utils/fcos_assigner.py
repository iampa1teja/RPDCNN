import torch 
import torch.nn as nn 
from typing import List, Tuple

class FCOSAssigner(nn.Module):
    def __init__(
        self, 
        in_strides: List[int], 
        regress_ranges: List[Tuple[float, float]], 
        center_sampling_radius: float = 1.5, 
        num_classes: int = 80,
        mask_branch_out_stride: int = 8,
    ):
        """
        FCOS Target Assigner for a batch of images.
        
        Args:
            in_strides: List of strides for each feature map level (e.g., [8, 16, 32, 64, 128])
            regress_ranges: Distance constraint bounds per level (e.g., [(-1, 64), (64, 128), ...])
            center_sampling_radius: Radius for center sampling in terms of stride pixels
            num_classes: Total number of target categories
            mask_branch_out_stride: stride at which gt_bitmasks (passed into
                _get_sample_region) were subsampled. Needed to convert the
                bitmask's own array-index grid into absolute pixel coordinates,
                since gt_boxes/points are always in absolute pixel space.
        """
        super().__init__()
        if len(in_strides) != len(regress_ranges): 
            raise ValueError(
                f"Mismatched lengths: in_strides has {len(in_strides)} levels."
                f"but regress ranges has {len(regress_ranges)}"
            )
        self.in_strides = list(in_strides) 
        self.regress_ranges = list(regress_ranges) 

        self.center_sampling_radius = center_sampling_radius 
        self.num_classes = num_classes 
        self.mask_branch_out_stride = mask_branch_out_stride
    
    def get_points(self, feature_shapes: List[Tuple[int, int]], device: torch.device) -> List[torch.Tensor]: 
        """
        Generates the absolute (x, y) center coordinates for every location across all feature levels.
        
        Args:
            feature_shapes: List of (H, W) pairs for each feature map level
            device: Device to allocate tensors on
            
        Returns:
            List of Tensors, where each tensor has shape (H * W, 2) representing (x, y) points
        """
        points_all_levels = [] 
        for stride, (h,w) in zip(self.in_strides, feature_shapes): 
            shifts_x = torch.arange(0, w * stride, step = stride, dtype=torch.float32, device=device) + (stride / 2.0) 
            shifts_y = torch.arange(0, h * stride, step = stride, dtype=torch.float32, device=device) + (stride / 2.0) 

            grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij") 
            points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim = -1) 
            points_all_levels.append(points) 

        return points_all_levels 
    
    def compute_centerness_targets(self, box_targets: torch.Tensor) -> torch.Tensor: 
        """
        Computes centerness targets for positive locations.
        
        Args:
            box_targets: Shape (N, 4) containing (l, t, r, b) distances
            
        Returns:
            Tensor of shape (N,) containing centerness values bounded between 0 and 1
        """
        left_right = box_targets[:, [0, 2]] 
        top_bottom = box_targets[:, [1, 3]] 

        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness) 
    
    def _get_sample_region(
        self, 
        gt_boxes: torch.Tensor, 
        stride: int, 
        num_points: int, 
        points: torch.Tensor,
        gt_bitmasks: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Determines which points fall within a sub-box centered around the ground truth center.
        Implemented following AdelaiDet's center-sampling optimization logic.
        """
        radius = self.center_sampling_radius * stride 

        if gt_bitmasks is not None and gt_bitmasks.numel() > 0:
            _, h, w = gt_bitmasks.size()

            # gt_bitmasks is sampled on a (h, w) grid at `mask_branch_out_stride`
            # pixels/cell (see RPDCNN._subsample_bitmasks), NOT at this level's
            # `stride`. Its own array indices are therefore in a different
            # coordinate space than gt_boxes/points, which are both in absolute
            # image-pixel coordinates. Convert the index grid to pixel space
            # (matching the point-generation convention: index*stride + stride/2)
            # before computing the centroid, or every comparison below silently
            # operates on incompatible scales and center-sampling matches nothing.
            bitmask_stride = self.mask_branch_out_stride
            ys = torch.arange(0, h, dtype=torch.float32, device=gt_bitmasks.device) * bitmask_stride + bitmask_stride / 2.0
            xs = torch.arange(0, w, dtype=torch.float32, device=gt_bitmasks.device) * bitmask_stride + bitmask_stride / 2.0
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

            m00 = gt_bitmasks.sum(dim=[-2, -1]).clamp(min=1e-6) 
            
            gt_cx = (gt_bitmasks * grid_x).sum(dim=[-2, -1]) / m00 
            gt_cy = (gt_bitmasks * grid_y).sum(dim=[-2, -1]) / m00 
        else:
            gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0 
            gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

        center_gts = torch.zeros_like(gt_boxes) 
        center_gts[:, 0] = torch.max(gt_boxes[:, 0], gt_cx - radius)
        center_gts[:, 1] = torch.max(gt_boxes[:, 1], gt_cy - radius)
        center_gts[:, 2] = torch.min(gt_boxes[:, 2], gt_cx + radius)
        center_gts[:, 3] = torch.min(gt_boxes[:, 3], gt_cy + radius)

        l = points[:, 0].unsqueeze(0) - center_gts[:, 0].unsqueeze(1) 
        t = points[:, 1].unsqueeze(0) - center_gts[:, 1].unsqueeze(1)
        r = center_gts[:, 2].unsqueeze(1) - points[:, 0].unsqueeze(0) 
        b = center_gts[:, 3].unsqueeze(1) - points[:, 1].unsqueeze(0) 

        center_deltas = torch.stack([l, t, r, b], dim=-1) 
        return center_deltas.min(dim=-1)[0] > 0
    
    def assign_single_image(
        self, 
        points_all_levels: List[torch.Tensor], 
        gt_boxes: torch.Tensor, 
        gt_labels: torch.Tensor,
        gt_bitmasks: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs target matching and label assigning for a single image.
        
        Args:
            points_all_levels: List of (H_i * W_i, 2) tensors for all grids
            gt_boxes: Tensor of shape (M, 4) in (x1, y1, x2, y2) format
            gt_labels: Tensor of shape (M,) containing integer class IDs
            
        Returns:
            labels: Shape (Total_Points,) tracking background (-1 or num_classes) or target IDs
            box_targets: Shape (Total_Points, 4) containing (l, t, r, b) displacements
            centerness_targets: Shape (Total_Points,) targets for classification quality validation
            matched_gt_inds: Shape (Total_Points,) tracking the matched ground truth index (-1 for background)
        """
        num_points_per_level = [p.shape[0] for p in points_all_levels]
        points_flat = torch.cat(points_all_levels, dim=0)
        num_total_points = points_flat.shape[0]
        num_gts = gt_boxes.shape[0]

        if num_gts == 0:
            return (
                torch.full((num_total_points,), self.num_classes, dtype=torch.long, device=points_flat.device),
                torch.zeros((num_total_points, 4), dtype=torch.float32, device=points_flat.device),
                torch.zeros((num_total_points,), dtype=torch.float32, device=points_flat.device),
                torch.full((num_total_points,), -1, dtype=torch.long, device=points_flat.device)
            )
        
        if gt_bitmasks is not None and gt_bitmasks.shape[0] == 0:
            gt_bitmasks = None

        l = points_flat[:, 0].unsqueeze(0) - gt_boxes[:, 0].unsqueeze(1)
        t = points_flat[:, 1].unsqueeze(0) - gt_boxes[:, 1].unsqueeze(1)
        r = gt_boxes[:, 2].unsqueeze(1) - points_flat[:, 0].unsqueeze(0)
        b = gt_boxes[:, 3].unsqueeze(1) - points_flat[:, 1].unsqueeze(0)
        box_deltas = torch.stack([l, t, r, b], dim=-1) 

        is_in_boxes = box_deltas.min(dim=-1)[0] > 0 

        is_in_center_regions = []
        for stride, num_points, points in zip(self.in_strides, num_points_per_level, points_all_levels):
            is_in_center = self._get_sample_region(gt_boxes, stride, num_points, points, gt_bitmasks)
            is_in_center_regions.append(is_in_center)
        is_in_center_regions = torch.cat(is_in_center_regions, dim=1) 

        max_regress_deltas = box_deltas.max(dim=-1)[0] 
        is_in_level_ranges = []
        for range_idx, (min_val, max_val) in enumerate(self.regress_ranges):
            in_range = (max_regress_deltas >= min_val) & (max_regress_deltas <= max_val)
            start_idx = sum(num_points_per_level[:range_idx])
            end_idx = start_idx + num_points_per_level[range_idx]
            is_in_level_ranges.append(in_range[:, start_idx:end_idx])
        is_in_level_ranges = torch.cat(is_in_level_ranges, dim=1)
        is_valid_candidate = is_in_boxes & is_in_center_regions & is_in_level_ranges

        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        inf_matrix = torch.full_like(is_valid_candidate, float('inf'), dtype=torch.float32)
        candidate_areas = torch.where(is_valid_candidate, gt_areas.unsqueeze(1), inf_matrix)
        
        min_areas, min_area_indices = candidate_areas.min(dim=0)

        labels = gt_labels[min_area_indices]
        labels[min_areas == float('inf')] = self.num_classes 

        matched_gt_inds = min_area_indices.clone()
        matched_gt_inds[min_areas == float('inf')] = -1

        box_targets = box_deltas[min_area_indices, torch.arange(num_total_points, device=points_flat.device)]
        centerness_targets = torch.zeros_like(labels, dtype=torch.float32)
        foreground_mask = labels != self.num_classes
        if foreground_mask.any():
            centerness_targets[foreground_mask] = self.compute_centerness_targets(box_targets[foreground_mask])

        return labels, box_targets, centerness_targets, matched_gt_inds

    def forward(
        self, 
        gt_boxes_batch: List[torch.Tensor], 
        gt_labels_batch: List[torch.Tensor], 
        feature_shapes: List[Tuple[int, int]],
        gt_bitmasks_batch: List[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batches execution of single image assignment over multi-level feature coordinates.
        
        Args:
            gt_boxes_batch: List of length B, containing Tensors of shape (M_i, 4)
            gt_labels_batch: List of length B, containing Tensors of shape (M_i,)
            feature_shapes: List of (H, W) shapes for all output blocks
            
        Returns:
            stacked_labels: Shape (B, Total_Points)
            stacked_box_targets: Shape (B, Total_Points, 4)
            stacked_centerness_targets: Shape (B, Total_Points)
            stacked_matched_inds: Shape (B, Total_Points) tracking matched GT indices
        """
        if not gt_boxes_batch:
            raise ValueError("Input batch lists cannot be empty.")
            
        device = gt_boxes_batch[0].device
        points_all_levels = self.get_points(feature_shapes, device)
        
        batch_labels = []
        batch_box_targets = []
        batch_centerness_targets = []
        batch_matched_inds = []
        bitmasks_iter = gt_bitmasks_batch if gt_bitmasks_batch is not None else [None] * len(gt_boxes_batch)
        
        for gt_boxes, gt_labels, gt_bitmasks in zip(gt_boxes_batch, gt_labels_batch, bitmasks_iter):
            labels, box_targets, centerness_targets, matched_gt_inds = self.assign_single_image(
                points_all_levels, gt_boxes, gt_labels, gt_bitmasks
            )
            batch_labels.append(labels)
            batch_box_targets.append(box_targets)
            batch_centerness_targets.append(centerness_targets)
            batch_matched_inds.append(matched_gt_inds)
            
        return (
            torch.stack(batch_labels, dim=0),
            torch.stack(batch_box_targets, dim=0),
            torch.stack(batch_centerness_targets, dim=0),
            torch.stack(batch_matched_inds, dim=0)
        )
