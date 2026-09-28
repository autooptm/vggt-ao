# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # randomly pick which True positions to keep
    sampled_indices = np.random.choice(true_indices, size=max_trues, replace=False)  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
            - y_coords (numpy.ndarray): Array of y coordinates for all frames
            - x_coords (numpy.ndarray): Array of x coordinates for all frames
            - f_coords (numpy.ndarray): Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf


def select_confident_points(points_3d, depth_conf, points_rgb, conf_thres_value, max_points, seed=42):
    """Keep at most max_points pixels whose depth confidence clears the threshold.

    This is create_pixel_coordinate_grid + a boolean mask + randomly_limit_trues +
    three gathers, done in one pass. When the inputs are CUDA tensors it runs on the
    device and only the selected rows cross PCIe, instead of copying the whole
    (S, H, W, 3) point map and confidence map back to the host to build an 80 MB
    coordinate grid there. The pixel coordinates are arithmetic on the flat index,
    so that grid is never materialized at all.

    Args:
        points_3d: (S, H, W, 3) world points, torch tensor or numpy array
        depth_conf: (S, H, W) confidence, same kind as points_3d
        points_rgb: (S, H, W, 3) uint8 colours, same kind as points_3d
        conf_thres_value: keep pixels whose confidence is >= this
        max_points: cap on the number of points kept
        seed: seeds the subsample taken when more pixels qualify than the cap allows

    Returns:
        (points_3d, points_xyf, points_rgb) as numpy arrays, in ascending pixel order
    """
    if not isinstance(depth_conf, torch.Tensor):
        points_xyf = create_pixel_coordinate_grid(*depth_conf.shape)
        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points)
        return points_3d[conf_mask], points_xyf[conf_mask], points_rgb[conf_mask]

    num_frames, height, width = depth_conf.shape
    flat = (depth_conf >= conf_thres_value).reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if flat.numel() > max_points:
        generator = torch.Generator(device=flat.device)
        generator.manual_seed(seed)
        pick = torch.randperm(flat.numel(), device=flat.device, generator=generator)[:max_points]
        flat = torch.sort(flat[pick]).values  # ascending, like the boolean mask it replaces

    within_frame = flat % (height * width)
    points_xyf = torch.stack(
        (
            (within_frame % width).float(),
            (within_frame // width).float(),
            (flat // (height * width)).float(),
        ),
        dim=1,
    )
    return (
        points_3d.reshape(-1, 3)[flat].cpu().numpy(),
        points_xyf.cpu().numpy(),
        points_rgb.reshape(-1, 3)[flat].cpu().numpy(),
    )
