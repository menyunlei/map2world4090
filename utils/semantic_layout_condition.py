"""Build per-latent-frame ego-aligned labels from a global semantic map.

Coordinates use image convention: route points are ``(x, y)`` in pixels and a
zero yaw points toward increasing x. Positive yaw rotates toward increasing y.
The output rows run from the configured lookahead to the configured rear range;
columns run from ego-left to ego-right.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticLayoutCondition:
    """Discrete ego crops and the poses used to produce them."""

    labels: torch.Tensor  # [B, F, crop_height, crop_width], torch.long
    route_progress: torch.Tensor  # [B, F]
    positions_xy: torch.Tensor  # [B, F, 2]
    yaw: torch.Tensor  # [B, F], radians in image coordinates


def load_projected_semantic_occupancy(
    path: str | Path,
    *,
    segment_index: int,
    pixel_frames_per_segment: int,
    num_latent_frames: int,
    target_size: Sequence[int],
    expected_channels: int = 4,
) -> dict[str, torch.Tensor]:
    """Load image-grid occupancy and align it with Wan patch tokens.

    Wan's temporal VAE layout is frame 0 followed by four-frame groups, so a
    121-frame segment uses source indices ``0, 1, 5, ..., 117``.  Spatial area
    interpolation maps the image occupancy grid (normally 44x80) to the DiT
    patch grid (normally 22x40) without converting overlapping classes to an
    argmax label.
    """

    array = np.load(Path(path), mmap_mode="r")
    if array.ndim != 4:
        raise ValueError(f"projected occupancy must have shape [T,C,H,W], got {array.shape}")
    if int(array.shape[1]) != int(expected_channels):
        raise ValueError(f"expected {expected_channels} occupancy channels, got {array.shape[1]}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("projected occupancy must use a floating dtype")
    segment_index = int(segment_index)
    pixel_frames_per_segment = int(pixel_frames_per_segment)
    num_latent_frames = int(num_latent_frames)
    if segment_index < 0 or pixel_frames_per_segment < 1 or num_latent_frames < 1:
        raise ValueError("segment_index must be non-negative and frame counts must be positive")
    local = np.concatenate(
        [np.zeros(1, dtype=np.int64), 1 + 4 * np.arange(num_latent_frames - 1, dtype=np.int64)]
    )
    if local[-1] >= pixel_frames_per_segment:
        raise ValueError(
            f"{num_latent_frames} latent frames require source index {local[-1]}, "
            f"outside a {pixel_frames_per_segment}-frame segment"
        )
    start = segment_index * max(pixel_frames_per_segment - 1, 1)
    indices = start + local
    if int(indices[-1]) >= int(array.shape[0]):
        raise ValueError(
            f"occupancy timeline has {array.shape[0]} frames but segment {segment_index} "
            f"requires source frame {indices[-1]}"
        )
    selected = torch.from_numpy(np.asarray(array[indices], dtype=np.float32).copy())
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("projected occupancy contains non-finite values")
    height, width = _validate_crop_size(target_size)
    selected = F.interpolate(selected, size=(height, width), mode="area")
    return {
        "features": selected.unsqueeze(0),
        "progress": torch.linspace(0, 1, num_latent_frames, dtype=torch.float32).unsqueeze(0),
        "source_indices": torch.from_numpy(indices),
    }


def build_ego_semantic_layout(
    global_labels: torch.Tensor,
    route_polyline: torch.Tensor,
    route_progress: torch.Tensor,
    *,
    crop_size: Sequence[int],
    lateral_extent: float,
    lookahead: float,
    rear_extent: float = 0.0,
    route_vertex_progress: torch.Tensor | None = None,
    yaw: torch.Tensor | None = None,
    outside_label: int = 0,
) -> SemanticLayoutCondition:
    """Sample global ``[H,W]`` labels into per-frame ego-aligned crops.

    Inputs may be unbatched or batched. Singleton batches are broadcast, while
    all non-singleton batch sizes must agree. Route progress must be in
    ``[0, 1]`` and is measured by polyline arc length unless normalized
    ``route_vertex_progress`` is supplied. If yaw is omitted, the local route
    tangent supplies it. Sampling is nearest-neighbor to preserve labels.
    """

    labels = torch.as_tensor(global_labels)
    route = torch.as_tensor(route_polyline)
    progress = torch.as_tensor(route_progress)
    vertex_progress = (
        None
        if route_vertex_progress is None
        else torch.as_tensor(route_vertex_progress)
    )
    if labels.ndim == 2:
        labels = labels.unsqueeze(0)
    if route.ndim == 2:
        route = route.unsqueeze(0)
    if progress.ndim == 1:
        progress = progress.unsqueeze(0)
    if vertex_progress is not None and vertex_progress.ndim == 1:
        vertex_progress = vertex_progress.unsqueeze(0)
    if labels.ndim != 3:
        raise ValueError(f"global_labels must have shape [H,W] or [B,H,W], got {tuple(labels.shape)}")
    if route.ndim != 3 or route.shape[-1] != 2:
        raise ValueError(f"route_polyline must have shape [N,2] or [B,N,2], got {tuple(route.shape)}")
    if progress.ndim != 2:
        raise ValueError(f"route_progress must have shape [F] or [B,F], got {tuple(progress.shape)}")
    if vertex_progress is not None and (
        vertex_progress.ndim != 2 or vertex_progress.shape[1] != route.shape[1]
    ):
        raise ValueError(
            "route_vertex_progress must have shape [N] or [B,N] matching route_polyline"
        )
    if route.shape[1] < 2:
        raise ValueError("route_polyline must contain at least two points")
    if labels.dtype.is_floating_point:
        raise TypeError("global_labels must use an integer or bool dtype")

    crop_height, crop_width = _validate_crop_size(crop_size)
    lateral_extent = float(lateral_extent)
    lookahead = float(lookahead)
    rear_extent = float(rear_extent)
    if lateral_extent < 0 or lookahead < 0 or rear_extent < 0:
        raise ValueError("crop extents must be non-negative")
    if lateral_extent == 0 and crop_width > 1:
        raise ValueError("lateral_extent must be positive when crop width is greater than one")
    if lookahead + rear_extent == 0 and crop_height > 1:
        raise ValueError("lookahead + rear_extent must be positive when crop height is greater than one")

    batch_sizes = [int(labels.shape[0]), int(route.shape[0]), int(progress.shape[0])]
    if vertex_progress is not None:
        batch_sizes.append(int(vertex_progress.shape[0]))
    batch = max(batch_sizes)
    labels = _broadcast_batch(labels, batch, "global_labels")
    route = _broadcast_batch(route, batch, "route_polyline").to(device=labels.device, dtype=torch.float32)
    progress = _broadcast_batch(progress, batch, "route_progress").to(device=labels.device, dtype=torch.float32)
    if not bool(torch.isfinite(route).all()) or not bool(torch.isfinite(progress).all()):
        raise ValueError("route_polyline and route_progress must contain only finite values")
    if bool(((progress < 0) | (progress > 1)).any()):
        raise ValueError("route_progress values must lie in [0, 1]")
    if vertex_progress is not None:
        vertex_progress = _broadcast_batch(
            vertex_progress,
            batch,
            "route_vertex_progress",
        ).to(device=labels.device, dtype=torch.float32)
        if not bool(torch.isfinite(vertex_progress).all()):
            raise ValueError("route_vertex_progress must contain only finite values")
        if not bool(torch.allclose(vertex_progress[:, 0], torch.zeros_like(vertex_progress[:, 0]))) or not bool(
            torch.allclose(vertex_progress[:, -1], torch.ones_like(vertex_progress[:, -1]))
        ):
            raise ValueError("route_vertex_progress must start at 0 and end at 1")
        if bool((vertex_progress[:, 1:] <= vertex_progress[:, :-1]).any()):
            raise ValueError("route_vertex_progress must be strictly increasing")

    positions, tangent_yaw = _interpolate_route(route, progress, vertex_progress)
    if yaw is None:
        frame_yaw = tangent_yaw
    else:
        frame_yaw = torch.as_tensor(yaw)
        if frame_yaw.ndim == 1:
            frame_yaw = frame_yaw.unsqueeze(0)
        if frame_yaw.ndim != 2 or int(frame_yaw.shape[1]) != int(progress.shape[1]):
            raise ValueError(f"yaw must have shape [F] or [B,F], got {tuple(frame_yaw.shape)}")
        frame_yaw = _broadcast_batch(frame_yaw, batch, "yaw").to(device=labels.device, dtype=torch.float32)
        if not bool(torch.isfinite(frame_yaw).all()):
            raise ValueError("yaw must contain only finite values")

    forward_distance = torch.linspace(
        lookahead, -rear_extent, crop_height, device=labels.device, dtype=torch.float32
    )
    lateral_distance = torch.linspace(
        -lateral_extent, lateral_extent, crop_width, device=labels.device, dtype=torch.float32
    )
    forward_grid, lateral_grid = torch.meshgrid(forward_distance, lateral_distance, indexing="ij")
    forward = torch.stack((torch.cos(frame_yaw), torch.sin(frame_yaw)), dim=-1)
    right = torch.stack((-forward[..., 1], forward[..., 0]), dim=-1)
    sample_xy = (
        positions[:, :, None, None, :]
        + forward[:, :, None, None, :] * forward_grid[None, None, :, :, None]
        + right[:, :, None, None, :] * lateral_grid[None, None, :, :, None]
    )

    map_height, map_width = int(labels.shape[-2]), int(labels.shape[-1])
    grid = sample_xy.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(map_width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(map_height - 1, 1) - 1.0
    frames = int(progress.shape[1])
    source = labels[:, None].expand(-1, frames, -1, -1).reshape(batch * frames, 1, map_height, map_width)
    sampled = F.grid_sample(
        source.to(torch.float64),
        grid.reshape(batch * frames, crop_height, crop_width, 2).to(torch.float64),
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch, frames, crop_height, crop_width)

    outside = (
        (sample_xy[..., 0] < 0)
        | (sample_xy[..., 0] > map_width - 1)
        | (sample_xy[..., 1] < 0)
        | (sample_xy[..., 1] > map_height - 1)
    )
    sampled = sampled.to(torch.long).masked_fill(outside, int(outside_label))
    return SemanticLayoutCondition(
        labels=sampled,
        route_progress=progress,
        positions_xy=positions,
        yaw=frame_yaw,
    )


def _interpolate_route(
    route: torch.Tensor,
    progress: torch.Tensor,
    vertex_progress: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    segments = route[:, 1:] - route[:, :-1]
    lengths = torch.linalg.vector_norm(segments, dim=-1)
    total = lengths.sum(dim=-1)
    if bool((total <= 0).any()):
        raise ValueError("each route polyline must have positive arc length")
    if vertex_progress is None:
        cumulative = torch.cat((torch.zeros_like(total[:, None]), lengths.cumsum(dim=-1)), dim=-1)
        target = progress * total[:, None]
    else:
        cumulative = vertex_progress
        target = progress
    segment_index = torch.searchsorted(cumulative.contiguous(), target.contiguous(), right=True) - 1
    segment_index = segment_index.clamp(0, route.shape[1] - 2)
    start_distance = cumulative.gather(1, segment_index)
    segment_length = (cumulative[:, 1:] - cumulative[:, :-1]).gather(1, segment_index)
    fraction = ((target - start_distance) / segment_length.clamp_min(torch.finfo(route.dtype).eps)).clamp(0, 1)
    gather_index = segment_index[..., None].expand(-1, -1, 2)
    start = route.gather(1, gather_index)
    delta = segments.gather(1, gather_index)
    positions = start + fraction[..., None] * delta
    frame_yaw = torch.atan2(delta[..., 1], delta[..., 0])
    return positions, frame_yaw


def _validate_crop_size(crop_size: Sequence[int]) -> tuple[int, int]:
    if len(crop_size) != 2:
        raise ValueError("crop_size must contain (height, width)")
    height, width = int(crop_size[0]), int(crop_size[1])
    if height < 1 or width < 1:
        raise ValueError("crop dimensions must be positive")
    return height, width


# ---------------------------------------------------------------------------
# Image-space semantic masks -> occupancy (the "true-layout" diagnostic path)
# ---------------------------------------------------------------------------

# Channel order MUST match experiments/mapstate/render_e0_occupancy_guides.GUIDE_CHANNELS.
_OCCUPANCY_CHANNELS = ("road", "water", "building", "vegetation")
# Map integer label IDs (experiments/geomap/scene_graph.LABEL_NAMES) to channels.
# 0=background, 1=building, 2=road, 3=greenspace, 4=water, 5=plaza
_LABEL_TO_OCCUPANCY_CHANNEL = {
    2: 0,  # road
    5: 0,  # plaza -> road (paved surface)
    4: 1,  # water
    1: 2,  # building
    3: 3,  # greenspace -> vegetation
}


def image_masks_to_occupancy(
    masks: "torch.Tensor | np.ndarray",
    *,
    target_size: Sequence[int],
    num_classes: int = 4,
) -> torch.Tensor:
    """Convert per-frame image-space semantic masks to one-hot occupancy.

    This is the missing converter for the "true-layout" diagnostic upper bound:
    it takes GROUND-TRUTH per-frame semantic masks (e.g. from a rendered scene)
    and produces the same ``[B, F, C, H, W]`` float occupancy tensor that
    ``ProjectedSemanticEncoder`` consumes, in the SAME channel order as the
    map-projected occupancy from ``render_e0_occupancy_guides`` (road, water,
    building, vegetation).

    Args:
        masks: ``[T, H, W]`` or ``[B, T, H, W]`` integer label map. Labels follow
            ``experiments/geomap/scene_graph.LABEL_NAMES``:
            0=background, 1=building, 2=road, 3=greenspace, 4=water, 5=plaza.
        target_size: ``(height, width)`` of the output occupancy grid. Use
            ``(22, 40)`` for the adapter's FiLM token grid or ``(44, 80)`` for
            the LayoutCritic.
        num_classes: number of occupancy channels (default 4).

    Returns:
        ``[1, T, C, H, W]`` float32 occupancy in [0, 1], area-interpolated to
        ``target_size`` (matching ``load_projected_semantic_occupancy``).
    """
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    masks = torch.as_tensor(masks)
    if masks.ndim == 3:
        masks = masks.unsqueeze(0)  # [1, T, H, W]
    if masks.ndim != 4:
        raise ValueError(f"masks must have shape [T,H,W] or [B,T,H,W], got {tuple(masks.shape)}")
    if masks.dtype.is_floating_point:
        raise TypeError("masks must use an integer or bool dtype")
    masks = masks.long()

    batch, frames, height, width = int(masks.shape[0]), int(masks.shape[1]), int(masks.shape[2]), int(masks.shape[3])
    out_h, out_w = _validate_crop_size(target_size)

    # one-hot encode: for each label, assign to the matching occupancy channel
    occupancy = torch.zeros(batch, frames, int(num_classes), height, width, dtype=torch.float32)
    for label_id, channel in _LABEL_TO_OCCUPANCY_CHANNEL.items():
        if channel >= num_classes:
            continue
        mask = (masks == label_id).float()  # [B, T, H, W]
        occupancy[:, :, channel] = torch.maximum(occupancy[:, :, channel], mask)

    # area-interpolate to target grid (preserves fractional coverage, like
    # load_projected_semantic_occupancy's mode="area")
    occupancy = occupancy.reshape(batch * frames, int(num_classes), height, width)
    occupancy = F.interpolate(occupancy, size=(out_h, out_w), mode="area")
    occupancy = occupancy.reshape(batch, frames, int(num_classes), out_h, out_w)
    return occupancy.clamp(0.0, 1.0)





def _broadcast_batch(value: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if int(value.shape[0]) == batch:
        return value
    if int(value.shape[0]) == 1:
        return value.expand(batch, *value.shape[1:])
    raise ValueError(f"{name} batch size {value.shape[0]} cannot broadcast to {batch}")
