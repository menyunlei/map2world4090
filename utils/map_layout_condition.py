"""Frame-aligned map layout conditions for MapAdapter/EMapCtrl.

This module builds the missing non-proxy part of map-to-layout control:

    global map + route/camera progress -> per-frame ego-aligned semantic grids

The output is deliberately passive data. It does not alter DreamX, PRoPE, VAE,
or prompt logic. A trained MapAdapter/EMapCtrl branch can consume these tensors;
with zero-initialized adapters they should be visually neutral.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, map_coordinates

from utils.map_trajectory import MapTrajectory
from utils.route_vectorizer import RoutePolyline, route_color_mask, tangent_at_progress
from wan.modules.emapctrl_branch import EMAPCTRL_CHANNELS


SEMANTIC_CHANNELS = tuple(EMAPCTRL_CHANNELS)
_CHANNEL_INDEX = {name: idx for idx, name in enumerate(SEMANTIC_CHANNELS)}


@dataclass(frozen=True)
class LayoutProjectionSettings:
    """Settings for top-down ego-aligned layout projection."""

    grid_height: int = 22
    grid_width: int = 40
    lateral_extent_px: float = 190.0
    lookahead_px: float = 360.0
    rear_px: float = 70.0
    route_width_px: int = 30
    event_radius_px: int = 34
    landmark_sigma_progress: float = 0.045


@dataclass(frozen=True)
class SemanticMap:
    """Global map semantic masks in image coordinates."""

    masks: dict[str, np.ndarray]
    image_size: tuple[int, int]
    channels: tuple[str, ...] = SEMANTIC_CHANNELS


@dataclass(frozen=True)
class FrameAlignedLayoutCondition:
    """Dense and sparse map conditions for one continuous trajectory."""

    features: np.ndarray
    channels: tuple[str, ...]
    frame_indices: np.ndarray
    frame_progress: np.ndarray
    landmark_progress: np.ndarray
    map_adapter_condition: dict[str, Any]
    settings: LayoutProjectionSettings

    def emapctrl_condition(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "channels": list(self.channels),
            "frame_indices": self.frame_indices,
            "frame_progress": self.frame_progress,
            "source": "frame_aligned_map_projection",
        }


def draw_polyline_mask(
    image_shape: tuple[int, int],
    points_xy: Sequence[Sequence[float]] | np.ndarray,
    *,
    width: int,
) -> np.ndarray:
    """Rasterize an image-space polyline into a binary mask."""

    height, width_px = int(image_shape[0]), int(image_shape[1])
    pts = [(float(x), float(y)) for x, y in np.asarray(points_xy, dtype=np.float64)]
    image = Image.new("L", (width_px, height), 0)
    draw = ImageDraw.Draw(image)
    if len(pts) >= 2:
        draw.line(pts, fill=255, width=max(1, int(width)), joint="curve")
    elif len(pts) == 1:
        x, y = pts[0]
        r = max(1, int(width) // 2)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def draw_event_disk_mask(
    image_shape: tuple[int, int],
    events: Sequence[Any],
    event_types: set[str],
    *,
    radius_px: int,
) -> np.ndarray:
    """Rasterize projected route events into a binary mask."""

    height, width_px = int(image_shape[0]), int(image_shape[1])
    image = Image.new("L", (width_px, height), 0)
    draw = ImageDraw.Draw(image)
    radius = max(1, int(radius_px))
    for event in events:
        event_type = _event_type(event)
        if event_type not in event_types:
            continue
        xy = _event_xy(event)
        if xy is None:
            continue
        x, y = xy
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def build_town_semantic_map(
    map_path: str | Path,
    route: RoutePolyline,
    events: Sequence[Any],
    *,
    settings: LayoutProjectionSettings | None = None,
) -> SemanticMap:
    """Build rough but stable semantic masks from the illustrated town map."""

    settings = settings or LayoutProjectionSettings()
    rgb = np.asarray(Image.open(map_path).convert("RGB"), dtype=np.float32)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    sat = maxc - minc

    try:
        route_pixels = route_color_mask(map_path).astype(bool)
    except Exception:
        route_pixels = np.zeros((height, width), dtype=bool)
    route_pixels = binary_dilation(route_pixels, iterations=4)

    water = (b > 145) & (b > r + 18) & (b > g + 4) & (g > 120) & ~route_pixels
    vegetation = (g > 130) & (g > r + 4) & (g > b - 18) & ~water & ~route_pixels

    purple_school = (b > 130) & (r > 105) & (g < 185) & (b > g + 20)
    school_event = draw_event_disk_mask(
        (height, width),
        events,
        {"school", "school_landmark"},
        radius_px=settings.event_radius_px + 24,
    ).astype(bool)
    school = purple_school | school_event

    route_corridor = draw_polyline_mask(
        (height, width),
        route.points_xy,
        width=settings.route_width_px,
    )
    route_corridor_bool = route_corridor > 0.5

    bridge_event = draw_event_disk_mask(
        (height, width),
        events,
        {"bridge", "bridge_landmark"},
        radius_px=settings.event_radius_px + 12,
    ).astype(bool)
    bridge_blue = (b > 95) & (g > 70) & (r < 135) & bridge_event
    bridge = bridge_event | bridge_blue

    low_sat = sat < 34
    not_semantic = ~(water | vegetation | school | bridge | route_pixels)
    light_surface = low_sat & (maxc > 170) & (maxc < 250) & not_semantic
    beige_bias = (r >= b - 4) & (g >= b - 14)
    building = light_surface & beige_bias & (maxc < 235)

    road_color = light_surface & (maxc >= 190) & ~building
    road = road_color | route_corridor_bool

    turn_events = draw_event_disk_mask(
        (height, width),
        events,
        {"left_turn", "right_turn", "left", "right", "turn_left", "turn_right"},
        radius_px=settings.event_radius_px,
    )
    curvature_events = _curvature_event_mask(route, radius_px=settings.event_radius_px)
    intersection = np.maximum(turn_events, curvature_events)

    masks = {
        "road": _float_mask(road),
        "water": _float_mask(water),
        "building": _float_mask(building),
        "route_corridor": _float_mask(route_corridor_bool),
        "intersection": _float_mask(intersection > 0.1),
        "school_landmark": _float_mask(school),
        "bridge_landmark": _float_mask(bridge),
        "route_progress": np.zeros((height, width), dtype=np.float32),
        "landmark_progress": np.zeros((height, width), dtype=np.float32),
    }
    return SemanticMap(masks=masks, image_size=(width, height))


def build_sparse_map_adapter_condition(
    traj: MapTrajectory,
    *,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build structured event tokens consumed by the sparse MapAdapter."""

    if frame_indices is None:
        frame_progress = np.asarray(traj.route_progress, dtype=np.float32)
    else:
        frame_progress = np.asarray(traj.route_progress[np.asarray(frame_indices, dtype=np.int64)], dtype=np.float32)

    events: list[dict[str, Any]] = [
        {
            "type": "route",
            "shape": "curve",
            "progress": 0.0,
            "route_progress": 0.0,
            "confidence": 1.0,
            "order": 0.0,
        }
    ]
    width, height = traj.route.image_size
    for order, event in enumerate(traj.events, start=1):
        x, y = event.projected_xy
        payload = {
            "type": _adapter_event_type(event.event_type),
            "progress": float(event.route_progress),
            "route_progress": float(event.route_progress),
            "x": float(x) / max(float(width), 1.0),
            "y": float(y) / max(float(height), 1.0),
            "confidence": max(0.0, 1.0 - float(event.distance_px) / 120.0),
            "order": float(order),
        }
        if event.side:
            payload["side"] = str(event.side)
        if event.event_type in {"left", "left_turn", "turn_left"}:
            payload["shape"] = "left_turn"
        elif event.event_type in {"right", "right_turn", "turn_right"}:
            payload["shape"] = "right_turn"
        elif event.event_type == "bridge":
            payload["shape"] = "corridor"
        events.append(payload)

    return {
        "events": events,
        "frame_progress": [float(x) for x in frame_progress.tolist()],
        "source": "frame_aligned_map_projection",
    }


def build_frame_aligned_layout_condition(
    traj: MapTrajectory,
    semantic_map: SemanticMap,
    *,
    settings: LayoutProjectionSettings | None = None,
    frame_indices: Sequence[int] | None = None,
) -> FrameAlignedLayoutCondition:
    """Project global map masks into ego-aligned per-frame EMapCtrl grids."""

    settings = settings or LayoutProjectionSettings()
    if frame_indices is None:
        indices = np.arange(len(traj.route_progress), dtype=np.int64)
    else:
        indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 1:
        raise ValueError("frame_indices must be a non-empty 1D sequence")
    if int(indices.min()) < 0 or int(indices.max()) >= len(traj.route_progress):
        raise ValueError("frame_indices contain values outside the trajectory")

    channels = tuple(semantic_map.channels)
    features = np.zeros(
        (1, int(indices.size), len(channels), int(settings.grid_height), int(settings.grid_width)),
        dtype=np.float32,
    )
    frame_progress = np.asarray(traj.route_progress[indices], dtype=np.float32)
    landmark_progress = _landmark_progress_values(frame_progress, traj.events, settings.landmark_sigma_progress)

    sample_masks = [
        semantic_map.masks[name]
        for name in channels
        if name not in {"route_progress", "landmark_progress"}
    ]
    sample_names = [
        name
        for name in channels
        if name not in {"route_progress", "landmark_progress"}
    ]
    for local_idx, frame_idx in enumerate(indices):
        progress = float(traj.route_progress[int(frame_idx)])
        center_xy = np.asarray(traj.route_xy[int(frame_idx)], dtype=np.float64)
        forward_xy = tangent_at_progress(traj.route, progress)
        crops = sample_ego_aligned_masks(
            sample_masks,
            center_xy=center_xy,
            forward_xy=forward_xy,
            grid_height=settings.grid_height,
            grid_width=settings.grid_width,
            lateral_extent_px=settings.lateral_extent_px,
            lookahead_px=settings.lookahead_px,
            rear_px=settings.rear_px,
        )
        for name, crop in zip(sample_names, crops):
            features[0, local_idx, _CHANNEL_INDEX[name]] = crop
        features[0, local_idx, _CHANNEL_INDEX["route_progress"]] = progress
        features[0, local_idx, _CHANNEL_INDEX["landmark_progress"]] = landmark_progress[local_idx]

    sparse = build_sparse_map_adapter_condition(traj, frame_indices=indices)
    return FrameAlignedLayoutCondition(
        features=features,
        channels=channels,
        frame_indices=indices.astype(np.int64),
        frame_progress=frame_progress,
        landmark_progress=landmark_progress.astype(np.float32),
        map_adapter_condition=sparse,
        settings=settings,
    )


def sample_ego_aligned_masks(
    masks: Sequence[np.ndarray],
    *,
    center_xy: Sequence[float],
    forward_xy: Sequence[float],
    grid_height: int,
    grid_width: int,
    lateral_extent_px: float,
    lookahead_px: float,
    rear_px: float,
) -> list[np.ndarray]:
    """Sample masks into a local route-aligned crop.

    Rows run from far-ahead to slightly behind the camera. Columns run from
    camera-left to camera-right.
    """

    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    forward = np.asarray(forward_xy, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-9:
        forward = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        forward = forward / norm
    right = np.asarray([-forward[1], forward[0]], dtype=np.float64)

    forward_values = np.linspace(float(lookahead_px), -float(rear_px), int(grid_height), dtype=np.float64)
    lateral_values = np.linspace(-float(lateral_extent_px), float(lateral_extent_px), int(grid_width), dtype=np.float64)
    ff, ll = np.meshgrid(forward_values, lateral_values, indexing="ij")
    xy = center[None, None, :] + ff[..., None] * forward[None, None, :] + ll[..., None] * right[None, None, :]

    coords = np.stack([xy[..., 1], xy[..., 0]], axis=0)
    crops = []
    for mask in masks:
        sampled = map_coordinates(
            np.asarray(mask, dtype=np.float32),
            coords,
            order=1,
            mode="constant",
            cval=0.0,
        )
        crops.append(np.clip(sampled, 0.0, 1.0).astype(np.float32))
    return crops


def summarize_layout_condition(condition: FrameAlignedLayoutCondition) -> dict[str, Any]:
    features = np.asarray(condition.features, dtype=np.float32)
    summary = {
        "source": "frame_aligned_map_projection",
        "features_shape": list(features.shape),
        "features_dtype": str(features.dtype),
        "channels": list(condition.channels),
        "frame_count": int(features.shape[1]),
        "frame_index_min": int(condition.frame_indices.min()),
        "frame_index_max": int(condition.frame_indices.max()),
        "route_progress_min": float(np.min(condition.frame_progress)),
        "route_progress_max": float(np.max(condition.frame_progress)),
        "route_progress_monotonic": bool(np.all(np.diff(condition.frame_progress) >= -1.0e-6)),
        "landmark_progress_max": float(np.max(condition.landmark_progress)),
        "settings": {
            "grid_height": int(condition.settings.grid_height),
            "grid_width": int(condition.settings.grid_width),
            "lateral_extent_px": float(condition.settings.lateral_extent_px),
            "lookahead_px": float(condition.settings.lookahead_px),
            "rear_px": float(condition.settings.rear_px),
            "route_width_px": int(condition.settings.route_width_px),
            "event_radius_px": int(condition.settings.event_radius_px),
        },
        "channel_stats": {},
        "map_adapter_event_count": len(condition.map_adapter_condition.get("events", [])),
    }
    for idx, name in enumerate(condition.channels):
        channel = features[0, :, idx]
        summary["channel_stats"][name] = {
            "min": float(np.min(channel)),
            "mean": float(np.mean(channel)),
            "max": float(np.max(channel)),
            "nonzero_fraction": float(np.mean(channel > 1.0e-4)),
        }
    return summary


def save_layout_condition_npz(
    path: str | Path,
    condition: FrameAlignedLayoutCondition,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=condition.features.astype(np.float32),
        frame_indices=condition.frame_indices.astype(np.int64),
        frame_progress=condition.frame_progress.astype(np.float32),
        landmark_progress=condition.landmark_progress.astype(np.float32),
        channels=np.asarray(condition.channels),
    )


def save_map_adapter_condition_json(
    path: str | Path,
    condition: FrameAlignedLayoutCondition,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(condition.map_adapter_condition, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def save_layout_condition_contact_sheet(
    path: str | Path,
    map_path: str | Path,
    traj: MapTrajectory,
    condition: FrameAlignedLayoutCondition,
    *,
    columns: Sequence[float] = (0.0, 0.25, 0.52, 0.68, 0.81, 1.0),
    rows: Sequence[str] = (
        "route_corridor",
        "road",
        "water",
        "building",
        "school_landmark",
        "bridge_landmark",
        "intersection",
    ),
) -> None:
    """Save a compact visual audit sheet for the dense layout condition."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_image = Image.open(map_path).convert("RGB")
    overlay = _map_overlay(map_image, traj)
    overlay.thumbnail((760, 570))

    cell_w = 160
    cell_h = max(88, int(round(cell_w * condition.settings.grid_height / condition.settings.grid_width)))
    pad = 12
    label_h = 26
    header_h = overlay.height + 2 * pad
    total_w = max(overlay.width + 2 * pad, pad + len(columns) * (cell_w + pad))
    total_h = header_h + label_h + len(rows) * (cell_h + label_h + pad) + pad
    sheet = Image.new("RGB", (total_w, total_h), (250, 250, 248))
    sheet.paste(overlay, (pad, pad))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((pad, overlay.height + pad + 2), "Frame-aligned EMapCtrl condition snapshots", fill=(20, 20, 20), font=font)

    selected = [_nearest_condition_frame(condition, progress) for progress in columns]
    channel_to_idx = {name: idx for idx, name in enumerate(condition.channels)}
    y = header_h + label_h
    for row_name in rows:
        x = pad
        draw.text((x, y - label_h + 6), row_name, fill=(20, 20, 20), font=font)
        for col_idx, local_frame in enumerate(selected):
            channel = condition.features[0, local_frame, channel_to_idx[row_name]]
            tile = _heatmap_tile(channel, row_name).resize((cell_w, cell_h), Image.Resampling.NEAREST)
            sheet.paste(tile, (x, y))
            progress = float(condition.frame_progress[local_frame])
            frame = int(condition.frame_indices[local_frame])
            draw.text((x + 4, y + 4), f"p={progress:.2f} f={frame}", fill=(255, 255, 255), font=font)
            x += cell_w + pad
        y += cell_h + label_h + pad

    sheet.save(path)


def _map_overlay(map_image: Image.Image, traj: MapTrajectory) -> Image.Image:
    image = map_image.copy()
    draw = ImageDraw.Draw(image)
    route_pts = [(float(x), float(y)) for x, y in traj.route.points_xy]
    if len(route_pts) >= 2:
        draw.line(route_pts, fill=(255, 0, 0), width=5, joint="curve")
    sampled = [(float(x), float(y)) for x, y in traj.route_xy]
    if len(sampled) >= 2:
        draw.line(sampled, fill=(0, 150, 255), width=3, joint="curve")
    _circle(draw, traj.route_xy[0], 10, (0, 220, 80))
    _circle(draw, traj.route_xy[-1], 10, (90, 160, 255))
    for event in traj.events:
        xy = np.asarray(event.projected_xy, dtype=np.float64)
        _circle(draw, xy, 8, (255, 255, 255))
        draw.text((float(xy[0]) + 10, float(xy[1]) - 8), f"{event.event_type}@{event.target_progress:.2f}", fill=(0, 0, 0))
    for idx in range(40, len(traj.route_xy) - 8, 160):
        _arrow(draw, traj.route_xy[idx], traj.route_xy[idx + 8], (0, 80, 220))
    return image


def _heatmap_tile(channel: np.ndarray, name: str) -> Image.Image:
    value = np.clip(np.asarray(channel, dtype=np.float32), 0.0, 1.0)
    palette = {
        "road": (210, 210, 210),
        "water": (80, 170, 240),
        "building": (205, 185, 170),
        "route_corridor": (255, 70, 30),
        "intersection": (255, 190, 40),
        "school_landmark": (150, 90, 210),
        "bridge_landmark": (40, 100, 190),
    }.get(name, (255, 255, 255))
    rgb = np.zeros((*value.shape, 3), dtype=np.uint8)
    for channel_idx, color in enumerate(palette):
        rgb[..., channel_idx] = np.asarray(value * float(color), dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _landmark_progress_values(
    frame_progress: np.ndarray,
    events: Sequence[Any],
    sigma: float,
) -> np.ndarray:
    landmark_events = [
        float(_event_progress(event))
        for event in events
        if _event_type(event) in {"school", "school_landmark", "bridge", "bridge_landmark"}
    ]
    if not landmark_events:
        return np.zeros_like(frame_progress, dtype=np.float32)
    sigma = max(float(sigma), 1.0e-6)
    values = np.zeros_like(frame_progress, dtype=np.float64)
    for progress in landmark_events:
        values = np.maximum(values, np.exp(-0.5 * ((frame_progress - progress) / sigma) ** 2))
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def _curvature_event_mask(route: RoutePolyline, *, radius_px: int) -> np.ndarray:
    pts = np.asarray(route.points_xy, dtype=np.float64)
    image_shape = (route.height, route.width)
    if pts.shape[0] < 5:
        return np.zeros(image_shape, dtype=np.float32)
    step = max(4, pts.shape[0] // 140)
    selected = []
    for idx in range(step, pts.shape[0] - step, step):
        prev = pts[idx] - pts[idx - step]
        nxt = pts[idx + step] - pts[idx]
        n0 = float(np.linalg.norm(prev))
        n1 = float(np.linalg.norm(nxt))
        if n0 < 1.0e-6 or n1 < 1.0e-6:
            continue
        angle = abs(math.atan2(prev[0] * nxt[1] - prev[1] * nxt[0], float(np.dot(prev, nxt))))
        if angle > math.radians(18.0):
            selected.append(pts[idx])
    if not selected:
        return np.zeros(image_shape, dtype=np.float32)
    return draw_polyline_mask(image_shape, np.asarray(selected, dtype=np.float64), width=max(1, int(radius_px)))


def _nearest_condition_frame(condition: FrameAlignedLayoutCondition, progress: float) -> int:
    return int(np.argmin(np.abs(condition.frame_progress - float(progress))))


def _float_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.float32).clip(0.0, 1.0)


def _circle(draw: ImageDraw.ImageDraw, point: Sequence[float], radius: int, fill: tuple[int, int, int]) -> None:
    x, y = float(point[0]), float(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(0, 0, 0), width=2)


def _arrow(draw: ImageDraw.ImageDraw, p0: Sequence[float], p1: Sequence[float], color: tuple[int, int, int]) -> None:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    draw.line((x0, y0, x1, y1), fill=color, width=3)
    angle = math.atan2(y1 - y0, x1 - x0)
    head = 12.0
    spread = 0.55
    left = (x1 - head * math.cos(angle - spread), y1 - head * math.sin(angle - spread))
    right = (x1 - head * math.cos(angle + spread), y1 - head * math.sin(angle + spread))
    draw.polygon([(x1, y1), left, right], fill=color)


def _event_type(event: Any) -> str:
    if isinstance(event, Mapping):
        return str(event.get("type", event.get("event_type", "event")))
    return str(getattr(event, "event_type", getattr(event, "type", "event")))


def _event_progress(event: Any) -> float:
    if isinstance(event, Mapping):
        return float(event.get("route_progress", event.get("progress", event.get("target_progress", 0.0))))
    return float(getattr(event, "route_progress", getattr(event, "progress", getattr(event, "target_progress", 0.0))))


def _event_xy(event: Any) -> tuple[float, float] | None:
    if isinstance(event, Mapping):
        xy = event.get("projected_xy", event.get("route_anchor_xy", event.get("xy")))
        if xy is None:
            return None
        return float(xy[0]), float(xy[1])
    xy = getattr(event, "projected_xy", None)
    if xy is None:
        xy = getattr(event, "route_anchor_xy", None)
    if xy is None:
        return None
    return float(xy[0]), float(xy[1])


def _adapter_event_type(event_type: str) -> str:
    aliases = {
        "left": "left_turn",
        "turn_left": "left_turn",
        "right": "right_turn",
        "turn_right": "right_turn",
        "school_landmark": "school",
        "bridge_landmark": "bridge",
    }
    return aliases.get(str(event_type), str(event_type))
