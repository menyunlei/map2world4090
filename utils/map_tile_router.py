"""Deterministic map-driven tile routing metadata for AR inference.

This module only produces routing decisions and debug artifacts. It does not
skip DiT, attention, FFN, VAE, or scheduler work.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is available in the project env.
    torch = None

from PIL import Image, ImageDraw


FULL_COMPUTE = "FULL_COMPUTE"
REFRESH = "REFRESH"
REUSE = "REUSE"
DECISIONS = (FULL_COMPUTE, REFRESH, REUSE)


@dataclass(frozen=True)
class RuntimeLatentGrid:
    """Spatial token grid inferred from the AR runtime latent tensor."""

    height: int
    width: int
    frame_seq_length: int
    latent_height: int
    latent_width: int
    token_downsample: int

    def to_dict(self) -> dict[str, int]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "frame_seq_length": int(self.frame_seq_length),
            "latent_height": int(self.latent_height),
            "latent_width": int(self.latent_width),
            "token_downsample": int(self.token_downsample),
        }


@dataclass(frozen=True)
class TileRange:
    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass
class MapTileRouterConfig:
    """Configuration for deterministic tile-decision generation."""

    latent_grid_height: int
    latent_grid_width: int
    frame_seq_length: int
    tile_height_tokens: int = 2
    tile_width_tokens: int = 4
    refresh_interval: int = 8
    road_class_ids: tuple[int, ...] = (1,)
    dynamic_threshold: float = 0.5
    uncertainty_threshold: float = 0.5
    turn_threshold_deg: float = 6.0
    translation_threshold: float = 1.0e-4
    progress_new_region_scale: float = 1.0
    motion_tile_margin: int = 1
    all_full_when_map_missing: bool = True
    debug_max_frames: int = 32

    def __post_init__(self) -> None:
        if int(self.latent_grid_height) < 1 or int(self.latent_grid_width) < 1:
            raise ValueError("latent grid height/width must be positive")
        if int(self.latent_grid_height) * int(self.latent_grid_width) != int(self.frame_seq_length):
            raise ValueError(
                "latent grid product does not match frame_seq_length: "
                f"{self.latent_grid_height}x{self.latent_grid_width} != {self.frame_seq_length}"
            )
        if int(self.tile_height_tokens) < 1 or int(self.tile_width_tokens) < 1:
            raise ValueError("tile size must be positive")
        if int(self.refresh_interval) < 0:
            raise ValueError("refresh_interval must be >= 0")
        if int(self.motion_tile_margin) < 0:
            raise ValueError("motion_tile_margin must be >= 0")

    @property
    def tile_size(self) -> tuple[int, int]:
        return int(self.tile_height_tokens), int(self.tile_width_tokens)


@dataclass
class MapTileRouterResult:
    schema_version: int
    runtime_latent_grid: dict[str, int]
    config: dict[str, Any]
    summary: dict[str, Any]
    frames: list[dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "runtime_latent_grid": dict(self.runtime_latent_grid),
            "config": dict(self.config),
            "summary": dict(self.summary),
            "frames": list(self.frames),
        }


def parse_tile_size(value: str | Sequence[int] | None, default: tuple[int, int] = (2, 4)) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.lower().replace(",", "x").replace(" ", "")
        if "x" not in text:
            raise ValueError("tile size must use HxW format, e.g. 2x4")
        h_text, w_text = text.split("x", 1)
        return int(h_text), int(w_text)
    if len(value) != 2:
        raise ValueError("tile size sequence must have length 2")
    return int(value[0]), int(value[1])


def infer_runtime_latent_grid(
    latent_height: int,
    latent_width: int,
    frame_seq_length: int,
) -> RuntimeLatentGrid:
    """Infer the per-frame token grid from runtime latent shape and sequence length."""

    latent_height = int(latent_height)
    latent_width = int(latent_width)
    frame_seq_length = int(frame_seq_length)
    if latent_height < 1 or latent_width < 1 or frame_seq_length < 1:
        raise ValueError("latent dimensions and frame_seq_length must be positive")

    if latent_height % 2 == 0 and latent_width % 2 == 0:
        token_h = latent_height // 2
        token_w = latent_width // 2
        if token_h * token_w == frame_seq_length:
            return RuntimeLatentGrid(
                height=token_h,
                width=token_w,
                frame_seq_length=frame_seq_length,
                latent_height=latent_height,
                latent_width=latent_width,
                token_downsample=2,
            )

    if latent_height * latent_width == frame_seq_length:
        return RuntimeLatentGrid(
            height=latent_height,
            width=latent_width,
            frame_seq_length=frame_seq_length,
            latent_height=latent_height,
            latent_width=latent_width,
            token_downsample=1,
        )

    raise ValueError(
        "Could not infer runtime latent token grid from "
        f"latent_shape=({latent_height}, {latent_width}) and frame_seq_length={frame_seq_length}"
    )


def infer_runtime_latent_grid_from_noise(noise: Any, frame_seq_length: int) -> RuntimeLatentGrid:
    shape = tuple(noise.shape)
    if len(shape) < 5:
        raise ValueError(f"expected AR noise tensor shape [B,T,C,H,W], got {shape}")
    return infer_runtime_latent_grid(shape[-2], shape[-1], frame_seq_length)


def load_spatial_array(path: str | Path | None) -> np.ndarray | None:
    """Load a .npy/.npz/.pt spatial tensor for semantic, dynamic, or uncertainty maps."""

    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"spatial array not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        loaded = np.load(path)
        if "array" in loaded:
            return loaded["array"]
        if "tensor" in loaded:
            return loaded["tensor"]
        first_key = sorted(loaded.files)[0]
        return loaded[first_key]
    if suffix in {".pt", ".pth"}:
        if torch is None:
            raise RuntimeError("torch is required to load .pt spatial arrays")
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            for key in ("semantic_map", "map", "mask", "tensor", "array"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        if torch.is_tensor(loaded):
            return loaded.detach().cpu().numpy()
        return np.asarray(loaded)
    raise ValueError(f"unsupported spatial array extension: {path.suffix}")


def normalize_semantic_frames(array: Any | None) -> np.ndarray | None:
    """Normalize semantic labels to [T,H,W] int32, with -1 for unknown."""

    if array is None:
        return None
    arr = _to_numpy(array)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        if int(arr.shape[1]) == 1:
            arr = arr[:, 0]
        elif int(arr.shape[-1]) == 1:
            arr = arr[..., 0]
        elif int(arr.shape[1]) <= 64:
            arr = np.argmax(arr, axis=1)
        else:
            arr = np.argmax(arr, axis=-1)
    else:
        raise ValueError(f"semantic map must have rank 2, 3, or 4, got {arr.shape}")

    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        out = np.full(arr.shape, -1, dtype=np.int32)
        finite = np.isfinite(arr)
        finite_values = arr[finite]
        if finite_values.size and float(finite_values.min()) >= 0.0 and float(finite_values.max()) <= 1.0:
            out[finite] = (arr[finite] > 0.5).astype(np.int32)
        else:
            out[finite] = np.rint(arr[finite]).astype(np.int32)
        return out
    return arr.astype(np.int32, copy=False)


def normalize_mask_frames(array: Any | None, threshold: float = 0.5) -> np.ndarray | None:
    """Normalize a mask to [T,H,W] bool."""

    if array is None:
        return None
    arr = _to_numpy(array)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        if int(arr.shape[1]) == 1:
            arr = arr[:, 0]
        elif int(arr.shape[-1]) == 1:
            arr = arr[..., 0]
        else:
            arr = np.max(arr, axis=1 if int(arr.shape[1]) <= 64 else -1)
    else:
        raise ValueError(f"mask must have rank 2, 3, or 4, got {arr.shape}")
    return np.asarray(arr > float(threshold), dtype=bool)


def camera_poses_from_cam_params(cam_params: Any, latent_frame_count: int) -> list[dict[str, float]]:
    """Convert DreamX camera-parameter rows to latent-aligned x/y/z/yaw poses."""

    arr = _to_numpy(cam_params)
    if arr.ndim != 2 or arr.shape[1] < 19:
        raise ValueError(f"expected camera params [N,>=19], got {arr.shape}")
    latent_frame_count = int(latent_frame_count)
    if latent_frame_count < 1:
        raise ValueError("latent_frame_count must be >= 1")
    aligned_indices = [0] + [1 + 4 * i for i in range(latent_frame_count - 1)]
    aligned_indices = [min(int(i), int(arr.shape[0]) - 1) for i in aligned_indices]
    return [normalize_camera_pose(arr[i]) for i in aligned_indices]


def normalize_camera_pose(pose: Any | None) -> dict[str, float] | None:
    """Normalize common pose formats to {x,y,z,yaw_rad}."""

    if pose is None:
        return None
    if isinstance(pose, dict):
        yaw = pose.get("yaw_rad", pose.get("yaw", 0.0))
        return {
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "z": float(pose.get("z", 0.0)),
            "yaw_rad": float(yaw),
        }

    arr = _to_numpy(pose).astype(np.float64, copy=False)
    if arr.shape == (4, 4):
        c2w = arr
        translation = c2w[:3, 3]
        yaw = math.atan2(float(c2w[0, 2]), float(c2w[2, 2]))
        return {"x": float(translation[0]), "y": float(translation[1]), "z": float(translation[2]), "yaw_rad": yaw}
    flat = arr.reshape(-1)
    if flat.size >= 19:
        w2c = flat[7:19].reshape(3, 4)
        w2c4 = np.eye(4, dtype=np.float64)
        w2c4[:3, :] = w2c
        c2w = np.linalg.inv(w2c4)
        translation = c2w[:3, 3]
        yaw = math.atan2(float(c2w[0, 2]), float(c2w[2, 2]))
        return {"x": float(translation[0]), "y": float(translation[1]), "z": float(translation[2]), "yaw_rad": yaw}
    if flat.size >= 3:
        return {"x": float(flat[0]), "y": 0.0, "z": float(flat[1]), "yaw_rad": float(flat[2])}
    if flat.size >= 2:
        return {"x": float(flat[0]), "y": 0.0, "z": float(flat[1]), "yaw_rad": 0.0}
    raise ValueError(f"unsupported camera pose shape: {arr.shape}")


def latent_route_progress(
    progress: Sequence[float] | None,
    *,
    latent_frame_count: int,
    pixel_frame_count: int | None = None,
) -> list[float]:
    """Return one normalized route-progress value per latent frame."""

    latent_frame_count = int(latent_frame_count)
    if latent_frame_count < 1:
        raise ValueError("latent_frame_count must be >= 1")
    if progress is None:
        if latent_frame_count == 1:
            return [0.0]
        return [float(i) / float(latent_frame_count - 1) for i in range(latent_frame_count)]

    values = [float(v) for v in progress]
    if len(values) == latent_frame_count:
        return [_clamp(v, 0.0, 1.0) for v in values]
    if pixel_frame_count is not None and len(values) == int(pixel_frame_count):
        aligned_indices = [0] + [1 + 4 * i for i in range(latent_frame_count - 1)]
        aligned_indices = [min(i, len(values) - 1) for i in aligned_indices]
        return [_clamp(values[i], 0.0, 1.0) for i in aligned_indices]
    if len(values) < 2:
        return [float(values[0])] * latent_frame_count

    src = np.linspace(0.0, 1.0, num=len(values), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, num=latent_frame_count, dtype=np.float64)
    interp = np.interp(dst, src, np.asarray(values, dtype=np.float64))
    return [_clamp(float(v), 0.0, 1.0) for v in interp.tolist()]


class MapTileRouter:
    """Build tile-level compute decisions without changing inference tensors."""

    def __init__(self, config: MapTileRouterConfig):
        self.config = config
        self._cache: dict[str, dict[str, Any]] = {}

    def reset_cache(self) -> None:
        self._cache.clear()

    def route(
        self,
        *,
        semantic_map: Any | None,
        num_frames: int,
        route_progress: Sequence[float] | None = None,
        camera_poses: Sequence[Any] | None = None,
        dynamic_mask: Any | None = None,
        uncertainty_mask: Any | None = None,
        task_id: str | None = None,
        map_frame_start: int = 0,
        map_frame_end: int | None = None,
    ) -> MapTileRouterResult:
        num_frames = int(num_frames)
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        semantic_frames = normalize_semantic_frames(semantic_map)
        dynamic_frames = normalize_mask_frames(dynamic_mask, self.config.dynamic_threshold)
        uncertainty_frames = normalize_mask_frames(uncertainty_mask, self.config.uncertainty_threshold)
        progress = latent_route_progress(route_progress, latent_frame_count=num_frames)
        poses = _normalize_pose_sequence(camera_poses, num_frames)

        if semantic_frames is None:
            if not self.config.all_full_when_map_missing:
                raise ValueError("semantic_map is required when all_full_when_map_missing=False")
            return self._route_all_full(num_frames=num_frames, progress=progress, poses=poses, task_id=task_id)

        tile_ranges = _tile_ranges(
            self.config.latent_grid_height,
            self.config.latent_grid_width,
            self.config.tile_height_tokens,
            self.config.tile_width_tokens,
        )
        frames: list[dict[str, Any]] = []
        total_counts = {decision: 0 for decision in DECISIONS}

        for frame_index in range(num_frames):
            map_index = _select_frame_index(
                frame_index,
                num_frames=num_frames,
                source_count=int(semantic_frames.shape[0]),
                progress=progress,
                map_frame_start=map_frame_start,
                map_frame_end=map_frame_end,
            )
            semantic_tokens = resize_labels_to_grid(
                semantic_frames[map_index],
                self.config.latent_grid_height,
                self.config.latent_grid_width,
            )
            dynamic_tokens = _select_optional_mask(
                dynamic_frames,
                frame_index=frame_index,
                num_frames=num_frames,
                target_h=self.config.latent_grid_height,
                target_w=self.config.latent_grid_width,
                progress=progress,
                map_frame_start=map_frame_start,
                map_frame_end=map_frame_end,
            )
            uncertainty_tokens = _select_optional_mask(
                uncertainty_frames,
                frame_index=frame_index,
                num_frames=num_frames,
                target_h=self.config.latent_grid_height,
                target_w=self.config.latent_grid_width,
                progress=progress,
                map_frame_start=map_frame_start,
                map_frame_end=map_frame_end,
            )
            boundary_tokens = semantic_boundary_mask(semantic_tokens, self.config.road_class_ids)
            new_visibility = self._new_visibility_masks(
                frame_index=frame_index,
                progress=progress,
                poses=poses,
            )
            tile_border_tokens = semantic_tile_border_mask(semantic_tokens, tile_ranges)
            tile_records: list[dict[str, Any]] = []
            counts = {decision: 0 for decision in DECISIONS}

            for tile in tile_ranges:
                token_slice = np.s_[tile.y0:tile.y1, tile.x0:tile.x1]
                tile_labels = semantic_tokens[token_slice]
                labels = sorted(int(v) for v in np.unique(tile_labels).tolist())
                reasons: list[str] = []
                if np.any(tile_labels < 0):
                    reasons.append("map_boundary")
                if np.any(boundary_tokens[token_slice]):
                    reasons.append("route_or_road_boundary")
                if len([v for v in labels if v >= 0]) > 1:
                    reasons.append("mixed_semantic_tile")
                if np.any(tile_border_tokens[token_slice]):
                    reasons.append("semantic_class_border")
                if dynamic_tokens is not None and np.any(dynamic_tokens[token_slice]):
                    reasons.append("dynamic_region")
                if uncertainty_tokens is not None and np.any(uncertainty_tokens[token_slice]):
                    reasons.append("uncertainty_region")
                for reason, mask in new_visibility.items():
                    if np.any(mask[token_slice]):
                        reasons.append(reason)

                signature = _semantic_signature(tile_labels)
                cache_key = f"r{tile.row}:c{tile.col}:s{signature}"
                cache_entry = self._cache.get(cache_key)
                cache_valid_before = cache_entry is not None
                force_full = bool(reasons)

                if force_full:
                    decision = FULL_COMPUTE
                elif not cache_valid_before:
                    decision = FULL_COMPUTE
                    reasons.append("missing_cache")
                elif self.config.refresh_interval > 0 and (
                    frame_index - int(cache_entry["last_compute_frame"])
                ) >= int(self.config.refresh_interval):
                    decision = REFRESH
                    reasons.append("refresh_interval")
                else:
                    decision = REUSE

                if decision in {FULL_COMPUTE, REFRESH}:
                    self._cache[cache_key] = {
                        "last_compute_frame": int(frame_index),
                        "semantic_signature": signature,
                    }

                counts[decision] += 1
                tile_records.append(
                    {
                        "tile_row": int(tile.row),
                        "tile_col": int(tile.col),
                        "token_y0": int(tile.y0),
                        "token_y1": int(tile.y1),
                        "token_x0": int(tile.x0),
                        "token_x1": int(tile.x1),
                        "decision": decision,
                        "reasons": sorted(set(reasons)),
                        "semantic_classes": labels,
                        "cache_key": cache_key,
                        "cache_valid_before": bool(cache_valid_before),
                    }
                )

            for decision in DECISIONS:
                total_counts[decision] += counts[decision]
            total_tiles = len(tile_records)
            active = counts[FULL_COMPUTE] + counts[REFRESH]
            frames.append(
                {
                    "frame_index": int(frame_index),
                    "map_frame_index": int(map_index),
                    "route_progress": float(progress[frame_index]),
                    "camera_pose": poses[frame_index],
                    "tile_rows": int(math.ceil(self.config.latent_grid_height / self.config.tile_height_tokens)),
                    "tile_cols": int(math.ceil(self.config.latent_grid_width / self.config.tile_width_tokens)),
                    "counts": counts,
                    "active_tile_ratio": float(active / max(1, total_tiles)),
                    "tiles": tile_records,
                }
            )

        total_tiles_all = sum(sum(frame["counts"].values()) for frame in frames)
        active_total = total_counts[FULL_COMPUTE] + total_counts[REFRESH]
        return MapTileRouterResult(
            schema_version=1,
            runtime_latent_grid={
                "height": int(self.config.latent_grid_height),
                "width": int(self.config.latent_grid_width),
                "frame_seq_length": int(self.config.frame_seq_length),
            },
            config={
                "tile_height_tokens": int(self.config.tile_height_tokens),
                "tile_width_tokens": int(self.config.tile_width_tokens),
                "refresh_interval": int(self.config.refresh_interval),
                "road_class_ids": [int(v) for v in self.config.road_class_ids],
                "turn_threshold_deg": float(self.config.turn_threshold_deg),
                "translation_threshold": float(self.config.translation_threshold),
                "progress_new_region_scale": float(self.config.progress_new_region_scale),
                "motion_tile_margin": int(self.config.motion_tile_margin),
                "debug_max_frames": int(self.config.debug_max_frames),
                "task_id": task_id,
            },
            summary={
                "frame_count": int(num_frames),
                "tile_count_per_frame": int(len(tile_ranges)),
                "total_tile_records": int(total_tiles_all),
                "counts": total_counts,
                "active_tile_ratio": float(active_total / max(1, total_tiles_all)),
            },
            frames=frames,
        )

    def _route_all_full(
        self,
        *,
        num_frames: int,
        progress: Sequence[float],
        poses: Sequence[dict[str, float] | None],
        task_id: str | None,
    ) -> MapTileRouterResult:
        tile_ranges = _tile_ranges(
            self.config.latent_grid_height,
            self.config.latent_grid_width,
            self.config.tile_height_tokens,
            self.config.tile_width_tokens,
        )
        frames: list[dict[str, Any]] = []
        for frame_index in range(num_frames):
            tiles = [
                {
                    "tile_row": int(tile.row),
                    "tile_col": int(tile.col),
                    "token_y0": int(tile.y0),
                    "token_y1": int(tile.y1),
                    "token_x0": int(tile.x0),
                    "token_x1": int(tile.x1),
                    "decision": FULL_COMPUTE,
                    "reasons": ["all_full_fallback", "missing_semantic_map"],
                    "semantic_classes": [],
                    "cache_key": None,
                    "cache_valid_before": False,
                }
                for tile in tile_ranges
            ]
            frames.append(
                {
                    "frame_index": int(frame_index),
                    "map_frame_index": None,
                    "route_progress": float(progress[frame_index]),
                    "camera_pose": poses[frame_index],
                    "tile_rows": int(math.ceil(self.config.latent_grid_height / self.config.tile_height_tokens)),
                    "tile_cols": int(math.ceil(self.config.latent_grid_width / self.config.tile_width_tokens)),
                    "counts": {FULL_COMPUTE: len(tiles), REFRESH: 0, REUSE: 0},
                    "active_tile_ratio": 1.0,
                    "tiles": tiles,
                }
            )
        total = len(tile_ranges) * int(num_frames)
        return MapTileRouterResult(
            schema_version=1,
            runtime_latent_grid={
                "height": int(self.config.latent_grid_height),
                "width": int(self.config.latent_grid_width),
                "frame_seq_length": int(self.config.frame_seq_length),
            },
            config={
                "tile_height_tokens": int(self.config.tile_height_tokens),
                "tile_width_tokens": int(self.config.tile_width_tokens),
                "refresh_interval": int(self.config.refresh_interval),
                "road_class_ids": [int(v) for v in self.config.road_class_ids],
                "debug_max_frames": int(self.config.debug_max_frames),
                "task_id": task_id,
                "fallback": "all_full_missing_semantic_map",
            },
            summary={
                "frame_count": int(num_frames),
                "tile_count_per_frame": int(len(tile_ranges)),
                "total_tile_records": int(total),
                "counts": {FULL_COMPUTE: int(total), REFRESH: 0, REUSE: 0},
                "active_tile_ratio": 1.0,
            },
            frames=frames,
        )

    def _new_visibility_masks(
        self,
        *,
        frame_index: int,
        progress: Sequence[float],
        poses: Sequence[dict[str, float] | None],
    ) -> dict[str, np.ndarray]:
        h = int(self.config.latent_grid_height)
        w = int(self.config.latent_grid_width)
        masks: dict[str, np.ndarray] = {}
        if int(frame_index) == 0:
            masks["newly_visible_initial_frame"] = np.ones((h, w), dtype=bool)
            return masks

        prev_progress = float(progress[frame_index - 1])
        cur_progress = float(progress[frame_index])
        delta_progress = max(0.0, cur_progress - prev_progress)
        if delta_progress > 1.0e-9 and self.config.progress_new_region_scale > 0.0:
            rows = int(math.ceil(delta_progress * h * float(self.config.progress_new_region_scale)))
            rows = max(1, min(h, rows))
            mask = np.zeros((h, w), dtype=bool)
            mask[:rows, :] = True
            masks["newly_visible_route_progress"] = mask

        prev_pose = poses[frame_index - 1]
        cur_pose = poses[frame_index]
        if prev_pose is None or cur_pose is None:
            return masks

        margin_h = max(1, int(self.config.motion_tile_margin) * int(self.config.tile_height_tokens))
        margin_w = max(1, int(self.config.motion_tile_margin) * int(self.config.tile_width_tokens))
        yaw_delta = _wrap_angle(float(cur_pose["yaw_rad"]) - float(prev_pose["yaw_rad"]))
        if abs(math.degrees(yaw_delta)) >= float(self.config.turn_threshold_deg):
            mask = np.zeros((h, w), dtype=bool)
            if yaw_delta > 0:
                mask[:, : min(w, margin_w)] = True
            else:
                mask[:, max(0, w - margin_w):] = True
            mask[: min(h, margin_h), :] = True
            masks["newly_visible_pose_turn"] = mask

        dx = float(cur_pose["x"]) - float(prev_pose["x"])
        dz = float(cur_pose["z"]) - float(prev_pose["z"])
        if math.hypot(dx, dz) >= float(self.config.translation_threshold):
            mask = np.zeros((h, w), dtype=bool)
            if abs(dz) >= abs(dx):
                if dz >= 0:
                    mask[: min(h, margin_h), :] = True
                else:
                    mask[max(0, h - margin_h):, :] = True
            else:
                if dx >= 0:
                    mask[:, max(0, w - margin_w):] = True
                else:
                    mask[:, : min(w, margin_w)] = True
            masks["newly_visible_pose_translation"] = mask
        return masks


def save_router_json(result: MapTileRouterResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_json_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def save_router_frame_jsons(result: MapTileRouterResult, directory: str | Path) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for frame in result.frames:
        path = directory / f"frame_{int(frame['frame_index']):04d}.json"
        payload = {
            "schema_version": int(result.schema_version),
            "runtime_latent_grid": dict(result.runtime_latent_grid),
            "config": dict(result.config),
            "frame": frame,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        paths.append(path)
    return paths


def render_router_debug_visualization(
    result: MapTileRouterResult,
    path: str | Path,
    *,
    cell_px: int = 10,
    frame_stride: int = 1,
) -> Path:
    """Render a contact sheet of tile decisions."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_h = int(result.runtime_latent_grid["height"])
    grid_w = int(result.runtime_latent_grid["width"])
    cell_px = max(2, int(cell_px))
    frame_stride = max(1, int(frame_stride))
    frames = result.frames[::frame_stride]
    max_frames = int(result.config.get("debug_max_frames", 32)) if "debug_max_frames" in result.config else 32
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    panel_w = grid_w * cell_px
    panel_h = grid_h * cell_px
    legend_h = 26
    gap = 8
    cols = min(4, max(1, len(frames)))
    rows = int(math.ceil(len(frames) / cols)) if frames else 1
    width = cols * panel_w + (cols - 1) * gap
    height = legend_h + rows * (panel_h + 18) + (rows - 1) * gap
    image = Image.new("RGB", (max(1, width), max(1, height)), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    colors = {
        FULL_COMPUTE: (220, 60, 55),
        REFRESH: (245, 170, 45),
        REUSE: (55, 165, 90),
    }
    x = 4
    for label in DECISIONS:
        draw.rectangle((x, 6, x + 16, 20), fill=colors[label])
        draw.text((x + 20, 5), label, fill=(235, 235, 235))
        x += 20 + len(label) * 7 + 16

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        ox = col * (panel_w + gap)
        oy = legend_h + row * (panel_h + 18 + gap)
        draw.text((ox, oy), f"f{frame['frame_index']} active={frame['active_tile_ratio']:.2f}", fill=(235, 235, 235))
        py = oy + 18
        draw.rectangle((ox, py, ox + panel_w - 1, py + panel_h - 1), fill=(45, 45, 45), outline=(90, 90, 90))
        for tile in frame["tiles"]:
            color = colors[tile["decision"]]
            x0 = ox + int(tile["token_x0"]) * cell_px
            x1 = ox + int(tile["token_x1"]) * cell_px - 1
            y0 = py + int(tile["token_y0"]) * cell_px
            y1 = py + int(tile["token_y1"]) * cell_px - 1
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=(16, 16, 16))
    image.save(path)
    return path


def resize_labels_to_grid(labels: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be [H,W], got {labels.shape}")
    target_h = int(target_h)
    target_w = int(target_w)
    out = np.full((target_h, target_w), -1, dtype=np.int32)
    src_h, src_w = labels.shape
    for y in range(target_h):
        y0, y1 = _scaled_bounds(y, src_h, target_h)
        for x in range(target_w):
            x0, x1 = _scaled_bounds(x, src_w, target_w)
            vals = labels[y0:y1, x0:x1].reshape(-1)
            vals = vals[np.isfinite(vals)] if np.issubdtype(vals.dtype, np.floating) else vals
            if vals.size == 0:
                continue
            uniq, counts = np.unique(vals.astype(np.int32), return_counts=True)
            out[y, x] = int(uniq[np.argmax(counts)])
    return out


def resize_mask_to_grid(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {mask.shape}")
    target_h = int(target_h)
    target_w = int(target_w)
    out = np.zeros((target_h, target_w), dtype=bool)
    src_h, src_w = mask.shape
    for y in range(target_h):
        y0, y1 = _scaled_bounds(y, src_h, target_h)
        for x in range(target_w):
            x0, x1 = _scaled_bounds(x, src_w, target_w)
            out[y, x] = bool(np.any(mask[y0:y1, x0:x1]))
    return out


def semantic_boundary_mask(labels: np.ndarray, road_class_ids: Sequence[int] = (1,)) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    boundary = labels < 0
    road = np.isin(labels, np.asarray(tuple(int(v) for v in road_class_ids), dtype=np.int32))
    for dy, dx in ((1, 0), (0, 1)):
        a = labels[:-dy or None, :-dx or None] if dy or dx else labels
        b = labels[dy:, dx:] if dy or dx else labels
        diff = a != b
        if dy:
            boundary[:-1, :] |= diff
            boundary[1:, :] |= diff
            road_diff = road[:-1, :] != road[1:, :]
            boundary[:-1, :] |= road_diff
            boundary[1:, :] |= road_diff
        if dx:
            boundary[:, :-1] |= diff
            boundary[:, 1:] |= diff
            road_diff = road[:, :-1] != road[:, 1:]
            boundary[:, :-1] |= road_diff
            boundary[:, 1:] |= road_diff
    return boundary


def semantic_tile_border_mask(labels: np.ndarray, tile_ranges: Sequence[TileRange]) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    major: dict[tuple[int, int], int] = {}
    for tile in tile_ranges:
        vals = labels[tile.y0:tile.y1, tile.x0:tile.x1].reshape(-1)
        vals = vals[vals >= 0]
        if vals.size == 0:
            major[(tile.row, tile.col)] = -1
        else:
            uniq, counts = np.unique(vals, return_counts=True)
            major[(tile.row, tile.col)] = int(uniq[np.argmax(counts)])

    by_pos = {(tile.row, tile.col): tile for tile in tile_ranges}
    out = np.zeros_like(labels, dtype=bool)
    for tile in tile_ranges:
        cls = major[(tile.row, tile.col)]
        for nr, nc in ((tile.row + 1, tile.col), (tile.row, tile.col + 1)):
            if (nr, nc) not in major:
                continue
            other = major[(nr, nc)]
            if cls == other:
                continue
            out[tile.y0:tile.y1, tile.x0:tile.x1] = True
            neighbor = by_pos[(nr, nc)]
            out[neighbor.y0:neighbor.y1, neighbor.x0:neighbor.x1] = True
    return out


def _select_optional_mask(
    frames: np.ndarray | None,
    *,
    frame_index: int,
    num_frames: int,
    target_h: int,
    target_w: int,
    progress: Sequence[float],
    map_frame_start: int,
    map_frame_end: int | None,
) -> np.ndarray | None:
    if frames is None:
        return None
    index = _select_frame_index(
        frame_index,
        num_frames=num_frames,
        source_count=int(frames.shape[0]),
        progress=progress,
        map_frame_start=map_frame_start,
        map_frame_end=map_frame_end,
    )
    return resize_mask_to_grid(frames[index], target_h, target_w)


def _select_frame_index(
    frame_index: int,
    *,
    num_frames: int,
    source_count: int,
    progress: Sequence[float],
    map_frame_start: int,
    map_frame_end: int | None,
) -> int:
    source_count = int(source_count)
    if source_count < 1:
        raise ValueError("source_count must be >= 1")
    if source_count == int(num_frames):
        return int(frame_index)
    start = max(0, min(int(map_frame_start), source_count - 1))
    end_exclusive = source_count if map_frame_end is None else int(map_frame_end)
    end_exclusive = max(start + 1, min(end_exclusive, source_count))
    last = end_exclusive - 1
    if last <= start:
        return start
    p = _clamp(float(progress[frame_index]), 0.0, 1.0)
    return int(round(start + p * float(last - start)))


def _tile_ranges(grid_h: int, grid_w: int, tile_h: int, tile_w: int) -> list[TileRange]:
    ranges: list[TileRange] = []
    tile_row = 0
    for y0 in range(0, int(grid_h), int(tile_h)):
        tile_col = 0
        y1 = min(int(grid_h), y0 + int(tile_h))
        for x0 in range(0, int(grid_w), int(tile_w)):
            x1 = min(int(grid_w), x0 + int(tile_w))
            ranges.append(TileRange(row=tile_row, col=tile_col, y0=y0, y1=y1, x0=x0, x1=x1))
            tile_col += 1
        tile_row += 1
    return ranges


def _scaled_bounds(index: int, source: int, target: int) -> tuple[int, int]:
    start = int(math.floor(float(index) * float(source) / float(target)))
    end = int(math.ceil(float(index + 1) * float(source) / float(target)))
    start = max(0, min(start, int(source) - 1))
    end = max(start + 1, min(end, int(source)))
    return start, end


def _semantic_signature(labels: np.ndarray) -> str:
    vals = sorted(int(v) for v in np.unique(labels).tolist())
    return ",".join(str(v) for v in vals)


def _normalize_pose_sequence(poses: Sequence[Any] | None, num_frames: int) -> list[dict[str, float] | None]:
    if poses is None:
        return [None] * int(num_frames)
    out = [normalize_camera_pose(pose) for pose in poses]
    if len(out) == int(num_frames):
        return out
    if len(out) < 1:
        return [None] * int(num_frames)
    if len(out) == 1:
        return out * int(num_frames)
    indices = np.linspace(0, len(out) - 1, num=int(num_frames))
    return [out[int(round(i))] for i in indices]


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))
