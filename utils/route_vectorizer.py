"""Vectorize the red route in a colored map image.

The extractor is intentionally deterministic: it thresholds the red/orange
route, traces start-to-goal through a dilated route corridor, then resamples the
ordered path by arc length. Coordinates are always image-space ``(x, y)``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt


@dataclass(frozen=True)
class RouteProjection:
    point_xy: tuple[float, float]
    progress: float
    distance_px: float
    segment_index: int


@dataclass(frozen=True)
class RoutePolyline:
    points_xy: np.ndarray
    progress: np.ndarray
    cumulative_px: np.ndarray
    length_px: float
    image_size: tuple[int, int]
    route_mask: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image_size[0])

    @property
    def height(self) -> int:
        return int(self.image_size[1])


def route_color_mask(path: str | Path, *, legend_cutoff: float = 0.85) -> np.ndarray:
    """Return a boolean mask for red/orange route pixels."""
    rgb = np.array(Image.open(path).convert("RGB")).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    route = (r > 180) & (g < 130) & (b < 130) & ((r - g) > 80) & ((r - b) > 80)

    cutoff = int(route.shape[0] * float(legend_cutoff))
    route[cutoff:, :] = False
    if not bool(route.any()):
        raise RuntimeError(f"No red/orange route pixels found in {path}")
    return route


def _auto_anchor_points(route: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = route.shape
    yy, xx = np.indices(route.shape)
    start_region = route & (xx < int(w * 0.30)) & (yy < int(h * 0.60))
    goal_region = route & (xx > int(w * 0.72)) & (yy < int(h * 0.35))
    if not bool(start_region.any()) or not bool(goal_region.any()):
        ys, xs = np.where(route)
        left_idx = int(np.argmin(xs))
        right_idx = int(np.argmax(xs))
        return (int(ys[left_idx]), int(xs[left_idx])), (int(ys[right_idx]), int(xs[right_idx]))

    sy, sx = np.where(start_region)
    min_start_x = int(sx.min())
    start_band = sx <= min_start_x + 3
    start = (int(round(float(np.median(sy[start_band])))), min_start_x)

    gy, gx = np.where(goal_region)
    min_goal_y = int(gy.min())
    goal_band = gy <= min_goal_y + 20
    goal = (int(round(float(np.median(gy[goal_band])))), int(round(float(np.median(gx[goal_band])))))
    return start, goal


def _nearest_mask_point(mask: np.ndarray, y: float, x: float) -> tuple[int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Cannot query nearest point on an empty mask")
    dist2 = (ys.astype(float) - float(y)) ** 2 + (xs.astype(float) - float(x)) ** 2
    idx = int(np.argmin(dist2))
    return int(ys[idx]), int(xs[idx])


def _shortest_path_on_mask(
    search_mask: np.ndarray,
    preferred_mask: np.ndarray,
    start_yx: tuple[int, int],
    goal_yx: tuple[int, int],
) -> np.ndarray:
    h, w = search_mask.shape
    start_yx = _nearest_mask_point(search_mask, *start_yx)
    goal_yx = _nearest_mask_point(search_mask, *goal_yx)
    start_id = start_yx[0] * w + start_yx[1]
    goal_id = goal_yx[0] * w + goal_yx[1]

    raw_distance = distance_transform_edt(~preferred_mask)
    center_distance = distance_transform_edt(search_mask)
    distances = np.full(h * w, np.inf, dtype=np.float64)
    previous = np.full(h * w, -1, dtype=np.int64)
    distances[start_id] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, start_yx[0], start_yx[1])]
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    while heap:
        dist, y, x = heapq.heappop(heap)
        idx = y * w + x
        if dist > distances[idx] + 1.0e-9:
            continue
        if idx == goal_id:
            break
        for dy, dx, step_cost in neighbors:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not bool(search_mask[ny, nx]):
                continue
            nidx = ny * w + nx
            raw_penalty = 0.0 if bool(preferred_mask[ny, nx]) else min(50.0, 4.0 * float(raw_distance[ny, nx]))
            center_penalty = 1.0 / (1.0 + float(center_distance[ny, nx]))
            next_dist = dist + step_cost * (1.0 + raw_penalty + center_penalty)
            if next_dist < distances[nidx]:
                distances[nidx] = next_dist
                previous[nidx] = idx
                heapq.heappush(heap, (next_dist, ny, nx))

    if not np.isfinite(distances[goal_id]):
        raise RuntimeError(f"No route path found from {start_yx} to {goal_yx}")

    path_ids: list[int] = []
    cursor = goal_id
    while cursor != -1:
        path_ids.append(cursor)
        cursor = int(previous[cursor])
    path_ids.reverse()
    return np.asarray([(idx % w, idx // w) for idx in path_ids], dtype=np.float64)


def trace_route_path(
    route_mask: np.ndarray,
    *,
    search_dilation: int = 24,
    start_xy: Sequence[float] | None = None,
    goal_xy: Sequence[float] | None = None,
) -> np.ndarray:
    """Trace an ordered route polyline from the red route mask."""
    if search_dilation < 1:
        raise ValueError("search_dilation must be positive")

    search_mask = binary_dilation(route_mask, iterations=int(search_dilation))
    auto_start, auto_goal = _auto_anchor_points(route_mask)
    if start_xy is None:
        start_yx = auto_start
    else:
        start_yx = _nearest_mask_point(route_mask, float(start_xy[1]), float(start_xy[0]))
    if goal_xy is None:
        goal_yx = auto_goal
    else:
        goal_yx = _nearest_mask_point(route_mask, float(goal_xy[1]), float(goal_xy[0]))
    return _shortest_path_on_mask(search_mask, route_mask, start_yx, goal_yx)


def cumulative_lengths(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {pts.shape}")
    if pts.shape[0] < 2:
        raise ValueError("A polyline needs at least two points")
    diffs = np.diff(pts, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    keep = np.concatenate([[True], lengths > 1.0e-9])
    if not bool(keep.all()):
        pts = pts[keep]
        diffs = np.diff(pts, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(lengths)])


def _drop_duplicate_points(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {pts.shape}")
    if pts.shape[0] < 2:
        raise ValueError("A polyline needs at least two points")
    keep = [pts[0]]
    for point in pts[1:]:
        if float(np.linalg.norm(point - keep[-1])) > 1.0e-9:
            keep.append(point)
    out = np.asarray(keep, dtype=np.float64)
    if out.shape[0] < 2:
        raise ValueError("Polyline collapsed to fewer than two unique points")
    return out


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return np.asarray(values, dtype=np.float64)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.stack(
        [np.convolve(padded[:, col], kernel, mode="valid") for col in range(values.shape[1])],
        axis=1,
    )


def resample_polyline(points_xy: np.ndarray, samples: int, *, smooth_window: int = 1) -> np.ndarray:
    """Resample a polyline to ``samples`` evenly spaced arc-length points."""
    if samples < 2:
        raise ValueError("samples must be >= 2")
    pts = _drop_duplicate_points(points_xy)
    distance = cumulative_lengths(pts)
    total = float(distance[-1])
    if total <= 0.0:
        raise ValueError("Polyline length must be positive")
    sample_distance = np.linspace(0.0, total, int(samples))
    x = np.interp(sample_distance, distance, pts[:, 0])
    y = np.interp(sample_distance, distance, pts[:, 1])
    return _moving_average(np.stack([x, y], axis=1), int(smooth_window))


def build_route_polyline(
    points_xy: np.ndarray,
    *,
    image_size: tuple[int, int],
    route_mask: np.ndarray,
) -> RoutePolyline:
    pts = _drop_duplicate_points(points_xy)
    distance = cumulative_lengths(pts)
    total = float(distance[-1])
    if total <= 0.0:
        raise ValueError("Route length must be positive")
    return RoutePolyline(
        points_xy=pts.astype(np.float64),
        progress=(distance / total).astype(np.float64),
        cumulative_px=distance.astype(np.float64),
        length_px=total,
        image_size=(int(image_size[0]), int(image_size[1])),
        route_mask=np.asarray(route_mask, dtype=bool),
    )


def vectorize_red_route(
    map_path: str | Path,
    *,
    search_dilation: int = 24,
    samples: int = 2000,
    smooth_window: int = 9,
    start_xy: Sequence[float] | None = None,
    goal_xy: Sequence[float] | None = None,
) -> RoutePolyline:
    """Extract and resample the red route from ``map_path``."""
    map_path = Path(map_path)
    route_mask = route_color_mask(map_path)
    raw_path = trace_route_path(
        route_mask,
        search_dilation=search_dilation,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )
    points = resample_polyline(raw_path, int(samples), smooth_window=int(smooth_window))
    with Image.open(map_path) as image:
        image_size = image.size
    return build_route_polyline(points, image_size=image_size, route_mask=route_mask)


def sample_route(route: RoutePolyline, progress: float | np.ndarray) -> np.ndarray:
    """Sample image-space route coordinates at normalized progress values."""
    p = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    x = np.interp(p, route.progress, route.points_xy[:, 0])
    y = np.interp(p, route.progress, route.points_xy[:, 1])
    return np.stack([x, y], axis=-1)


def tangent_at_progress(route: RoutePolyline, progress: float, *, eps: float = 1.0e-4) -> np.ndarray:
    p0 = max(0.0, float(progress) - eps)
    p1 = min(1.0, float(progress) + eps)
    if p1 <= p0:
        p0 = max(0.0, min(1.0, float(progress)) - eps)
        p1 = min(1.0, p0 + eps)
    pts = sample_route(route, np.asarray([p0, p1], dtype=np.float64))
    tangent = pts[1] - pts[0]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1.0e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return tangent / norm


def project_point_to_route(route: RoutePolyline, point_xy: Sequence[float]) -> RouteProjection:
    """Project an image-space point onto the route polyline."""
    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"point_xy must have shape (2,), got {point.shape}")

    pts = route.points_xy
    best_distance = float("inf")
    best_progress = 0.0
    best_point = pts[0].copy()
    best_segment = 0
    seg_lengths = np.diff(route.cumulative_px)
    for idx, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        v = b - a
        denom = float(np.dot(v, v))
        t = 0.0 if denom <= 1.0e-12 else float(np.clip(np.dot(point - a, v) / denom, 0.0, 1.0))
        q = a + t * v
        dist = float(np.linalg.norm(point - q))
        if dist < best_distance:
            arc = float(route.cumulative_px[idx] + t * seg_lengths[idx])
            best_distance = dist
            best_progress = arc / float(route.length_px)
            best_point = q
            best_segment = idx

    return RouteProjection(
        point_xy=(float(best_point[0]), float(best_point[1])),
        progress=float(np.clip(best_progress, 0.0, 1.0)),
        distance_px=float(best_distance),
        segment_index=int(best_segment),
    )
