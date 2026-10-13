"""Metric-aware route parameterization for DreamX-World map-driven generation.

The legacy route consumers (``route_to_actions_by_heading_delta_subsegment`` and
``build_segment_semantic_layout``) advance the route by *normalized* arc length,
which means the entire polyline is always consumed regardless of clip duration.
This module provides the bridge to *metric* parameterization: given a real-world
speed (m/s) and a map scale (meters-per-pixel), compute how far the camera
travels per clip and derive the effective ``route_duration_s`` that the existing
windowing logic can consume.

Key formula::

    route_metric_length_m = polyline_arc_length_px * meters_per_pixel
    clip_distance_m        = speed_mps * clip_duration_s
    route_duration_s       = route_metric_length_m / speed_mps

If the route is shorter than one clip's travel distance, the whole route is
consumed in the first clip (route_duration_s = clip_duration_s, progress_per_chunk = 1.0).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def polyline_arc_length_px(polyline: np.ndarray) -> float:
    """Total arc length of a polyline in pixel units.

    Args:
        polyline: ``(N, 2)`` array of route vertices in image xy.

    Returns:
        Sum of Euclidean segment lengths.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        return 0.0
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(seg_lens.sum())


def route_metric_length_m(
    polyline: np.ndarray,
    meters_per_pixel: float,
) -> float:
    """Total route length in meters.

    Args:
        polyline: ``(N, 2)`` array of route vertices in image xy.
        meters_per_pixel: Map scale: how many meters one pixel represents.

    Returns:
        ``polyline_arc_length_px(polyline) * meters_per_pixel``.
    """
    return polyline_arc_length_px(polyline) * float(meters_per_pixel)


def compute_metric_route_duration_s(
    polyline: np.ndarray,
    meters_per_pixel: float,
    speed_mps: float,
    clip_duration_s: float,
    *,
    min_duration_s: float = 0.1,
) -> Optional[float]:
    """Compute the effective ``route_duration_s`` for metric parameterization.

    This is the value to pass to ``route_to_actions_by_heading_delta_subsegment``
    (and the ego-layout windowing) so that each clip advances only
    ``speed_mps * clip_duration_s`` meters along the route.

    The existing windowing logic computes::

        progress_per_chunk = clip_duration_s / route_duration_s

    so each clip consumes a fraction ``clip_duration_s / route_duration_s`` of
    the total route arc length.  For metric mode we want::

        clip_distance_m = speed_mps * clip_duration_s
        route_distance_m = arc_length_px * meters_per_pixel
        progress_per_chunk = clip_distance_m / route_distance_m

    which gives::

        route_duration_s = route_distance_m / speed_mps

    If the route is shorter than one clip's travel distance, the entire route
    is consumed in clip 0 (route_duration_s = clip_duration_s).

    Args:
        polyline: ``(N, 2)`` route vertices in image xy.
        meters_per_pixel: Map scale (m/px).
        speed_mps: Travel speed in m/s.
        clip_duration_s: Duration of one diffusion segment in seconds.
        min_duration_s: Floor to avoid division-by-zero on degenerate routes.

    Returns:
        Effective route duration in seconds, or ``None`` if metric mode cannot
        be applied (e.g. zero speed or zero route length — caller should fall
        back to legacy behavior).
    """
    speed = float(speed_mps)
    mpp = float(meters_per_pixel)
    if speed <= 0.0 or mpp <= 0.0:
        return None

    total_m = route_metric_length_m(polyline, mpp)
    if total_m < 1.0e-6:
        return None

    route_duration = total_m / speed

    # If the route is shorter than one clip, clamp so the whole route is
    # consumed in clip 0 (progress_per_chunk = 1.0).
    if route_duration < float(clip_duration_s):
        route_duration = float(clip_duration_s)

    return max(route_duration, min_duration_s)


def segment_progress_window(
    seg_idx: int,
    num_segments: int,
    route_duration_s: Optional[float],
    clip_duration_s: float,
) -> tuple[float, float]:
    """Compute the ``[start, end]`` normalized route-progress window for a segment.

    This mirrors the windowing in
    ``route_to_actions_by_heading_delta_subsegment`` so the ego-layout path can
    use the *same* window as the action-plan path.

    Args:
        seg_idx: Segment index (0-based).
        num_segments: Total number of segments.
        route_duration_s: Effective route duration (from metric or explicit).
            If ``None``, falls back to legacy ``1/num_segments`` per segment.
        clip_duration_s: Duration of one segment in seconds.

    Returns:
        ``(start, end)`` in normalized route progress ``[0, 1]``.
    """
    if route_duration_s is not None and route_duration_s > 0:
        progress_per_chunk = max(
            1.0e-6, min(1.0, float(clip_duration_s) / float(route_duration_s))
        )
    else:
        progress_per_chunk = 1.0 / float(num_segments) if num_segments > 0 else 1.0

    start = min(1.0, float(seg_idx) * progress_per_chunk)
    end = min(1.0, float(seg_idx + 1) * progress_per_chunk)
    return start, end


def frame_progress_for_segment(
    seg_idx: int,
    num_segments: int,
    num_latent_frames: int,
    route_duration_s: Optional[float],
    clip_duration_s: float,
) -> np.ndarray:
    """Per-latent-frame normalized route progress for one segment.

    Returns an array of length ``num_latent_frames`` linearly spaced from the
    segment's start progress to its end progress.  This replaces the legacy
    ``torch.linspace(seg_idx/num_segments, (seg_idx+1)/num_segments, F)`` with a
    metric-aware window when ``route_duration_s`` is supplied.

    Args:
        seg_idx: Segment index (0-based).
        num_segments: Total number of segments.
        num_latent_frames: Number of latent frames in this segment.
        route_duration_s: Effective route duration (metric or explicit), or ``None``.
        clip_duration_s: Duration of one segment in seconds.

    Returns:
        ``(num_latent_frames,)`` float64 array in ``[0, 1]``.
    """
    start, end = segment_progress_window(
        seg_idx, num_segments, route_duration_s, clip_duration_s
    )
    return np.linspace(start, end, int(num_latent_frames), dtype=np.float64)
