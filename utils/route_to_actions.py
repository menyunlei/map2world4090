"""Deterministic route-polyline -> DreamX camera action-sequence converter.

In map-driven camera mode the turn sequence is derived from the route polyline
itself (the map's plan-view path), then handed to the existing
``ActionToPoseFromID`` -> ``GetPoseEmbedsFromPosesPrope`` -> PRoPE pathway
exactly as a manually-authored ``action_seq`` would be. This closes the route-
information-leakage gap identified in ``ROUTE_INFORMATION_LEAKAGE.md``: turns
are no longer taken from a hand-written ``action_seq`` that secretly encodes
the route; they are read off the map route.

Action vocabulary (``utils/inference_utils.py:ACTION_DICT``):
    w  -> forward
    wj -> forward + left_rot   (yaw += rotation_value  -> left turn)
    wl -> forward + right_rot  (yaw -= rotation_value  -> right turn)

Turn sign convention (``utils/inference_utils.py:141-144``):
    left_rot  -> +yaw  -> ``wj``
    right_rot -> -yaw  -> ``wl``

The converter is pure and deterministic: same polyline in -> same action
sequence out. It does not touch the DiT, MapCtrl, or any token bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

# Action vocabulary (mirrors utils/inference_utils.py:ACTION_DICT).
ACTION_FORWARD = "w"
ACTION_LEFT = "wj"   # forward + left_rot
ACTION_RIGHT = "wl"  # forward + right_rot
_VALID_ACTIONS = {ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}

# Below this |signed turn angle| (radians) a segment is treated as straight.
STRAIGHT_THRESHOLD_RAD = math.radians(5.0)


@dataclass
class RouteToActionResult:
    """Output of :func:`route_to_actions`.

    Attributes:
        actions: per-segment DreamX action tokens (``w``/``wj``/``wl``).
        turn_angles_deg: signed turn angle at each segment in degrees
            (``0`` for straight segments). Sign follows the runtime convention:
            positive = left, negative = right.
        progress: normalized route progress ``in [0, 1]`` at each segment
            midpoint.
        n_segments: number of segments.
        turn_sequence: human-readable turn sequence, e.g. ``"straight,left,right"``.
    """

    actions: list[str] = field(default_factory=list)
    turn_angles_deg: list[float] = field(default_factory=list)
    progress: list[float] = field(default_factory=list)
    n_segments: int = 0
    turn_sequence: str = ""
    action_speed_list: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    trajectory_xy: np.ndarray | None = None
    trajectory_yaw: np.ndarray | None = None

    def to_dict(self) -> dict:
        out = {
            "actions": list(self.actions),
            "turn_angles_deg": [round(a, 3) for a in self.turn_angles_deg],
            "progress": [round(p, 4) for p in self.progress],
            "n_segments": self.n_segments,
            "turn_sequence": self.turn_sequence,
        }
        if self.action_speed_list:
            out["action_speed_list"] = [round(float(s), 4) for s in self.action_speed_list]
        if self.metrics:
            out["metrics"] = self.metrics
        if self.trajectory_xy is not None:
            out["trajectory_xy_shape"] = list(self.trajectory_xy.shape)
        if self.trajectory_yaw is not None:
            out["trajectory_yaw_shape"] = list(self.trajectory_yaw.shape)
        return out


def _wrap_rad(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _labels_from_actions(actions: Sequence[str]) -> list[str]:
    labels = []
    for action in actions:
        if action == ACTION_LEFT:
            labels.append("left")
        elif action == ACTION_RIGHT:
            labels.append("right")
        elif action == ACTION_FORWARD:
            labels.append("straight")
        else:
            raise ValueError(f"Unknown action {action!r}")
    return labels


def _signed_turn_angle(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> float:
    """Signed turn angle (radians) at vertex ``cur`` on the polyline.

    Positive -> left turn, negative -> right turn, matching the runtime yaw
    convention (``left_rot`` = +yaw, ``right_rot`` = -yaw).

    Computed as the angle from heading ``prev->cur`` to heading ``cur->nxt``
    using the 2D cross product for the sign and the dot product for the
    magnitude. Image/array coordinates are assumed (x right, y down); the
    cross-product sign is flipped so that "turning toward +x while heading
    +y" reads as a right turn, consistent with a top-down map where the camera
    yaws about the vertical axis.
    """
    v1 = np.asarray(cur, dtype=np.float64) - np.asarray(prev, dtype=np.float64)
    v2 = np.asarray(nxt, dtype=np.float64) - np.asarray(cur, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    mag = math.acos(cos_a)
    # 2D cross z-component: >0 means v2 is counter-clockwise from v1.
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    # In image coords (y down), CCW visually = clockwise in math sense, so flip
    # so that a visually left turn yields +angle (matching +yaw = left_rot).
    return -mag if cross > 0 else mag


def _segment_yaws_image_xy(points_xy: np.ndarray) -> np.ndarray:
    """Return per-edge yaw in image coordinates.

    The runtime convention uses +yaw for ``left_rot``. In image coordinates
    (x right, y down), heading yaw is therefore ``atan2(-dy, dx)``.
    """
    diffs = np.diff(np.asarray(points_xy, dtype=np.float64), axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    yaws = np.zeros(diffs.shape[0], dtype=np.float64)
    valid = lengths > 1.0e-9
    yaws[valid] = np.arctan2(-diffs[valid, 1], diffs[valid, 0])
    if not bool(valid.all()):
        for idx in range(len(yaws)):
            if valid[idx]:
                continue
            prev_valid = np.where(valid[:idx])[0]
            next_valid = np.where(valid[idx + 1 :])[0]
            if len(prev_valid):
                yaws[idx] = yaws[int(prev_valid[-1])]
            elif len(next_valid):
                yaws[idx] = yaws[int(idx + 1 + next_valid[0])]
    return yaws


def _normalize_polyline(
    polyline: np.ndarray,
    progress: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Coerce input to an ``(N, 2)`` float64 array, dropping duplicate points.

    Consecutive points within ``1e-9`` of the previous kept point are dropped.
    When ``progress`` is supplied, it is filtered by the SAME keep-mask so the
    returned progress stays aligned with the deduped polyline (lengths match).
    This matters for sources like ``route_plan.route_samples`` whose polylines
    contain repeated consecutive vertices: deduping only the points would leave
    ``progress`` one-per-original-vertex and trigger the length-mismatch guard
    in :func:`_progress_for_polyline`.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"polyline must have shape (N, 2); got {pts.shape}")
    if pts.shape[0] < 2:
        raise ValueError("polyline must have at least 2 points")
    keep_idx = [0]
    for i in range(1, pts.shape[0]):
        if np.linalg.norm(pts[i] - pts[keep_idx[-1]]) > 1e-9:
            keep_idx.append(i)
    out = pts[keep_idx]
    if out.shape[0] < 2:
        raise ValueError("polyline collapsed to a single point after dedup")
    out_progress: Optional[np.ndarray] = None
    if progress is not None:
        prog = np.asarray(progress, dtype=np.float64)
        if prog.shape[0] != pts.shape[0]:
            raise ValueError(
                f"progress length {prog.shape[0]} != input polyline length {pts.shape[0]}"
            )
        out_progress = prog[keep_idx]
    return out, out_progress


def route_to_actions(
    polyline: Sequence[Sequence[float]] | np.ndarray,
    n_segments: int,
    progress: Optional[Sequence[float]] = None,
    straight_threshold_rad: float = STRAIGHT_THRESHOLD_RAD,
) -> RouteToActionResult:
    """Convert an ordered 2D route polyline into a DreamX action sequence.

    Parameters:
        polyline: ordered ``(N, 2)`` route points in any consistent planar
            coordinate frame (map pixels, meters, or normalized [0,1]). Only
            the relative geometry matters.
        n_segments: number of camera action segments to emit (one action per
            segment, matching the existing per-segment plan in
            ``inference_long_video.py``).
        progress: optional normalized route progress ``in [0, 1]`` for each
            polyline point. If ``None`` it is derived from cumulative arc
            length. If provided it must have the same length as ``polyline``
            and be monotonically non-decreasing in ``[0, 1]``.
        straight_threshold_rad: below this |signed angle| a segment is
            classified as straight (``w``).

    Returns:
        :class:`RouteToActionResult` with one entry per segment.

    The segment->route mapping: each segment ``s`` covers route progress
    ``[s/n, (s+1)/n]``; the representative turn for the segment is the signed
    turn angle at the polyline vertex whose progress is closest to the segment
    midpoint ``(s+0.5)/n``. This mirrors the nearest-event scheme already used
    by ``experiments/geomap/scene_graph.py:SceneGraph.to_action_seq`` so the
    map-driven camera spine is consistent with the scene-graph compiler.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    pts, filtered_progress = _normalize_polyline(polyline, progress)
    n = pts.shape[0]

    # Per-point normalized progress.
    if filtered_progress is None:
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = float(cum[-1])
        if total < 1e-9:
            # Degenerate route: all segments straight.
            res = RouteToActionResult(
                actions=[ACTION_FORWARD] * n_segments,
                turn_angles_deg=[0.0] * n_segments,
                progress=[(s + 0.5) / n_segments for s in range(n_segments)],
                n_segments=n_segments,
                turn_sequence=",".join(["straight"] * n_segments),
            )
            return res
        pts_progress = cum / total
    else:
        pts_progress = filtered_progress
        if pts_progress[0] < -1e-9 or pts_progress[-1] > 1.0 + 1e-9:
            raise ValueError("progress must lie in [0, 1]")
        if np.any(np.diff(pts_progress) < -1e-9):
            raise ValueError("progress must be monotonically non-decreasing")
        pts_progress = np.clip(pts_progress, 0.0, 1.0)

    # Signed turn angle at every interior vertex; endpoints carry 0.
    vertex_angles = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        vertex_angles[i] = _signed_turn_angle(pts[i - 1], pts[i], pts[i + 1])

    actions: list[str] = []
    turn_angles_deg: list[float] = []
    seg_progress: list[float] = []
    turn_labels: list[str] = []
    for s in range(n_segments):
        mid = (s + 0.5) / n_segments
        seg_progress.append(float(mid))
        # nearest interior vertex to the segment midpoint
        best_i = 1
        best_d = abs(float(pts_progress[1]) - mid)
        for i in range(2, n - 1):
            d = abs(float(pts_progress[i]) - mid)
            if d < best_d:
                best_d = d
                best_i = i
        ang = float(vertex_angles[best_i])
        if ang > straight_threshold_rad:
            actions.append(ACTION_LEFT)
            turn_labels.append("left")
        elif ang < -straight_threshold_rad:
            actions.append(ACTION_RIGHT)
            turn_labels.append("right")
        else:
            actions.append(ACTION_FORWARD)
            turn_labels.append("straight")
        turn_angles_deg.append(math.degrees(ang))

    return RouteToActionResult(
        actions=actions,
        turn_angles_deg=turn_angles_deg,
        progress=seg_progress,
        n_segments=n_segments,
        turn_sequence=",".join(turn_labels),
    )


def route_to_actions_by_heading_delta(
    polyline: Sequence[Sequence[float]] | np.ndarray,
    n_segments: int,
    progress: Optional[Sequence[float]] = None,
    straight_threshold_rad: float = math.radians(10.0),
) -> RouteToActionResult:
    """Convert route geometry to actions by integrating yaw changes per segment.

    This is the better fit for smooth vectorized map routes. The older
    ``route_to_actions`` looks at a single nearest vertex; after smoothing, the
    per-vertex angles can be too small and the whole route collapses to
    ``w``. This function sums the signed heading changes whose vertices fall
    inside each output segment, matching the controlled-Markov interpretation:
    the map supplies the route tangent over time, then each 5s video segment is
    quantized into the native DreamX action vocabulary.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    pts, filtered_progress = _normalize_polyline(polyline, progress)
    n = pts.shape[0]

    if filtered_progress is None:
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = float(cum[-1])
        if total < 1e-9:
            return RouteToActionResult(
                actions=[ACTION_FORWARD] * n_segments,
                turn_angles_deg=[0.0] * n_segments,
                progress=[(s + 0.5) / n_segments for s in range(n_segments)],
                n_segments=n_segments,
                turn_sequence=",".join(["straight"] * n_segments),
            )
        pts_progress = cum / total
    else:
        pts_progress = filtered_progress
        if pts_progress[0] < -1e-9 or pts_progress[-1] > 1.0 + 1e-9:
            raise ValueError("progress must lie in [0, 1]")
        if np.any(np.diff(pts_progress) < -1e-9):
            raise ValueError("progress must be monotonically non-decreasing")
        pts_progress = np.clip(pts_progress, 0.0, 1.0)

    edge_yaws = np.unwrap(_segment_yaws_image_xy(pts))
    vertex_delta = np.zeros(n, dtype=np.float64)
    if n > 2:
        vertex_delta[1:-1] = _wrap_rad(np.diff(edge_yaws))

    actions: list[str] = []
    turn_angles_deg: list[float] = []
    seg_progress: list[float] = []
    for seg_idx in range(n_segments):
        start_p = seg_idx / float(n_segments)
        end_p = (seg_idx + 1) / float(n_segments)
        mid_p = (seg_idx + 0.5) / float(n_segments)
        seg_progress.append(mid_p)

        if seg_idx == n_segments - 1:
            mask = (pts_progress >= start_p - 1.0e-9) & (pts_progress <= end_p + 1.0e-9)
        else:
            mask = (pts_progress >= start_p - 1.0e-9) & (pts_progress < end_p - 1.0e-9)
        # Exclude endpoints, which have no turn angle.
        mask[0] = False
        mask[-1] = False
        signed_delta = float(np.sum(vertex_delta[mask]))

        if signed_delta > straight_threshold_rad:
            actions.append(ACTION_LEFT)
        elif signed_delta < -straight_threshold_rad:
            actions.append(ACTION_RIGHT)
        else:
            actions.append(ACTION_FORWARD)
        turn_angles_deg.append(math.degrees(signed_delta))

    labels = _labels_from_actions(actions)
    return RouteToActionResult(
        actions=actions,
        turn_angles_deg=turn_angles_deg,
        progress=seg_progress,
        n_segments=n_segments,
        turn_sequence=",".join(labels),
    )


def route_to_actions_by_heading_delta_subsegment(
    polyline: Sequence[Sequence[float]] | np.ndarray,
    n_segments: int,
    sub_segments_per_segment: int = 5,
    progress: Optional[Sequence[float]] = None,
    straight_threshold_rad: float = math.radians(10.0),
) -> tuple[list[str], list[float], list[int]]:
    """Heading-delta plan with finer within-segment time resolution (scheme A+B).

    Like :func:`route_to_actions_by_heading_delta` this integrates signed yaw
    changes whose route-progress vertices fall inside each output segment. The
    difference is temporal granularity: instead of ONE action token per 5s
    diffusion segment (whose yaw delta the pose model then spreads as a single
    linear ramp across the whole segment), this emits ``sub_segments_per_segment``
    tokens per segment (≈1s each at 5s/seg). ``ActionToPoseFromID`` already
    auto-divides a segment's frames evenly across the tokens it is handed, so
    passing the per-segment sub-token list yields ~1s turn resolution without any
    pose-model change.

    Angles are REAL (un-quantized, un-clipped) per sub-segment: the turn
    magnitude is recovered from the route geometry, not from the
    ``clip(|angle|/10, min, max)`` speed rule. Thresholding is still applied for
    the action *label* (``wj``/``wl``/``w``) so sub-segments inherit the smooth
    straight/turn policy.

    Args:
        polyline: route vertices ``(N, 2)`` in image xy.
        n_segments: number of 5s diffusion segments.
        sub_segments_per_segment: sub-tokens per segment (5 -> ~1s at 5s/seg).
        progress: optional per-vertex normalized progress in ``[0, 1]``. If
            ``None``, progress is cumulative arc length.
        straight_threshold_rad: below this |signed angle| a sub-segment is ``w``.

    Returns:
        ``(actions, turn_angles_deg, segment_offsets)`` where ``actions`` and
        ``turn_angles_deg`` are flat across all segments × sub-segments
        (length ``n_segments * sub_segments_per_segment``), and
        ``segment_offsets`` has length ``n_segments + 1`` with
        ``segment_offsets[s]``..``segment_offsets[s+1]`` indexing segment ``s``'s
        sub-tokens. The caller groups sub-tokens per segment and passes each
        group to ``ActionToPoseFromID``.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    if sub_segments_per_segment < 1:
        raise ValueError("sub_segments_per_segment must be >= 1")
    pts, filtered_progress = _normalize_polyline(polyline, progress)
    n = pts.shape[0]
    pts_progress = _progress_for_polyline(pts, filtered_progress)

    edge_yaws = np.unwrap(_segment_yaws_image_xy(pts))
    vertex_delta = np.zeros(n, dtype=np.float64)
    if n > 2:
        vertex_delta[1:-1] = _wrap_rad(np.diff(edge_yaws))

    total_sub = n_segments * sub_segments_per_segment
    actions: list[str] = []
    turn_angles_deg: list[float] = []
    segment_offsets: list[int] = [0]

    for seg_idx in range(n_segments):
        seg_start = seg_idx / float(n_segments)
        seg_end = (seg_idx + 1) / float(n_segments)
        for sub_idx in range(sub_segments_per_segment):
            sub_start = seg_start + (seg_end - seg_start) * (sub_idx / float(sub_segments_per_segment))
            sub_end = seg_start + (seg_end - seg_start) * ((sub_idx + 1) / float(sub_segments_per_segment))
            is_last = (seg_idx == n_segments - 1) and (sub_idx == sub_segments_per_segment - 1)
            if is_last:
                mask = (pts_progress >= sub_start - 1.0e-9) & (pts_progress <= sub_end + 1.0e-9)
            else:
                mask = (pts_progress >= sub_start - 1.0e-9) & (pts_progress < sub_end - 1.0e-9)
            # Exclude polyline endpoints, which carry no turn angle.
            mask[0] = False
            mask[-1] = False
            signed_delta = float(np.sum(vertex_delta[mask]))

            if signed_delta > straight_threshold_rad:
                actions.append(ACTION_LEFT)
            elif signed_delta < -straight_threshold_rad:
                actions.append(ACTION_RIGHT)
            else:
                actions.append(ACTION_FORWARD)
            turn_angles_deg.append(math.degrees(signed_delta))
        segment_offsets.append(len(actions))

    return actions, turn_angles_deg, segment_offsets


def action_speeds_from_turn_angles(
    actions: Sequence[str],
    turn_angles_deg: Sequence[float],
    *,
    straight_speed: float = 4.0,
    min_turn_speed: float = 2.0,
    max_turn_speed: float = 5.2,
) -> list[float]:
    """Match the previous local-yaw quantizer's speed policy."""
    out: list[float] = []
    for action, angle_deg in zip(actions, turn_angles_deg):
        if action == ACTION_FORWARD:
            out.append(float(straight_speed))
        else:
            out.append(float(np.clip(abs(float(angle_deg)) / 10.0, min_turn_speed, max_turn_speed)))
    return out


def _progress_for_polyline(pts: np.ndarray, progress: Optional[Sequence[float]]) -> np.ndarray:
    if progress is None:
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = float(cum[-1])
        if total < 1.0e-9:
            return np.linspace(0.0, 1.0, pts.shape[0], dtype=np.float64)
        return cum / total
    out = np.asarray(progress, dtype=np.float64)
    if out.shape[0] != pts.shape[0]:
        raise ValueError(f"progress length {out.shape[0]} != polyline length {pts.shape[0]}")
    if out[0] < -1.0e-9 or out[-1] > 1.0 + 1.0e-9:
        raise ValueError("progress must lie in [0, 1]")
    if np.any(np.diff(out) < -1.0e-9):
        raise ValueError("progress must be monotonically non-decreasing")
    return np.clip(out, 0.0, 1.0)


def _sample_polyline_at_progress(
    pts: np.ndarray,
    pts_progress: np.ndarray,
    sample_progress: Sequence[float],
) -> np.ndarray:
    sample_progress = np.asarray(sample_progress, dtype=np.float64)
    return np.stack(
        [
            np.interp(sample_progress, pts_progress, pts[:, 0]),
            np.interp(sample_progress, pts_progress, pts[:, 1]),
        ],
        axis=1,
    )


def _target_xy_yaw(
    polyline: Sequence[Sequence[float]] | np.ndarray,
    *,
    progress: Optional[Sequence[float]],
    target_frame_progress: Optional[Sequence[float]],
    total_frames: int,
    route_length_units: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts, filtered_progress = _normalize_polyline(
        np.asarray(polyline, dtype=np.float64), progress
    )
    pts_progress = _progress_for_polyline(pts, filtered_progress)
    if target_frame_progress is None:
        frame_progress = np.linspace(0.0, 1.0, int(total_frames), dtype=np.float64)
    else:
        frame_progress = np.asarray(target_frame_progress, dtype=np.float64)
        if frame_progress.shape[0] != int(total_frames):
            raise ValueError(
                f"target_frame_progress length {frame_progress.shape[0]} != total_frames {total_frames}"
            )
    target_xy_px = _sample_polyline_at_progress(pts, pts_progress, frame_progress)
    route_len_px = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    scale = 1.0 if route_len_px < 1.0e-9 else float(route_length_units) / route_len_px
    target_xy = (target_xy_px - target_xy_px[0]) * scale

    dx = np.gradient(target_xy[:, 0].astype(np.float64))
    dy = np.gradient(target_xy[:, 1].astype(np.float64))
    target_yaw = np.unwrap(np.arctan2(-dx, dy))
    return target_xy, target_yaw, frame_progress, pts_progress


def _action_yaw_delta_rad(action: str, speed: float) -> float:
    if action == ACTION_LEFT:
        return math.radians(float(speed) * 10.0)
    if action == ACTION_RIGHT:
        return -math.radians(float(speed) * 10.0)
    if action == ACTION_FORWARD:
        return 0.0
    raise ValueError(f"Unknown action {action!r}")


def simulate_native_action_plan(
    actions: Sequence[str],
    speeds: Sequence[float],
    *,
    frames_per_segment: int = 81,
    start_xy: Sequence[float] = (0.0, 0.0),
    start_yaw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the existing DreamX native camera primitives in SE(2).

    This mirrors ``ActionToPoseFromID`` semantics closely enough for planning:
    each segment translates along the segment's initial heading while yaw is
    updated linearly. Because inference passes one action into an 81-frame
    segment and keeps the first 81 poses, each segment contributes 80 intervals
    with a native duration denominator of 81.
    """
    if len(actions) != len(speeds):
        raise ValueError(f"actions length {len(actions)} != speeds length {len(speeds)}")
    if frames_per_segment < 2:
        raise ValueError("frames_per_segment must be >= 2")

    intervals = int(frames_per_segment) - 1
    native_duration = float(frames_per_segment)
    xy = np.asarray(start_xy, dtype=np.float64).reshape(2)
    yaw = float(start_yaw)
    xy_frames = [xy.copy()]
    yaw_frames = [yaw]

    for action, speed in zip(actions, speeds):
        speed = float(speed)
        forward = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        step_xy = forward * (speed / native_duration)
        step_yaw = _action_yaw_delta_rad(action, speed) / native_duration
        for _ in range(intervals):
            xy = xy + step_xy
            yaw = yaw + step_yaw
            xy_frames.append(xy.copy())
            yaw_frames.append(yaw)

    return np.asarray(xy_frames, dtype=np.float64), np.asarray(yaw_frames, dtype=np.float64)


def _event_frame_and_progress(event: Any) -> tuple[int | None, float | None]:
    if isinstance(event, dict):
        frame = event.get("target_frame")
        progress = event.get("route_progress", event.get("progress"))
    else:
        frame = getattr(event, "target_frame", None)
        progress = getattr(event, "route_progress", None)
        if progress is None:
            progress = getattr(event, "target_progress", None)
    return (None if frame is None else int(frame), None if progress is None else float(progress))


def _nearest_route_progress(points_xy: np.ndarray, target_xy: np.ndarray, frame_progress: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        _, indices = cKDTree(target_xy).query(points_xy)
        return frame_progress[np.asarray(indices, dtype=np.int64)]
    except Exception:
        out = []
        for point in points_xy:
            idx = int(np.argmin(np.sum((target_xy - point[None, :]) ** 2, axis=1)))
            out.append(float(frame_progress[idx]))
        return np.asarray(out, dtype=np.float64)


def evaluate_native_action_plan(
    actions: Sequence[str],
    speeds: Sequence[float],
    polyline: Sequence[Sequence[float]] | np.ndarray,
    *,
    progress: Optional[Sequence[float]] = None,
    target_frame_progress: Optional[Sequence[float]] = None,
    events: Optional[Sequence[Any]] = None,
    frames_per_segment: int = 81,
    route_length_units: Optional[float] = None,
    position_weight: float = 5.0,
    yaw_weight: float = 1.0,
    event_weight: float = 10.0,
    switch_weight: float = 0.02,
) -> dict[str, Any]:
    """Score a native action plan against a target vectorized route."""
    n_segments = len(actions)
    total_frames = n_segments * (int(frames_per_segment) - 1) + 1
    straight_ref = float(np.mean([s for a, s in zip(actions, speeds) if a == ACTION_FORWARD] or [4.0]))
    if route_length_units is None:
        route_length_units = n_segments * straight_ref * (int(frames_per_segment) - 1) / float(frames_per_segment)

    target_xy, target_yaw, frame_progress, _ = _target_xy_yaw(
        polyline,
        progress=progress,
        target_frame_progress=target_frame_progress,
        total_frames=total_frames,
        route_length_units=float(route_length_units),
    )
    sim_xy, sim_yaw = simulate_native_action_plan(
        actions,
        speeds,
        frames_per_segment=frames_per_segment,
        start_xy=target_xy[0],
        start_yaw=float(target_yaw[0]),
    )
    sim_yaw = np.unwrap(sim_yaw)

    pos_err = np.linalg.norm(sim_xy - target_xy, axis=1)
    yaw_err = np.asarray(_wrap_rad(sim_yaw - target_yaw), dtype=np.float64)
    pos_rmse = float(np.sqrt(np.mean(pos_err**2)))
    yaw_rmse_rad = float(np.sqrt(np.mean(yaw_err**2)))
    route_length_units = float(max(route_length_units, 1.0e-9))
    position_mse_norm = float(np.mean(pos_err**2) / (route_length_units**2))
    yaw_mse_norm = float(np.mean(yaw_err**2) / (math.pi**2))

    event_records = []
    event_errors = []
    if events:
        for event in events:
            frame, event_progress = _event_frame_and_progress(event)
            if frame is None or event_progress is None or frame < 0 or frame >= len(sim_xy):
                continue
            predicted = float(_nearest_route_progress(sim_xy[frame : frame + 1], target_xy, frame_progress)[0])
            err = predicted - float(event_progress)
            event_errors.append(err)
            event_type = event.get("type", "event") if isinstance(event, dict) else getattr(event, "event_type", "event")
            event_records.append(
                {
                    "type": str(event_type),
                    "target_frame": int(frame),
                    "target_progress": round(float(event_progress), 6),
                    "predicted_progress": round(predicted, 6),
                    "error": round(float(err), 6),
                }
            )
    event_mse = float(np.mean(np.asarray(event_errors, dtype=np.float64) ** 2)) if event_errors else 0.0
    event_rmse = float(math.sqrt(event_mse))
    switch_count = int(sum(1 for prev, cur in zip(actions[:-1], actions[1:]) if prev != cur))
    cost = (
        float(position_weight) * position_mse_norm
        + float(yaw_weight) * yaw_mse_norm
        + float(event_weight) * event_mse
        + float(switch_weight) * switch_count
    )
    final_pos_error = float(np.linalg.norm(sim_xy[-1] - target_xy[-1]))
    final_yaw_error_deg = float(abs(math.degrees(float(_wrap_rad(sim_yaw[-1] - target_yaw[-1])))))

    return {
        "cost": round(float(cost), 8),
        "position_rmse_units": round(pos_rmse, 6),
        "position_rmse_normalized": round(pos_rmse / route_length_units, 8),
        "yaw_rmse_deg": round(math.degrees(yaw_rmse_rad), 6),
        "event_progress_rmse": round(event_rmse, 8),
        "switch_count": switch_count,
        "final_position_error_units": round(final_pos_error, 6),
        "final_yaw_error_deg": round(final_yaw_error_deg, 6),
        "route_length_units": round(route_length_units, 6),
        "events": event_records,
        "_trajectory_xy": sim_xy,
        "_trajectory_yaw": sim_yaw,
        "_target_xy": target_xy,
        "_target_yaw": target_yaw,
        "_frame_progress": frame_progress,
    }


def fit_actions_global_motion_primitives(
    polyline: Sequence[Sequence[float]] | np.ndarray,
    n_segments: int,
    *,
    progress: Optional[Sequence[float]] = None,
    target_frame_progress: Optional[Sequence[float]] = None,
    events: Optional[Sequence[Any]] = None,
    frames_per_segment: int = 81,
    straight_speed: float = 4.0,
    straight_speeds: Optional[Sequence[float]] = None,
    turn_speeds: Optional[Sequence[float]] = None,
    route_length_units: Optional[float] = None,
    beam_width: int = 96,
    position_weight: float = 5.0,
    yaw_weight: float = 1.0,
    event_weight: float = 10.0,
    switch_weight: float = 0.02,
) -> RouteToActionResult:
    """Fit a global native ``w``/``wj``/``wl`` plan with beam search.

    Unlike local yaw quantization, this scores complete motion histories:
    position error, tangent-yaw error, event-anchor progress error, and action
    switching are accumulated over the whole route.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    if frames_per_segment < 2:
        raise ValueError("frames_per_segment must be >= 2")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    intervals = int(frames_per_segment) - 1
    total_frames = int(n_segments) * intervals + 1
    if route_length_units is None:
        route_length_units = int(n_segments) * float(straight_speed) * intervals / float(frames_per_segment)

    target_xy, target_yaw, frame_progress, _ = _target_xy_yaw(
        polyline,
        progress=progress,
        target_frame_progress=target_frame_progress,
        total_frames=total_frames,
        route_length_units=float(route_length_units),
    )
    event_by_segment: dict[int, list[tuple[int, float]]] = {}
    if events:
        for event in events:
            frame, event_progress = _event_frame_and_progress(event)
            if frame is None or event_progress is None or frame <= 0 or frame >= total_frames:
                continue
            segment = min(int(n_segments) - 1, (frame - 1) // intervals)
            event_by_segment.setdefault(segment, []).append((int(frame), float(event_progress)))

    straight_candidates = [float(x) for x in (straight_speeds or [straight_speed])]
    turn_candidates = [float(x) for x in (turn_speeds or [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.2])]
    candidates = [(ACTION_FORWARD, s) for s in straight_candidates]
    candidates.extend((ACTION_LEFT, s) for s in turn_candidates)
    candidates.extend((ACTION_RIGHT, s) for s in turn_candidates)

    beams = [
        {
            "xy": target_xy[0].copy(),
            "yaw": float(target_yaw[0]),
            "actions": [],
            "speeds": [],
            "cost": 0.0,
            "trajectory_xy": [target_xy[0].copy()],
            "trajectory_yaw": [float(target_yaw[0])],
        }
    ]

    route_length_sq = float(max(route_length_units, 1.0e-9)) ** 2
    for segment in range(int(n_segments)):
        start = segment * intervals
        end = start + intervals
        target_seg_xy = target_xy[start + 1 : end + 1]
        target_seg_yaw = target_yaw[start + 1 : end + 1]
        next_beams = []
        for beam in beams:
            prev_action = beam["actions"][-1] if beam["actions"] else None
            for action, speed in candidates:
                seg_xy, seg_yaw = simulate_native_action_plan(
                    [action],
                    [speed],
                    frames_per_segment=frames_per_segment,
                    start_xy=beam["xy"],
                    start_yaw=float(beam["yaw"]),
                )
                sim_xy = seg_xy[1:]
                sim_yaw = np.unwrap(seg_yaw[1:])
                pos_err = np.linalg.norm(sim_xy - target_seg_xy, axis=1)
                yaw_err = np.asarray(_wrap_rad(sim_yaw - target_seg_yaw), dtype=np.float64)
                pos_cost = float(np.mean(pos_err**2) / route_length_sq)
                yaw_cost = float(np.mean(yaw_err**2) / (math.pi**2))
                event_cost = 0.0
                if event_by_segment.get(segment):
                    predicted_progress = _nearest_route_progress(
                        np.asarray([sim_xy[frame - start - 1] for frame, _ in event_by_segment[segment]]),
                        target_xy,
                        frame_progress,
                    )
                    target_progress = np.asarray([p for _, p in event_by_segment[segment]], dtype=np.float64)
                    event_cost = float(np.mean((predicted_progress - target_progress) ** 2))
                switch_cost = float(prev_action is not None and prev_action != action)
                total_cost = (
                    float(beam["cost"])
                    + float(position_weight) * pos_cost
                    + float(yaw_weight) * yaw_cost
                    + float(event_weight) * event_cost
                    + float(switch_weight) * switch_cost
                )
                next_beams.append(
                    {
                        "xy": sim_xy[-1].copy(),
                        "yaw": float(sim_yaw[-1]),
                        "actions": beam["actions"] + [action],
                        "speeds": beam["speeds"] + [float(speed)],
                        "cost": total_cost,
                        "trajectory_xy": beam["trajectory_xy"] + [p.copy() for p in sim_xy],
                        "trajectory_yaw": beam["trajectory_yaw"] + [float(y) for y in sim_yaw],
                    }
                )
        next_beams.sort(key=lambda item: float(item["cost"]))
        beams = next_beams[: int(beam_width)]

    best = beams[0]
    metrics = evaluate_native_action_plan(
        best["actions"],
        best["speeds"],
        polyline,
        progress=progress,
        target_frame_progress=target_frame_progress,
        events=events,
        frames_per_segment=frames_per_segment,
        route_length_units=float(route_length_units),
        position_weight=position_weight,
        yaw_weight=yaw_weight,
        event_weight=event_weight,
        switch_weight=switch_weight,
    )
    trajectory_xy = metrics.pop("_trajectory_xy")
    trajectory_yaw = metrics.pop("_trajectory_yaw")
    metrics.pop("_target_xy", None)
    metrics.pop("_target_yaw", None)
    metrics.pop("_frame_progress", None)

    turn_angles = [
        math.degrees(_action_yaw_delta_rad(action, speed))
        for action, speed in zip(best["actions"], best["speeds"])
    ]
    labels = _labels_from_actions(best["actions"])
    return RouteToActionResult(
        actions=list(best["actions"]),
        turn_angles_deg=turn_angles,
        progress=[(idx + 0.5) / float(n_segments) for idx in range(int(n_segments))],
        n_segments=int(n_segments),
        turn_sequence=",".join(labels),
        action_speed_list=[float(s) for s in best["speeds"]],
        metrics=metrics,
        trajectory_xy=trajectory_xy,
        trajectory_yaw=trajectory_yaw,
    )


def load_route_polyline_from_poses_csv(path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a route polyline (and optional progress) from a MapCtrl poses CSV.

    The CSV is the same file consumed by the MapCtrl wrapper
    (``experiments/mapctrl_benchmark/mapctrl_inference_wrapper.py``) and has
    columns ``frame, x, y, theta_rad, ...``. The ``(x, y)`` columns form the
    ordered route polyline; the integer ``frame`` column, when present, is
    used to derive normalized progress so segment midpoints align with the
    route even when samples are unevenly spaced.

    Returns:
        (polyline ``(N, 2)``, progress ``(N,)`` or ``None``).
    """
    import csv

    rows: list[tuple[float, float]] = []
    frames: list[int] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError(f"poses CSV {path} missing required 'x'/'y' columns")
        for row in reader:
            rows.append((float(row["x"]), float(row["y"])))
            f = row.get("frame")
            frames.append(int(float(f)) if f not in (None, "") else -1)
    if not rows:
        raise ValueError(f"poses CSV {path} has no rows")
    polyline = np.asarray(rows, dtype=np.float64)
    progress: Optional[np.ndarray] = None
    if frames and frames[0] >= 0 and frames[-1] > frames[0]:
        progress = np.asarray(frames, dtype=np.float64)
        progress = (progress - progress[0]) / (progress[-1] - progress[0])
    return polyline, progress
