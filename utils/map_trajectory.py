"""Map-conditioned continuous camera trajectory construction.

This module turns an ordered 2D route polyline into a deterministic camera
trajectory for DreamX PRoPE conditioning. The map supplies the global route;
the local state is the smooth camera state ``(x, y, progress, yaw, speed)``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from utils.route_vectorizer import (
    RoutePolyline,
    RouteProjection,
    project_point_to_route,
    sample_route,
    tangent_at_progress,
)


@dataclass(frozen=True)
class TrajectorySettings:
    fps: int = 16
    duration_seconds: float = 100.0
    segment_seconds: float = 5.0
    camera_height_m: float = 1.55
    pitch_deg: float = -5.0
    route_length_m: float | None = 244.3
    yaw_smooth_window: int = 17
    max_yaw_rate_deg_s: float = 60.0

    @property
    def frames_per_segment(self) -> int:
        return int(round(self.segment_seconds * self.fps)) + 1

    @property
    def frame_step_per_segment(self) -> int:
        return self.frames_per_segment - 1

    @property
    def total_intervals(self) -> int:
        return int(round(self.duration_seconds * self.fps))

    @property
    def total_frames(self) -> int:
        return self.total_intervals + 1

    @property
    def segment_count(self) -> int:
        return int(math.ceil(self.total_intervals / float(self.frame_step_per_segment)))


@dataclass(frozen=True)
class ProjectedEvent:
    event_type: str
    target_progress: float
    route_progress: float
    target_frame: int
    projected_xy: tuple[float, float]
    route_anchor_xy: tuple[float, float] | None
    distance_px: float
    side: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "type": self.event_type,
            "target_progress": round(float(self.target_progress), 6),
            "route_progress": round(float(self.route_progress), 6),
            "target_frame": int(self.target_frame),
            "projected_xy": [round(float(self.projected_xy[0]), 3), round(float(self.projected_xy[1]), 3)],
            "distance_px": round(float(self.distance_px), 3),
        }
        if self.route_anchor_xy is not None:
            out["route_anchor_xy"] = [round(float(self.route_anchor_xy[0]), 3), round(float(self.route_anchor_xy[1]), 3)]
        if self.side:
            out["side"] = self.side
        return out


@dataclass(frozen=True)
class MapTrajectory:
    settings: TrajectorySettings
    route: RoutePolyline
    events: list[ProjectedEvent]
    video_progress: np.ndarray
    route_progress: np.ndarray
    route_xy: np.ndarray
    states: np.ndarray
    centers_w: np.ndarray
    w2c_abs: np.ndarray
    yaw_rate_deg_s: np.ndarray
    speed_mps: np.ndarray
    meters_per_pixel: float
    progress_map_video: np.ndarray
    progress_map_route: np.ndarray


def load_metadata(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def settings_from_metadata(metadata: dict[str, Any]) -> TrajectorySettings:
    trajectory = metadata.get("trajectory", {})
    controller = metadata.get("controller", {})
    return TrajectorySettings(
        fps=int(trajectory.get("fps", 16)),
        duration_seconds=float(trajectory.get("duration_seconds", trajectory.get("duration_s", 100.0))),
        segment_seconds=float(trajectory.get("segment_seconds", trajectory.get("segment_s", 5.0))),
        camera_height_m=float(trajectory.get("camera_height_m", 1.55)),
        pitch_deg=float(trajectory.get("pitch_deg", -5.0)),
        route_length_m=(
            None if trajectory.get("route_length_m") is None else float(trajectory.get("route_length_m", 244.3))
        ),
        yaw_smooth_window=int(controller.get("yaw_smooth_window", 17)),
        max_yaw_rate_deg_s=float(controller.get("max_yaw_rate_deg_s", 60.0)),
    )


def _wrap_rad(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _rate_limit_yaw(yaw_raw: np.ndarray, dt: float, max_yaw_rate_deg_s: float, smooth_window: int) -> np.ndarray:
    unwrapped = np.unwrap(np.asarray(yaw_raw, dtype=np.float64))
    smoothed = _moving_average_1d(unwrapped, int(smooth_window))
    if max_yaw_rate_deg_s <= 0.0:
        return smoothed

    max_step = math.radians(float(max_yaw_rate_deg_s)) * float(dt)
    limited = np.empty_like(smoothed)
    limited[0] = smoothed[0]
    for idx in range(1, len(smoothed)):
        delta = float(_wrap_rad(smoothed[idx] - limited[idx - 1]))
        limited[idx] = limited[idx - 1] + float(np.clip(delta, -max_step, max_step))
    return limited


def _euler_to_quaternion(pitch_deg: float, yaw_deg: float, roll_deg: float) -> list[float]:
    pitch, yaw, roll = np.radians([pitch_deg, yaw_deg, roll_deg])
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return [
        cy * cp * cr + sy * sp * sr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
        cy * cp * sr - sy * sp * cr,
    ]


def _quaternion_to_rotation_matrix(q: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.asarray(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def _w2c_from_center_yaw_pitch(center_w: np.ndarray, yaw_rad: float, pitch_deg: float) -> np.ndarray:
    yaw_deg = math.degrees(float(yaw_rad))
    rot = _quaternion_to_rotation_matrix(_euler_to_quaternion(float(pitch_deg), yaw_deg, 0.0))
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = -rot @ np.asarray(center_w, dtype=np.float64)
    return mat


def default_prope_K(num_frames: int) -> np.ndarray:
    """Match the normalized PRoPE intrinsics used by GetPoseEmbedsFromPosesPrope."""
    fx_norm = 969.6969696969696 / (960.0 * 2.0)
    fy_norm = 969.6969696969696 / (540.0 * 2.0)
    K = np.zeros((int(num_frames), 3, 3), dtype=np.float32)
    K[:, 0, 0] = fx_norm
    K[:, 1, 1] = fy_norm
    K[:, 2, 2] = 1.0
    return K


def _anchor_to_image_xy(anchor: Sequence[float], image_size: tuple[int, int]) -> tuple[float, float]:
    if len(anchor) != 2:
        raise ValueError(f"route_anchor must be [x, y], got {anchor}")
    x = float(anchor[0])
    y = float(anchor[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= float(image_size[0])
        y *= float(image_size[1])
    return x, y


def project_route_events(
    route: RoutePolyline,
    events: Sequence[dict[str, Any]],
    *,
    total_intervals: int,
) -> list[ProjectedEvent]:
    projected: list[ProjectedEvent] = []
    for event in events:
        target_progress = float(event.get("progress", 0.0))
        if not 0.0 <= target_progress <= 1.0:
            raise ValueError(f"Event progress must be in [0, 1], got {target_progress}")

        route_anchor_xy: tuple[float, float] | None
        if event.get("route_anchor") is not None:
            route_anchor_xy = _anchor_to_image_xy(event["route_anchor"], route.image_size)
            projection = project_point_to_route(route, route_anchor_xy)
        else:
            route_anchor_xy = None
            xy = sample_route(route, target_progress)
            projection = RouteProjection(
                point_xy=(float(xy[0]), float(xy[1])),
                progress=target_progress,
                distance_px=0.0,
                segment_index=0,
            )
        projected.append(
            ProjectedEvent(
                event_type=str(event.get("type", "event")),
                target_progress=target_progress,
                route_progress=float(projection.progress),
                target_frame=int(round(target_progress * int(total_intervals))),
                projected_xy=projection.point_xy,
                route_anchor_xy=route_anchor_xy,
                distance_px=float(projection.distance_px),
                side=event.get("side"),
            )
        )

    projected.sort(key=lambda item: item.target_progress)
    return projected


def _progress_mapping(events: Sequence[ProjectedEvent]) -> tuple[np.ndarray, np.ndarray]:
    video_points = [0.0]
    route_points = [0.0]
    last_video = 0.0
    last_route = 0.0
    for event in events:
        if event.target_progress <= last_video + 1.0e-9:
            raise ValueError("Event target progress values must be strictly increasing")
        if event.route_progress <= last_route + 1.0e-9:
            raise ValueError(
                "Projected event route progress values must be strictly increasing; "
                f"{event.event_type} projected to {event.route_progress:.4f} after {last_route:.4f}"
            )
        video_points.append(float(event.target_progress))
        route_points.append(float(event.route_progress))
        last_video = float(event.target_progress)
        last_route = float(event.route_progress)
    if last_video >= 1.0 - 1.0e-9 or last_route >= 1.0 - 1.0e-9:
        raise ValueError("Route events must leave room for the final endpoint at progress 1.0")
    video_points.append(1.0)
    route_points.append(1.0)
    return np.asarray(video_points, dtype=np.float64), np.asarray(route_points, dtype=np.float64)


def build_map_trajectory(
    route: RoutePolyline,
    metadata: dict[str, Any],
    *,
    settings: TrajectorySettings | None = None,
) -> MapTrajectory:
    settings = settings or settings_from_metadata(metadata)
    if settings.frames_per_segment < 2:
        raise ValueError("frames_per_segment must be at least 2")

    events = project_route_events(
        route,
        metadata.get("route_events", []),
        total_intervals=settings.total_intervals,
    )
    progress_video_knots, progress_route_knots = _progress_mapping(events)

    video_progress = np.linspace(0.0, 1.0, settings.total_frames, dtype=np.float64)
    route_progress = np.interp(video_progress, progress_video_knots, progress_route_knots)
    route_progress = np.maximum.accumulate(np.clip(route_progress, 0.0, 1.0))
    route_xy = sample_route(route, route_progress)

    route_length_m = float(settings.route_length_m) if settings.route_length_m is not None else float(route.length_px)
    meters_per_pixel = route_length_m / float(route.length_px)
    start_xy = route_xy[0]
    x_m = (route_xy[:, 0] - float(start_xy[0])) * meters_per_pixel
    y_m = (route_xy[:, 1] - float(start_xy[1])) * meters_per_pixel

    dx = np.gradient(route_xy[:, 0].astype(np.float64))
    dy = np.gradient(route_xy[:, 1].astype(np.float64))
    yaw_raw = np.arctan2(-dx, dy)
    dt = 1.0 / float(settings.fps)
    yaw = _rate_limit_yaw(
        yaw_raw,
        dt=dt,
        max_yaw_rate_deg_s=settings.max_yaw_rate_deg_s,
        smooth_window=settings.yaw_smooth_window,
    )

    centers_w = np.stack(
        [x_m, np.full_like(x_m, float(settings.camera_height_m)), y_m],
        axis=1,
    )
    w2c_abs = np.stack(
        [_w2c_from_center_yaw_pitch(center, yaw_i, settings.pitch_deg) for center, yaw_i in zip(centers_w, yaw)],
        axis=0,
    )

    dist_m = np.zeros(settings.total_frames, dtype=np.float64)
    dist_m[1:] = np.linalg.norm(np.diff(centers_w[:, [0, 2]], axis=0), axis=1)
    speed_mps = np.gradient(route_progress) * route_length_m / dt
    speed_mps[0] = dist_m[1] / dt if settings.total_frames > 1 else 0.0
    speed_mps[-1] = dist_m[-1] / dt if settings.total_frames > 1 else 0.0
    yaw_rate = np.zeros(settings.total_frames, dtype=np.float64)
    yaw_rate[1:] = np.diff(np.unwrap(yaw)) / dt * 180.0 / math.pi

    states = np.stack([x_m, y_m, route_progress, _wrap_rad(yaw), speed_mps], axis=1)
    return MapTrajectory(
        settings=settings,
        route=route,
        events=events,
        video_progress=video_progress,
        route_progress=route_progress,
        route_xy=route_xy,
        states=states.astype(np.float64),
        centers_w=centers_w.astype(np.float64),
        w2c_abs=w2c_abs.astype(np.float64),
        yaw_rate_deg_s=yaw_rate.astype(np.float64),
        speed_mps=speed_mps.astype(np.float64),
        meters_per_pixel=float(meters_per_pixel),
        progress_map_video=progress_video_knots,
        progress_map_route=progress_route_knots,
    )


def latent_frame_indices_for_segment(traj: MapTrajectory, segment_index: int) -> np.ndarray:
    settings = traj.settings
    if segment_index < 0 or segment_index >= settings.segment_count:
        raise IndexError(f"segment_index {segment_index} outside [0, {settings.segment_count})")
    start = segment_index * settings.frame_step_per_segment
    end = min(start + settings.frame_step_per_segment, settings.total_frames - 1)
    segment_frames = end - start + 1
    latent_count = 1 + (segment_frames - 1) // 4
    return np.rint(np.linspace(start, end, latent_count)).astype(np.int64)


def segment_camera_condition_arrays(traj: MapTrajectory, segment_index: int) -> dict[str, np.ndarray]:
    indices = latent_frame_indices_for_segment(traj, segment_index)
    abs_w2c = traj.w2c_abs[indices]
    first_c2w = np.linalg.inv(abs_w2c[0])
    viewmats = np.einsum("tij,jk->tik", abs_w2c, first_c2w).astype(np.float32)
    return {
        "viewmats": viewmats,
        "K": default_prope_K(len(indices)),
        "frame_indices": indices.astype(np.int64),
    }


def all_segment_camera_condition_arrays(traj: MapTrajectory) -> dict[str, np.ndarray]:
    segments = [segment_camera_condition_arrays(traj, idx) for idx in range(traj.settings.segment_count)]
    return {
        "viewmats": np.stack([seg["viewmats"] for seg in segments], axis=0),
        "K": np.stack([seg["K"] for seg in segments], axis=0),
        "frame_indices": np.stack([seg["frame_indices"] for seg in segments], axis=0),
    }


def validate_se3(mats: np.ndarray, *, atol: float = 1.0e-4) -> dict[str, float]:
    mats = np.asarray(mats, dtype=np.float64)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"Expected SE(3) matrices with shape (N, 4, 4), got {mats.shape}")
    R = mats[:, :3, :3]
    should_be_I = np.einsum("tji,tjk->tik", R, R)
    eye = np.eye(3, dtype=np.float64)[None]
    orth_err = float(np.max(np.abs(should_be_I - eye)))
    det = np.linalg.det(R)
    det_err = float(np.max(np.abs(det - 1.0)))
    bottom = mats[:, 3, :]
    bottom_err = float(np.max(np.abs(bottom - np.asarray([0.0, 0.0, 0.0, 1.0]))))
    if orth_err > atol or det_err > atol or bottom_err > atol:
        raise ValueError(
            f"Invalid SE(3): orth_err={orth_err:.6g}, det_err={det_err:.6g}, bottom_err={bottom_err:.6g}"
        )
    return {
        "orthonormal_error": orth_err,
        "det_error": det_err,
        "bottom_row_error": bottom_err,
    }


def build_segment_manifest(traj: MapTrajectory) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    settings = traj.settings
    for idx in range(settings.segment_count):
        start = idx * settings.frame_step_per_segment
        end = min(start + settings.frame_step_per_segment, settings.total_frames - 1)
        latent_indices = latent_frame_indices_for_segment(traj, idx)
        is_last = idx == settings.segment_count - 1
        events = [
            event.to_dict()
            for event in traj.events
            if start <= event.target_frame and (event.target_frame < end or is_last)
        ]
        out.append(
            {
                "segment": idx,
                "start_frame": int(start),
                "end_frame": int(end),
                "start_time_s": round(start / float(settings.fps), 6),
                "end_time_s": round(end / float(settings.fps), 6),
                "frames": int(end - start + 1),
                "latent_frames": int(len(latent_indices)),
                "latent_frame_indices": [int(x) for x in latent_indices],
                "start_route_progress": round(float(traj.route_progress[start]), 6),
                "end_route_progress": round(float(traj.route_progress[end]), 6),
                "events": events,
            }
        )
    return out


def trajectory_summary(traj: MapTrajectory) -> dict[str, Any]:
    se3_stats = validate_se3(traj.w2c_abs)
    segment_conditions = all_segment_camera_condition_arrays(traj)
    first_segment_se3 = validate_se3(segment_conditions["viewmats"].reshape(-1, 4, 4))
    settings = traj.settings
    return {
        "fps": int(settings.fps),
        "duration_seconds": float(settings.duration_seconds),
        "total_frames": int(settings.total_frames),
        "unique_frames": int(settings.total_frames),
        "frames_per_segment": int(settings.frames_per_segment),
        "segment_count": int(settings.segment_count),
        "latent_frames_per_segment": int(segment_conditions["viewmats"].shape[1]),
        "route_length_px": round(float(traj.route.length_px), 3),
        "route_length_m": round(float(traj.route.length_px * traj.meters_per_pixel), 3),
        "meters_per_pixel": round(float(traj.meters_per_pixel), 8),
        "speed_mps": {
            "min": round(float(np.min(traj.speed_mps)), 4),
            "mean": round(float(np.mean(traj.speed_mps)), 4),
            "max": round(float(np.max(traj.speed_mps)), 4),
        },
        "yaw_rate_deg_s": {
            "min": round(float(np.min(traj.yaw_rate_deg_s)), 4),
            "max": round(float(np.max(traj.yaw_rate_deg_s)), 4),
        },
        "start_xy_px": [round(float(x), 3) for x in traj.route_xy[0]],
        "goal_xy_px": [round(float(x), 3) for x in traj.route_xy[-1]],
        "events": [event.to_dict() for event in traj.events],
        "progress_mapping": [
            {
                "video_progress": round(float(video_p), 6),
                "route_progress": round(float(route_p), 6),
            }
            for video_p, route_p in zip(traj.progress_map_video, traj.progress_map_route)
        ],
        "se3": {
            "absolute_w2c": se3_stats,
            "segment_relative_viewmats": first_segment_se3,
        },
        "state_columns": ["x_m", "y_m", "progress", "yaw_rad", "speed_mps"],
        "camera_center_columns": ["world_x_m", "world_y_height_m", "world_z_m"],
    }


def write_camera_poses_w2c(path: str | Path, traj: MapTrajectory) -> None:
    path = Path(path)
    lines = ["# frame fx fy cx cy 0 0 w2c_3x4(row-major)\n"]
    intrinsic = [0.8, 0.5, 0.5, 0.5]
    for frame, mat in enumerate(traj.w2c_abs):
        row = [frame] + intrinsic + [0, 0] + mat[:3, :].reshape(-1).tolist()
        lines.append(" ".join(str(float(x)) if idx > 0 else str(int(x)) for idx, x in enumerate(row)) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_debug_overlay(map_path: str | Path, output_path: str | Path, traj: MapTrajectory) -> None:
    image = Image.open(map_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    route_points = [(float(x), float(y)) for x, y in traj.route.points_xy]
    if len(route_points) >= 2:
        draw.line(route_points, fill=(255, 220, 0), width=4)
    traj_points = [(float(x), float(y)) for x, y in traj.route_xy]
    if len(traj_points) >= 2:
        draw.line(traj_points, fill=(0, 210, 255), width=3)

    def circle(x: float, y: float, radius: int, fill: tuple[int, int, int]) -> None:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=(0, 0, 0), width=2)

    circle(float(traj.route_xy[0, 0]), float(traj.route_xy[0, 1]), 9, (0, 220, 80))
    circle(float(traj.route_xy[-1, 0]), float(traj.route_xy[-1, 1]), 9, (80, 150, 255))
    draw.text((float(traj.route_xy[0, 0]) + 12, float(traj.route_xy[0, 1]) - 10), "start", fill=(0, 0, 0))
    draw.text((float(traj.route_xy[-1, 0]) + 12, float(traj.route_xy[-1, 1]) - 10), "goal", fill=(0, 0, 0))

    for event in traj.events:
        x, y = event.projected_xy
        circle(float(x), float(y), 8, (255, 255, 255))
        label = f"{event.event_type}@{event.target_progress:.2f}"
        draw.text((float(x) + 12, float(y) - 10), label, fill=(0, 0, 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_navigation_outputs(output_dir: str | Path, map_path: str | Path, traj: MapTrajectory) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_path = output_dir / "trajectory_debug.png"
    summary_path = output_dir / "trajectory_summary.json"
    manifest_path = output_dir / "segment_manifest.json"
    poses_path = output_dir / "camera_poses_w2c.txt"
    conditions_path = output_dir / "segment_camera_conditions.npz"

    write_debug_overlay(map_path, debug_path, traj)
    write_camera_poses_w2c(poses_path, traj)
    summary_path.write_text(json.dumps(trajectory_summary(traj), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(build_segment_manifest(traj), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    segment_conditions = all_segment_camera_condition_arrays(traj)
    np.savez_compressed(
        conditions_path,
        viewmats=segment_conditions["viewmats"],
        K=segment_conditions["K"],
        frame_indices=segment_conditions["frame_indices"],
        states=traj.states.astype(np.float32),
        centers_w=traj.centers_w.astype(np.float32),
        route_xy=traj.route_xy.astype(np.float32),
        route_progress=traj.route_progress.astype(np.float32),
    )
    return {
        "debug": debug_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "poses": poses_path,
        "conditions": conditions_path,
    }
