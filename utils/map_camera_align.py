"""World-frame camera/map alignment for training-free PSCAR.

This module is the shared geometric infrastructure that
``GEOMAP_TRAININGFREE_PROPOSAL.md`` flags as the "first engineering risk": a
single, authoritative mapping from map-plane semantics to image-token
occupancy via the camera projection ``K @ T_world->cam @ [x_w, y_w, z_ground, 1]``.

It is intentionally side-effect free and torch/CV-stack free (numpy only) so it
can be unit-tested without loading the DiT. Two pieces are provided:

1. ``default_K_normalized`` -- the authoritative PRoPE intrinsics. The model is
   trained with ``fx = 969.69 / (960*2) ~ 0.505``, ``fy = 969.69 / (540*2) ~
   0.898``, ``cx = cy = 0.5`` (pixel coordinates normalized to [0, 1]); see
   ``inference_ar_forcing.py:140-149`` and ``utils/map_trajectory.py:196``. The
   ``[0.8, 0.5, 0.5, 0.5]`` literal at ``map_trajectory.py:492`` is only the
   ``camera_poses_w2c.txt`` row format, NOT what the model consumes, so it is
   not used here.

2. ``bev_ego_crop_to_image_occupancy`` -- the projection kernel. The existing
   ``utils/map_layout_condition.sample_ego_aligned_masks`` already produces a
   BEV ego crop (a top-down sampling of the map plane in camera
   forward/lateral metres). This kernel re-projects that BEV grid onto the
   image-token grid so a frozen text cross-attention can be spatially routed
   per PSCAR, without any training.

The BEV crop geometry (``LayoutProjectionSettings``) samples the ground plane
at ``forward`` in ``[lookahead_px, -rear_px]`` (rows, far -> near) and
``lateral`` in ``[-lateral_extent_px, +lateral_extent_px]`` (cols, left ->
right), in map pixels. Combined with ``meters_per_pixel`` this becomes metric
ground-plane points ``(x_forward_m, y_lateral_m)`` relative to the camera.
With a camera at height ``h`` pitched down by ``pitch_deg``, a ground point
``(x, y, 0)`` projects to normalized pixels and hence to token ``(row, col)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Authoritative intrinsics
# ---------------------------------------------------------------------------

# Focal length in pixels of the Wan2.x 5B-Cam training config, before the
# /1920, /1080 normalization. Kept as a named constant so the projection kernel
# and any future re-derivation share one source of truth.
_FOCAL_PX = 969.6969696969696
_TRAIN_W_PX = 1920.0
_TRAIN_H_PX = 1080.0


@dataclass(frozen=True)
class NormalizedIntrinsics:
    """Pinhole intrinsics with pixel coordinates normalized to [0, 1].

    ``u = fx * X_cam + cx``, ``v = fy * Y_cam + cy`` where ``(u, v)`` are in
    [0, 1]. This matches the PRoPE conditioning convention used by the model.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def default_K_normalized() -> NormalizedIntrinsics:
    """The intrinsics the DiT was trained with (normalized pixel units).

    Equals ``inference_ar_forcing.py:140-149`` and
    ``map_trajectory.default_prope_K``. Use this for every map->image
    projection so PSCAR occupancy lines up with the camera branch the model
    actually consumes.
    """
    return NormalizedIntrinsics(
        fx=_FOCAL_PX / _TRAIN_W_PX,
        fy=_FOCAL_PX / _TRAIN_H_PX,
        cx=0.5,
        cy=0.5,
    )


# ---------------------------------------------------------------------------
# SE(2) (map-pixel) <-> SE(3) (metric world) bridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapToWorldCalibration:
    """Rigid transform from map-pixel frame to metric world frame.

    The map-pixel frame has its origin at the route start pixel and y growing
    downward (image convention). The metric world frame follows
    ``map_trajectory.build_map_trajectory``: ``x_w`` along route-start tangent,
    ``z_w`` along the ground plane's second axis, ``y_w`` up. We keep the
    bridge explicit and minimal: a per-pixel scale and a 2D rotation/translation
    in the ground plane.
    """

    meters_per_pixel: float
    origin_px: tuple[float, float]
    # yaw mapping: map-pixel tangent angle theta = atan2(dy, dx) (image coords,
    # +x right, +y down) -> world yaw. build_map_trajectory uses
    # yaw = arctan2(-dx, dy), i.e. world x = -sin(yaw), world z = cos(yaw) for a
    # unit forward. We expose the sign convention so callers can stay
    # consistent without re-deriving it.

    def pixel_to_world(
        self, x_px: np.ndarray | float, y_px: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map pixel (x right, y down) -> metric world ground (x_w, z_w).

        Supports vectorized inputs (broadcast as-is; ``float`` scalar inputs
        return 0-d arrays, which is fine for downstream numpy ops).
        """
        x_px = np.asarray(x_px, dtype=np.float64)
        y_px = np.asarray(y_px, dtype=np.float64)
        dx = (x_px - float(self.origin_px[0])) * float(self.meters_per_pixel)
        dy = (y_px - float(self.origin_px[1])) * float(self.meters_per_pixel)
        # build_map_trajectory: x_m from dx, y_m from dy, then centers_w =
        # [x_m, height, y_m]; yaw = arctan2(-dx, dy). So world ground axes are
        # aligned with pixel deltas directly (x_w<-dx_px, z_w<-dy_px in metres).
        return dx, dy


def se2_poses_to_world_centers(
    poses: np.ndarray,
    *,
    meters_per_pixel: float,
    origin_px: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``auto_pipeline_mapctrl.resample_path_to_poses`` output to
    metric world-frame ground centers and yaws.

    ``poses`` is ``[T, 3]`` of ``(x_px, y_px, theta)`` with
    ``theta = atan2(dy, dx)`` in image (y-down) coordinates.

    Returns
    -------
    centers_w : ndarray [T, 3]
        Metric world camera centers ``(x_w, height, z_w)`` with ``height = 0``
        (caller adds camera_height_m). Ground plane only.
    yaws_w : ndarray [T]
        World yaw in radians, using the ``arctan2(-dx, dy)`` convention from
        ``map_trajectory`` so the resulting w2c matches the PRoPE branch.
    """

    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"poses must be [T, 3], got {poses.shape}")
    if origin_px is None:
        origin_px = (float(poses[0, 0]), float(poses[0, 1]))
    calib = MapToWorldCalibration(
        meters_per_pixel=float(meters_per_pixel),
        origin_px=origin_px,
    )
    x_w, z_w = calib.pixel_to_world(poses[:, 0], poses[:, 1])
    # x_w, z_w are [T]; stack along a new axis to get [T, 3].
    centers_w = np.stack(
        [np.atleast_1d(x_w), np.zeros_like(np.atleast_1d(x_w)), np.atleast_1d(z_w)],
        axis=1,
    )
    # theta is atan2(dy, dx) in image coords; world yaw = arctan2(-dx, dy)
    # with dx, dy the pixel deltas from the origin (map_trajectory convention).
    dx_px = poses[:, 0] - float(origin_px[0])
    dy_px = poses[:, 1] - float(origin_px[1])
    yaws_w = np.arctan2(-dx_px, dy_px)
    return centers_w, yaws_w


# ---------------------------------------------------------------------------
# BEV ego crop -> image-token occupancy (the PSCAR projection kernel)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BEVCropGeometry:
    """Geometry of a ``sample_ego_aligned_masks`` BEV crop in metric units.

    Mirrors ``map_layout_condition.LayoutProjectionSettings`` but in metres and
    with the convention made explicit: rows run far-ahead -> near (positive
    forward -> negative forward), columns run camera-left -> camera-right.
    """

    grid_height: int
    grid_width: int
    lookahead_m: float
    rear_m: float
    lateral_extent_m: float

    def forward_axis_m(self) -> np.ndarray:
        """Per-row forward distance, far -> near (lookahead -> -rear)."""
        return np.linspace(self.lookahead_m, -self.rear_m, self.grid_height, dtype=np.float64)

    def lateral_axis_m(self) -> np.ndarray:
        """Per-column lateral distance, left -> right."""
        return np.linspace(-self.lateral_extent_m, self.lateral_extent_m, self.grid_width, dtype=np.float64)


@dataclass(frozen=True)
class CameraGroundParams:
    """Extrinsic ground-camera parameters used by the projection kernel.

    The camera sits at height ``height_m`` above the ground origin, pitched
    down by ``pitch_deg`` (negative pitch looks down). Yaw is handled by the
    caller rotating the BEV points into the camera frame, so this struct only
    describes the elevation geometry that is invariant across frames.
    """

    height_m: float = 1.55
    pitch_deg: float = -5.0


def _rotation_x(deg: float) -> np.ndarray:
    """Rotation about the x-axis (camera pitch), radians.

    Convention: a negative ``deg`` (camera pitching down) rotates the world so
    that ground points ahead move to larger image v (toward the bottom), which
    matches the OpenCV-style camera frame used below (+X right, +Y down,
    +Z forward).
    """
    rad = math.radians(float(deg))
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=np.float64)


def project_ground_points_to_image(
    points_forward_m: np.ndarray,
    points_lateral_m: np.ndarray,
    camera: CameraGroundParams,
    K: NormalizedIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ground-plane points to normalized image coordinates.

    Uses the OpenCV-style camera frame: +X right, +Y down, +Z forward, so a
    point in front of the camera has positive depth. Pixel projection is
    ``u = fx * X / Z + cx``, ``v = fy * Y / Z + cy`` with ``(u, v)`` in [0, 1].

    Parameters
    ----------
    points_forward_m, points_lateral_m : ndarray, broadcastable
        Ground-plane points in the camera's forward/lateral frame (camera at
        origin, looking along +forward, +lateral to the camera's right).
    camera : CameraGroundParams
    K : NormalizedIntrinsics

    Returns
    -------
    u, v : ndarray
        Normalized pixel coordinates in [0, 1] (may fall outside for points
        behind/above the horizon).
    depth : ndarray
        Camera-frame depth along +Z; points in front of the camera have
        positive depth and should be kept; non-positive depth is masked out.
    """

    fwd = np.asarray(points_forward_m, dtype=np.float64)
    lat = np.asarray(points_lateral_m, dtype=np.float64)
    # Ground point (forward, lateral, 0) relative to camera at height h:
    #   X_cam = lateral           (right)
    #   Y_cam = +height           (ground is below the camera, +Y is down)
    #   Z_cam = forward           (forward is into the scene, +Z)
    X = lat
    Y = np.full_like(fwd, float(camera.height_m))
    Z = fwd
    cam = np.stack([X, Y, Z], axis=-1)  # ..., 3
    # Apply pitch about X: negative pitch (looking down) rotates world points so
    # that ahead-ground points gain Y (move toward image bottom).
    R = _rotation_x(camera.pitch_deg)
    cam = cam @ R.T
    depth = cam[..., 2]
    safe_depth = np.where(np.abs(depth) < 1.0e-9, 1.0e-9, depth)
    u = K.fx * cam[..., 0] / safe_depth + K.cx
    v = K.fy * cam[..., 1] / safe_depth + K.cy
    return u, v, depth


def bev_ego_crop_to_image_occupancy(
    bev_grid: np.ndarray,
    bev_geom: BEVCropGeometry,
    camera: CameraGroundParams,
    *,
    image_grid_h: int,
    image_grid_w: int,
    K: NormalizedIntrinsics | None = None,
    spatial_blur_px: float = 1.5,
    behind_camera_margin_m: float = 0.05,
) -> np.ndarray:
    """Re-project a BEV ego-aligned semantic crop onto the image-token grid.

    This is the PSCAR projection kernel. For every BEV cell (a ground-plane
    point in camera forward/lateral metres) and every semantic channel, it
    accumulates the BEV value into the image-token cell that the ground point
    projects to. The result is a per-channel image-plane occupancy that a
    frozen text cross-attention can be spatially routed against.

    Parameters
    ----------
    bev_grid : ndarray [T, C, H_bev, W_bev] or [C, H_bev, W_bev]
        BEV ego crop from ``sample_ego_aligned_masks`` (one per frame) or a
        single frame. Values are semantic occupancies in [0, 1].
    bev_geom : BEVCropGeometry
        Metric geometry of the BEV crop.
    camera : CameraGroundParams
    image_grid_h, image_grid_w : int
        Output image-token grid (e.g. 352 x 640 for 704x1280 video with
        patch (1,2,2); the wrapper passes the runtime grid_sizes).
    K : NormalizedIntrinsics, optional
        Defaults to ``default_K_normalized()``.
    spatial_blur_px : float
        Sigma of a light Gaussian blur applied in image-token space so a single
        BEV cell does not alias onto one token.
    behind_camera_margin_m : float
        BEV cells with forward distance below this margin (slightly behind the
        camera) are dropped.

    Returns
    -------
    occupancy : ndarray [T, C, image_grid_h, image_grid_w] (or [C, ...] for a
        single-frame input), float32 in [0, 1].
    """

    bev = np.asarray(bev_grid, dtype=np.float32)
    single = bev.ndim == 3
    if single:
        bev = bev[None]
    if bev.ndim != 4:
        raise ValueError(f"bev_grid must be [T,C,H,W] or [C,H,W], got {bev.shape}")
    T, C, H_bev, W_bev = bev.shape
    if H_bev != bev_geom.grid_height or W_bev != bev_geom.grid_width:
        raise ValueError(
            f"bev_grid HxW {(H_bev, W_bev)} != bev_geom {(bev_geom.grid_height, bev_geom.grid_width)}"
        )
    K = K or default_K_normalized()

    fwd = bev_geom.forward_axis_m()  # [H_bev], far -> near
    lat = bev_geom.lateral_axis_m()  # [W_bev], left -> right
    ff, ll = np.meshgrid(fwd, lat, indexing="ij")  # [H_bev, W_bev]
    u, v, depth = project_ground_points_to_image(ff, ll, camera, K)  # [H_bev, W_bev]

    # Map normalized pixels to token grid and mask out non-visible cells.
    # A ground point is visible if depth > 0 (in front of camera) and its pixel
    # lands inside the [0,1]^2 image. The far rows of the BEV crop project near
    # the horizon (v -> 0); near rows project toward the bottom (v -> 1).
    col = np.floor(u * image_grid_w).astype(np.int64)
    row = np.floor(v * image_grid_h).astype(np.int64)
    visible = (depth > behind_camera_margin_m) & (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)

    occupancy = np.zeros((T, C, image_grid_h, image_grid_w), dtype=np.float32)
    # Scatter-add each BEV cell's value into its image-token cell, then
    # normalize per token cell by the count of contributors. This is a
    # many-to-one map (distant BEV rows compress toward the horizon), so we
    # take a max over contributors rather than a mean to preserve "there is
    # water somewhere ahead" rather than diluting it.
    visible_idx = np.nonzero(visible)
    r_idx = row[visible_idx]
    c_idx = col[visible_idx]
    for t in range(T):
        for ch in range(C):
            vals = bev[t, ch][visible_idx]
            if vals.size == 0:
                continue
            np.maximum.at(occupancy[t, ch], (r_idx, c_idx), vals)

    if spatial_blur_px > 0:
        occupancy = _gaussian_blur_2d_per_channel(occupancy, sigma=spatial_blur_px)

    return occupancy[0] if single else occupancy


def _gaussian_blur_2d_per_channel(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur over the last two dims, edge-replicated.

    Pure-numpy so the module stays torch-free. ``arr`` is [T, C, H, W].
    """
    if sigma <= 0:
        return arr
    radius = max(1, int(math.ceil(float(sigma) * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * float(sigma) ** 2))
    kernel /= kernel.sum()

    def _blur_axis(a: np.ndarray, axis: int) -> np.ndarray:
        pad = [(0, 0)] * a.ndim
        pad[axis] = (radius, radius)
        ap = np.pad(a, pad, mode="edge")
        out = np.zeros_like(a, dtype=np.float64)
        size = int(a.shape[axis])
        for i, w in enumerate(kernel):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + size)
            out = out + float(w) * ap[tuple(sl)]
        return out

    out = _blur_axis(arr.astype(np.float64), -1)
    out = _blur_axis(out, -2)
    return out.astype(arr.dtype)


# ---------------------------------------------------------------------------
# Occupancy -> per-token routing weights (PSCAR softmax over classes)
# ---------------------------------------------------------------------------


def occupancy_to_routing_weights(
    occupancy: np.ndarray,
    *,
    tau: float = 0.1,
    base_floor: float = 0.05,
) -> np.ndarray:
    """Convert per-class image occupancy to PSCAR routing weights.

    For each image-token cell, ``w_c = softmax_c(occ_c / tau)``. A small
    ``base_floor`` is added to every class so that tokens with no map evidence
    route to the base prompt rather than to an arbitrary class. ``base`` is
    expected to be the last channel; if not, callers can append a zeros channel
    before calling this.

    Parameters
    ----------
    occupancy : ndarray [T, C, H, W] or [C, H, W]
        Per-class image-plane occupancy in [0, 1].
    tau : float
        Temperature; smaller -> sharper routing.
    base_floor : float
        Constant added to every class's logit so empty cells fall back to base.

    Returns
    -------
    weights : ndarray, same shape as ``occupancy``, summing to 1 over the
        class axis (axis 1 or 0).
    """

    occ = np.asarray(occupancy, dtype=np.float64)
    single = occ.ndim == 3
    if single:
        occ = occ[None]
    if occ.ndim != 4:
        raise ValueError(f"occupancy must be [T,C,H,W] or [C,H,W], got {occ.shape}")
    logits = occ / max(float(tau), 1.0e-6) + float(base_floor)
    # Numerically stable softmax over the class axis (axis=1).
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    weights = exp / exp.sum(axis=1, keepdims=True)
    return weights[0] if single else weights


def occupancy_to_blend_mask(
    occupancy: np.ndarray,
    *,
    class_index: int,
    threshold: float = 0.0,
) -> np.ndarray:
    """Single-class soft blend mask in [0, 1] for the POI-style router.

    The existing ``mapctrl_inference_wrapper`` POI path blends two contexts with
    a single ``blend`` mask. This helper extracts one class's occupancy as that
    blend, clamped to [0, 1], so the 1-class template can be reused unchanged
    for a first training-free PSCAR smoke test.
    """

    occ = np.asarray(occupancy, dtype=np.float32)
    if occ.ndim == 4:
        occ = occ[:, class_index]
    elif occ.ndim == 3:
        occ = occ[class_index]
    else:
        raise ValueError(f"occupancy must be [T,C,H,W] or [C,H,W], got {occ.shape}")
    return np.clip(occ - float(threshold), 0.0, 1.0).astype(np.float32)
