"""World-canvas: persistent appearance memory indexed by WORLD coordinates.

Why this module exists
----------------------
A map is a world: a persistent, place-indexed description that does not move
when the camera does. The long-video pipeline, however, chains segments purely
in TIME (segment k+1 starts from segment k's last frame and its poses restart
at the origin), so appearance memory is indexed by time, not place. A revisit
therefore renders whatever the *previous frame* looked like, not what the
*place* looked like, and drift compounds without bound along the route.

The world canvas re-indexes that memory from time to place, training-free and
outside the network (P1 in MAP_WORLD_ALIGNMENT_STRATEGY.md):

  WRITE (after segment k)  run the frozen SegFormer on the segment's saved
      frames, unproject each ground-plane pixel through the segment's EXECUTED
      poses (same ``ActionToPoseFromID`` the PRoPE branch consumed), and splat
      per-pixel semantics into a global BEV grid with per-cell confidence.
      World coordinates come from exact SE(3) chaining of each segment's end
      c2w, so "world" == the executed trajectory's frame, not map pixels.

  READ (before segment k+1)  project observed canvas cells through the NEXT
      segment's executed poses into screen space and emit a per-latent-frame
      occupancy guide in the LayoutCritic E's channel space
      [road, building, vegetation, sky]. The existing Route-B gradient hook
      (``pipeline._layout_guidance_config``) then steers early denoising steps
      toward it. First visits read nothing (canvas empty ahead); revisits read
      what was written at first visit -- loop consistency becomes table lookup.

Determinism under multi-GPU: rank0 paints the guide to ``.npz`` on disk; every
rank loads the same bytes (same pattern the proxy-layout path already uses).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants pinned to the rest of the repo
# ---------------------------------------------------------------------------

# SegFormer-b0-Cityscapes id space (verified from the cached config's
# id2label; see experiments/geomap/layout_extractor.py).
CITY_ROAD = 0
CITY_SIDEWALK = 1
CITY_BUILDING = 2
CITY_VEGETATION = 8
CITY_SKY = 10
CITY_CAR = 13

# LayoutCritic E output channels (experiments/mapstate/layout_critic.py):
CRITIC_CHANNELS = ("road", "building", "vegetation", "sky")

# Canvas planes we store (superset of E's fillable channels).
# 0 road, 1 building, 2 vegetation, 3 sky (reserved), 4 any-observed conf.
_CANVAS_ROAD, _CANVAS_BUILDING, _CANVAS_VEG, _CANVAS_SKY, _CANVAS_CONF = 0, 1, 2, 3, 4
N_CANVAS_PLANES = 5

# SegFormer id -> canvas plane. Sky never reaches the ground plane; its plane
# stays empty and the critic's sky channel is zero-weighted anyway.
_SEG_TO_PLANE = {
    CITY_ROAD: _CANVAS_ROAD,
    CITY_BUILDING: _CANVAS_BUILDING,
    CITY_VEGETATION: _CANVAS_VEG,
}


def _default_K():
    from utils.map_camera_align import default_K_normalized

    return default_K_normalized()


def _default_cam(height_m: float, pitch_deg: float):
    from utils.map_camera_align import CameraGroundParams

    return CameraGroundParams(height_m=height_m, pitch_deg=pitch_deg)


# ---------------------------------------------------------------------------
# Executed-pose construction: EXACTLY what the PRoPE branch consumes
# ---------------------------------------------------------------------------


def build_segment_local_poses(action_seq, action_speed_list, video_length, yaw_deltas=None):
    """Return (T,4,4) local c2w matrices + (T,) yaw_deg for ONE segment.

    Calls the repo's own ``ActionToPoseFromID`` with the same arguments
    ``get_camera_sequence`` (inference_long_video.py) passes it: duration =
    ceil(video_length/len(action_seq)), rows truncated to video_length. The
    parsed rows are the exact pose strings ``GetPoseEmbedsFromPosesPrope``
    turns into the PRoPE control the DiT consumes.
    """
    from utils.inference_utils import ActionToPoseFromID, Camera

    duration = math.ceil(video_length / len(action_seq))
    pose_kwargs = {"duration": duration}
    if yaw_deltas is not None:
        pose_kwargs["yaw_deltas"] = list(yaw_deltas)
    rows = ActionToPoseFromID(list(action_seq), list(action_speed_list), **pose_kwargs)
    rows = rows[:video_length]

    cams = [Camera([float(x) for x in r.split()]) for r in rows]
    c2w = np.stack([c.c2w_mat.astype(np.float64) for c in cams])  # (T,4,4)
    yaw_deg = c2w_yaw(c2w)
    return c2w, yaw_deg


def c2w_yaw(global_c2w: np.ndarray) -> np.ndarray:
    """(T,4,4) c2w -> (T,) yaw_deg about world +Y (yaw 0 faces +Z).

    forward_w = R_w2c^T @ [0,0,1] = c2w rotation @ e_z; the repo's forward
    vector convention is (-sin yaw, *, cos yaw), so yaw = atan2(-fx, fz).
    """
    fwd = np.asarray(global_c2w)[:, :3, :3] @ np.array([0.0, 0.0, 1.0])
    return np.degrees(np.arctan2(-fwd[:, 0], fwd[:, 2]))


def chain_c2w(local_c2w: np.ndarray, prev_end_c2w: Optional[np.ndarray]) -> np.ndarray:
    """Place a segment's local poses into the global world frame.

    Segment k's poses restart at the origin; the executed global trajectory
    chains them by composing with segment k-1's END c2w. Composition of rigid
    transforms is exact (no linearization): segment k's local frame coincides
    with segment k-1's end camera frame at the boundary, and the w/wj/wl
    vocabulary keeps pitch=roll=0, so the shared ground plane is preserved.

    ``prev_end_c2w=None`` (segment 0) returns the local poses unchanged.
    """
    if prev_end_c2w is None:
        return np.asarray(local_c2w, dtype=np.float64).copy()
    prev = np.asarray(prev_end_c2w, dtype=np.float64)
    return np.einsum("ij,tjk->tik", prev, local_c2w)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


@dataclass
class WorldCanvasConfig:
    cell_size_m: float = 1.0
    half_extent_m: float = 400.0
    camera_height_m: float = 1.55
    camera_pitch_deg: float = -5.0
    # Only pixels below this normalized v are splatted: near the horizon the
    # ground-plane inverse depth explodes and a 1-pixel error moves the cell
    # by tens of meters.
    write_max_v: float = 0.60
    min_forward_m: float = 0.5
    max_forward_m: float = 60.0
    max_lateral_m: float = 30.0
    fresh_conf: float = 0.9
    # How a segment's fresh observations fuse into cells that already hold
    # older content. The revisit failure mode of the frozen A/B was traced to
    # "union": a re-observed cell keeps EVERY class ever seen there at full
    # confidence, so a revisit's guide superimposes all past hypotheses and
    # the model falls back to autoregressive context. The alternatives:
    #   "first"  first visit owns the cell forever -- revisits read the
    #            original first-visit layout (clean, unambiguous);
    #   "latest" most recent visit owns the cell -- revisits read what the
    #            previous pass just wrote (unambiguous, but endorses drift).
    # "union" must stay bit-identical to the original direct per-frame splat:
    # the frozen A/B arms depend on it.
    memory_mode: str = "union"

    def __post_init__(self):
        if self.memory_mode not in ("union", "first", "latest"):
            raise ValueError(
                f"unknown memory_mode {self.memory_mode!r}: "
                "expected 'union', 'first' or 'latest'"
            )

    @property
    def grid_shape(self) -> tuple:
        n = int(round(2.0 * self.half_extent_m / self.cell_size_m))
        return (n, n)


@dataclass
class WorldCanvas:
    """Persistent BEV semantic grid in the executed-trajectory world frame."""

    config: WorldCanvasConfig = field(default_factory=WorldCanvasConfig)
    # planes[c] : (N, N) float32, rows = -z (north), cols = +x (east)
    planes: Optional[list] = None
    obs_count: Optional[np.ndarray] = None
    # global end c2w of each written segment (for chaining)
    segment_end_c2w: list = field(default_factory=list)
    # audit trail
    segments_written: list = field(default_factory=list)

    def __post_init__(self):
        if self.planes is None:
            self.planes = [
                np.zeros(self.config.grid_shape, dtype=np.float32)
                for _ in range(N_CANVAS_PLANES)
            ]
        if self.obs_count is None:
            self.obs_count = np.zeros(self.config.grid_shape, dtype=np.int32)

    # ----------------------------------------------------------------- write
    def write_segment(self, seg_idx: int, global_c2w: np.ndarray, frame_semantics) -> dict:
        """Splat one segment's frames onto the canvas.

        ``global_c2w``: (T,4,4) executed poses in the GLOBAL frame (output of
        :func:`chain_c2w`). ``frame_semantics``: per-extracted-frame dicts
        ``{"frame_idx": int, "seg": (H,W) int Cityscapes id map}`` at the
        decoded video resolution; coordinates are normalized, so any
        resolution works.
        """
        by_frame = {}
        for d in frame_semantics:
            if d.get("seg") is not None:
                by_frame[int(d["frame_idx"])] = np.asarray(d["seg"])
        if not by_frame:
            return {"seg_idx": int(seg_idx), "frames_written": 0, "cells_written": 0}

        cells_written = 0
        frames_used = 0
        # Per-segment staging: every frame splats into scratch buffers first,
        # then ONE commit applies the memory_mode fusion rule. Within a
        # segment fusion is always max-at (identical ops to the original
        # direct splat, just into scratch), so a "union" commit reproduces
        # the old canvas bit-for-bit; "first"/"latest" only change how the
        # staged content meets older content at commit time.
        shape = self.config.grid_shape
        st_planes = [np.zeros(shape, dtype=np.float32) for _ in range(N_CANVAS_PLANES)]
        st_obs = np.zeros(shape, dtype=np.int32)
        for fidx in sorted(by_frame):
            if fidx >= len(global_c2w):
                continue
            n = self._splat_frame(global_c2w[fidx], by_frame[fidx], st_planes, st_obs)
            cells_written += n
            frames_used += 1 if n > 0 else 0
        self._commit_segment(st_planes, st_obs)

        self.segment_end_c2w.append(np.asarray(global_c2w[-1], dtype=np.float64))
        rec = {
            "seg_idx": int(seg_idx),
            "frames_written": frames_used,
            "cells_written": int(cells_written),
            "observed_cells_total": int((self.obs_count > 0).sum()),
        }
        self.segments_written.append(rec)
        return rec

    def _splat_frame(self, c2w: np.ndarray, sem: np.ndarray,
                     st_planes: list, st_obs: np.ndarray) -> int:
        """Inverse-project one frame's ground pixels into the staging buffer."""
        cfg = self.config
        H, W = sem.shape
        K = _default_K()

        cam_pos = c2w[:3, 3]
        R = c2w[:3, :3]
        fwd_w = R @ np.array([0.0, 0.0, 1.0])
        yaw = math.atan2(-fwd_w[0], fwd_w[2])

        vs = (np.arange(H, dtype=np.float64) + 0.5) / H
        us = (np.arange(W, dtype=np.float64) + 0.5) / W
        vv, uu = np.meshgrid(vs, us, indexing="ij")

        keep = vv <= cfg.write_max_v
        v = vv[keep]
        u = uu[keep]
        ids = sem[keep]
        if v.size == 0:
            return 0

        # Analytic inverse of project_ground_points_to_image. Its forward map
        # for a ground point (fwd=Z, lat=X, height h):
        #   cam-space before pitch: (X, +h, Z)
        #   pitch about X by p (negative = look down): X'=X,
        #       Y' = cos p*h + sin p*Z,  Z' = -sin p*h + cos p*Z
        #   u = fx*X'/Z' + cx ; v = fy*Y'/Z' + cy
        # From v, with q = (v-cy)/fy = Y'/Z':
        #   q*(-sp*h + cp*Z) = cp*h + sp*Z
        #   Z*(q*cp - sp) = h*(cp + q*sp)
        p = math.radians(cfg.camera_pitch_deg)
        cp, sp = math.cos(p), math.sin(p)
        h = cfg.camera_height_m
        q = (v - K.cy) / K.fy
        denom = q * cp - sp
        good = np.abs(denom) > 1.0e-9
        Z = np.zeros_like(q)
        Z[good] = h * (cp + q[good] * sp) / denom[good]
        good &= (Z >= cfg.min_forward_m) & (Z <= cfg.max_forward_m)
        if not good.any():
            return 0
        # From u: X' = (u-cx)/fx * Z' and X' = X, so lat = (u-cx)/fx * Z'.
        lat = (u - K.cx) / K.fx * (-sp * h + cp * Z)
        good &= np.abs(lat) <= cfg.max_lateral_m
        if not good.any():
            return 0

        # Camera-frame (fwd, lat) -> world (x, z) with the executed yaw:
        # forward_w(yaw) = (-sin yaw, cos yaw); right_w(yaw) = (cos yaw, sin yaw)
        fwd2 = np.array([-math.sin(yaw), math.cos(yaw)])
        right2 = np.array([math.cos(yaw), math.sin(yaw)])
        wx = cam_pos[0] + fwd2[0] * Z + right2[0] * lat
        wz = cam_pos[2] + fwd2[1] * Z + right2[1] * lat

        n = cfg.grid_shape[0]
        col = np.floor((wx + cfg.half_extent_m) / cfg.cell_size_m).astype(np.int64)
        row = np.floor((cfg.half_extent_m - wz) / cfg.cell_size_m).astype(np.int64)
        inb = good & (col >= 0) & (col < n) & (row >= 0) & (row < n)
        if not inb.any():
            return 0
        rows_i = row[inb]
        cols_i = col[inb]
        ids_i = ids[inb]

        conf = cfg.fresh_conf
        written = 0
        for seg_id, plane_idx in _SEG_TO_PLANE.items():
            sel = ids_i == seg_id
            if not sel.any():
                continue
            r = rows_i[sel]
            c = cols_i[sel]
            plane = st_planes[plane_idx]
            # within-segment fusion: a fresh observation can raise a cell
            # toward conf but never lower it.
            np.maximum.at(plane, (r, c), conf)
            written += int(len(r))
        np.maximum.at(st_planes[_CANVAS_CONF], (rows_i, cols_i), conf)
        np.add.at(st_obs, (rows_i, cols_i), 1)
        return written

    def _commit_segment(self, st_planes: list, st_obs: np.ndarray) -> None:
        """Fuse one segment's staging buffer into the canvas (memory_mode).

        All modes keep ``obs_count`` a plain observation counter: the read
        path only ever tests ``> 0``, so it stays mode-independent.
        """
        mode = self.config.memory_mode
        if mode == "union":
            # max of maxes == the original per-frame direct max-at (values
            # are only ever {0, conf}); frozen-behaviour-preserving.
            for pi in range(N_CANVAS_PLANES):
                np.maximum(self.planes[pi], st_planes[pi], out=self.planes[pi])
        elif mode == "first":
            # cells never observed before take this segment's content;
            # already-observed cells keep their first-visit layout forever.
            fresh = (self.obs_count == 0) & (st_obs > 0)
            if fresh.any():
                for pi in range(N_CANVAS_PLANES):
                    self.planes[pi][fresh] = st_planes[pi][fresh]
        elif mode == "latest":
            # cells observed this segment are REPLACED by its content
            # (including classes it did not see, which are cleared).
            touched = st_obs > 0
            if touched.any():
                for pi in range(N_CANVAS_PLANES):
                    self.planes[pi][touched] = st_planes[pi][touched]
        self.obs_count += st_obs

    # ------------------------------------------------------------------ read
    def read_guide(self, global_c2w_next: np.ndarray, num_latent_frames: int,
                   grid: tuple = (44, 80)) -> Optional[dict]:
        """Project observed canvas cells into the NEXT segment's frusta.

        Returns ``{"guide": (F,4,gr,gc) float32 in E's channel order,
        "mask": (F,1,gr,gc) float32 (1 = observed), "coverage": (F,)}`` or
        ``None`` when the canvas has nothing in view (first visit). ``grid``
        defaults to the critic's 44x80 token grid for 704x1280 videos.
        """
        if (self.obs_count > 0).sum() == 0:
            return None
        T = len(global_c2w_next)
        if T == 0:
            return None
        # latent frames sample the executed pose track the way the VAE does:
        # latent frame f ~ video frame 4f.
        F = int(num_latent_frames)
        frame_ids = np.clip(np.arange(F, dtype=np.int64) * 4, 0, T - 1)

        any_obs = False
        guides, masks, coverage = [], [], []
        for f in range(F):
            g, m = self._read_frame(global_c2w_next[int(frame_ids[f])], grid=grid)
            guides.append(g)
            masks.append(m)
            coverage.append(float((m > 0).mean()))
            any_obs = any_obs or coverage[-1] > 0
        if not any_obs:
            return None
        return {
            "guide": np.stack(guides).astype(np.float32),
            "mask": np.stack(masks).astype(np.float32),
            "coverage": np.asarray(coverage, dtype=np.float32),
            "frame_indices": [int(x) for x in frame_ids],
        }

    def _read_frame(self, c2w: np.ndarray, grid: tuple = (44, 80)):
        """One frame: canvas cells -> (4,gr,gc) guide + (1,gr,gc) mask."""
        from utils.map_camera_align import project_ground_points_to_image

        cfg = self.config
        K = _default_K()
        cam = _default_cam(cfg.camera_height_m, cfg.camera_pitch_deg)
        gr, gc = grid
        empty = (np.zeros((4, gr, gc)), np.zeros((1, gr, gc)))

        rows, cols = np.nonzero(self.obs_count > 0)
        if rows.size == 0:
            return empty
        wx = (cols + 0.5) * cfg.cell_size_m - cfg.half_extent_m
        wz = cfg.half_extent_m - (rows + 0.5) * cfg.cell_size_m

        cam_pos = c2w[:3, 3]
        R = c2w[:3, :3]
        fwd_w = R @ np.array([0.0, 0.0, 1.0])
        yaw = math.atan2(-fwd_w[0], fwd_w[2])
        fwd2 = np.array([-math.sin(yaw), math.cos(yaw)])
        right2 = np.array([math.cos(yaw), math.sin(yaw)])
        dx = wx - cam_pos[0]
        dz = wz - cam_pos[2]
        forward = dx * fwd2[0] + dz * fwd2[1]
        lateral = dx * right2[0] + dz * right2[1]

        keep = (
            (forward >= cfg.min_forward_m)
            & (forward <= cfg.max_forward_m)
            & (np.abs(lateral) <= cfg.max_lateral_m)
        )
        if not keep.any():
            return empty
        u, v, depth = project_ground_points_to_image(forward[keep], lateral[keep], cam, K)
        inb = (depth > 0) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        if not inb.any():
            return empty

        gu = np.clip((u[inb] * gc).astype(np.int64), 0, gc - 1)
        gv = np.clip((v[inb] * gr).astype(np.int64), 0, gr - 1)
        flat = gv * gc + gu
        guide = np.zeros((4, gr, gc), dtype=np.float32)
        mask = np.zeros((1, gr, gc), dtype=np.float32)

        plane_to_critic = {_CANVAS_ROAD: 0, _CANVAS_BUILDING: 1, _CANVAS_VEG: 2}
        r_sel = rows[keep][inb]
        c_sel = cols[keep][inb]
        for plane_idx, e_ch in plane_to_critic.items():
            vals = self.planes[plane_idx][r_sel, c_sel]
            act = vals > 0
            if not act.any():
                continue
            f_idx = flat[act]
            w_sum = np.bincount(f_idx, weights=vals[act].astype(np.float64), minlength=gr * gc)
            n_hit = np.bincount(f_idx, minlength=gr * gc)
            avg = (w_sum / np.maximum(n_hit, 1)).reshape(gr, gc)
            guide[e_ch] = np.maximum(guide[e_ch], avg.astype(np.float32))
        n_any = np.bincount(flat, minlength=gr * gc).reshape(gr, gc)
        mask[0] = (n_any > 0).astype(np.float32)
        return guide, mask

    # -------------------------------------------------------------- persist
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            planes=np.stack(self.planes).astype(np.float16),
            obs_count=self.obs_count,
            end_c2w=np.stack(self.segment_end_c2w) if self.segment_end_c2w else np.zeros((0, 4, 4)),
            config=json.dumps({
                "cell_size_m": self.config.cell_size_m,
                "half_extent_m": self.config.half_extent_m,
                "camera_height_m": self.config.camera_height_m,
                "camera_pitch_deg": self.config.camera_pitch_deg,
                "write_max_v": self.config.write_max_v,
                "min_forward_m": self.config.min_forward_m,
                "max_forward_m": self.config.max_forward_m,
                "max_lateral_m": self.config.max_lateral_m,
                "fresh_conf": self.config.fresh_conf,
                "memory_mode": self.config.memory_mode,
            }),
        )

    @classmethod
    def load(cls, path) -> "WorldCanvas":
        data = np.load(path, allow_pickle=False)
        cfg = WorldCanvasConfig(**json.loads(str(data["config"])))
        canvas = cls(config=cfg)
        canvas.planes = [p.astype(np.float32) for p in data["planes"]]
        canvas.obs_count = data["obs_count"].astype(np.int32)
        end = data["end_c2w"]
        canvas.segment_end_c2w = [np.asarray(m, dtype=np.float64) for m in end]
        return canvas


# ---------------------------------------------------------------------------
# Frozen SegFormer frame extractor (cached Cityscapes b0)
# ---------------------------------------------------------------------------


class FrozenLayoutExtractor:
    """SegFormer-b0-Cityscapes -> per-frame id maps for the WRITE path."""

    MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

    def __init__(self, device: str = "cuda"):
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self.proc = SegformerImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(self.MODEL_NAME)
            .to(device)
            .eval()
        )
        self.device = device

    def segment_id_maps(self, video_path, every: int = 8, batch: int = 8) -> list:
        import torch
        import torch.nn.functional as F

        frames = _read_video_frames(video_path, every=every)
        if not frames:
            return []
        out = []
        for i in range(0, len(frames), batch):
            chunk = frames[i : i + batch]
            from PIL import Image

            imgs = [Image.fromarray(rgb) for _, rgb in chunk]
            inputs = self.proc(images=imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            up = F.interpolate(
                logits, size=imgs[0].size[::-1], mode="bilinear", align_corners=False
            )
            seg = up.argmax(dim=1).cpu().numpy()
            for (fidx, _), s in zip(chunk, seg):
                out.append({"frame_idx": int(fidx), "seg": s.astype(np.int16)})
        return out


def _read_video_frames(video_path, every: int = 8):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every == 0:
            frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Guide persistence (rank0 writes, every rank reads identical bytes)
# ---------------------------------------------------------------------------


def save_guide_npz(path, guide_dict: dict, seg_idx: int, meta: dict = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seg_idx": np.int64(seg_idx),
        "guide": guide_dict["guide"],
        "mask": guide_dict["mask"],
        "coverage": guide_dict["coverage"],
        "meta": json.dumps(meta or {}),
    }
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def load_guide_npz(path) -> Optional[dict]:
    data = np.load(path, allow_pickle=False)
    if "guide" not in data:
        return None
    return {
        "seg_idx": int(data["seg_idx"]),
        "guide": data["guide"].astype(np.float32),
        "mask": data["mask"].astype(np.float32),
        "coverage": data["coverage"].astype(np.float32),
        "meta": json.loads(str(data["meta"])) if "meta" in data else {},
    }
