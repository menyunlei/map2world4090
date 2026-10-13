"""R1b — world-indexed warm start: replay instead of regenerate, at segment
granularity, on the Wan2_2_CameraPipeline path (loop_150s's true generation
path: 20 full-window denoise steps per 121-frame segment).

Mechanism. Every generated segment is inserted into a world index keyed by
(world tile, view bin) over its GROUND-BAND pixels (the region that the
repo's camera model can inverse-project: 0.5-150 m forward, below horizon).
Before segment k runs, the index warps everything it has from segments < k
into segment k's planned camera poses (ground-plane compensated warp — the
same canvas math tools/measure_world_revisit_quality.py validated at PSNR
median 17.4 dB vs naive same-screen paste's SSIM 0.34-0.60), producing:
  * a pixel composite of 31 latent-anchor frames (expanded to the causal VAE
    121-frame layout [a0, a1 x4, ...] and VAE-encoded by the caller), and
  * a per-latent-pixel coverage mask (16x16 mean-pool of the covered ground
    band; frame 0 zeroed — it is locked to the start image anyway).
The pipeline then replaces the pure-noise init with a noise-consistent
SDEdit blend at sigma(t_{s0}) and runs only the tail s0..N-1 of the
schedule (pre-registered: s0 chosen by sigma, not by step count).

Index semantics (from the pre-measurement): tile 16 m, view bin 15 deg,
adjacent-bin lookup, |dyaw| <= 7.5 deg, sources strictly earlier segments,
newest-first greedy fill, candidate cap 8. A segment is warm iff the mean
ground-band coverage over anchors 1..30 is >= min_coverage (0.5 default;
pre-measured warm fraction ~0.90 on the Egypt loop route).

Multi-rank determinism: the transformer head all-gathers its output across
the SP group (models/wan_transformer3d.py `all_gather(x, dim=1)`), so the
decoded sample is IDENTICAL on every rank; each rank maintains its own
index from the same samples with zero communication.

CPU-only, numpy. ~6-9 s per segment at 704x1280 (vs ~146 s baseline DiT
time per segment) — reported separately, never charged to the DiT bar.
"""

import math
from collections import defaultdict

import numpy as np
import torch

from utils.map_camera_align import (
    CameraGroundParams,
    default_K_normalized,
    project_ground_points_to_image,
)
from utils.world_canvas import c2w_yaw

# Copied from tools/measure_world_revisit_quality.py (pre-registered values).
MIN_FORWARD_M = 0.5
MAX_FORWARD_M = 150.0
CAMERA_BIN_DEG = 15.0
MAX_VIEW_DELTA_DEG = 7.5
N_YAW_BINS = int(round(360.0 / CAMERA_BIN_DEG))


def bilinear_sample(frame_f, u, v):
    """Sample a float32 frame at normalized (u, v); out-of-bounds -> invalid."""
    H, W = frame_f.shape[:2]
    x = u * W - 0.5
    y = v * H - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    valid = (x0 >= 0) & (x0 < W - 1) & (y0 >= 0) & (y0 < H - 1)
    xi = np.clip(x0, 0, W - 2)
    yi = np.clip(y0, 0, H - 2)
    fx = np.clip(x - xi, 0.0, 1.0)
    fy = np.clip(y - yi, 0.0, 1.0)
    c00 = frame_f[yi, xi]
    c01 = frame_f[yi, xi + 1]
    c10 = frame_f[yi + 1, xi]
    c11 = frame_f[yi + 1, xi + 1]
    out = (c00 * ((1 - fy) * (1 - fx))[..., None]
           + c01 * ((1 - fy) * fx)[..., None]
           + c10 * (fy * (1 - fx))[..., None]
           + c11 * (fy * fx)[..., None])
    return out, valid


class GroundProjector:
    """Analytic pixel <-> ground-plane round trip using the repo camera model.

    Faithful copy of tools/measure_world_revisit_quality.py:GroundProjector —
    the math that produced the pre-registered warp-quality measurements.
    """

    def __init__(self, height_m, pitch_deg):
        self.h = float(height_m)
        p = math.radians(pitch_deg)
        self.cp, self.sp = math.cos(p), math.sin(p)
        self.K = default_K_normalized()
        self.cam = CameraGroundParams(height_m=height_m, pitch_deg=pitch_deg)

    def pixels_to_world(self, u, v, cam_pos, yaw):
        q = (v - self.K.cy) / self.K.fy
        denom = q * self.cp - self.sp
        good = np.abs(denom) > 1.0e-9
        Z = np.zeros_like(q)
        Z[good] = self.h * (self.cp + q[good] * self.sp) / denom[good]
        good &= (Z >= MIN_FORWARD_M) & (Z <= MAX_FORWARD_M)
        lat = (u - self.K.cx) / self.K.fx * (-self.sp * self.h + self.cp * Z)
        fwd2 = np.array([-math.sin(yaw), math.cos(yaw)])
        right2 = np.array([math.cos(yaw), math.sin(yaw)])
        wx = cam_pos[0] + fwd2[0] * Z + right2[0] * lat
        wz = cam_pos[2] + fwd2[1] * Z + right2[1] * lat
        return wx, wz, good

    def world_to_pixels(self, wx, wz, cam_pos, yaw):
        fwd2 = np.array([-math.sin(yaw), math.cos(yaw)])
        right2 = np.array([math.cos(yaw), math.sin(yaw)])
        dx = wx - cam_pos[0]
        dz = wz - cam_pos[2]
        Z = dx * fwd2[0] + dz * fwd2[1]
        lat = dx * right2[0] + dz * right2[1]
        u, v, depth = project_ground_points_to_image(Z, lat, self.cam, self.K)
        # Source-side depth window ADDED vs the measurement copy: a ground
        # point beyond the source camera's own [0.5, 150] m band projects
        # above its horizon and would paste sky/building pixels into a ground
        # slot — wrong-content prior, strictly worse than leaving it empty.
        good = (depth >= MIN_FORWARD_M) & (depth <= MAX_FORWARD_M) \
            & np.isfinite(u) & np.isfinite(v)
        return u, v, good


def _wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


class _Anchor:
    __slots__ = ("aid", "seg_idx", "g", "frame_u8", "cam_pos", "yaw_deg", "yaw_rad", "bin")

    def __init__(self, aid, seg_idx, g, frame_u8, cam_pos, yaw_deg):
        self.aid = aid
        self.seg_idx = seg_idx
        self.g = g
        self.frame_u8 = frame_u8
        self.cam_pos = cam_pos
        self.yaw_deg = yaw_deg
        self.yaw_rad = math.radians(yaw_deg)
        self.bin = int(round(((yaw_deg % 360.0) / CAMERA_BIN_DEG))) % N_YAW_BINS


class WorldWarmStartIndex:
    """World index over rendered segments; builds warm-start composites.

    add_segment(seg_idx, sample, poses): sample [B,C,T,H,W] in [0,1] (torch,
    identical on all SP ranks), poses [T,4,4] executed global c2w. Registers
    the segment's 31 latent-anchor frames (pixel frames 0,4,8,...) under the
    (tile, view-bin) keys their ground bands touch.

    build_warm_composite(target_poses, min_coverage): warps everything from
    STRICTLY EARLIER segments into the target segment's poses. Returns None
    if the index is empty (segment 0), else a pack dict; the CALLER applies
    the min_coverage threshold (one place, printed either way).
    """

    def __init__(self, tile_m=16.0, height_m=1.55, pitch_deg=-5.0,
                 height_px=704, width_px=1280, latent_stride=16,
                 max_candidates=8, coverage_early_exit=0.995,
                 frame_cache_size=64):
        self.tile_m = float(tile_m)
        self.proj = GroundProjector(height_m, pitch_deg)
        self.H = int(height_px)
        self.W = int(width_px)
        self.latent_stride = int(latent_stride)
        self.max_candidates = int(max_candidates)
        self.coverage_early_exit = float(coverage_early_exit)
        self.frame_cache_size = int(frame_cache_size)

        self._anchors = []
        self._tile_index = defaultdict(list)  # (wtx, wty, bin) -> [aid]
        self._frame_f_cache = {}
        self._frame_f_order = []
        self._build_valid_band()

    # ---- ground band: validity depends on v only -> precompute once -------
    def _build_valid_band(self):
        vs = (np.arange(self.H) + 0.5) / self.H
        us = (np.arange(self.W) + 0.5) / self.W
        q = (vs - self.proj.K.cy) / self.proj.K.fy
        denom = q * self.proj.cp - self.proj.sp
        good_rows = np.abs(denom) > 1.0e-9
        Z = np.zeros_like(q)
        Z[good_rows] = self.proj.h * (self.proj.cp + q[good_rows] * self.proj.sp) \
            / denom[good_rows]
        good_rows &= (Z >= MIN_FORWARD_M) & (Z <= MAX_FORWARD_M)
        self.valid_rows = np.nonzero(good_rows)[0]
        if self.valid_rows.size == 0:
            raise ValueError("[world_warm] camera model yields an empty ground band")
        uu, vv = np.meshgrid(us, (self.valid_rows + 0.5) / self.H)
        self.band_u = uu.ravel().astype(np.float64)
        self.band_v = vv.ravel().astype(np.float64)
        self.band_flat = (self.valid_rows[:, None] * self.W
                          + np.arange(self.W)[None, :]).ravel()
        self.n_band = int(self.band_u.size)

    def _frame_f(self, aid):
        f = self._frame_f_cache.get(aid)
        if f is None:
            f = self._anchors[aid].frame_u8.astype(np.float32)
            self._frame_f_cache[aid] = f
            self._frame_f_order.append(aid)
            while len(self._frame_f_order) > self.frame_cache_size:
                evict = self._frame_f_order.pop(0)
                self._frame_f_cache.pop(evict, None)
        return f

    def _band_world(self, cam_pos, yaw_rad):
        wx, wz, _ = self.proj.pixels_to_world(self.band_u, self.band_v, cam_pos, yaw_rad)
        tiles = np.unique(
            np.stack([np.floor(wx / self.tile_m).astype(np.int64),
                      np.floor(wz / self.tile_m).astype(np.int64)]),
            axis=1,
        )
        return wx, wz, tiles

    # ---- index write -------------------------------------------------------
    def add_segment(self, seg_idx, sample, poses):
        """sample: torch [B,C,T,H,W] in [0,1] (identical on all SP ranks) or a
        numpy [T,H,W,3] uint8 array (analysis/smoke-test path)."""
        if isinstance(sample, np.ndarray):
            frames = np.ascontiguousarray(sample).astype(np.uint8)
        else:
            s = sample[0].permute(1, 2, 3, 0).detach().cpu().float().clamp(0, 1).numpy()
            frames = (s * 255.0).round().clip(0, 255).astype(np.uint8)
        T = frames.shape[0]
        poses = np.asarray(poses)
        yaws_deg = c2w_yaw(poses)
        n_anchors = 0
        for p in range(0, T, 4):  # latent anchors: pixel frames 0, 4, 8, ...
            cam_pos = np.asarray(poses[p][:3, 3], dtype=np.float64)
            yaw_deg = float(yaws_deg[p])
            aid = len(self._anchors)
            anch = _Anchor(aid, int(seg_idx), p // 4, frames[p], cam_pos, yaw_deg)
            self._anchors.append(anch)
            _, _, tiles = self._band_world(cam_pos, anch.yaw_rad)
            for tx, ty in tiles.T:
                self._tile_index[(int(tx), int(ty), anch.bin)].append(aid)
            n_anchors += 1
        return n_anchors

    # ---- candidate lookup ---------------------------------------------------
    def _candidates(self, tiles, yaw_deg):
        bin0 = int(round(((yaw_deg % 360.0) / CAMERA_BIN_DEG))) % N_YAW_BINS
        bins = {(bin0 + d) % N_YAW_BINS for d in (-1, 0, 1)}
        seen = set()
        for tx, ty in tiles:
            for b in bins:
                for aid in self._tile_index.get((int(tx), int(ty), b), ()):
                    if aid not in seen:
                        seen.add(aid)
        out = []
        for aid in seen:
            a = self._anchors[aid]
            if abs(_wrap_deg(a.yaw_deg - yaw_deg)) > MAX_VIEW_DELTA_DEG:
                continue
            out.append(aid)
        # newest first: latest segment, then latest anchor within it
        out.sort(key=lambda aid: (self._anchors[aid].seg_idx, self._anchors[aid].g),
                 reverse=True)
        return out[: self.max_candidates]

    # ---- index read: build the warm composite --------------------------------
    def build_warm_composite(self, target_poses, min_coverage=0.5):
        if not self._anchors:
            return None
        target_poses = np.asarray(target_poses)
        T = target_poses.shape[0]
        T_a = (T - 1) // 4 + 1
        yaws_t = c2w_yaw(target_poses)
        H, W = self.H, self.W

        comp = np.zeros((T_a, H, W, 3), np.float32)
        cov_hw = np.zeros((T_a, H, W), np.float32)
        per_anchor_cov = np.zeros(T_a, np.float32)
        n_cands = np.zeros(T_a, np.int64)

        for g in range(T_a):
            p = min(4 * g, T - 1)
            cam_pos = np.asarray(target_poses[p][:3, 3], dtype=np.float64)
            yaw_deg = float(yaws_t[p])
            wx, wz, tiles = self._band_world(cam_pos, math.radians(yaw_deg))
            cands = self._candidates(tiles.T, yaw_deg)
            n_cands[g] = len(cands)
            if not cands:
                continue
            flat = np.zeros((self.n_band, 3), np.float32)
            filled = np.zeros(self.n_band, bool)
            used = 0
            for aid in cands:
                a = self._anchors[aid]
                src = self._frame_f(aid)
                u_s, v_s, good = self.proj.world_to_pixels(
                    wx, wz, a.cam_pos, a.yaw_rad)
                samp, inb = bilinear_sample(src, u_s, v_s)
                take = good & inb & ~filled
                if not take.any():
                    continue
                flat[take] = samp[take]
                filled[take] = True
                used += 1
                if filled.mean() >= self.coverage_early_exit:
                    break
            comp[g].reshape(-1, 3)[self.band_flat] = flat
            cov_hw[g].reshape(-1)[self.band_flat] = filled
            per_anchor_cov[g] = float(filled.mean())

        lat_h = H // self.latent_stride
        lat_w = W // self.latent_stride
        cov_lat = cov_hw.reshape(T_a, lat_h, self.latent_stride,
                                 lat_w, self.latent_stride).mean(axis=(2, 4))
        cov_lat = (cov_lat >= 0.5).astype(np.float32)
        cov_lat[0] = 0.0  # frame 0 is locked to the start image; never warm

        seg_cov = float(per_anchor_cov[1:].mean()) if T_a > 1 else 0.0
        return {
            "anchors": np.clip(comp, 0, 255).astype(np.uint8),  # [T_a,H,W,3]
            "coverage_latent": cov_lat,                          # [T_a,lat_h,lat_w]
            "coverage_hw": cov_hw,                               # debug only
            "coverage": seg_cov,
            "anchor_coverage": per_anchor_cov,
            "n_candidates": n_cands,
            "n_anchors_hit": int((per_anchor_cov[1:] > 0).sum()) if T_a > 1 else 0,
            "n_anchors_total": int(T_a),
            "min_coverage": float(min_coverage),
        }
