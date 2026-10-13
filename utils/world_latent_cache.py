"""R1c-latent — world-indexed LATENT cache: lossless revisit identity.

Companion to utils/world_warm_start.py (R1b pixel-warp).  R1b was closed on
ALL injection mechanisms because the ground-plane bilinear pixel warp is a
13-17 dB low-pass prior — no injection mechanism (SDEdit blend, per-step
hard re-lock, anchor) can exceed it.  This module bypasses the warp entirely:

  * Stores the clean **denoised VAE latent** (not pixels) per anchor frame,
    keyed by (world tile, view bin) — the same index as R1b.
  * On revisit, retrieves the exact latent for matching (tile, view bin)
    within |Δyaw| ≤ 7.5°.  No warp, no VAE re-encode — the cached latent IS
    the generated content, at full sharpness.
  * Coverage is per-anchor binary (full-frame latent or nothing), not the
    per-pixel ground-band mask of R1b.

Trade-off vs R1b pixel warp:
  + Lossless retrieval (vs 13-17 dB warp)
  + Full-frame coverage (vs ground-band-only)
  - Only exact-view matches (same tile, same view bin); no cross-view warp.
    For loop routes (closure ratio ~0.02) the revisit segments hit the same
    tiles with ~same view → high coverage.  For novel-view segments the cache
    falls cold (zero prior), which is the correct behaviour: we do not fake
    identity we do not have.

The returned pack is consumed by run_segment() the same way as R1b's
build_warm_composite pack, except the latent is already in latent space
(key "warm_latent" instead of "anchors") so run_segment skips the VAE encode.
"""

import math
from collections import defaultdict

import numpy as np
import torch

from utils.world_warm_start import (
    GroundProjector,
    _wrap_deg,
    CAMERA_BIN_DEG,
    MAX_VIEW_DELTA_DEG,
    N_YAW_BINS,
)
from utils.world_canvas import c2w_yaw


class _LatentAnchor:
    __slots__ = ("aid", "seg_idx", "g", "cam_pos", "yaw_deg", "yaw_rad", "bin")

    def __init__(self, aid, seg_idx, g, cam_pos, yaw_deg):
        self.aid = aid
        self.seg_idx = seg_idx
        self.g = g
        self.cam_pos = cam_pos
        self.yaw_deg = yaw_deg
        self.yaw_rad = math.radians(yaw_deg)
        self.bin = int(round(((yaw_deg % 360.0) / CAMERA_BIN_DEG))) % N_YAW_BINS


class WorldLatentCache:
    """World index over denoised latents; retrieves losslessly on revisit.

    add_segment(seg_idx, latent, poses): latent [B,C,T_lat,H_lat,W_lat] or
    [C,T_lat,H_lat,W_lat] torch tensor (the clean post-denoise latent from
    the pipeline).  Stores per-anchor latent frames [C,H_lat,W_lat] on CPU,
    keyed by (tile, view bin).

    build_warm_latent(target_poses, min_coverage): for each target anchor,
    finds the newest cached latent at the same (tile, view bin) within
    |Δyaw| ≤ 7.5°.  Returns a pack with the assembled warm latent + binary
    coverage mask, or None if the index is empty.
    """

    def __init__(self, tile_m=16.0, height_m=1.55, pitch_deg=-5.0,
                 height_px=704, width_px=1280, latent_stride=16,
                 max_candidates=8):
        self.tile_m = float(tile_m)
        self.proj = GroundProjector(height_m, pitch_deg)
        self.H = int(height_px)
        self.W = int(width_px)
        self.latent_stride = int(latent_stride)
        self.max_candidates = int(max_candidates)

        self._anchors: list[_LatentAnchor] = []
        self._latents: list[torch.Tensor] = []  # [C, H_lat, W_lat] CPU
        self._tile_index: dict = defaultdict(list)
        self._build_valid_band()

    # ---- ground band: same as WorldWarmStartIndex (determines tile keys) ----
    def _build_valid_band(self):
        vs = (np.arange(self.H) + 0.5) / self.H
        us = (np.arange(self.W) + 0.5) / self.W
        q = (vs - self.proj.K.cy) / self.proj.K.fy
        denom = q * self.proj.cp - self.proj.sp
        good_rows = np.abs(denom) > 1.0e-9
        Z = np.zeros_like(q)
        Z[good_rows] = self.proj.h * (self.proj.cp + q[good_rows] * self.proj.sp) \
            / denom[good_rows]
        good_rows &= (Z >= 0.5) & (Z <= 150.0)
        self.valid_rows = np.nonzero(good_rows)[0]
        if self.valid_rows.size == 0:
            raise ValueError("[world_latent_cache] empty ground band")
        uu, vv = np.meshgrid(us, (self.valid_rows + 0.5) / self.H)
        self.band_u = uu.ravel().astype(np.float64)
        self.band_v = vv.ravel().astype(np.float64)
        self.n_band = int(self.band_u.size)

    def _band_tiles(self, cam_pos, yaw_rad):
        """Return the set of (wtx, wty) tiles this camera's ground band touches."""
        wx, wz, _ = self.proj.pixels_to_world(
            self.band_u, self.band_v, cam_pos, yaw_rad)
        tiles = np.unique(
            np.stack([np.floor(wx / self.tile_m).astype(np.int64),
                      np.floor(wz / self.tile_m).astype(np.int64)]),
            axis=1,
        )
        return tiles  # (2, n_tiles)

    # ---- index write -------------------------------------------------------
    def add_segment(self, seg_idx, latent, poses):
        """latent: torch [B,C,T_lat,H_lat,W_lat] or [C,T_lat,H_lat,W_lat].
        poses: [T,4,4] executed global c2w (T = 4*(T_lat-1)+1 pixel frames)."""
        if latent.dim() == 5:
            latent = latent[0]  # [C, T_lat, H_lat, W_lat]
        # Move to CPU for storage (frees GPU memory).
        latent = latent.detach().cpu()
        T_lat = latent.shape[1]
        poses = np.asarray(poses)
        yaws_deg = c2w_yaw(poses)
        n_anchors = 0
        for g in range(T_lat):
            p = min(4 * g, poses.shape[0] - 1)
            cam_pos = np.asarray(poses[p][:3, 3], dtype=np.float64)
            yaw_deg = float(yaws_deg[p])
            aid = len(self._anchors)
            anch = _LatentAnchor(aid, int(seg_idx), g, cam_pos, yaw_deg)
            self._anchors.append(anch)
            self._latents.append(latent[:, g].clone())  # [C, H_lat, W_lat]
            tiles = self._band_tiles(cam_pos, anch.yaw_rad)
            for tx, ty in tiles.T:
                self._tile_index[(int(tx), int(ty), anch.bin)].append(aid)
            n_anchors += 1
        return n_anchors

    # ---- candidate lookup (same logic as WorldWarmStartIndex) --------------
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
        out.sort(key=lambda aid: (self._anchors[aid].seg_idx,
                                  self._anchors[aid].g), reverse=True)
        return out[: self.max_candidates]

    # ---- index read: retrieve cached latents losslessly --------------------
    def build_warm_latent(self, target_poses, min_coverage=0.5):
        if not self._anchors:
            return None
        target_poses = np.asarray(target_poses)
        T = target_poses.shape[0]
        T_lat = (T - 1) // 4 + 1
        yaws_t = c2w_yaw(target_poses)

        C = self._latents[0].shape[0]
        H_lat = self._latents[0].shape[1]
        W_lat = self._latents[0].shape[2]
        ref_dtype = self._latents[0].dtype

        warm_latent = torch.zeros(C, T_lat, H_lat, W_lat, dtype=ref_dtype)
        cov_mask = torch.zeros(1, T_lat, H_lat, W_lat, dtype=torch.float32)
        per_anchor_cov = np.zeros(T_lat, dtype=np.float32)
        n_cands = np.zeros(T_lat, dtype=np.int64)

        for g in range(T_lat):
            p = min(4 * g, T - 1)
            cam_pos = np.asarray(target_poses[p][:3, 3], dtype=np.float64)
            yaw_deg = float(yaws_t[p])
            tiles = self._band_tiles(cam_pos, math.radians(yaw_deg))
            cands = self._candidates(tiles.T, yaw_deg)
            n_cands[g] = len(cands)
            if cands:
                # Take the newest matching candidate — its latent IS the
                # generated content at this (tile, view), no warp needed.
                aid = cands[0]
                warm_latent[:, g] = self._latents[aid]
                cov_mask[0, g] = 1.0  # full-frame coverage
                per_anchor_cov[g] = 1.0

        # Frame 0 is locked to the start image by the pipeline; never warm.
        warm_latent[:, 0] = 0
        cov_mask[0, 0] = 0.0
        per_anchor_cov[0] = 0.0  # frame 0 locked by pipeline

        seg_cov = float(per_anchor_cov[1:].mean()) if T_lat > 1 else 0.0
        return {
            "warm_latent": warm_latent.unsqueeze(0),       # [1,C,T_lat,H_lat,W_lat]
            "coverage_mask": cov_mask.unsqueeze(0),        # [1,1,T_lat,H_lat,W_lat]
            "coverage": seg_cov,
            "anchor_coverage": per_anchor_cov,
            "n_candidates": n_cands,
            "n_anchors_hit": int((per_anchor_cov[1:] > 0).sum()) if T_lat > 1 else 0,
            "n_anchors_total": int(T_lat),
            "min_coverage": float(min_coverage),
        }
