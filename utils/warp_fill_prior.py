"""Warp+Fill prior for SDEdit-style truncated DiT warm start.

Instead of sourcing warm-start content from earlier segments' cached frames
(WorldWarmStartIndex), this warps the SEGMENT'S OWN start_image forward to
each anchor frame using MiDaS metric depth + the segment's pose chain. The
result is a structural prior that DiT sharpens via a truncated denoise
schedule.

Two warp modes:
  - "single": warp from the start_image to ALL anchors using ONE depth map.
    Fails at ~0.9 m camera displacement (camera moves past visible geometry,
    warp produces 0% valid pixels). Suitable for slow / lateral motion.
  - "chain": autoregressive warp — re-estimate MiDaS depth on each filled
    anchor before warping to the next. Each step is ~0.5 m (one anchor
    interval), keeping disocclusion small. Valid to ~2 m+ total displacement.
    Costs one extra MiDaS call per anchor (~0.3 s each).

Production flow (per segment):
  1. Load start_image (PNG from previous segment's last frame)
  2. Compute MiDaS metric depth on it
  3. Forward-warp to each anchor frame (every 4th frame in causal VAE layout)
     — single mode: always from start_image; chain mode: from previous anchor
  4. Fill holes (horizontal nearest-neighbor)
  5. Package as warm_start_pack (same format as WorldWarmStartIndex)

The pack is consumed by run_segment → pipeline_dreamxworld, which:
  - Expands [T_a,H,W,3] anchors to causal layout [a0, a1×4, ...]
  - Encodes to latent space
  - Blends: latents = (1-sigma) * (mask * prior) + sigma * noise
  - Runs timesteps[s0:] (truncated denoise)

Per-segment coverage gate: if the mean valid-pixel fraction (excluding
frame 0) is below ``min_seg_coverage``, ``build_warm_composite`` returns
``None`` and the pipeline runs the FULL denoise schedule (no warm-start).
This prevents the failure mode where a truncated schedule with an invalid
prior produces worse results than full denoise.

Per-anchor coverage: the coverage_latent mask is built from the ACTUAL
per-pixel warp validity (downsampled to latent resolution), NOT hardcoded
to 1.0. Anchors with valid fraction below ``min_anchor_coverage`` get a
zero mask (no warm-start for that anchor).
"""
from __future__ import annotations
import os, time, math
import numpy as np
import cv2
import torch

from tools.mapstream_forward_warp import (forward_warp, forward_warp_cuda,
                                          fill_holes, fill_holes_fast)
from tools.midas_depth_adapter import midas_metric_depth_any, depth_to_warp_norm


def _c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Convert camera-to-world [4,4] to world-to-camera [4,4]."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = c2w[:3, :3].T
    w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]
    return w2c


def _load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB uint8 [H,W,3]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class WarpFillPriorIndex:
    """Builds warp+fill warm-start priors from a segment's start_image.

    Unlike WorldWarmStartIndex (cross-segment cached frames), this warps the
    segment's OWN start_image forward using MiDaS depth + pose chain.

    warp_mode="chain" (default) re-estimates MiDaS depth at each anchor,
    keeping per-step displacement small and valid to ~2 m+ total. This is
    the production-safe mode for forward cycling motion.

    warp_mode="single" warps from the start_image only; valid to ~0.9 m
    before the camera moves past visible geometry. Suitable for slow or
    lateral motion only.
    """

    def __init__(self, H: int, W: int, K_np: np.ndarray,
                 min_d: float = 0.5, max_d: float = 80.0,
                 cam_h: float = 1.55, latent_stride: int = 16,
                 warp_mode: str = "chain",
                 min_seg_coverage: float = 0.10,
                 min_anchor_coverage: float = 0.02,
                 midas_skip: int = 1,
                 use_cuda_warp: bool = True,
                 device: str = "cuda:0"):
        self.H = H
        self.W = W
        self.K = K_np
        self.fy_px = float(K_np[1, 1] * H)
        self.min_d = min_d
        self.max_d = max_d
        self.cam_h = cam_h
        self.latent_stride = latent_stride
        self.warp_mode = warp_mode
        self.min_seg_coverage = min_seg_coverage
        self.min_anchor_coverage = min_anchor_coverage
        self.midas_skip = max(1, int(midas_skip))
        self.use_cuda_warp = use_cuda_warp
        self.device = device
        self._midas_loaded = False

    def _ensure_midas(self):
        if not self._midas_loaded:
            from tools.streaming.extract_world_geometry import load_midas
            load_midas()
            self._midas_loaded = True

    def build_warm_composite(self, target_poses: np.ndarray,
                              start_image_path: str,
                              min_coverage: float = 0.0) -> dict | None:
        """Warp start_image forward to each anchor frame.

        Args:
            target_poses: [T, 4, 4] c2w poses for all T frames in the segment
            start_image_path: path to the segment's start_image PNG
            min_coverage: legacy compatibility (unused; see min_seg_coverage)

        Returns:
            warm_start_pack dict matching WorldWarmStartIndex format, or
            ``None`` if the per-segment coverage gate rejects the segment
            (mean valid-pixel fraction below ``min_seg_coverage``). When
            ``None`` is returned, the pipeline runs the FULL denoise schedule.
        """
        self._ensure_midas()
        target_poses = np.asarray(target_poses, dtype=np.float64)
        T = target_poses.shape[0]
        T_a = (T - 1) // 4 + 1  # anchor frames in causal VAE layout

        # Choose warp function: CUDA for speed, numpy fallback
        _warp_fn = forward_warp_cuda if self.use_cuda_warp else forward_warp

        # Load start image and compute MiDaS depth
        src_rgb = _load_image_rgb(start_image_path)
        if src_rgb.shape[:2] != (self.H, self.W):
            src_rgb = cv2.resize(src_rgb, (self.W, self.H))

        Z_src = midas_metric_depth_any(src_rgb, self.H, self.W,
                                       self.fy_px, cam_h=self.cam_h)
        D_src = depth_to_warp_norm(Z_src, self.min_d, self.max_d)
        w2c_src = _c2w_to_w2c(target_poses[0])

        # Warp to each anchor frame (frame 0, 4, 8, ..., 4*(T_a-1))
        comp = np.zeros((T_a, self.H, self.W, 3), dtype=np.uint8)
        per_anchor_valid = np.zeros((T_a, self.H, self.W), dtype=np.float32)
        per_anchor_cov = np.zeros(T_a, dtype=np.float32)
        n_cands = np.zeros(T_a, dtype=np.int64)

        # Chain state: for warp_mode="chain", each anchor warps from the
        # PREVIOUS anchor's filled image + freshly estimated MiDaS depth.
        # This keeps per-step displacement small (~0.5 m per anchor) so
        # disocclusion stays low. For "single" mode, every anchor warps
        # from the start_image's single depth map.
        chain_rgb = src_rgb
        chain_D = D_src
        chain_w2c = w2c_src

        for g in range(T_a):
            p = min(4 * g, T - 1)
            if g == 0:
                # Anchor 0 = the start image itself
                comp[g] = src_rgb
                per_anchor_valid[g] = 1.0
                per_anchor_cov[g] = 1.0
                n_cands[g] = 1
                continue

            w2c_tgt = _c2w_to_w2c(target_poses[p])

            if self.warp_mode == "chain":
                warp_src_rgb = chain_rgb
                warp_src_D = chain_D
                warp_src_w2c = chain_w2c
            else:
                warp_src_rgb = src_rgb
                warp_src_D = D_src
                warp_src_w2c = w2c_src

            if self.use_cuda_warp:
                warped, omega, valid = _warp_fn(
                    warp_src_rgb, warp_src_D, warp_src_w2c, w2c_tgt,
                    self.K, self.H, self.W, self.min_d, self.max_d,
                    device=self.device)
            else:
                warped, omega, valid = _warp_fn(
                    warp_src_rgb, warp_src_D, warp_src_w2c, w2c_tgt,
                    self.K, self.H, self.W, self.min_d, self.max_d)
            filled = fill_holes_fast(warped, valid, device=self.device,
                                     use_cuda=self.use_cuda_warp)

            comp[g] = filled
            per_anchor_valid[g] = valid.astype(np.float32)
            per_anchor_cov[g] = 1.0 - float(omega)
            n_cands[g] = 1

            # Advance chain state: re-estimate MiDaS depth on the filled
            # image so the next warp step has a depth map that matches the
            # current viewpoint (not the stale start-image depth).
            # With midas_skip > 1, skip re-estimation at most anchors to
            # save time — the depth changes slowly over ~0.8m steps.
            if self.warp_mode == "chain":
                chain_rgb = filled
                chain_w2c = w2c_tgt
                if g % self.midas_skip == 0:
                    Z_chain = midas_metric_depth_any(
                        filled, self.H, self.W,
                        self.fy_px, cam_h=self.cam_h)
                    chain_D = depth_to_warp_norm(Z_chain, self.min_d, self.max_d)
                # else: reuse chain_D from previous step (stale but close)

        # Per-segment coverage gate: if the mean valid-pixel fraction
        # (excluding frame 0) is too low, the truncated denoise schedule
        # would produce worse results than full denoise. Return None so
        # the pipeline runs cold (full steps).
        seg_cov = float(per_anchor_cov[1:].mean()) if T_a > 1 else 0.0
        if T_a > 1 and seg_cov < self.min_seg_coverage:
            return None

        # Build the per-pixel coverage mask at LATENT resolution from the
        # actual warp validity — NOT hardcoded to 1.0. Each latent cell
        # gets the mean valid fraction of its constituent pixels (INTER_AREA
        # = box-average downsample). Anchors below the per-anchor threshold
        # get a zero mask (no warm-start for unreliable anchors). Frame 0
        # is always zero (locked to start_image by the pipeline).
        lat_h = self.H // self.latent_stride
        lat_w = self.W // self.latent_stride
        cov_lat = np.zeros((T_a, lat_h, lat_w), dtype=np.float32)
        for g in range(1, T_a):
            if per_anchor_cov[g] < self.min_anchor_coverage:
                continue  # cov_lat[g] stays 0 — unreliable warp
            cov_lat[g] = cv2.resize(
                per_anchor_valid[g], (lat_w, lat_h),
                interpolation=cv2.INTER_AREA).astype(np.float32)

        return {
            "anchors": comp,                                    # [T_a,H,W,3]
            "coverage_latent": cov_lat,                         # [T_a,lat_h,lat_w]
            "coverage_hw": np.repeat(per_anchor_cov[:, None, None],
                                     self.H * self.W, axis=1).reshape(T_a, self.H, self.W),
            "coverage": seg_cov,
            "anchor_coverage": per_anchor_cov,
            "n_candidates": n_cands,
            "n_anchors_hit": int((per_anchor_cov[1:] > 0).sum()) if T_a > 1 else 0,
            "n_anchors_total": int(T_a),
            "min_coverage": float(min_coverage),
            "_source": f"warp_fill_{self.warp_mode}",
        }


def build_warp_fill_prior(target_poses: np.ndarray, start_image_path: str,
                          H: int, W: int, K_np: np.ndarray,
                          min_d: float = 0.5, max_d: float = 80.0,
                          cam_h: float = 1.55, latent_stride: int = 16,
                          warp_mode: str = "chain",
                          min_seg_coverage: float = 0.10,
                          min_anchor_coverage: float = 0.02,
                          midas_skip: int = 1,
                          use_cuda_warp: bool = True,
                          device: str = "cuda:0") -> dict | None:
    """Convenience function: build a warp+fill warm-start pack.

    Wraps WarpFillPriorIndex for one-shot use (no state across segments).
    For multi-segment runs, instantiate WarpFillPriorIndex once and reuse
    (MiDaS loads once).
    """
    idx = WarpFillPriorIndex(H, W, K_np, min_d, max_d, cam_h,
                             latent_stride, warp_mode,
                             min_seg_coverage, min_anchor_coverage,
                             midas_skip, use_cuda_warp, device)
    return idx.build_warm_composite(target_poses, start_image_path)
