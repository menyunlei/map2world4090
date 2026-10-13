"""World 3D map (brick 1 of the 3D pivot): full-frame geometrically-correct
novel-view priors from MVS depth, replacing world_warm_start's ground-plane
warp composite (which collapsed into radial anisotropic smear under forward
translation — PIVOT_IDENTITY_VERDICT.md §5).

Mechanism. Every generated segment's latent-anchor frames are registered with
their EXACT executed c2w poses + per-pixel metric depth (COLMAP known-pose MVS
on this route's bank keyframes; the pose source is the plan chain, not SfM).
To build a prior for target anchor pose P_t, the n nearest earlier source
anchors are unprojected with their depth into the world and z-buffer-splatted
into P_t (nearest-z wins). Full frame — buildings and facades included, not
just the ground band; points beyond max_depth_m (sky misfires) are dropped.

Constraint parity with the 2.5D arms (scientific comparability): sources are
restricted to STRICTLY EARLIER segments (per-segment source gating is the
caller's job via seg_idx on each registered anchor).

All CPU + torch float32, no CUDA: the DiT ranks run with full CPU offload and
any transient CUDA allocation risks OOM. Measured cost target: <30 s per
segment (31 anchors) on this box.
"""

import os
from collections import OrderedDict

import cv2
import numpy as np
import torch


def read_colmap_depth(path):
    """COLMAP 3.12 'W&H&C&' ascii-header float32 single-channel bin."""
    data = open(path, "rb").read()
    i, vals = 0, []
    for _ in range(3):
        j = data.index(b"&", i)
        vals.append(int(data[i:j]))
        i = j + 1
    W, H, C = vals
    return np.frombuffer(data[i:], dtype=np.float32).reshape(H, W, C)[..., 0]


class World3DMapIndex:
    """Depth-registered anchor store + z-buffer novel-view splat renderer."""

    def __init__(self, fx, fy, cx, cy, H=704, W=1280,
                 min_depth_m=0.5, max_depth_m=150.0,
                 n_src=8, max_pose_dist_m=80.0, max_yaw_delta_deg=110.0,
                 substep=1, dilate=1, cache_size=24):
        self.fx, self.fy, self.cx, self.cy = map(float, (fx, fy, cx, cy))
        self.H, self.W = int(H), int(W)
        self.min_d, self.max_d = float(min_depth_m), float(max_depth_m)
        self.n_src = int(n_src)
        self.max_dist = float(max_pose_dist_m)
        self.max_dyaw = float(max_yaw_delta_deg)
        self.substep = int(substep)          # source pixel subsample (1 = full)
        self.dilate = int(dilate)            # coverage/hole infill dilation px
        self.cache_size = int(cache_size)
        self._anchors = []                   # dicts: seg_idx, g, c2w, pos, yaw, rgb_path, depth_path
        self._cache = OrderedDict()          # aid -> (rgb_f32_sub, depth_sub)
        us = (torch.arange(self.W) + 0.5)[None, :].expand(self.H, self.W)
        vs = (torch.arange(self.H)[:, None] + 0.5).expand(self.H, self.W)
        self._xn = ((us - self.cx) / self.fx)[:: self.substep, :: self.substep].reshape(-1)
        self._yn = ((vs - self.cy) / self.fy)[:: self.substep, :: self.substep].reshape(-1)

    # ------------------------------------------------------------------ store
    def _load(self, aid):
        if aid in self._cache:
            self._cache.move_to_end(aid)
            return self._cache[aid]
        a = self._anchors[aid]
        rgb = cv2.imread(a["rgb_path"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        d = read_colmap_depth(a["depth_path"]).astype(np.float32)
        s = self.substep
        if s > 1:
            rgb = rgb[::s, ::s]
            d = d[::s, ::s]
        out = (torch.from_numpy(np.ascontiguousarray(rgb)).reshape(-1, 3),
               torch.from_numpy(np.ascontiguousarray(d)).reshape(-1))
        self._cache[aid] = out
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return out

    def add_frame(self, seg_idx, g, c2w, rgb_path, depth_path):
        c2w = np.asarray(c2w, dtype=np.float64)
        pos = c2w[:3, 3]
        fwd = c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])
        yaw = float(np.degrees(np.arctan2(-fwd[0], fwd[2])))
        aid = len(self._anchors)
        self._anchors.append({"seg_idx": int(seg_idx), "g": int(g), "c2w": c2w,
                              "pos": pos, "yaw": yaw,
                              "rgb_path": rgb_path, "depth_path": depth_path})
        return aid

    # -------------------------------------------------------------- selection
    def candidates(self, target_pos, target_yaw, seg_lo=0, seg_hi=None):
        seg_hi = 10 ** 9 if seg_hi is None else seg_hi
        out = []
        for aid, a in enumerate(self._anchors):
            if not (seg_lo <= a["seg_idx"] < seg_hi):
                continue
            d = float(np.linalg.norm(a["pos"] - target_pos))
            dy = abs((a["yaw"] - target_yaw + 180.0) % 360.0 - 180.0)
            if d < self.max_dist and dy < self.max_dyaw:
                out.append((d, aid))
        out.sort()
        return [aid for _, aid in out[: self.n_src]]

    # ---------------------------------------------------------------- render
    def render(self, target_c2w, seg_lo=0, seg_hi=None, return_sources=False):
        """Splat the selected earlier sources into the target pose.
        Returns (rgb_u8 [H,W,3], coverage [H,W] f32)."""
        target_c2w = np.asarray(target_c2w, dtype=np.float64)
        w2c = torch.from_numpy(np.linalg.inv(target_c2w).astype(np.float32))
        tpos = target_c2w[:3, 3]
        fwd = target_c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])
        tyaw = float(np.degrees(np.arctan2(-fwd[0], fwd[2])))
        cands = self.candidates(tpos, tyaw, seg_lo=seg_lo, seg_hi=seg_hi)

        zs = torch.full((self.H * self.W,), float("inf"))
        rgb = torch.zeros((self.H * self.W, 3))
        dirx = torch.full_like(zs, -1)

        for aid in cands:
            a = self._anchors[aid]
            c2w_s = torch.from_numpy(a["c2w"].astype(np.float32))
            src_rgb, d = self._load(aid)
            valid = (d > self.min_d) & (d < self.max_d)
            if not valid.any():
                continue
            xn, yn = self._xn[valid], self._yn[valid]
            dv = d[valid]
            pc = torch.stack([xn * dv, yn * dv, dv], 1)            # cam frame
            pw = pc @ c2w_s[:3, :3].T + c2w_s[:3, 3]               # world
            pt = pw @ w2c[:3, :3].T + w2c[:3, 3]                   # target cam
            z = pt[:, 2]
            ok = z > 0.05
            if not ok.any():
                continue
            z = z[ok]
            u = (self.fx * pt[ok, 0] / z + self.cx)
            v = (self.fy * pt[ok, 1] / z + self.cy)
            ui = torch.floor(u).long()
            vi = torch.floor(v).long()
            inb = (ui >= 0) & (ui < self.W) & (vi >= 0) & (vi < self.H)
            if not inb.any():
                continue
            lin = (vi[inb] * self.W + ui[inb])
            zz = z[inb]
            col = src_rgb[valid][ok][inb]
            # ascending-order write: nearest z survives (plain scatter assign)
            ord_ = torch.argsort(zz, descending=True)  # farthest first
            lin, zz, col = lin[ord_], zz[ord_], col[ord_]
            # 2px splat suppresses 1px z-buffer holes (each write re-checks nearest-z)
            for off in (0, 1, self.W, self.W + 1):
                l = lin + off
                lim = l < self.H * self.W
                l, z2, c2 = l[lim], zz[lim], col[lim]
                better = z2 < zs[l]
                idx2 = l[better]
                zs[idx2] = z2[better]
                rgb[idx2] = c2[better]
                dirx[idx2] = aid

        cov = torch.isfinite(zs).float().reshape(self.H, self.W)
        img = rgb.reshape(self.H, self.W, 3)
        if self.dilate > 0:
            cv = cov.numpy()[None, None]
            k = 2 * self.dilate + 1
            dil = torch.nn.functional.max_pool2d(torch.from_numpy(cv), k,
                                                 stride=1, padding=self.dilate)[0, 0]
            grow = (dil > 0) & (cov == 0)
            if grow.any():
                kk = np.ones((k, k), np.float32)
                s = torch.from_numpy(np.ones_like(cv[0])) 
                num = cv2.filter2D((img * cov[..., None]).numpy(), -1, kk)
                den = cv2.filter2D(cov.numpy(), -1, kk)
                den = np.maximum(den, 1e-6)
                fill = torch.from_numpy((num / den[..., None]).astype(np.float32))
                img[grow] = fill[grow]
                cov = dil
        out = (img.clamp(0, 1) * 255.0).round().byte().numpy()
        # 3x3 median only where covered: crushes z-fighting salt noise without
        # touching genuinely sharp content much; uncovered stays exactly zero.
        med = cv2.medianBlur(out, 3)
        out = np.where((cov > 0)[..., None], med, out).astype(np.uint8)
        if return_sources:
            return out, cov.numpy(), cands
        return out, cov.numpy()


class FusedVoxelMap:
    """Offline-fused metric voxel map (tools/fuse_voxel_map.py output).

    Single-valued coherent splat source: no per-source z-fighting, no runtime
    depth needed. minseg gating (voxels with min_seg < seg_hi) preserves the
    strictly-earlier-segments rule of the 2.5D arms. CPU torch only."""

    def __init__(self, npz_path, splat=1, z_tol=0.05, chunk=6_000_000,
                 min_cnt=1, min_support=2.5, tol_slope=0.03, post_median=5):
        d = np.load(npz_path)
        xyz = np.asarray(d["vox_xyz"])
        rgb = np.asarray(d["vox_rgb"])
        cnt = np.asarray(d["vox_cnt"])
        okc = cnt >= int(min_cnt)
        if min_support > 0.0 and "vox_support" in d.files:
            # neighbor-supported evidence only: kills airborne speckle while
            # keeping thin early-segment surfaces a cnt filter would drop
            okc &= np.asarray(d["vox_support"]) >= float(min_support)
        self.xyz = torch.from_numpy(xyz[okc])                            # float32 [M,3]
        self.rgb = torch.from_numpy(rgb[okc]).float() / 255.0
        self.minseg = torch.from_numpy(np.asarray(d["vox_minseg"])[okc]).long()
        K = np.asarray(d["K_px"])
        self.fx, self.fy, self.cx, self.cy = map(float, K)
        H, W = np.asarray(d["HW"])
        self.H, self.W = int(H), int(W)
        self.splat = int(splat)
        self.z_tol = float(z_tol)          # additive floor of the z acceptance window
        self.tol_slope = float(tol_slope)  # depth-proportional part (noise ~ z)
        self.chunk = int(chunk)
        self.post_median = int(post_median)  # covered-only despeckle; 0 disables

    def render(self, target_c2w, seg_hi):
        """-> (rgb_u8 [H,W,3], coverage [H,W] float32)

        One projection, chunked: pass 1 builds the z-buffer (with 2px splat
        growth), caching per-chunk (lin, z, color); pass 2 paints points whose
        z is within z_tol of the winning buffer value, using the same splat
        offsets so splat-grown pixels get coherent colors.
        """
        w2c = torch.from_numpy(
            np.linalg.inv(np.asarray(target_c2w, np.float64)).astype(np.float32))
        keep = torch.nonzero(self.minseg < int(seg_hi)).reshape(-1)
        H, W = self.H, self.W
        offs = (0,) if not self.splat else (0, 1, W, W + 1)
        zs = torch.full((H * W,), float("inf"))
        cache = []
        for s in range(0, keep.numel(), self.chunk):
            sub = keep[s: s + self.chunk]
            pt = self.xyz[sub] @ w2c[:3, :3].T + w2c[:3, 3]
            z = pt[:, 2]
            ok = z > 0.1
            if not ok.any():
                continue
            z = z[ok]
            pt = pt[ok]
            col = self.rgb[sub[ok]]
            u = self.fx * pt[:, 0] / z + self.cx
            v = self.fy * pt[:, 1] / z + self.cy
            ui = torch.floor(u).long()
            vi = torch.floor(v).long()
            inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not inb.any():
                continue
            lin = vi[inb] * W + ui[inb]
            zz = z[inb]
            col = col[inb]
            for off in offs:
                l = lin + off
                sel = l < H * W
                l, z2 = l[sel], zz[sel]
                upd = z2 < zs[l]
                zs[l[upd]] = z2[upd]
            cache.append((lin, zz, col))
        img = torch.zeros((H * W, 3))
        cov = torch.zeros(H * W)
        for lin, zz, col in cache:
            for off in offs:
                l = lin + off
                sel = l < H * W
                l, z2, c2 = l[sel], zz[sel], col[sel]
                near = z2 <= zs[l] + self.z_tol + self.tol_slope * z2
                if near.any():
                    ln = l[near]
                    img[ln] = c2[near]
                    cov[ln] = 1.0
        img = img.reshape(H, W, 3).clamp(0, 1)
        out = (img * 255.0).round().byte().numpy()
        cov = cov.reshape(H, W).numpy()
        if self.post_median >= 3 and (cov > 0).any():
            import cv2 as _cv2
            blur = _cv2.medianBlur(out, self.post_median)
            cm = cov > 0
            out[cm] = blur[cm]
        return out, cov

    def render_soft(self, target_c2w, seg_hi):
        """Slab-averaged variant: pass 2 AVERAGES colors of all voxels within
        z_tol + tol_slope*z of the z-buffer winner instead of taking the
        nearest, so per-view depth noise interleaved across voxels averages
        back to the surface colour. A covered-only 3x3 median then removes
        the residual airborne speckle. -> (rgb_u8 [H,W,3], coverage [H,W] f32)
        """
        import cv2 as _cv2
        w2c = torch.from_numpy(
            np.linalg.inv(np.asarray(target_c2w, np.float64)).astype(np.float32))
        keep = torch.nonzero(self.minseg < int(seg_hi)).reshape(-1)
        H, W = self.H, self.W
        HWn = H * W
        zs = torch.full((HWn,), float("inf"))
        cache = []
        for s in range(0, keep.numel(), self.chunk):
            sub = keep[s: s + self.chunk]
            pt = self.xyz[sub] @ w2c[:3, :3].T + w2c[:3, 3]
            z = pt[:, 2]
            ok = z > 0.1
            if not ok.any():
                continue
            z = z[ok]
            pt = pt[ok]
            col = self.rgb[sub[ok]]
            u = self.fx * pt[:, 0] / z + self.cx
            v = self.fy * pt[:, 1] / z + self.cy
            ui = torch.floor(u).long()
            vi = torch.floor(v).long()
            inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not inb.any():
                continue
            lin = vi[inb] * W + ui[inb]
            zz = z[inb]
            col = col[inb]
            upd = zz < zs[lin]
            zs[lin[upd]] = zz[upd]
            cache.append((lin, zz, col))
        img_sum = torch.zeros((HWn, 3))
        cnt = torch.zeros(HWn)
        for lin, zz, col in cache:
            near = zz <= zs[lin] + self.z_tol + self.tol_slope * zz
            if near.any():
                img_sum.index_add_(0, lin[near], col[near])
                cnt.index_add_(0, lin[near], torch.ones(int(near.sum())))
        cov = (cnt > 0).float()
        img = torch.zeros((HWn, 3))
        m = cnt > 0
        img[m] = img_sum[m] / cnt[m].unsqueeze(1)
        out = (img.reshape(H, W, 3).clamp(0, 1) * 255.0).round().byte().numpy()
        covn = cov.reshape(H, W).numpy()
        if m.any():
            blur = _cv2.medianBlur(out, 3)
            cm = covn > 0
            out[cm] = blur[cm]
        return out, covn


class BakedPriorIndex:
    """Offline-baked composite prior (tools/bake_composites.py output dir).

    The map-driven pose chain is deterministic, so the per-segment anchor
    composites are pre-rendered from strictly-earlier bank keyframes with the
    per-source z-buffer splat renderer. Runtime = PNG loads only. Same pack
    contract as WorldWarmStartIndex.build_warm_composite; seg gating is baked
    into the files, add_segment just advances the cursor.
    """

    def __init__(self, bake_dir, latent_stride=16, lowpass=4):
        self.dir = str(bake_dir)
        self.latent_stride = int(latent_stride)
        self.lowpass = int(lowpass)   # 4 = quarter-res low-pass: give the DiT
        self.seg = 0                  # structure+colour, let 10 repaint steps
        self._anchors = []            # synthesise texture (salt priors get
                                      # followed instead of cleaned)

    def add_segment(self, seg_idx, sample, poses):
        self.seg = int(seg_idx) + 1
        self._anchors.append(int(seg_idx))

    def build_warm_composite(self, target_poses, min_coverage=0.5):
        T = np.asarray(target_poses).shape[0]
        T_a = (T - 1) // 4 + 1
        H, W = 704, 1280
        comp = np.zeros((T_a, H, W, 3), np.uint8)
        cov_hw = np.zeros((T_a, H, W), np.float32)
        per_anchor_cov = np.zeros(T_a, np.float32)
        import cv2 as _cv2
        for g in range(1, T_a):
            stem = os.path.join(self.dir, f"seg{self.seg:02d}_a{g:02d}")
            rgb_p, cov_p = stem + ".png", stem + "_cov.png"
            if not (os.path.exists(rgb_p) and os.path.exists(cov_p)):
                continue
            img = _cv2.cvtColor(_cv2.imread(rgb_p), _cv2.COLOR_BGR2RGB)
            cov = _cv2.imread(cov_p, _cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if img is None or img.shape[:2] != (H, W):
                continue
            if self.lowpass > 1:
                img = _cv2.resize(img, (W // self.lowpass, H // self.lowpass),
                                  interpolation=_cv2.INTER_AREA)
                img = _cv2.resize(img, (W, H), interpolation=_cv2.INTER_CUBIC)
            comp[g] = img
            cov_hw[g] = cov
            per_anchor_cov[g] = float((cov > 0).mean())
        lat_h, lat_w = H // self.latent_stride, W // self.latent_stride
        cov_lat = cov_hw.reshape(T_a, lat_h, self.latent_stride,
                                 lat_w, self.latent_stride).mean(axis=(2, 4))
        cov_lat = (cov_lat >= 0.5).astype(np.float32)
        cov_lat[0] = 0.0  # frame 0 locked to the start image
        return {
            "anchors": comp,
            "coverage_latent": cov_lat,
            "coverage_hw": cov_hw,
            "coverage": float(per_anchor_cov[1:].mean()) if T_a > 1 else 0.0,
            "anchor_coverage": per_anchor_cov,
            "n_anchors_total": int(T_a),
            "n_anchors_hit": int((per_anchor_cov[1:] > 0).sum()),
            "n_candidates": np.zeros(T_a, np.int64),
        }


class Render3DPriorIndex:
    """Drop-in for WorldWarmStartIndex backed by an offline-fused voxel map.

    Same pack contract as WorldWarmStartIndex.build_warm_composite; gating by
    seg_hi = current segment preserves the strictly-earlier rule. add_segment
    only advances the segment cursor (content lives in the bank-fused map).
    """

    def __init__(self, fused_npz, latent_stride=16):
        self.map = FusedVoxelMap(fused_npz)
        self.latent_stride = int(latent_stride)
        self.seg = 0
        self._anchors = []   # exposed for the cold-log's len() usage

    def add_segment(self, seg_idx, sample, poses):
        self.seg = int(seg_idx) + 1
        self._anchors.append(int(seg_idx))

    def build_warm_composite(self, target_poses, min_coverage=0.5):
        target_poses = np.asarray(target_poses)
        T = target_poses.shape[0]
        T_a = (T - 1) // 4 + 1
        H, W = self.map.H, self.map.W
        comp = np.zeros((T_a, H, W, 3), np.uint8)
        cov_hw = np.zeros((T_a, H, W), np.float32)
        per_anchor_cov = np.zeros(T_a, np.float32)
        for g in range(T_a):
            p = min(4 * g, T - 1)
            img, cov = self.map.render(target_poses[p], seg_hi=self.seg)
            comp[g] = img
            cov_hw[g] = cov
            per_anchor_cov[g] = float(cov.mean())
        lat_h = H // self.latent_stride
        lat_w = W // self.latent_stride
        cov_lat = cov_hw.reshape(T_a, lat_h, self.latent_stride,
                                 lat_w, self.latent_stride).mean(axis=(2, 4))
        cov_lat = (cov_lat >= 0.5).astype(np.float32)
        cov_lat[0] = 0.0  # frame 0 locked to the start image
        seg_cov = float(per_anchor_cov[1:].mean()) if T_a > 1 else 0.0
        return {
            "anchors": comp,
            "coverage_latent": cov_lat,
            "coverage_hw": cov_hw,
            "coverage": seg_cov,
            "anchor_coverage": per_anchor_cov,
            "n_candidates": np.zeros(T_a, np.int64),
            "n_anchors_hit": int((per_anchor_cov[1:] > 0).sum()) if T_a > 1 else 0,
            "n_anchors_total": int(T_a),
            "min_coverage": float(min_coverage),
        }
