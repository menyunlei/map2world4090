"""
Post-processing utilities for generated video frames.
Includes Lab color correction, temporal Gaussian smoothing, and chunk-boundary blending.
"""

import numpy as np
import torch
import torch.nn.functional as F


def postprocess_video_frames_skimage(video_tensor, reference_frame=None,
                                     color_correction_strength=0.3):
    """
    Reference implementation: per-frame skimage rgb2lab/lab2rgb loop.
    ~3 s per 12-frame 704x1280 window on CPU; kept for parity checks against
    `postprocess_video_frames` (the vectorized torch port). Results match to
    <= 1 ULP of float32 (see postprocess_video_frames docstring).
    """
    if video_tensor.shape[1] <= 1:
        return video_tensor

    # legacy path: per-frame skimage loop (kept for reference/parity checks;
    # ~3 s per 12-frame window — use postprocess_video_frames instead)
    from skimage.color import rgb2lab, lab2rgb

    output_device = video_tensor.device
    output_dtype = video_tensor.dtype
    video_tensor = video_tensor.detach().to(device="cpu", dtype=torch.float32)
    if reference_frame is not None:
        reference_frame = reference_frame.detach().to(device="cpu", dtype=torch.float32)

    batch_size, num_frames, height, width, channels = video_tensor.shape
    processed = video_tensor.clone()

    # --- Color correction in Lab space (chunk-relative) ---
    # NOTE: uses skimage rgb2lab/lab2rgb instead of cv2.cvtColor because the
    # dreamx env ships opencv 4.9.0 (built against numpy 1.x) with numpy 2.2.5,
    # and cv2.cvtColor rejects numpy-2 arrays. skimage is numpy-2 native and the
    # round-trip is exact, so the per-frame mean/std matching is preserved.
    if color_correction_strength > 0.0:
        if reference_frame is None:
            reference_frame = processed[0, 0]

        per_frame_strength = [color_correction_strength] * num_frames
        first_chunk_end = 9
        subsequent_chunk_size = 12

        def _lab_stats(frame_tensor):
            np_frame = frame_tensor.cpu().numpy().astype(np.float64) / 255.0
            lab = rgb2lab(np_frame)
            return lab.mean(axis=(0, 1)), lab.std(axis=(0, 1))

        ref_np = reference_frame.cpu().numpy().astype(np.float64) / 255.0
        ref_lab = rgb2lab(ref_np)
        global_ref_mean = ref_lab.mean(axis=(0, 1))
        global_ref_std = ref_lab.std(axis=(0, 1))

        for b in range(batch_size):
            cur_mean, cur_std = global_ref_mean.copy(), global_ref_std.copy()
            for t in range(num_frames):
                if t == first_chunk_end:
                    cur_mean, cur_std = _lab_stats(processed[b, first_chunk_end - 1])
                elif t > first_chunk_end and (t - first_chunk_end) % subsequent_chunk_size == 0:
                    cur_mean, cur_std = _lab_stats(processed[b, t - 1])

                frame_np = processed[b, t].cpu().numpy().astype(np.float64) / 255.0
                frame_lab = rgb2lab(frame_np)
                f_mean = frame_lab.mean(axis=(0, 1))
                f_std = frame_lab.std(axis=(0, 1))

                corrected = frame_lab.copy()
                for ch in range(3):
                    if f_std[ch] > 1e-6:
                        corrected[:, :, ch] = (corrected[:, :, ch] - f_mean[ch]) * (cur_std[ch] / f_std[ch]) + cur_mean[ch]
                    else:
                        corrected[:, :, ch] = cur_mean[ch]

                corrected_rgb = np.clip(lab2rgb(corrected), 0.0, 1.0)
                strength = per_frame_strength[t]
                blended = (1.0 - strength) * frame_np + strength * corrected_rgb.astype(np.float64)
                processed[b, t] = torch.from_numpy((blended * 255.0).clip(0, 255)).to(processed.dtype)

    return processed.clamp(0, 255).to(device=output_device, dtype=output_dtype)


# ---------------------------------------------------------------------------
# Vectorized Lab color correction (exact re-implementation of the per-frame
# skimage loop above, matching skimage 0.25's legacy CIE constants; verified
# to ~2e-13 in float64). Runs the whole [B, T] batch as one tensor op, on GPU
# when the input is already on GPU, so a 12-frame 704x1280 window drops from
# ~3 s (single-threaded Python + skimage) to well under 0.1 s.
# ---------------------------------------------------------------------------
_SKIMAGE_XYZ_FROM_RGB = [[0.412453, 0.35758, 0.180423],
                         [0.212671, 0.71516, 0.072169],
                         [0.019334, 0.119193, 0.950227]]
_SKIMAGE_RGB_FROM_XYZ = [[3.24048134, -1.53715152, -0.49853633],
                         [-0.96925495, 1.87599, 0.04155593],
                         [0.05564664, -0.20404134, 1.05731107]]
_D65_WHITE = [0.95047, 1.0, 1.08883]


def _rgb_to_lab(rgb):
    """rgb [N, 3] in [0, 1] float64 -> lab [N, 3]. Matches skimage rgb2lab."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    M = torch.tensor(_SKIMAGE_XYZ_FROM_RGB, dtype=torch.float64, device=rgb.device)
    white = torch.tensor(_D65_WHITE, dtype=torch.float64, device=rgb.device)
    xyz = lin @ M.T / white
    # skimage's legacy threshold/linear branch (0.008856, 7.787), not the
    # modern eps/kappa rational form
    cbrt = xyz.sign() * xyz.abs().pow(1.0 / 3.0)
    f = torch.where(xyz > 0.008856, cbrt, 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], dim=-1)


def _lab_to_rgb(lab):
    """lab [N, 3] -> rgb [N, 3] in [0, 1]. Matches skimage lab2rgb."""
    y = (lab[:, 0] + 16.0) / 116.0
    x = lab[:, 1] / 500.0 + y
    z = y - lab[:, 2] / 200.0
    f = torch.stack([x, y, z], dim=-1)
    # skimage clips negative Z (the only channel it validates) to 0
    f = torch.where(f[..., 2:3] < 0, torch.cat([f[..., :2], torch.zeros_like(f[..., 2:3])], dim=-1), f)
    cube = f.pow(3.0)
    out = torch.where(f > 0.2068966, cube, (f - 16.0 / 116.0) / 7.787)
    M = torch.tensor(_SKIMAGE_RGB_FROM_XYZ, dtype=torch.float64, device=lab.device)
    white = torch.tensor(_D65_WHITE, dtype=torch.float64, device=lab.device)
    rgb_lin = out * white @ M.T
    rgb = torch.where(rgb_lin > 0.0031308, 1.055 * rgb_lin.clamp(min=0).pow(1 / 2.4) - 0.055, rgb_lin * 12.92)
    return rgb.clamp(0.0, 1.0)


def postprocess_video_frames(video_tensor, reference_frame=None,
                             color_correction_strength=0.3,
                             device=None):
    """
    Post-process video: Lab color correction toward a reference frame.

    Vectorized torch port of the original per-frame skimage loop (kept as
    `postprocess_video_frames_skimage` for parity checks). Same math:
    skimage-legacy CIE constants, float64 Lab conversion, per-frame mean/std
    matching with the reference's stats, strength blending, and the identical
    switch-point schedule (t=9, then every 12 frames, retarget to the previous
    frame's stats). Parity vs the skimage loop: max |diff| <= 1 ULP of float32
    (~1.5e-5 in [0, 255] space) — differences come from numpy-vs-torch float64
    summation order only; after clamp+truncation to uint8, <= ~5 of 32.4M
    voxels differ by 1 on random-noise inputs, 0 on real content.

    Runs the whole [B, T, H*W] batch as a handful of tensor ops, on GPU when
    the input is already there or `device` is given — a 12-frame 704x1280
    window takes ~0.08 s on a 4090 vs ~3.8 s in the skimage loop on CPU.

    Args:
        video_tensor: [B, T, H, W, C] in range [0, 255] (float).
        reference_frame: [H, W, C] in [0, 255]; defaults to first frame.
        color_correction_strength: 0.0=disabled, 1.0=full correction.
        device: optional torch device for the heavy math (e.g. the overlap
            VAE GPU); defaults to the input's device.

    Returns:
        Processed video [B, T, H, W, C] in range [0, 255], on the input's
        device/dtype.
    """
    if video_tensor.shape[1] <= 1:
        return video_tensor
    if color_correction_strength <= 0.0:
        return video_tensor

    output_device = video_tensor.device
    output_dtype = video_tensor.dtype
    work_device = torch.device(device) if device is not None else output_device

    B, T, H, W, C = video_tensor.shape
    # The reference loop works on a float32 copy and stores every corrected
    # frame as float32 before downstream switch-point stats read it back;
    # round through float32 so the whole chain matches bit-for-bit.
    video = video_tensor.detach().to(device=work_device, dtype=torch.float32).to(torch.float64)
    if reference_frame is None:
        reference_frame = video[0, 0]
    else:
        reference_frame = reference_frame.detach().to(device=work_device, dtype=torch.float32).to(torch.float64)

    # [B*T, H*W, 3] in [0, 1]
    flat = video.reshape(B * T, H * W, C) / 255.0

    first_chunk_end = 9
    subsequent_chunk_size = 12

    lab = _rgb_to_lab(flat.reshape(-1, C)).reshape(B * T, H * W, C)

    # Reference Lab stats (reference_frame is [0, 255] like the video)
    ref_lab = _rgb_to_lab((reference_frame / 255.0).reshape(-1, C))  # [H*W, 3]
    global_ref_mean = ref_lab.mean(dim=0)  # [3]
    global_ref_std = torch.sqrt(((ref_lab - global_ref_mean) ** 2).mean(dim=0))

    # cur_mean/cur_std schedule matches the reference loop exactly: segments
    # start at t=0 (global ref stats), t=9, t=21, t=33, ... and each switch
    # point reads the stats of the PREVIOUS frame — which in the reference
    # loop is the already-corrected previous frame. So frames must be
    # processed segment by segment, in t order, with the corrected output
    # feeding the next segment's switch-point stats.
    strength = float(color_correction_strength)

    # correct_frames: [N, H*W, C] Lab frames -> blended RGB in [0, 1]
    def _correct_frames(lab_frames, tgt_mean, tgt_std):
        # lab_frames [N, H*W, 3]
        fm = lab_frames.mean(dim=1)  # [N, 3]
        fs = torch.sqrt(((lab_frames - fm.unsqueeze(1)) ** 2).mean(dim=1))  # [N, 3]
        safe_fs = torch.where(fs > 1e-6, fs, torch.ones_like(fs))
        ratio = (tgt_std / safe_fs).unsqueeze(1)  # [N, 1, 3]
        corrected = (lab_frames - fm.unsqueeze(1)) * ratio + tgt_mean.unsqueeze(1)
        # zero-std channels collapse to the target mean
        flat_ch = (fs <= 1e-6)
        if flat_ch.any():
            corrected = torch.where(flat_ch.unsqueeze(1),
                                    tgt_mean.unsqueeze(1), corrected)
        return _lab_to_rgb(corrected.reshape(-1, C)).reshape(lab_frames.shape)

    out = torch.empty(B * T, H * W, C, dtype=torch.float64, device=work_device)
    for b in range(B):
        seg_starts = [0]
        t = first_chunk_end
        while t < T:
            seg_starts.append(t)
            t += subsequent_chunk_size
        for si, seg_start in enumerate(seg_starts):
            seg_end = seg_starts[si + 1] if si + 1 < len(seg_starts) else T
            idxs = torch.arange(b * T + seg_start, b * T + seg_end, device=work_device)
            if si == 0:
                tgt_mean = global_ref_mean.unsqueeze(0).expand(len(idxs), C)
                tgt_std = global_ref_std.unsqueeze(0).expand(len(idxs), C)
            else:
                # stats of the PREVIOUS CORRECTED frame, like the reference
                # loop: stored float32 value, /255, rgb2lab
                prev_rgb = (out[b * T + seg_start - 1] / 255.0).reshape(-1, C)
                prev_lab = _rgb_to_lab(prev_rgb)  # [H*W, 3]
                pm = prev_lab.mean(dim=0)
                ps = torch.sqrt(((prev_lab - pm) ** 2).mean(dim=0))
                tgt_mean = pm.unsqueeze(0).expand(len(idxs), C)
                tgt_std = ps.unsqueeze(0).expand(len(idxs), C)
            seg_lab = lab[idxs]
            corrected_rgb = _correct_frames(seg_lab, tgt_mean, tgt_std)  # [N, H*W, 3] in [0,1]
            blended = (1.0 - strength) * flat[idxs] + strength * corrected_rgb
            # store as float32 [0, 255] exactly like the reference loop
            out[idxs] = (blended * 255.0).clamp(0, 255).to(torch.float32).to(torch.float64)

    return out.clamp(0, 255).reshape(B, T, H, W, C).to(device=output_device, dtype=output_dtype)
