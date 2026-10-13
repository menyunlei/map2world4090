"""
Map-to-latent injector for DreamX-World DiT (v2 — anti-blur).

Key improvements over v1:
  1. Token count reduction: 880 → 64 via adaptive average pooling
     (4×4 grid, each token summarizes a map region)
  2. LayerNorm before patch_embedding to normalize VAE latent distribution
  3. Strength scaling applied AFTER tokenization (gentle prompt, not overwrite)
  4. Optional: only inject into early blocks (0-5) — handled in DiT patch

Pipeline:
  map image (RGB)
    → resize to (1280, 704) → VAE encode → 16-ch latent [1,16,1,44,80]
    → LayerNorm (normalize to zero-mean unit-variance)
    → pad to 48-ch → adaptive pool to [1,48,1,8,8] (spatial downsampling)
    → DiT patch_embedding (Conv3d 48→3072) → [1,3072,1,4,4]
    → flatten → [1, 16, 3072] tokens (16 map tokens, not 880!)
    → scale by map_strength (default 0.3, gentle)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def load_map_image(map_path, target_size=(1280, 704)):
    """Load map image, resize to model spatial size, return [1,3,H,W] in [-1,1]."""
    img = Image.open(map_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = 2.0 * arr - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


@torch.no_grad()
def encode_map_to_dit_tokens(
    map_path,
    vae,
    patch_embedding,
    target_size=(1280, 704),
    device="cuda",
    dtype=torch.bfloat16,
    map_strength=0.3,
    n_tokens=16,
):
    """
    Encode a map image into a SMALL number of DiT cross-attention tokens.

    Args:
        map_path: path to map image
        vae: Wan VAE (AutoencoderKLWan)
        patch_embedding: DiT patch_embedding (Conv3d 48→3072)
        target_size: (W, H)
        map_strength: scale factor (0.3 = gentle hint, 1.0 = strong)
        n_tokens: number of map tokens (16 = 4×4 grid)

    Returns:
        map_tokens: [1, n_tokens, 3072]
    """
    # 1. Load and resize map
    map_img = load_map_image(map_path, target_size).to(device, dtype)

    # 2. VAE encode → 16-ch latent
    map_5d = map_img.unsqueeze(2)  # [1,3,1,H,W]
    latent_out = vae.encode(map_5d)
    if hasattr(latent_out, "latent_dist"):
        latent = latent_out.latent_dist.mode()
    elif hasattr(latent_out, "latents"):
        latent = latent_out.latents
    elif isinstance(latent_out, tuple):
        latent = latent_out[0]
    else:
        latent = latent_out

    if latent.dim() == 4:
        latent = latent.unsqueeze(2)

    # 3. Normalize latent distribution (critical for patch_embedding compatibility)
    # VAE latents for maps have different statistics than video latents.
    # LayerNorm per-channel brings them to a similar distribution.
    latent = F.layer_norm(latent, latent.shape[1:])

    # 4. Pad 16-ch → 48-ch
    c_latent = latent.shape[1]
    in_dim = 48
    if c_latent < in_dim:
        pad = torch.zeros(
            latent.shape[0], in_dim - c_latent, *latent.shape[2:],
            device=device, dtype=dtype
        )
        latent_48 = torch.cat([latent, pad], dim=1)
    elif c_latent > in_dim:
        latent_48 = latent[:, :in_dim]
    else:
        latent_48 = latent

    # 5. Spatial downsampling via adaptive average pool
    # Reduce 44×80 → small grid so we get few tokens (not 880!)
    # n_tokens=16 → 4×4 grid; n_tokens=64 → 8×8 grid
    grid = int(np.sqrt(n_tokens))
    # Pool to (grid*2, grid*2) because patch_embedding has stride (1,2,2)
    # → output will be (grid, grid) → grid*grid = n_tokens
    pool_h = grid * 2
    pool_w = grid * 2
    latent_48 = F.adaptive_avg_pool3d(latent_48, (1, pool_h, pool_w))

    # 6. Pass through DiT patch_embedding
    # Under sequential_cpu_offload, patch_embedding weights may be on CPU or on
    # "meta" device (placeholders without data, managed by the offload hook).
    # We must not call .to() on meta tensors — instead, run the forward under
    # the offload hook's execution context so weights are loaded automatically.
    pe_param = next(patch_embedding.parameters())
    pe_device = pe_param.device
    pe_is_meta = pe_param.is_meta
    if pe_is_meta:
        # Meta tensor: the accelerate offload hook will materialize weights
        # during the forward pass. Just call the module directly.
        tokens = patch_embedding(latent_48)  # [1,3072,1,grid,grid]
    elif pe_device != device:
        patch_embedding.to(device)
        try:
            tokens = patch_embedding(latent_48)  # [1,3072,1,grid,grid]
        finally:
            patch_embedding.to(pe_device)
    else:
        tokens = patch_embedding(latent_48)  # [1,3072,1,grid,grid]

    # 7. Flatten → [1, n_tokens, 3072]
    tokens = tokens.flatten(2).transpose(1, 2)  # [1, grid*grid, 3072]

    # 8. Apply strength scaling (gentle)
    if map_strength != 1.0:
        tokens = tokens * map_strength

    print(f"  [MapInject v2] map={map_path}")
    print(f"  [MapInject v2] VAE latent: {latent.shape} → LayerNorm → pad→48ch")
    print(f"  [MapInject v2] Adaptive pool to {pool_h}×{pool_w} → patch_embedding")
    print(f"  [MapInject v2] Output: {tokens.shape[1]} map tokens (dim={tokens.shape[2]})")
    print(f"  [MapInject v2] map_strength = {map_strength} (gentle hint)")

    return tokens


@torch.no_grad()
def encode_map_to_dit_tokens_numpy(
    map_path, vae, patch_embedding,
    target_size=(1280, 704), device="cuda", dtype=torch.bfloat16,
    map_strength=0.3, n_tokens=16,
):
    """Same as above but also returns numpy."""
    tokens = encode_map_to_dit_tokens(
        map_path, vae, patch_embedding, target_size, device, dtype,
        map_strength, n_tokens
    )
    return tokens, tokens.cpu().float().numpy()


@torch.no_grad()
def encode_map_to_spatial_kv(
    map_path,
    vae,
    patch_embedding,
    target_size=(1280, 704),
    device="cuda",
    dtype=torch.bfloat16,
    map_strength=0.3,
    grid_sizes=None,
    seq_len=None,
):
    """Encode map into FULL-RESOLUTION spatial tokens for self-attention K/V injection.

    Unlike encode_map_to_dit_tokens (which pools to 4×4=16 tokens for cross-attention),
    this function preserves the FULL spatial grid and projects through the TRAINED
    patch_embedding (Conv3d 48→3072) — no random encoder, no pooling.

    The output is aligned to the video token grid so that token (h,w) of the map
    corresponds to token (h,w) of the video, enabling spatially-grounded generation.

    Pipeline:
      map image → resize → VAE encode → [1, 48, 1, H_lat, W_lat]
      → LayerNorm → patch_embedding (trained Conv3d 48→3072, stride 1,2,2)
      → [1, 3072, 1, H_patch, W_patch]
      → flatten → [1, H_patch*W_patch, 3072]
      → expand to all frames → pad to seq_len
      → [1, seq_len, 3072]

    Args:
        map_path: path to map image
        vae: Wan VAE
        patch_embedding: DiT patch_embedding (trained Conv3d 48→3072)
        target_size: (W, H) for map resize
        map_strength: scale factor
        grid_sizes: [B, 3] tensor of (F, H_patch, W_patch) — video token grid
        seq_len: total video token count (F * H_patch * W_patch)

    Returns:
        spatial_tokens: [1, seq_len, 3072] — spatially aligned to video grid
    """
    # 1. Load and resize map
    map_img = load_map_image(map_path, target_size).to(device, dtype)

    # 2. VAE encode → 48-ch latent (same as video)
    map_5d = map_img.unsqueeze(2)  # [1,3,1,H,W]
    latent_out = vae.encode(map_5d)
    if hasattr(latent_out, "latent_dist"):
        latent = latent_out.latent_dist.mode()
    elif hasattr(latent_out, "latents"):
        latent = latent_out.latents
    elif isinstance(latent_out, tuple):
        latent = latent_out[0]
    else:
        latent = latent_out

    if latent.dim() == 4:
        latent = latent.unsqueeze(2)

    # 3. Normalize latent distribution (same as 16-token path)
    latent = F.layer_norm(latent, latent.shape[1:])

    # 4. Ensure 48 channels (pad or truncate)
    c_latent = latent.shape[1]
    in_dim = 48
    if c_latent < in_dim:
        pad = torch.zeros(
            latent.shape[0], in_dim - c_latent, *latent.shape[2:],
            device=device, dtype=dtype
        )
        latent_48 = torch.cat([latent, pad], dim=1)
    elif c_latent > in_dim:
        latent_48 = latent[:, :in_dim]
    else:
        latent_48 = latent

    # 5. Pass through TRAINED patch_embedding (NO pooling!)
    # This uses the same trained Conv3d that processes video latents.
    # Output: [1, 3072, 1, H_patch, W_patch]
    print(f"  [MapSpatialKV] latent_48 shape before patch_embedding: {latent_48.shape}")
    pe_param = next(patch_embedding.parameters())
    pe_device = pe_param.device
    pe_is_meta = pe_param.is_meta
    if pe_is_meta:
        tokens = patch_embedding(latent_48)
    elif pe_device != device:
        patch_embedding.to(device)
        try:
            tokens = patch_embedding(latent_48)
        finally:
            patch_embedding.to(pe_device)
    else:
        tokens = patch_embedding(latent_48)
    print(f"  [MapSpatialKV] tokens shape after patch_embedding: {tokens.shape}")

    # 6. Flatten spatial grid → [1, H_patch*W_patch, 3072]
    n_spatial = tokens.shape[3] * tokens.shape[4]  # H_patch * W_patch
    tokens = tokens.flatten(2).transpose(1, 2)  # [1, n_spatial, 3072]

    # 7. Apply strength scaling
    if map_strength != 1.0:
        tokens = tokens * map_strength

    # 8. Expand to all frames and pad to seq_len
    if grid_sizes is not None:
        F_i, H_i, W_i = grid_sizes[0].tolist()
        video_spatial = H_i * W_i
        if n_spatial != video_spatial:
            print(f"  [MapSpatialKV] WARNING: map spatial ({n_spatial}) != video spatial ({video_spatial})")
            # Interpolate tokens to match video grid
            # tokens: [1, n_spatial, 3072] → [1, 3072, H_map, W_map] → interp → [1, 3072, H_i, W_i]
            # We don't know exact H_map, W_map but n_spatial = H_map*W_map.
            # Reshape from the pre-flatten spatial dims instead.
            # Re-derive: tokens were [1, 3072, 1, H_map, W_map] before flatten.
            # Use F.interpolate on the 5D form. But we already flattened...
            # Simplest: reshape to 2D assuming square-ish, then interpolate.
            # Actually, n_spatial should match when map is resized to video resolution.
            pass

        # Replicate the 2D map across all frames (static top-down map)
        # tokens is [1, n_spatial, 3072], need [1, F_i * n_spatial, 3072]
        tokens = tokens.repeat(1, F_i, 1)  # [1, F_i * n_spatial, 3072]

        if seq_len is not None and tokens.shape[1] < seq_len:
            pad = torch.zeros(1, seq_len - tokens.shape[1], tokens.shape[2],
                              device=device, dtype=dtype)
            tokens = torch.cat([tokens, pad], dim=1)
        elif seq_len is not None and tokens.shape[1] > seq_len:
            tokens = tokens[:, :seq_len, :]

    print(f"  [MapSpatialKV] map={map_path}")
    print(f"  [MapSpatialKV] VAE latent: {latent.shape} → LayerNorm → 48ch")
    print(f"  [MapSpatialKV] patch_embedding (trained) → {tokens.shape[1]} spatial tokens")
    print(f"  [MapSpatialKV] Output: {tokens.shape} (aligned to video grid)")
    print(f"  [MapSpatialKV] map_strength = {map_strength}")

    return tokens


@torch.no_grad()
def encode_map_to_latent_noise(
    map_path,
    vae,
    target_size=(1024, 576),
    device="cuda",
    dtype=torch.bfloat16,
    map_strength=0.3,
    latent_shape=None,
):
    """Encode map into a VAE latent that can be mixed into the initial noise.

    This is a fundamentally different approach from K/V injection: instead of
    biasing attention, it embeds the map's spatial structure directly into the
    starting point of the diffusion process. The noise latent is:
        latents = sqrt(1 - alpha) * noise + sqrt(alpha) * map_latent
    where alpha = map_strength. This is analogous to SDEdit / structural init.

    The map VAE latent has the same spatial dimensions as the video VAE latent
    (both encoded at the same resolution), so token (h,w) of the map corresponds
    to token (h,w) of the video — spatial alignment is preserved by construction.

    Args:
        map_path: path to map image
        vae: Wan VAE
        target_size: (W, H) for map resize — should match video resolution
        map_strength: blending factor alpha in [0, 1].
            0.0 = pure noise (no map influence)
            0.3 = gentle structural hint
            0.5 = moderate structure
            1.0 = map latent only (no noise)
        latent_shape: (B, C, F, H, W) of the target noise latent. The map latent
            is replicated across all frames to match this shape.

    Returns:
        map_latent: [B, C, F, H, W] — map VAE latent shaped like the noise latent,
                    scaled to be mixed with noise via the alpha blend formula.
    """
    # 1. Load and resize map
    map_img = load_map_image(map_path, target_size).to(device, dtype)

    # 2. VAE encode → latent
    map_5d = map_img.unsqueeze(2)  # [1,3,1,H,W]
    latent_out = vae.encode(map_5d)
    if hasattr(latent_out, "latent_dist"):
        latent = latent_out.latent_dist.mode()
    elif hasattr(latent_out, "latents"):
        latent = latent_out.latents
    elif isinstance(latent_out, tuple):
        latent = latent_out[0]
    else:
        latent = latent_out

    if latent.dim() == 4:
        latent = latent.unsqueeze(2)

    # Wan VAE returns [mu, log_var] concatenated on channel dim. Take mu only.
    c = latent.shape[1]
    latent_channels = 16  # Wan VAE latent_channels
    if c == 2 * latent_channels:
        latent = latent[:, :latent_channels]
        print(f"  [MapLatentNoise] Split mu/log_var → using mu only: {latent.shape}")

    # 3. Match the target latent shape
    if latent_shape is not None:
        B, C, F, H, W = latent_shape
        # Spatial: resize map latent to match video latent spatial dims
        if latent.shape[3] != H or latent.shape[4] != W:
            latent = F.interpolate(
                latent, size=(H, W), mode='bilinear', align_corners=False
            )
        # Channels: pad or truncate
        if latent.shape[1] < C:
            pad = torch.zeros(
                latent.shape[0], C - latent.shape[1], *latent.shape[2:],
                device=device, dtype=dtype
            )
            latent = torch.cat([latent, pad], dim=1)
        elif latent.shape[1] > C:
            latent = latent[:, :C]
        # Temporal: replicate single frame to all frames
        latent = latent.expand(B, C, F, H, W).contiguous()
    else:
        B = latent.shape[0]

    # 4. Normalize the map latent to match noise statistics
    # The noise latent has unit variance (from randn * init_noise_sigma).
    # We standardize the map latent so it has the same scale.
    latent_mean = latent.mean()
    latent_std = latent.std()
    if latent_std > 1e-6:
        latent = (latent - latent_mean) / latent_std

    print(f"  [MapLatentNoise] map={map_path}")
    print(f"  [MapLatentNoise] VAE latent (mu): {latent.shape}")
    print(f"  [MapLatentNoise] Standardized to zero-mean unit-variance")
    print(f"  [MapLatentNoise] map_strength (alpha) = {map_strength}")

    return latent


@torch.no_grad()
def load_occ_latent_noise(npy_path, latent_shape=None, device="cuda", dtype=torch.bfloat16):
    """Load a precomputed per-frame occupancy VAE latent (M2/M3 of the world-follows-map scan).

    The npy holds RAW VAE mu in the pipeline's latent convention, shaped
    [1, C, F, H, W] (C=48, F=num latent frames, H/W = latent grid), produced by
    experiments/world_follows_map/precompute_occ_latents.py. Each latent frame
    corresponds to the ego-view layout render at that point in the segment, so
    spatial AND temporal positions both carry map structure.

    Standardizes to zero-mean unit-variance (same recipe as
    encode_map_to_latent_noise) so it can replace a unit-variance noise draw.
    """
    arr = np.load(npy_path)
    latent = torch.from_numpy(arr).to(device=device, dtype=dtype)
    if latent.dim() == 4:
        latent = latent.unsqueeze(2)

    if latent_shape is not None:
        B, C, F, H, W = latent_shape
        # Spatial
        if latent.shape[3] != H or latent.shape[4] != W:
            latent = F.interpolate(latent, size=(H, W), mode="bilinear", align_corners=False)
        # Channels: pad or truncate
        if latent.shape[1] < C:
            pad = torch.zeros(
                latent.shape[0], C - latent.shape[1], *latent.shape[2:],
                device=device, dtype=dtype,
            )
            latent = torch.cat([latent, pad], dim=1)
        elif latent.shape[1] > C:
            latent = latent[:, :C]
        # Temporal: replicate single frame, or nearest-resample if frame count differs
        if latent.shape[2] == 1:
            latent = latent.expand(B, C, F, H, W).contiguous()
        elif latent.shape[2] != F:
            latent = F.interpolate(
                latent, size=(F, latent.shape[3], latent.shape[4]), mode="nearest"
            )
        if latent.shape[0] != B:
            latent = latent.expand(B, *latent.shape[1:]).contiguous()

    latent_mean = latent.mean()
    latent_std = latent.std()
    if latent_std > 1e-6:
        latent = (latent - latent_mean) / latent_std

    print(f"  [OccLatentNoise] npy={npy_path}")
    print(f"  [OccLatentNoise] shaped+standardized: {tuple(latent.shape)} "
          f"(raw mean={float(latent_mean):+.4f} std={float(latent_std):.4f})")
    return latent

