"""Runtime helpers for the MapFirstAdapter (map -> token residual).

Checkpoint-aware: V1.1 ckpts carry meta {blocks, amplitude, differential} and
are rebuilt with those semantics; V1 ckpts lack the keys and fall back to
plain all-ones-amplitude, non-differential behavior.
"""
import importlib.util
import os
import numpy as np
import torch
import torch.nn.functional as F

# class-name -> semantic-layout label id (LABEL_NAMES: bg0 bld1 road2 green3 water4 plaza5)
_NAME_TO_LABEL = {'background': 0, 'building': 1, 'road': 2, 'greenspace': 3,
                  'vegetation': 3, 'tree': 3, 'grass': 3, 'water': 4,
                  'sidewalk': 5, 'plaza': 5, 'crosswalk': 2}


def _wrap_residual_probe(adapter):
    """Residual Decomposition Probe hooks (opt-in via env, no retraining).

    MFA_RESIDUAL_CENTER=1 -> subtract the per-(latent-frame, channel) spatial
        mean from every injected residual (removes the global/style-shift
        component; exactly preserves the spatially varying and flip-odd parts)
    MFA_RESIDUAL_SCALE=x  -> multiply every injected residual by x
    Defaults are no-ops; the wrapper is only installed when explicitly set.
    """
    center = os.environ.get('MFA_RESIDUAL_CENTER', '0') == '1'
    try:
        scale = float(os.environ.get('MFA_RESIDUAL_SCALE', '1.0'))
    except ValueError:
        scale = 1.0
    if not center and scale == 1.0:
        return
    orig = adapter.per_block_list

    def per_block_list_probe(labels, token_hw, num_layers):
        out = orig(labels, token_hw, num_layers)
        th, tw = token_hw
        s = th * tw
        for i, o in enumerate(out):
            if o is None:
                continue
            if center:
                B, L, C = o.shape
                Fr = labels.shape[1]
                if o.is_contiguous():
                    # in-place on a view: zero extra allocation (~110MB/block saved)
                    o4 = o.view(B, Fr, s, C)
                    o4.sub_(o4.mean(dim=2, keepdim=True))
                else:
                    o4 = o.reshape(B, Fr, s, C)
                    o = o4.sub_(o4.mean(dim=2, keepdim=True)).reshape(B, L, C)
            if scale != 1.0:
                o = o.mul_(scale)
            out[i] = o
        return out

    adapter.per_block_list = per_block_list_probe
    print(f"[map_first_adapter] residual probe hooks: center={center} scale={scale}")


def load_adapter(ckpt_path, device):
    spec = importlib.util.spec_from_file_location(
        'train_map_first_adapter', 'tools/train_map_first_adapter.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    meta = ck.get('meta', {})
    blocks = tuple(meta.get('blocks') or (0, 3, 6, 9, 12))
    amplitude = meta.get('amplitude')          # None -> all-ones (V1 semantics)
    differential = bool(meta.get('differential', False))
    learnable_gates = bool(meta.get('learnable_gates', False))
    # bf16: the per-block scaled residual copies produced by per_block_list stay
    # resident across all denoise steps (~5x [L,3072]); at 121-frame inference
    # fp32 costs >1.6GB and OOMs. bf16 also makes the transformer's per-layer
    # `.to(x.dtype)` a no-op instead of a fresh copy each step.
    adapter = mod.MapFirstAdapter(inject_blocks=blocks, amplitude=amplitude,
                                  differential=differential,
                                  learnable_gates=learnable_gates).to(
                                      device=device, dtype=torch.bfloat16).eval()
    missing, unexpected = adapter.load_state_dict(ck['adapter'], strict=False)
    if unexpected or [k for k in missing if k != 'gate_param']:
        raise RuntimeError(f'incompatible adapter state in {ckpt_path}: '
                           f'missing {missing}, unexpected {list(unexpected)}')
    for p in adapter.parameters():
        p.requires_grad_(False)
    _wrap_residual_probe(adapter)
    return adapter, meta


def build_labels(occupancy, class_names, v_start, v_end, latent_frames, device):
    """occupancy [T,C,44,80] ego-aligned -> labels [1,F,16,16] long."""
    occ = torch.from_numpy(np.asarray(occupancy, dtype=np.float32))
    T, C, H, W = occ.shape
    ids = torch.tensor([_NAME_TO_LABEL.get(str(n).lower(), 0) for n in class_names])
    win = min(v_end, T) - v_start
    stride = (win - 1) / max(latent_frames - 1, 1)
    frames = []
    for l in range(latent_frames):
        vf = int(round(v_start + l * stride))
        vf = min(max(vf, 0), T - 1)
        crops = occ[vf].unsqueeze(0)                       # [1,C,44,80]
        small = F.adaptive_avg_pool2d(crops, (16, 16))[0]  # [C,16,16]
        idx = (small * (ids.view(-1, 1, 1) + 1)).argmax(0) # prefer non-bg classes
        frames.append(torch.where(small.max(0).values > 0.05, ids[idx], torch.zeros_like(idx)))
    return torch.stack(frames).unsqueeze(0).to(device=device, dtype=torch.long)


def build_external_latents(adapter, labels, token_hw, num_layers, blocks=(0, 3, 6, 9, 12)):
    with torch.no_grad():
        # per_block_list applies the ckpt's amplitude schedule internally
        if hasattr(adapter, 'per_block_list'):
            return adapter.per_block_list(labels, token_hw, num_layers)
        res = adapter(labels, token_hw)                    # [1, F*toks, 3072]
    out = [None] * num_layers
    for b in blocks:
        if b < num_layers:
            out[b] = res
    return out
