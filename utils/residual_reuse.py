from typing import Iterable, List, Optional, Sequence, Tuple

import torch


def latent_action_keys_from_pixel_keys(
    per_frame_keys: Sequence[str],
    latent_frame_count: int,
) -> List[str]:
    """Map pixel-frame action labels to the latent frame indices used by PRoPE."""
    if latent_frame_count <= 0:
        return []
    if not per_frame_keys:
        return [""] * int(latent_frame_count)
    aligned_indices = [0] + [1 + 4 * i for i in range(int(latent_frame_count) - 1)]
    last_index = len(per_frame_keys) - 1
    return [str(per_frame_keys[min(index, last_index)]) for index in aligned_indices]


def camera_discontinuity_chunks(
    latent_action_keys: Sequence[str],
    chunk_size: int,
) -> Tuple[List[int], List[dict]]:
    """Return AR chunks that contain the first latent frame after an action change."""
    chunk_size = max(1, int(chunk_size))
    disabled_chunks = set()
    events = []
    for latent_index in range(1, len(latent_action_keys)):
        prev_action = str(latent_action_keys[latent_index - 1])
        action = str(latent_action_keys[latent_index])
        if action == prev_action:
            continue
        chunk_index = latent_index // chunk_size
        disabled_chunks.add(chunk_index)
        events.append({
            "latent_frame_index": int(latent_index),
            "chunk_index": int(chunk_index),
            "previous_action": prev_action,
            "action": action,
        })
    return sorted(disabled_chunks), events


def chunk_difficulty_classes(
    latent_action_keys: Sequence[str],
    chunk_size: int,
    *,
    easy_step_count: int = 3,
    hard_step_count: int = 4,
    turn_keys: Sequence[str] = ("j", "l", "i", "k", "a", "d"),
) -> List[dict]:
    """Classify AR chunks as map-easy (straight) or map-hard (turn/lateral).

    A chunk is hard when any of its latent actions contains a turn/lateral key
    or when the action changes inside the chunk (action-change chunks are the
    same discontinuity set that disables residual reuse). Easy chunks keep the
    native step list; hard chunks run the reduced easy step count.
    """
    chunk_size = max(1, int(chunk_size))
    turn_key_set = {str(key).lower() for key in turn_keys}
    decisions: List[dict] = []
    for latent_index, raw_key in enumerate(latent_action_keys):
        chunk_index = int(latent_index) // chunk_size
        key = str(raw_key).lower()
        prev_key = str(latent_action_keys[latent_index - 1]).lower() if latent_index > 0 else None
        is_hard = any(turn_char in key for turn_char in turn_key_set)
        if not is_hard and prev_key is not None and key != prev_key:
            is_hard = True
        while len(decisions) <= chunk_index:
            decisions.append({
                "chunk_index": len(decisions),
                "difficulty": "hard",
                "step_count": int(hard_step_count),
                "reason": "uncovered",
            })
        if is_hard:
            decisions[chunk_index] = {
                "chunk_index": chunk_index,
                "difficulty": "hard",
                "step_count": int(hard_step_count),
                "reason": "turn_key" if any(turn_char in key for turn_char in turn_key_set)
                else "action_change",
            }
        elif decisions[chunk_index]["reason"] == "uncovered":
            decisions[chunk_index] = {
                "chunk_index": chunk_index,
                "difficulty": "easy",
                "step_count": int(easy_step_count),
                "reason": "straight",
            }
    return decisions


def residual_relative_l1_change(
    current_latents: torch.Tensor,
    previous_latents: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Relative L1 change used as the conservative TeaCache-style gate."""
    current = current_latents.detach().float()
    previous = previous_latents.detach().float()
    numerator = torch.mean(torch.abs(current - previous))
    denominator = torch.mean(torch.abs(previous)).clamp_min(float(eps))
    return float((numerator / denominator).item())


def residual_reuse_decision(
    *,
    enabled: bool,
    threshold: float,
    chunk_disabled: bool,
    step_index: int,
    step_count: int,
    has_cached_residual: bool,
    metric: Optional[float],
) -> Tuple[bool, str]:
    """Decide whether this denoising step may reuse a cached residual."""
    if not enabled:
        return False, "feature_disabled"
    if chunk_disabled:
        return False, "camera_action_discontinuity"
    if step_index <= 0:
        return False, "first_step"
    if step_index >= step_count - 1:
        return False, "final_step"
    if not has_cached_residual:
        return False, "no_cached_residual"
    if metric is None:
        return False, "missing_metric"
    if float(metric) <= float(threshold):
        return True, "metric_below_threshold"
    return False, "metric_above_threshold"


def normalize_disabled_chunks(chunks: Optional[Iterable[int]]) -> set:
    if chunks is None:
        return set()
    return {int(chunk) for chunk in chunks}
