"""Shared semantic-map label definitions and visualization helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PIL import Image


LABEL_NAMES = ("background", "building", "road", "greenspace", "water", "plaza")
NUM_LABELS = len(LABEL_NAMES)

LABEL_COLORS = {
    0: (128, 128, 128),
    1: (180, 120, 80),
    2: (80, 80, 80),
    3: (60, 160, 60),
    4: (100, 180, 220),
    5: (220, 200, 100),
}

ADE20K_LABEL_COLORS = {
    0: (0, 0, 0),
    1: (120, 120, 120),
    2: (140, 140, 140),
    3: (4, 200, 3),
    4: (6, 230, 230),
    5: (204, 5, 255),
}


def validate_label_map(label_map: np.ndarray) -> np.ndarray:
    """Return a 2-D integer label array after strict schema validation."""
    labels = np.asarray(label_map)
    if labels.ndim != 2:
        raise ValueError(f"semantic label map must be two-dimensional, got {labels.shape}")
    if labels.dtype.kind not in "biu":
        raise TypeError(f"semantic label map must use an integer dtype, got {labels.dtype}")
    labels = labels.astype(np.int64, copy=False)
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= NUM_LABELS):
        values = np.unique(labels)
        raise ValueError(
            f"semantic labels must lie in [0, {NUM_LABELS - 1}], got {values[:12].tolist()}"
        )
    return labels


def semantic_map_to_rgb(
    label_map: np.ndarray,
    *,
    use_ade20k: bool = False,
    palette: Mapping[int, tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Render a semantic ID raster with the repository's stable display palette."""
    labels = validate_label_map(label_map)
    colors = dict(palette or (ADE20K_LABEL_COLORS if use_ade20k else LABEL_COLORS))
    missing = sorted(set(np.unique(labels).tolist()) - set(colors))
    if missing:
        raise ValueError(f"palette has no colors for labels {missing}")
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in colors.items():
        rgb[labels == int(label)] = tuple(int(channel) for channel in color)
    return Image.fromarray(rgb, mode="RGB")


__all__ = [
    "ADE20K_LABEL_COLORS",
    "LABEL_COLORS",
    "LABEL_NAMES",
    "NUM_LABELS",
    "semantic_map_to_rgb",
    "validate_label_map",
]
