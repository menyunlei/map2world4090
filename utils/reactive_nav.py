"""Deterministic semantic-map reactive navigation for long-video inference.

Coordinates stored in the navigation state are normalized image coordinates
(``x`` grows right and ``y`` grows down).  A yaw of zero faces toward the top
of the map; positive yaw turns left, matching DreamX's ``j`` action.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from os import PathLike

import numpy as np
from PIL import Image


LABEL_NAMES = ("background", "building", "road", "greenspace", "water", "plaza")
_LABEL_COLORS = {
    0: (128, 128, 128),
    1: (180, 120, 80),
    2: (80, 80, 80),
    3: (60, 160, 60),
    4: (100, 180, 220),
    5: (220, 200, 100),
}
DEFAULT_WALKABLE_LABELS = frozenset({2, 5})
DEFAULT_OBSTACLE_LABELS = frozenset({1, 3, 4})

_RGB_TO_LABEL = {tuple(color): label for label, color in _LABEL_COLORS.items()}
_LABEL_ALIASES = {
    **{name: index for index, name in enumerate(LABEL_NAMES)},
    "green": 3,
    "park": 3,
    "open_land": 5,
    "open-land": 5,
}


def parse_label_set(value, default) -> set[int]:
    """Parse numeric IDs or canonical label names into a validated label set."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return {int(label) for label in default}
    values = value.split(",") if isinstance(value, str) else value
    if not isinstance(values, Iterable):
        values = (values,)
    parsed = set()
    for raw in values:
        token = str(raw).strip().lower()
        if not token:
            continue
        try:
            label = int(token)
        except ValueError:
            if token not in _LABEL_ALIASES:
                raise ValueError(f"unknown semantic label {raw!r}") from None
            label = _LABEL_ALIASES[token]
        if not 0 <= label < len(LABEL_NAMES):
            raise ValueError(f"semantic label ID must lie in [0, {len(LABEL_NAMES) - 1}], got {label}")
        parsed.add(label)
    return parsed


def _validate_label_map(label_map) -> np.ndarray:
    labels = np.asarray(label_map)
    if labels.ndim != 2:
        raise ValueError("semantic label map must be two-dimensional")
    if labels.dtype.kind not in "biu":
        raise TypeError("semantic label map must use an integer dtype")
    labels = labels.astype(np.int64, copy=False)
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= len(LABEL_NAMES)):
        raise ValueError(f"semantic labels must lie in [0, {len(LABEL_NAMES) - 1}]")
    return labels


def load_label_map(path: str | PathLike, sample_size) -> np.ndarray:
    """Load a semantic label PNG and resize it with nearest-neighbour sampling.

    Grayscale images are interpreted strictly as class IDs.  RGB/RGBA images
    must use the repository's standard six-class RGB palette; unknown colours
    raise ``ValueError`` rather than being silently interpreted as IDs.
    ``sample_size`` follows inference_long_video's ``(height, width)`` order.
    """
    with Image.open(path) as source:
        if source.mode in {"1", "L", "I", "I;16", "I;16B", "I;16L"}:
            image = source.copy()
            grayscale = True
        elif source.mode in {"RGB", "RGBA"}:
            image = source.convert("RGB")
            grayscale = False
        else:
            raise ValueError(
                f"unsupported semantic label-map mode {source.mode!r}; "
                "expected grayscale IDs or RGB palette colours"
            )

    if sample_size is not None:
        try:
            height, width = (int(sample_size[0]), int(sample_size[1]))
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError("sample_size must be a (height, width) pair") from exc
        if height <= 0 or width <= 0:
            raise ValueError("sample_size dimensions must be positive")
        nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        image = image.resize((width, height), resample=nearest)

    array = np.asarray(image)
    if grayscale:
        if array.ndim != 2:
            raise ValueError("grayscale semantic label map must be two-dimensional")
        return _validate_label_map(array)

    labels = np.full(array.shape[:2], -1, dtype=np.int64)
    for color, label in _RGB_TO_LABEL.items():
        labels[np.all(array == color, axis=-1)] = label
    if np.any(labels < 0):
        colors = np.unique(array[labels < 0].reshape(-1, 3), axis=0)
        preview = ", ".join(str(tuple(int(v) for v in color)) for color in colors[:5])
        raise ValueError(f"unknown semantic palette colour(s): {preview}")
    return labels


def make_reactive_nav_state(item: dict | None) -> dict:
    """Create a mutable, JSON-safe navigation state from an inference item."""
    item = item or {}
    start_xy = item.get("reactive_nav_start_xy", (0.5, 0.82))
    if not isinstance(start_xy, (list, tuple)) or len(start_xy) != 2:
        raise ValueError("reactive_nav_start_xy must contain [x, y]")
    prefer_turn = str(item.get("reactive_nav_prefer_turn", "right")).strip().lower()
    if prefer_turn not in {"left", "right"}:
        prefer_turn = "right"
    target = item.get("reactive_nav_target_yaw_abs")
    target = None if target is None else abs(float(target))
    return {
        "x": float(start_xy[0]),
        "y": float(start_xy[1]),
        "yaw_deg": float(item.get("reactive_nav_start_yaw_deg", 0.0)),
        "target_yaw_abs": target,
        "prefer_turn": prefer_turn,
    }


def _region_ratios(label_map, state, walkable_labels, obstacle_labels):
    """Sample fixed egocentric regions around the current normalized pose."""
    height, width = label_map.shape
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    dx = xs[None, :] - float(state["x"])
    dy = ys[:, None] - float(state["y"])

    yaw = math.radians(float(state["yaw_deg"]))
    # Forward points up at yaw=0; right is its clockwise perpendicular.
    forward_x, forward_y = -math.sin(yaw), -math.cos(yaw)
    right_x, right_y = math.cos(yaw), -math.sin(yaw)
    longitudinal = dx * forward_x + dy * forward_y
    lateral = dx * right_x + dy * right_y

    regions = {
        "ground": (np.abs(lateral) <= 0.045) & (longitudinal >= -0.025) & (longitudinal < 0.035),
        "local_front": (np.abs(lateral) <= 0.055) & (longitudinal >= 0.035) & (longitudinal < 0.11),
        "front": (np.abs(lateral) <= 0.065) & (longitudinal >= 0.11) & (longitudinal < 0.26),
        "left": (lateral >= -0.18) & (lateral < -0.055) & (longitudinal >= 0.025) & (longitudinal < 0.21),
        "right": (lateral > 0.055) & (lateral <= 0.18) & (longitudinal >= 0.025) & (longitudinal < 0.21),
    }
    walkable = np.isin(label_map, tuple(walkable_labels))
    obstacle = np.isin(label_map, tuple(obstacle_labels))
    result = {}
    for name, mask in regions.items():
        count = int(mask.sum())
        result[f"{name}_walkable"] = float(walkable[mask].mean()) if count else 0.0
        result[f"{name}_obstacle"] = float(obstacle[mask].mean()) if count else 0.0
    return result


def _arg(args, name, default):
    return getattr(args, name, default)


def _turn_with_yaw_cap(direction, speed, state):
    """Apply the configured absolute-yaw cap, reversing at an active cap."""
    yaw = float(state["yaw_deg"])
    cap = state.get("target_yaw_abs")
    sign = 1.0 if direction == "left" else -1.0
    if cap is not None:
        cap = abs(float(cap))
        if (direction == "left" and yaw >= cap) or (direction == "right" and yaw <= -cap):
            direction = "right" if direction == "left" else "left"
            sign = -sign
        room = cap - sign * yaw
        if room >= 0.0:
            speed = min(speed, room / 10.0)
    state["yaw_deg"] = float(yaw + sign * speed * 10.0)
    return direction, float(speed)


def choose_reactive_actions(
    action_seq,
    action_speed_list,
    label_map,
    state,
    args,
    walkable_labels=DEFAULT_WALKABLE_LABELS,
    obstacle_labels=DEFAULT_OBSTACLE_LABELS,
):
    """Choose clear, creep, or avoidance-turn actions and update *state*.

    The returned tuple is ``(actions, speeds, metrics)``.  The caller-owned
    state dictionary is updated in place using only JSON-serializable values.
    """
    labels = np.asarray(label_map)
    if labels.ndim != 2:
        raise ValueError("label_map must be a two-dimensional class-ID array")
    walkable_labels = parse_label_set(walkable_labels, DEFAULT_WALKABLE_LABELS)
    obstacle_labels = parse_label_set(obstacle_labels, DEFAULT_OBSTACLE_LABELS)

    metrics = _region_ratios(labels, state, walkable_labels, obstacle_labels)
    forward_clearance = float(_arg(args, "reactive_forward_clearance", 0.48))
    ground_clearance = float(_arg(args, "reactive_ground_clearance", 0.62))
    creep_speed = max(0.0, float(_arg(args, "reactive_creep_speed", 0.32)))
    turn_speed = max(0.0, float(_arg(args, "reactive_turn_speed", 2.8)))

    ground_clear = metrics["ground_walkable"] >= ground_clearance
    local_clear = metrics["local_front_walkable"] >= forward_clearance
    front_clear = metrics["front_walkable"] >= forward_clearance

    if ground_clear and local_clear and front_clear:
        actions = list(action_seq)
        speeds = [float(speed) for speed in action_speed_list]
        if not actions:
            actions, speeds = ["w"], [creep_speed]
        elif not speeds:
            speeds = [creep_speed] * len(actions)
        metrics["decision"] = "clear"
        movement_speed = speeds[0]
        # Planned composite turns also change the pose used by the next map
        # sample.  Route actions are normally one item per call, and the first
        # item is the segment whose motion is being committed here.
        first_action = str(actions[0])
        if ("j" in first_action) != ("l" in first_action):
            planned_direction = "left" if "j" in first_action else "right"
            actual_direction, movement_speed = _turn_with_yaw_cap(
                planned_direction, movement_speed, state
            )
            old_key = "j" if planned_direction == "left" else "l"
            new_key = "j" if actual_direction == "left" else "l"
            actions[0] = first_action.replace(old_key, new_key)
            speeds[0] = movement_speed
    elif ground_clear and local_clear:
        # Preserve planned turn direction during creep: if the route plan
        # asks for a turn (wj/wl), keep turning at creep speed so the camera
        # follows the map route instead of locking yaw until a wall forces it.
        planned = str(action_seq[0]) if action_seq else "w"
        if "j" in planned or "l" in planned:
            pdir = "left" if "j" in planned else "right"
            pdir, creep_speed = _turn_with_yaw_cap(pdir, creep_speed, state)
            actions = ["wj" if pdir == "left" else "wl"]
            speeds = [creep_speed]
        else:
            actions, speeds = ["w"], [creep_speed]
        metrics["decision"] = "creep"
        movement_speed = creep_speed
    else:
        left = metrics["left_walkable"]
        right = metrics["right_walkable"]
        if left > right:
            direction = "left"
        elif right > left:
            direction = "right"
        else:
            direction = state.get("prefer_turn", "right")
        direction, actual_turn_speed = _turn_with_yaw_cap(direction, turn_speed, state)
        actions = ["wj" if direction == "left" else "wl"]
        speeds = [actual_turn_speed]
        metrics["decision"] = f"turn_{direction}"
        movement_speed = actual_turn_speed

    # Keep the top-down probe moving with the generated camera.  The bounded
    # scale prevents legacy high action speeds from jumping across the map.
    yaw = math.radians(float(state["yaw_deg"]))
    step = min(max(float(movement_speed), 0.0) * 0.04, 0.08)
    state["x"] = float(min(1.0, max(0.0, float(state["x"]) - math.sin(yaw) * step)))
    state["y"] = float(min(1.0, max(0.0, float(state["y"]) - math.cos(yaw) * step)))
    state["yaw_deg"] = float(state["yaw_deg"])
    if state.get("target_yaw_abs") is not None:
        state["target_yaw_abs"] = float(state["target_yaw_abs"])
    state["prefer_turn"] = str(state.get("prefer_turn", "right"))

    metrics.update({
        "state_x": state["x"],
        "state_y": state["y"],
        "yaw_deg": state["yaw_deg"],
    })
    return actions, speeds, metrics


__all__ = [
    "DEFAULT_OBSTACLE_LABELS",
    "DEFAULT_WALKABLE_LABELS",
    "parse_label_set",
    "load_label_map",
    "make_reactive_nav_state",
    "choose_reactive_actions",
]
