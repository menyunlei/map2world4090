"""Map route parser for the focused MapCtrl adapter path.

The parser is deterministic: it converts a route polyline into turn events for
the structured map adapter and into DreamX camera actions consumed by the
existing PRoPE camera pathway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from utils.route_to_actions import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    RouteToActionResult,
    route_to_actions,
)


@dataclass
class ParsedMapRoute:
    actions: list[str] = field(default_factory=list)
    action_speed_list: list[int] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    progress: list[float] = field(default_factory=list)
    turn_angles_deg: list[float] = field(default_factory=list)
    turn_sequence: str = ""

    def to_dict(self) -> dict:
        return {
            "actions": list(self.actions),
            "action_speed_list": list(self.action_speed_list),
            "events": [dict(event) for event in self.events],
            "progress": [float(value) for value in self.progress],
            "turn_angles_deg": [float(value) for value in self.turn_angles_deg],
            "turn_sequence": self.turn_sequence,
        }


def parse_route_polyline(
    polyline: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_segments: int,
    progress: Optional[Sequence[float]] = None,
    default_speed: int = 6,
    straight_threshold_rad=None,
) -> ParsedMapRoute:
    kwargs = {}
    if straight_threshold_rad is not None:
        kwargs["straight_threshold_rad"] = float(straight_threshold_rad)
    result: RouteToActionResult = route_to_actions(
        polyline,
        n_segments=n_segments,
        progress=progress,
        **kwargs,
    )
    events = []
    for index, (action, progress_value, angle_deg) in enumerate(
        zip(result.actions, result.progress, result.turn_angles_deg)
    ):
        if action == ACTION_LEFT:
            event_type = "left_turn"
        elif action == ACTION_RIGHT:
            event_type = "right_turn"
        else:
            event_type = "straight"
        events.append(
            {
                "type": event_type,
                "progress": float(progress_value),
                "route_progress": float(progress_value),
                "order": int(index),
                "turn_angle_deg": float(angle_deg),
            }
        )

    return ParsedMapRoute(
        actions=list(result.actions),
        action_speed_list=[int(default_speed)] * int(n_segments),
        events=events,
        progress=list(result.progress),
        turn_angles_deg=list(result.turn_angles_deg),
        turn_sequence=result.turn_sequence,
    )


def route_events_to_actions(events: Sequence[dict]) -> list[str]:
    actions = []
    for event in events:
        event_type = str(event.get("type", "")).strip().lower()
        if event_type in {"left", "left_turn", "turn_left"}:
            actions.append(ACTION_LEFT)
        elif event_type in {"right", "right_turn", "turn_right"}:
            actions.append(ACTION_RIGHT)
        else:
            actions.append(ACTION_FORWARD)
    return actions
