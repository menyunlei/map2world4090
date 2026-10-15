"""Deterministic, inference-only same-session visual-context selection.

This module does not generate images, access GPUs, launch services, or pretend
that a selected latent is a geometrically registered 3D scene. A caller must
verify the latent hash before conditioning native inference on it.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SelectionPolicy:
    radius_m: float = 2.5
    yaw_tolerance_degrees: float = 20.0
    minimum_age_latents: int = 15
    max_records: int = 512

    def __post_init__(self):
        if not (0 < self.radius_m <= 10):
            raise ValueError("radius_m must be positive and at most10")
        if not (0 < self.yaw_tolerance_degrees < 90):
            raise ValueError("Require a similar heading, never an opposite-view match")
        if self.minimum_age_latents < 12:
            raise ValueError("Memory retrieval must exclude the rolling local window")
        if self.max_records < 1:
            raise ValueError("max_records must be positive")


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    world_id: str
    map_revision_id: str
    latent_start: int
    latent_count: int
    position_xyz: tuple
    yaw_degrees: float
    latent_path: str
    latent_sha256: str
    camera_path: str
    camera_sha256: str

    def __post_init__(self):
        if self.latent_count != 3 or self.latent_start < 3:
            raise ValueError("Store completed native3-latent generated chunks, excludingI0")
        if len(self.position_xyz) != 3 or not all(math.isfinite(float(x)) for x in self.position_xyz):
            raise ValueError("A finite global model-space XYZ pose is required")
        if not math.isfinite(self.yaw_degrees):
            raise ValueError("Finite yaw required")
        for value in (self.latent_sha256, self.camera_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("PayloadSHA256 required")


def angular_distance_degrees(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


class PoseMemoryIndex:
    """Bounded CPU metadata; saved latent payloads remain outside GPU memory.

    world_id denotes one continuous camera coordinate world. Explicit new-map
    or new-I0 resets must get a new world_id. A continuation can reuse it only
    if the camera world is actually preserved.
    """
    def __init__(self, policy=None):
        self.policy = policy or SelectionPolicy()
        self.records = []

    def commit(self, record):
        if any(old.record_id == record.record_id for old in self.records):
            raise ValueError("Duplicate memory transaction")
        if self.records and record.world_id == self.records[-1].world_id:
            if record.latent_start < self.records[-1].latent_start + self.records[-1].latent_count:
                raise ValueError("Memory commits must be chronologically nonoverlapping")
        self.records.append(record)
        evicted = self.records[:-self.policy.max_records]
        self.records = self.records[-self.policy.max_records:]
        return [item.record_id for item in evicted]

    def query(self, *, world_id, map_revision_id, current_latent_start,
              position_xyz, yaw_degrees, active_record_id=None):
        if len(position_xyz) != 3 or not all(math.isfinite(float(x)) for x in position_xyz):
            raise ValueError("Finite queryXYZ required")
        if not math.isfinite(yaw_degrees):
            raise ValueError("Finite queryyaw required")
        candidates = []
        rejected = {"world_or_map": 0, "recent_or_future": 0,
                    "distance": 0, "heading": 0}
        for record in self.records:
            if record.world_id != world_id or record.map_revision_id != map_revision_id:
                rejected["world_or_map"] += 1
                continue
            if record.latent_start + record.latent_count > current_latent_start - self.policy.minimum_age_latents:
                rejected["recent_or_future"] += 1
                continue
            distance = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(record.position_xyz, position_xyz)))
            if distance > self.policy.radius_m:
                rejected["distance"] += 1
                continue
            yaw = angular_distance_degrees(record.yaw_degrees, yaw_degrees)
            if yaw > self.policy.yaw_tolerance_degrees:
                rejected["heading"] += 1
                continue
            # Fixed deterministic ordering, no image/VLM/quality/seed selection.
            candidates.append((distance, yaw, record.latent_start, record.record_id, record))
        candidates.sort(key=lambda item: item[:4])
        record = candidates[0][-1] if candidates else None
        return record, {
            "selected_record_id": record.record_id if record else None,
            "distance_m": candidates[0][0] if candidates else None,
            "yaw_difference_degrees": candidates[0][1] if candidates else None,
            "same_active_record": record is not None and record.record_id == active_record_id,
            "eligible_count": len(candidates), "rejected": rejected,
            "policy": asdict(self.policy),
            "selection_basis": "same world/map, completed oldchunk, Euclidean modelspaceXYZ, similar heading",
            "opposite_or_unseen_view_guaranteed": False,
        }
