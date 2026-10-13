"""World-coordinate latent tile cache for Map Tile Router outputs.

This cache stores only latent tiles. It is intentionally passive: callers may
record and look up cached tiles, but this module does not skip DiT computation
or patch attention/FFN code.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is available in the project env.
    torch = None


MISS_DYNAMIC_REGION = "dynamic_region_never_reuse"
MISS_MISSING_ENTRY = "missing_entry"
MISS_INCOMPATIBLE_WORLD = "incompatible_world_coordinate"
MISS_INCOMPATIBLE_VIEW = "incompatible_camera_direction"
MISS_INCOMPATIBLE_SHAPE = "incompatible_latent_shape"
MISS_SEMANTIC_VERSION = "semantic_map_version_mismatch"
MISS_INVALIDATED = "invalidated_world_tile"
MISS_LATENT_LEVEL = "latent_frame_level_mismatch"


@dataclass(frozen=True)
class WorldTileCacheConfig:
    """Cache compatibility rules.

    ``camera_bin_degrees`` controls the key space. ``max_view_delta_degrees`` is
    a second verification step so a same-bin lookup still checks the actual
    camera direction.
    """

    camera_bin_degrees: float = 15.0
    max_view_delta_degrees: float = 7.5
    clone_on_store: bool = True
    clone_on_retrieve: bool = True

    def __post_init__(self) -> None:
        if float(self.camera_bin_degrees) <= 0.0 or float(self.camera_bin_degrees) > 360.0:
            raise ValueError("camera_bin_degrees must be in (0, 360]")
        if float(self.max_view_delta_degrees) < 0.0:
            raise ValueError("max_view_delta_degrees must be >= 0")


@dataclass(frozen=True)
class WorldTileCacheKey:
    world_tile_x: int
    world_tile_y: int
    camera_view_bin: int
    latent_frame_level: int
    semantic_map_version: str

    def to_tuple(self) -> tuple[int, int, int, int, str]:
        return (
            int(self.world_tile_x),
            int(self.world_tile_y),
            int(self.camera_view_bin),
            int(self.latent_frame_level),
            str(self.semantic_map_version),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_tile_x": int(self.world_tile_x),
            "world_tile_y": int(self.world_tile_y),
            "camera_view_bin": int(self.camera_view_bin),
            "latent_frame_level": int(self.latent_frame_level),
            "semantic_map_version": str(self.semantic_map_version),
        }


@dataclass
class WorldTileCacheEntry:
    key: WorldTileCacheKey
    camera_yaw_rad: float
    latent_shape: tuple[int, ...]
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldTileStoreResult:
    stored: bool
    reason: str
    key: WorldTileCacheKey | None = None


@dataclass
class WorldTileLookupResult:
    hit: bool
    value: Any = None
    reason: str | None = None
    key: WorldTileCacheKey | None = None
    entry_metadata: dict[str, Any] = field(default_factory=dict)


class WorldTileCache:
    """In-memory cache keyed by world tile, camera bin, latent level and map version."""

    def __init__(self, config: WorldTileCacheConfig | None = None):
        self.config = config or WorldTileCacheConfig()
        self._entries: dict[WorldTileCacheKey, WorldTileCacheEntry] = {}
        self._invalidated_tiles: set[tuple[tuple[int, int], str | None]] = set()
        self._requests = 0
        self._hits = 0
        self._misses = 0
        self._miss_reasons: Counter[str] = Counter()
        self._stores = 0
        self._store_skips: Counter[str] = Counter()
        self._invalidations = 0

    def store(
        self,
        *,
        latent_tile: Any,
        world_tile_coord: Sequence[int] | Mapping[str, Any],
        camera_yaw_rad: float,
        latent_frame_level: int,
        semantic_map_version: Hashable,
        dynamic: bool = False,
        router_tile: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorldTileStoreResult:
        """Store a latent tile unless the tile is dynamic."""

        if dynamic or router_tile_is_dynamic(router_tile):
            self._store_skips[MISS_DYNAMIC_REGION] += 1
            return WorldTileStoreResult(stored=False, reason=MISS_DYNAMIC_REGION)

        latent_shape = latent_shape_of(latent_tile)
        key = self.make_key(
            world_tile_coord=world_tile_coord,
            camera_yaw_rad=camera_yaw_rad,
            latent_frame_level=latent_frame_level,
            semantic_map_version=semantic_map_version,
        )
        entry = WorldTileCacheEntry(
            key=key,
            camera_yaw_rad=float(camera_yaw_rad),
            latent_shape=latent_shape,
            value=clone_latent_tile(latent_tile) if self.config.clone_on_store else latent_tile,
            metadata=dict(metadata or {}),
        )
        self._entries[key] = entry
        coord = (int(key.world_tile_x), int(key.world_tile_y))
        version = str(key.semantic_map_version)
        self._invalidated_tiles.discard((coord, version))
        self._invalidated_tiles.discard((coord, None))
        self._stores += 1
        return WorldTileStoreResult(stored=True, reason="stored", key=key)

    def retrieve(
        self,
        *,
        world_tile_coord: Sequence[int] | Mapping[str, Any],
        camera_yaw_rad: float,
        latent_frame_level: int,
        semantic_map_version: Hashable,
        expected_latent_shape: Sequence[int],
        dynamic: bool = False,
        router_tile: Mapping[str, Any] | None = None,
    ) -> WorldTileLookupResult:
        """Retrieve a compatible cached latent tile."""

        self._requests += 1
        if dynamic or router_tile_is_dynamic(router_tile):
            return self._miss(MISS_DYNAMIC_REGION)

        coord = normalize_world_tile_coord(world_tile_coord)
        version = str(semantic_map_version)
        if (coord, version) in self._invalidated_tiles or (coord, None) in self._invalidated_tiles:
            return self._miss(MISS_INVALIDATED)

        key = self.make_key(
            world_tile_coord=coord,
            camera_yaw_rad=camera_yaw_rad,
            latent_frame_level=latent_frame_level,
            semantic_map_version=version,
        )
        entry = self._entries.get(key)
        if entry is None:
            return self._miss(self._diagnose_missing_entry(key, float(camera_yaw_rad), tuple(expected_latent_shape)))

        reason = self._verify_entry(
            entry,
            world_tile_coord=coord,
            camera_yaw_rad=float(camera_yaw_rad),
            latent_frame_level=int(latent_frame_level),
            semantic_map_version=version,
            expected_latent_shape=tuple(int(v) for v in expected_latent_shape),
        )
        if reason is not None:
            return self._miss(reason)

        self._hits += 1
        value = clone_latent_tile(entry.value) if self.config.clone_on_retrieve else entry.value
        return WorldTileLookupResult(
            hit=True,
            value=value,
            reason=None,
            key=entry.key,
            entry_metadata=dict(entry.metadata),
        )

    def invalidate(
        self,
        *,
        world_tile_coord: Sequence[int] | Mapping[str, Any],
        semantic_map_version: Hashable | None = None,
    ) -> int:
        """Invalidate one world-map tile.

        If ``semantic_map_version`` is omitted, entries for all semantic versions
        of the world tile are removed.
        """

        coord = normalize_world_tile_coord(world_tile_coord)
        version = None if semantic_map_version is None else str(semantic_map_version)
        to_delete = [
            key
            for key in self._entries
            if (int(key.world_tile_x), int(key.world_tile_y)) == coord
            and (version is None or str(key.semantic_map_version) == version)
        ]
        for key in to_delete:
            del self._entries[key]
        self._invalidated_tiles.add((coord, version))
        self._invalidations += 1
        return len(to_delete)

    def clear(self) -> None:
        self._entries.clear()
        self._invalidated_tiles.clear()
        self._requests = 0
        self._hits = 0
        self._misses = 0
        self._miss_reasons.clear()
        self._stores = 0
        self._store_skips.clear()
        self._invalidations = 0

    def make_key(
        self,
        *,
        world_tile_coord: Sequence[int] | Mapping[str, Any],
        camera_yaw_rad: float,
        latent_frame_level: int,
        semantic_map_version: Hashable,
    ) -> WorldTileCacheKey:
        x, y = normalize_world_tile_coord(world_tile_coord)
        return WorldTileCacheKey(
            world_tile_x=int(x),
            world_tile_y=int(y),
            camera_view_bin=camera_view_bin(float(camera_yaw_rad), self.config.camera_bin_degrees),
            latent_frame_level=int(latent_frame_level),
            semantic_map_version=str(semantic_map_version),
        )

    def stats(self) -> dict[str, Any]:
        hit_rate = float(self._hits / self._requests) if self._requests else 0.0
        return {
            "entries": int(len(self._entries)),
            "requests": int(self._requests),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "hit_rate": hit_rate,
            "miss_reasons": dict(self._miss_reasons),
            "stores": int(self._stores),
            "store_skips": dict(self._store_skips),
            "invalidations": int(self._invalidations),
        }

    def _miss(self, reason: str) -> WorldTileLookupResult:
        self._misses += 1
        self._miss_reasons[str(reason)] += 1
        return WorldTileLookupResult(hit=False, value=None, reason=str(reason))

    def _verify_entry(
        self,
        entry: WorldTileCacheEntry,
        *,
        world_tile_coord: tuple[int, int],
        camera_yaw_rad: float,
        latent_frame_level: int,
        semantic_map_version: str,
        expected_latent_shape: tuple[int, ...],
    ) -> str | None:
        if (int(entry.key.world_tile_x), int(entry.key.world_tile_y)) != world_tile_coord:
            return MISS_INCOMPATIBLE_WORLD
        if int(entry.key.latent_frame_level) != int(latent_frame_level):
            return MISS_LATENT_LEVEL
        if str(entry.key.semantic_map_version) != str(semantic_map_version):
            return MISS_SEMANTIC_VERSION
        if tuple(entry.latent_shape) != tuple(expected_latent_shape):
            return MISS_INCOMPATIBLE_SHAPE
        view_delta = abs_angle_delta_degrees(float(entry.camera_yaw_rad), float(camera_yaw_rad))
        if view_delta > float(self.config.max_view_delta_degrees):
            return MISS_INCOMPATIBLE_VIEW
        return None

    def _diagnose_missing_entry(
        self,
        key: WorldTileCacheKey,
        camera_yaw_rad: float,
        expected_latent_shape: tuple[int, ...],
    ) -> str:
        if not self._entries:
            return MISS_MISSING_ENTRY

        same_coord = [
            entry for entry in self._entries.values()
            if (int(entry.key.world_tile_x), int(entry.key.world_tile_y))
            == (int(key.world_tile_x), int(key.world_tile_y))
        ]
        if not same_coord:
            return MISS_INCOMPATIBLE_WORLD

        same_version = [
            entry for entry in same_coord
            if str(entry.key.semantic_map_version) == str(key.semantic_map_version)
        ]
        if not same_version:
            return MISS_SEMANTIC_VERSION

        same_level = [
            entry for entry in same_version
            if int(entry.key.latent_frame_level) == int(key.latent_frame_level)
        ]
        if not same_level:
            return MISS_LATENT_LEVEL

        same_shape = [entry for entry in same_level if tuple(entry.latent_shape) == tuple(expected_latent_shape)]
        if not same_shape:
            return MISS_INCOMPATIBLE_SHAPE

        if not any(
            abs_angle_delta_degrees(float(entry.camera_yaw_rad), float(camera_yaw_rad))
            <= float(self.config.max_view_delta_degrees)
            for entry in same_shape
        ):
            return MISS_INCOMPATIBLE_VIEW
        return MISS_MISSING_ENTRY


def normalize_world_tile_coord(value: Sequence[int] | Mapping[str, Any]) -> tuple[int, int]:
    """Normalize world-map tile coordinates to an integer `(x, y)` tuple."""

    if isinstance(value, Mapping):
        if "world_tile_x" in value and "world_tile_y" in value:
            return int(value["world_tile_x"]), int(value["world_tile_y"])
        if "x" in value and "y" in value:
            return int(value["x"]), int(value["y"])
        if "coord" in value:
            return normalize_world_tile_coord(value["coord"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"unsupported world tile coordinate: {value!r}")


def camera_view_bin(camera_yaw_rad: float, camera_bin_degrees: float = 15.0) -> int:
    """Quantize yaw to a deterministic circular view bin."""

    bin_degrees = float(camera_bin_degrees)
    bins = max(1, int(round(360.0 / bin_degrees)))
    yaw_deg = math.degrees(float(camera_yaw_rad)) % 360.0
    return int(round(yaw_deg / bin_degrees)) % bins


def abs_angle_delta_degrees(a_rad: float, b_rad: float) -> float:
    delta = (float(a_rad) - float(b_rad) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def latent_shape_of(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(v) for v in value.shape)
    raise ValueError("latent_tile must expose a shape")


def clone_latent_tile(value: Any) -> Any:
    if torch is not None and torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if hasattr(value, "copy"):
        return value.copy()
    return value


def router_tile_is_dynamic(router_tile: Mapping[str, Any] | None) -> bool:
    if router_tile is None:
        return False
    if bool(router_tile.get("dynamic", False)):
        return True
    reasons = router_tile.get("reasons", [])
    return "dynamic_region" in set(str(reason) for reason in reasons)


def world_tile_coord_from_router_tile(
    router_tile: Mapping[str, Any],
    *,
    map_tile_origin: Sequence[int] = (0, 0),
    tile_world_scale: int = 1,
) -> tuple[int, int]:
    """Derive a deterministic world tile coordinate from a router tile record.

    This helper is deliberately simple: map-projection code can pass explicit
    world coordinates directly to ``store``/``retrieve``. When only a router
    tile record is available, this maps its regular tile row/column into a
    world-map tile coordinate under a caller-provided integer origin and scale.
    """

    if "world_tile_coord" in router_tile:
        return normalize_world_tile_coord(router_tile["world_tile_coord"])
    origin_x, origin_y = normalize_world_tile_coord(map_tile_origin)
    scale = max(1, int(tile_world_scale))
    return (
        origin_x + int(router_tile["tile_col"]) * scale,
        origin_y + int(router_tile["tile_row"]) * scale,
    )
