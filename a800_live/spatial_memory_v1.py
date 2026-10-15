"""Experimental pose-selected clean-latent sink context for native DreamXAR.

No playback, training, 3D reconstruction, changed sampler or generated-latent
overwrite. Native scratch recaching runs only after a similar-pose oldmemory
hit. Longterm payloads live onCPU/disk; active native localKV remains bounded.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    import torch
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def angular_distance_degrees(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _valid_pose(position, yaw):
    if len(position) != 3 or not all(math.isfinite(float(x)) for x in position):
        raise ValueError("A finite global modelspaceXYZ is required")
    if not math.isfinite(float(yaw)):
        raise ValueError("A finite global modelspace yaw is required")


@dataclass(frozen=True)
class MemoryEntry:
    record_id: str
    session_id: str
    map_revision_id: str
    latent_start: int
    latent_count: int
    pose_xyz: tuple
    yaw_degrees: float
    model_time_seconds: float
    payload_file: str
    payload_sha256: str
    raw_latent_sha256: str
    camera_sha256: dict


class SpatialMemoryBank:
    """One map and one continuous session; no geographic visibility claim.

    add() accepts completed nativechunks only. query() changes no GPU state.
    mark_applied() starts the fixed cooldown only after successful recaching.
    clear() isolates a newI0/mapworld; oldpayloads remain as immutable evidence.
    """
    def __init__(self, root, *, session_id, map_revision_id,
                 radius_m=0.75, yaw_tolerance_degrees=20.0,
                 minimum_age_latents=15, cooldown_seconds=1.5,
                 max_records=512):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.radius_m = float(radius_m)
        self.yaw_tolerance_degrees = float(yaw_tolerance_degrees)
        self.minimum_age_latents = int(minimum_age_latents)
        self.cooldown_seconds = float(cooldown_seconds)
        self.max_records = int(max_records)
        if self.radius_m <= 0 or not 0 < self.yaw_tolerance_degrees < 90:
            raise ValueError("Positive pose radius and similarheading tolerance required")
        if self.minimum_age_latents < 12 or self.cooldown_seconds < 0 or self.max_records < 1:
            raise ValueError("Exclude native localwindow and bound metadata")
        self.records = []
        self.total_committed = 0
        self.total_evicted_from_index = 0
        self.query_count = self.hit_count = self.applied_count = 0
        self.active_record_id = None
        self.last_applied_model_time = -math.inf
        self.last_query = None
        self.session_id = str(session_id); self.map_revision_id = str(map_revision_id)
        self._assert_scope()
        if (self.root / "INDEX.json").exists():
            old = json.loads((self.root / "INDEX.json").read_text())
            if old["session_id"] != self.session_id or old["map_revision_id"] != self.map_revision_id:
                raise ValueError("Use clear() or a newmemorydirectory for a newworld")
            expected_policy = {"radius_m": self.radius_m, "yaw_tolerance_degrees": self.yaw_tolerance_degrees,
                               "minimum_age_latents": self.minimum_age_latents,
                               "cooldown_seconds": self.cooldown_seconds, "max_records": self.max_records}
            if old.get("selection_policy") != expected_policy:
                raise ValueError("Do not silently change a persistedmemorypolicy")
            self.records = [MemoryEntry(**{**item, "pose_xyz": tuple(item["pose_xyz"])})
                            for item in old["records"]]
            for key in ("total_committed", "total_evicted_from_index", "query_count", "hit_count", "applied_count"):
                setattr(self, key, int(old[key]))
            self.active_record_id = old["active_record_id"]
            saved_time = old.get("last_applied_model_time")
            self.last_applied_model_time = -math.inf if saved_time is None else float(saved_time)
        self._db_lock = threading.RLock()
        self._db = sqlite3.connect(self.root / "SPATIAL_INDEX.sqlite3", timeout=5,
                                   check_same_thread=False)
        with self._db:
            self._db.execute("""CREATE TABLE IF NOT EXISTS memory_entries (
                record_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                map_revision_id TEXT NOT NULL, latent_start INTEGER NOT NULL,
                latent_count INTEGER NOT NULL, model_time REAL NOT NULL,
                x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
                yaw REAL NOT NULL, entry_json TEXT NOT NULL)""")
            self._db.execute("""CREATE INDEX IF NOT EXISTS memory_scope_spatial
                ON memory_entries(session_id, map_revision_id, x, y, z, latent_start)""")
        # Additive migration of a pre-SQL boundedmetadata index. No latent
        # payloads are loaded intoRAM and no historical rows are deleted.
        for record in self.records:
            self._index_entry(record)
        self._persist()

    def _index_entry(self, entry):
        encoded = json.dumps(asdict(entry), sort_keys=True, separators=(",", ":"))
        with self._db_lock, self._db:
            old = self._db.execute("SELECT entry_json FROM memory_entries WHERE record_id=?",
                                   (entry.record_id,)).fetchone()
            if old is not None:
                if old[0] != encoded:
                    raise ValueError("Immutable diskmemory metadata collision")
                return
            self._db.execute("INSERT INTO memory_entries VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                entry.record_id, entry.session_id, entry.map_revision_id, entry.latent_start,
                entry.latent_count, entry.model_time_seconds, *entry.pose_xyz,
                entry.yaw_degrees % 360.0, encoded))

    def _disk_candidates(self, *, current_start, pose_xyz, yaw_degrees, model_time_seconds):
        x, y, z = map(float, pose_xyz)
        radius = self.radius_m
        parameters = {"session": self.session_id, "map": self.map_revision_id,
                      "cutoff": int(current_start) - self.minimum_age_latents,
                      "time": float(model_time_seconds), "qx": x, "qy": y, "qz": z,
                      "xmin": x - radius, "xmax": x + radius,
                      "ymin": y - radius, "ymax": y + radius,
                      "zmin": z - radius, "zmax": z + radius,
                      "yaw": float(yaw_degrees) % 360.0,
                      "tol": self.yaw_tolerance_degrees,
                      "wrap_tol": 360.0 - self.yaw_tolerance_degrees}
        with self._db_lock:
            rows = self._db.execute("""SELECT entry_json FROM memory_entries
                WHERE session_id=:session AND map_revision_id=:map
                  AND latent_start + latent_count <= :cutoff AND model_time <= :time
                  AND x BETWEEN :xmin AND :xmax AND y BETWEEN :ymin AND :ymax
                  AND z BETWEEN :zmin AND :zmax
                  AND (abs(yaw-:yaw)<=:tol OR abs(yaw-:yaw)>=:wrap_tol)
                ORDER BY (x-:qx)*(x-:qx)+(y-:qy)*(y-:qy)+(z-:qz)*(z-:qz),
                  min(abs(yaw-:yaw), 360.0-abs(yaw-:yaw)), latent_start, record_id
                LIMIT 512""", parameters).fetchall()
        return [MemoryEntry(**{**json.loads(row[0]), "pose_xyz": tuple(json.loads(row[0])["pose_xyz"])})
                for row in rows]

    def _assert_scope(self):
        if not self.session_id or not self.map_revision_id:
            raise ValueError("Explicit session and maprevision required")

    def _persist(self):
        _save_json(self.root / "INDEX.json", {
            **self.status(), "records": [asdict(record) for record in self.records]})

    def clear(self, *, session_id, map_revision_id):
        self.session_id = str(session_id); self.map_revision_id = str(map_revision_id)
        self._assert_scope()
        self.records = []; self.active_record_id = None
        self.last_applied_model_time = -math.inf
        self.last_query = {"reason": "explicit_newI0_or_map_scope"}
        self._persist()

    def add(self, chunk, start, *, camera, pose_xyz, yaw_degrees,
            model_time_seconds):
        import torch
        _valid_pose(pose_xyz, yaw_degrees)
        if not math.isfinite(float(model_time_seconds)):
            raise ValueError("Finite querymodeltime required")
        start = int(start)
        if start < 3:
            return {"committed": False, "reason": "protect_I0_chunk"}
        if (tuple(chunk.shape[:3]) != (1, 3, 48) or tuple(chunk.shape[3:]) not in {(24, 40), (28, 48)}
                or chunk.dtype != torch.bfloat16):
            raise ValueError("Expected completed nativeBF16chunk at640x384 or768x448")
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("Memorycleanlatent contains nonfinite values")
        if self.records and start < self.records[-1].latent_start + 3:
            raise ValueError("Memorycommit is duplicate, overlapping or outoforder")
        if set(camera) != {"viewmats", "K"}:
            raise ValueError("Save the actual native viewmats andK")
        tokens = 3 * (chunk.shape[3] // 2) * (chunk.shape[4] // 2)
        expected = {"viewmats": (1, tokens, 4, 4), "K": (1, tokens, 3, 3)}
        for key, value in camera.items():
            if tuple(value.shape) != expected[key] or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid original perchunk camera payload")
        payload = {
            "latent": chunk.detach().to("cpu").clone().contiguous(),
            "camera": {key: value.detach().to("cpu").clone().contiguous()
                       for key, value in camera.items()},
            "session_id": self.session_id, "map_revision_id": self.map_revision_id,
            "latent_start": start, "latent_count": 3,
            "pose_xyz": list(map(float, pose_xyz)), "yaw_degrees": float(yaw_degrees),
            "model_time_seconds": float(model_time_seconds),
        }
        if not math.isfinite(float(model_time_seconds)):
            raise ValueError("Finite modeltime required")
        scope = hashlib.sha256((self.session_id + "\0" + self.map_revision_id).encode()).hexdigest()[:16]
        record_id = f"{scope}_{start:09d}"
        destination = self.root / f"{record_id}.pt"
        if destination.exists():
            raise ValueError("Immutable memorypayload alreadyexists")
        temporary = destination.with_suffix(".pt.partial")
        torch.save(payload, temporary); temporary.replace(destination)
        entry = MemoryEntry(record_id, self.session_id, self.map_revision_id, start, 3,
                            tuple(map(float, pose_xyz)), float(yaw_degrees), float(model_time_seconds),
                            destination.name, _sha(destination), tensor_sha256(payload["latent"]),
                            {key: tensor_sha256(value) for key, value in payload["camera"].items()})
        self._index_entry(entry)
        self.records.append(entry); self.total_committed += 1
        if len(self.records) > self.max_records:
            self.total_evicted_from_index += len(self.records) - self.max_records
            self.records = self.records[-self.max_records:]
        self._persist()
        return {"committed": True, "record_id": record_id,
                "payload_sha256": entry.payload_sha256, "cpu_payload": True}

    def query(self, *, current_start, pose_xyz, yaw_degrees,
              model_time_seconds, session_id=None, map_revision_id=None):
        _valid_pose(pose_xyz, yaw_degrees)
        if not math.isfinite(float(model_time_seconds)):
            raise ValueError("Finite querymodeltime required")
        self.query_count += 1
        audit = {"current_start": int(current_start), "pose_xyz": list(pose_xyz),
                 "yaw_degrees": float(yaw_degrees), "model_time_seconds": float(model_time_seconds),
                 "selected_record_id": None, "reason": "no_eligible_sameview_memory",
                 "radius_m": self.radius_m, "yaw_tolerance_degrees": self.yaw_tolerance_degrees,
                 "minimum_age_latents": self.minimum_age_latents,
                 "cooldown_seconds": self.cooldown_seconds,
                 "disk_spatial_index": True, "disk_candidate_limit": 512}
        if (session_id is not None and session_id != self.session_id) or (
                map_revision_id is not None and map_revision_id != self.map_revision_id):
            audit["reason"] = "scope_mismatch"
        elif float(model_time_seconds) - self.last_applied_model_time < self.cooldown_seconds:
            audit["reason"] = "cooldown"
        else:
            candidates = []
            disk_records = self._disk_candidates(current_start=current_start, pose_xyz=pose_xyz,
                                                 yaw_degrees=yaw_degrees,
                                                 model_time_seconds=model_time_seconds)
            audit["disk_candidates_read"] = len(disk_records)
            for record in disk_records:
                if record.session_id != self.session_id or record.map_revision_id != self.map_revision_id:
                    continue
                if record.latent_start + 3 > int(current_start) - self.minimum_age_latents:
                    continue
                if record.model_time_seconds > float(model_time_seconds):
                    continue
                distance = math.sqrt(sum((a - float(b)) ** 2 for a, b in zip(record.pose_xyz, pose_xyz)))
                yaw = angular_distance_degrees(record.yaw_degrees, yaw_degrees)
                if distance <= self.radius_m and yaw <= self.yaw_tolerance_degrees:
                    candidates.append((distance, yaw, record.latent_start, record.record_id, record))
            candidates.sort(key=lambda item: item[:4])
            if candidates:
                candidate = candidates[0]
                audit.update(selected_record_id=candidate[-1].record_id,
                             distance_m=candidate[0], yaw_difference_degrees=candidate[1])
                if candidate[-1].record_id == self.active_record_id:
                    audit["reason"] = "same_active_sink"
                else:
                    audit["reason"] = "eligible_sameview_memory"
                    self.hit_count += 1; self.last_query = audit
                    return candidate[-1], audit
        self.last_query = audit
        return None, audit

    def load(self, entry):
        import torch
        if entry.session_id != self.session_id or entry.map_revision_id != self.map_revision_id:
            raise ValueError("Refuse memoryfromanothermap/world")
        path = self.root / entry.payload_file
        if path.parent != self.root or _sha(path) != entry.payload_sha256:
            raise ValueError("Memorypayload path/hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["session_id"] != self.session_id or payload["map_revision_id"] != self.map_revision_id:
            raise ValueError("Memoryscope metadata mismatch")
        if tensor_sha256(payload["latent"]) != entry.raw_latent_sha256 or any(
            tensor_sha256(payload["camera"][key]) != digest for key, digest in entry.camera_sha256.items()):
            raise ValueError("Memorytensor hashes mismatch")
        payload["record_id"] = entry.record_id
        payload["payload_sha256"] = entry.payload_sha256
        return payload

    def mark_applied(self, entry, *, model_time_seconds):
        if entry.session_id != self.session_id or entry.map_revision_id != self.map_revision_id:
            raise ValueError("Cannotmarkanotherworld applied")
        self.active_record_id = entry.record_id
        self.last_applied_model_time = float(model_time_seconds)
        self.applied_count += 1
        self._persist()

    def status(self):
        if hasattr(self, "_db"):
            with self._db_lock:
                disk_total = self._db.execute("SELECT count(*) FROM memory_entries").fetchone()[0]
                disk_scope = self._db.execute("SELECT count(*) FROM memory_entries WHERE session_id=? AND map_revision_id=?",
                                               (self.session_id, self.map_revision_id)).fetchone()[0]
        else:
            disk_total = disk_scope = 0
        return {"kind": "experimental_pose_selected_native_sink_context",
                "session_id": self.session_id, "map_revision_id": self.map_revision_id,
                "records_in_RAM": len(self.records), "max_records": self.max_records,
                "latent_payloads_in_RAM": 0, "total_committed": self.total_committed,
                "total_evicted_from_index": self.total_evicted_from_index,
                "total_evicted_from_RAM": self.total_evicted_from_index,
                "disk_index_total_records": disk_total, "disk_index_scope_records": disk_scope,
                "disk_spatial_query_candidate_limit": 512,
                "query_count": self.query_count, "hit_count": self.hit_count,
                "applied_count": self.applied_count, "active_record_id": self.active_record_id,
                "last_applied_model_time": self.last_applied_model_time if math.isfinite(self.last_applied_model_time) else None,
                "selection_policy": {"radius_m": self.radius_m, "yaw_tolerance_degrees": self.yaw_tolerance_degrees,
                                     "minimum_age_latents": self.minimum_age_latents,
                                     "cooldown_seconds": self.cooldown_seconds, "max_records": self.max_records},
                "last_query": self.last_query, "old_payloads_preserved_on_disk": True,
                "opposite_heading_or_unseen_side_consistency_guaranteed": False}


class NativeSinkMemoryAdapter:
    """Rebuild native un-RoPE sinkcontext, never copy oldforeign timelineKV.

    ScratchKV is reused. No new PRoPEcache is introduced: the current native
    camera branch remains perchunk. This is a conditioningpolicy change.
    """
    def __init__(self, pipeline, *, verify_tensor_hashes=True):
        self.pipeline = pipeline
        self.verify_tensor_hashes = bool(verify_tensor_hashes)
        self.scratch = None
        self.calls = 0

    def recache(self, conditional_dict, payload):
        import torch
        began = time.perf_counter()
        pipeline = self.pipeline
        live = pipeline.kv_cache1
        if len(live) != 30 or pipeline.frame_seq_length not in {240, 336}:
            raise ValueError("Expected native30blocks and240or336tokens/frame")
        sink_tokens = 3 * pipeline.frame_seq_length
        for block in pipeline.generator.model.blocks:
            if int(block.self_attn.sink_size) != 3:
                raise ValueError("Preserve native3framesink contract")
        if any("prope_k" in cache for cache in live):
            raise ValueError("Thisadapter doesnot invent globalcameraKV semantics")
        if not all(cache["is_init"] for cache in pipeline.crossattn_cache):
            raise ValueError("Native text/mapcrosscontext must alreadybe initialized")
        latent = payload["latent"]
        if (tuple(latent.shape[:3]) != (1, 3, 48) or tuple(latent.shape[3:]) not in {(24, 40), (28, 48)}
                or latent.dtype != torch.bfloat16):
            raise ValueError("Unexpected cleanmemorylatent")
        expected_frame_tokens = (latent.shape[3] // 2) * (latent.shape[4] // 2)
        if expected_frame_tokens != pipeline.frame_seq_length:
            raise ValueError("Memory and activegeneration native gridsmustmatch")
        expected_camera = {"viewmats": (1, sink_tokens, 4, 4), "K": (1, sink_tokens, 3, 3)}
        if set(payload["camera"]) != set(expected_camera) or any(
                tuple(payload["camera"][key].shape) != shape for key, shape in expected_camera.items()):
            raise ValueError("Memorycamera griddoesnotmatch theactivenativegrid")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("Nonfinite memorylatent")
        device = live[0]["k"].device
        if self.scratch is None:
            self.scratch = [{"k": torch.zeros_like(cache["k"]),
                             "v": torch.zeros_like(cache["v"]),
                             "global_end_index": torch.zeros(1, dtype=torch.long, device="cpu"),
                             "local_end_index": torch.zeros(1, dtype=torch.long, device="cpu")}
                            for cache in live]
        for cache in self.scratch:
            for key in ("k", "v", "global_end_index", "local_end_index"):
                cache[key].zero_()
        # Snapshot RNG before any nativecall. Caller retains exact futuret/noise.
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        cursor_before = [(int(cache["global_end_index"].item()), int(cache["local_end_index"].item()))
                         for cache in live]
        local_hash_before = ([{key: tensor_sha256(cache[key][:, sink_tokens:]) for key in ("k", "v")}
                              for cache in live] if self.verify_tensor_hashes else None)
        cross_hash_before = ([{key: tensor_sha256(cache[key]) for key in ("k", "v")}
                              | {"is_init": bool(cache["is_init"])} for cache in pipeline.crossattn_cache]
                             if self.verify_tensor_hashes else None)
        source_hash = tensor_sha256(latent)
        rng_changed_inside = None
        try:
            latent_gpu = latent.to(device=device, dtype=live[0]["k"].dtype)
            camera_gpu = {key: value.to(device=device, dtype=live[0]["k"].dtype)
                          for key, value in payload["camera"].items()}
            # Native timestep inherits BF16 from the first-frame mask before
            # ones_like(timestep)*context_noise; preserve that scalar rounding.
            timestep = torch.ones((1, sink_tokens), device=device, dtype=latent_gpu.dtype) * pipeline.args.context_noise
            with torch.no_grad():
                pipeline.generator(noisy_image_or_video=latent_gpu,
                                   conditional_dict=conditional_dict, y=None,
                                   y_camera=camera_gpu, timestep=timestep,
                                   kv_cache=self.scratch, crossattn_cache=pipeline.crossattn_cache,
                                   current_start=0)
            if any(int(cache["global_end_index"].item()) != sink_tokens or
                   int(cache["local_end_index"].item()) != sink_tokens for cache in self.scratch):
                raise AssertionError("Scratchnativecache didnotcommit exactlythreeframes")
            if tensor_sha256(latent) != source_hash:
                raise AssertionError("Immutable sourcecleanlatent wasmodified")
            if self.verify_tensor_hashes:
                for before, cache in zip(cross_hash_before, pipeline.crossattn_cache):
                    after = {key: tensor_sha256(cache[key]) for key in ("k", "v")} | {"is_init": bool(cache["is_init"])}
                    if before != after:
                        raise AssertionError("Native memoryrecache changed text/mapcrosscontext")
            with torch.no_grad():
                for cache, scratch in zip(live, self.scratch):
                    cache["k"][:, :sink_tokens].copy_(scratch["k"][:, :sink_tokens])
                    cache["v"][:, :sink_tokens].copy_(scratch["v"][:, :sink_tokens])
            after_cursors = [(int(cache["global_end_index"].item()), int(cache["local_end_index"].item()))
                             for cache in live]
            local_hash_after = ([{key: tensor_sha256(cache[key][:, sink_tokens:]) for key in ("k", "v")}
                                 for cache in live] if self.verify_tensor_hashes else None)
            if after_cursors != cursor_before or (self.verify_tensor_hashes and local_hash_after != local_hash_before):
                raise AssertionError("Memoryrecache modified activecursors or rollinglocalKV")
            rng_changed_inside = not torch.equal(cpu_rng, torch.get_rng_state())
            if cuda_rng is not None:
                rng_changed_inside |= any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
            self.calls += 1
            result = {"record_id": payload.get("record_id"), "payload_sha256": payload.get("payload_sha256"),
                      "source_clean_latent_sha256": source_hash, "native_context_noise": float(pipeline.args.context_noise),
                      "native_context_timestep_dtype": str(timestep.dtype),
                      "native_frame_tokens": pipeline.frame_seq_length,
                      "replaced_sink_tokens_per_layer": sink_tokens, "native_layer_count": len(live),
                      "live_cursors_unchanged": True, "rolling_localKV_unchanged": True,
                      "cross_text_mapKV_unchanged": True, "source_clean_latent_immutable": True,
                      "non_sink_and_crossKV_hashes_verified": self.verify_tensor_hashes,
                      "rng_changed_inside_native_call": bool(rng_changed_inside),
                      "sink_hashes": ([{key: tensor_sha256(cache[key][:, :sink_tokens]) for key in ("k", "v")}
                                       for cache in live] if self.verify_tensor_hashes else None),
                      "camera_memory_is_global_prope_registration": False,
                      "conditioning_policy": "experimental pose_selected regenerated_native3frame_sink",
                      "seconds": time.perf_counter() - began}
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        result["cpu_rng_restored"] = bool(torch.equal(cpu_rng, torch.get_rng_state()))
        result["cuda_rng_restored"] = cuda_rng is None or all(
            torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
        return result
