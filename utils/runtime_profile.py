import json
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch


def _round_sec(value: float) -> float:
    return round(float(value), 6)


def _round_gib(value: float) -> float:
    return round(float(value), 4)


class RuntimeProfiler:
    """Passive JSONL profiler for AR chunk runtime measurements."""

    schema_version = 1

    def __init__(
        self,
        path: Optional[str] = None,
        enabled: bool = True,
        device: Optional[torch.device] = None,
        block_group_size: int = 5,
    ):
        self.path = path
        self.enabled = bool(enabled)
        self.device = device
        self.block_group_size = max(1, int(block_group_size))
        self.records = []
        self._records_by_chunk = {}
        self._item_metadata: Dict[str, Any] = {}
        self._shared_stages: Dict[str, Dict[str, Any]] = {}
        self._pending_transfer_events = []
        self._item_start_sec = None

    def reset_output(self) -> None:
        if not self.enabled or not self.path:
            return
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8"):
            pass

    def start_item(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self.records = []
        self._records_by_chunk = {}
        self._item_metadata = dict(metadata or {})
        self._shared_stages = {}
        self._pending_transfer_events = []
        self._item_start_sec = time.perf_counter()

    def start_timer(self, reset_peak: bool = False) -> Optional[float]:
        if not self.enabled:
            return None
        self._synchronize()
        if reset_peak:
            self._reset_peak_memory()
        return time.perf_counter()

    def stop_timer(self, timer: Optional[float]) -> Tuple[float, Dict[str, float]]:
        if not self.enabled or timer is None:
            return 0.0, {}
        self._synchronize()
        return time.perf_counter() - timer, self._memory_snapshot()

    def measure(
        self,
        fn: Callable[[], Any],
        reset_peak: bool = False,
    ) -> Tuple[Any, float, Dict[str, float]]:
        if not self.enabled:
            return fn(), 0.0, {}
        timer = self.start_timer(reset_peak=reset_peak)
        result = fn()
        elapsed, peak = self.stop_timer(timer)
        return result, elapsed, peak

    def set_shared_stage(
        self,
        name: str,
        seconds: float,
        peak: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled:
            return
        self._shared_stages[name] = {
            "seconds": _round_sec(seconds),
            "peak": dict(peak or {}),
        }

    def add_pending_transfer_event(
        self,
        name: str,
        seconds: float,
        peak: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled:
            return
        self._pending_transfer_events.append({
            "name": name,
            "duration_sec": _round_sec(seconds),
        })
        if peak:
            self._pending_transfer_events[-1].update(self._prefixed_peak(peak))

    def start_chunk(
        self,
        chunk_index: int,
        latent_start_frame: int,
        latent_end_frame: int,
        total_chunks: int,
        denoising_step_count: int,
        denoising_timesteps,
    ) -> None:
        if not self.enabled:
            return
        record = {
            "schema_version": self.schema_version,
            **self._item_metadata,
            "shared_stage_policy": "once_per_item; *_once_per_item_sec fields are duplicated on every chunk for reference",
            "chunk_index": int(chunk_index),
            "total_chunks": int(total_chunks),
            "latent_start_frame": int(latent_start_frame),
            "latent_end_frame_exclusive": int(latent_end_frame),
            "latent_frame_count": int(latent_end_frame - latent_start_frame),
            "denoising_step_count": int(denoising_step_count),
            "denoising_timesteps": [float(x) for x in denoising_timesteps],
            "text_encoding_sec": 0.0,
            "image_vae_encode_sec": 0.0,
            "cpu_gpu_transfer_sec": 0.0,
            "dit_chunk_sec": 0.0,
            "denoising_total_sec": 0.0,
            "kv_cache_update_sec": 0.0,
            "vae_decode_sec": 0.0,
            "postprocess_sec": 0.0,
            "video_encoding_sec": 0.0,
            "total_profiled_sec": 0.0,
            "peak_vram_allocated_gib": 0.0,
            "peak_vram_reserved_gib": 0.0,
            "denoising_steps": [],
            "dit_block_groups": [],
            "transfer_events": [],
            "vae_decode_chunks": [],
            "video_encoding_events": [],
        }
        for name, stage in self._shared_stages.items():
            shared_key = f"{name}_shared_sec"
            record[shared_key] = stage["seconds"]
            record[f"{name}_once_per_item_sec"] = stage["seconds"]
            if int(chunk_index) == 0:
                record[f"{name}_sec"] = stage["seconds"]
                self._merge_peak(record, stage.get("peak"))
        if int(chunk_index) == 0 and self._pending_transfer_events:
            for event in self._pending_transfer_events:
                self._append_transfer(record, event)
            self._pending_transfer_events = []
        self.records.append(record)
        self._records_by_chunk[int(chunk_index)] = record

    def add_transfer_event(
        self,
        chunk_index: Optional[int],
        name: str,
        seconds: float,
        peak: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "name": name,
            "duration_sec": _round_sec(seconds),
        }
        if peak:
            event.update(self._prefixed_peak(peak))
        record = self._record_or_pending(chunk_index)
        if record is None:
            self._pending_transfer_events.append(event)
            return
        self._append_transfer(record, event)
        self._merge_peak(record, peak)

    def add_dit_block_group_timing(
        self,
        chunk_index: Optional[int],
        phase: str,
        denoising_step_index: Optional[int],
        layer_start: int,
        layer_end: int,
        seconds: float,
        peak: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled or chunk_index is None:
            return
        record = self._records_by_chunk.get(int(chunk_index))
        if record is None:
            return
        entry = {
            "phase": phase,
            "denoising_step_index": None if denoising_step_index is None else int(denoising_step_index),
            "layer_start": int(layer_start),
            "layer_end_exclusive": int(layer_end),
            "duration_sec": _round_sec(seconds),
        }
        record["dit_block_groups"].append(entry)
        self._merge_peak(record, peak)

    def add_denoising_step(
        self,
        chunk_index: int,
        step_index: int,
        timestep: float,
        model_forward_sec: float,
        scheduler_update_sec: float,
        total_sec: float,
        peak: Optional[Dict[str, float]] = None,
        residual_reuse_decision: Optional[str] = None,
        residual_reuse_reason: Optional[str] = None,
        residual_reuse_metric: Optional[float] = None,
        residual_reuse_threshold: Optional[float] = None,
        camera_discontinuity_disabled: Optional[bool] = None,
    ) -> None:
        if not self.enabled:
            return
        record = self._records_by_chunk.get(int(chunk_index))
        if record is None:
            return
        step_record = {
            "step_index": int(step_index),
            "timestep": float(timestep),
            "model_forward_sec": _round_sec(model_forward_sec),
            "scheduler_update_sec": _round_sec(scheduler_update_sec),
            "total_sec": _round_sec(total_sec),
        }
        if residual_reuse_decision is not None:
            step_record["residual_reuse_decision"] = str(residual_reuse_decision)
            step_record["residual_reuse_reason"] = str(residual_reuse_reason)
            step_record["residual_reuse_metric"] = None if residual_reuse_metric is None else float(residual_reuse_metric)
            step_record["residual_reuse_threshold"] = None if residual_reuse_threshold is None else float(residual_reuse_threshold)
            step_record["camera_discontinuity_disabled"] = bool(camera_discontinuity_disabled)
        record["denoising_steps"].append(step_record)
        record["denoising_total_sec"] = _round_sec(
            record["denoising_total_sec"] + float(total_sec)
        )
        self._merge_peak(record, peak)

    def add_kv_cache_update(
        self,
        chunk_index: int,
        seconds: float,
        peak: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled:
            return
        record = self._records_by_chunk.get(int(chunk_index))
        if record is None:
            return
        record["kv_cache_update_sec"] = _round_sec(seconds)
        self._merge_peak(record, peak)

    def finish_dit_chunk(
        self,
        chunk_index: int,
        seconds: float,
        peak: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled:
            return
        record = self._records_by_chunk.get(int(chunk_index))
        if record is None:
            return
        record["dit_chunk_sec"] = _round_sec(seconds)
        self._merge_peak(record, peak)

    def add_vae_decode_chunk(
        self,
        chunk_index: Optional[int],
        latent_start_frame: int,
        latent_end_frame: int,
        seconds: float,
        peak: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled:
            return
        record = self._record_or_first(chunk_index)
        if record is None:
            return
        record["vae_decode_chunks"].append({
            "latent_start_frame": int(latent_start_frame),
            "latent_end_frame_exclusive": int(latent_end_frame),
            "duration_sec": _round_sec(seconds),
        })
        record["vae_decode_sec"] = _round_sec(record["vae_decode_sec"] + float(seconds))
        self._merge_peak(record, peak)

    def add_postprocess(
        self,
        chunk_index: Optional[int],
        seconds: float,
        peak: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled:
            return
        record = self._record_or_first(chunk_index)
        if record is None:
            return
        record["postprocess_sec"] = _round_sec(record["postprocess_sec"] + float(seconds))
        self._merge_peak(record, peak)

    def add_video_encoding(
        self,
        chunk_index: Optional[int],
        seconds: float,
        frames: int = 0,
        event: str = "append_frames",
        peak: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled:
            return
        record = self._record_or_first(chunk_index)
        if record is None:
            return
        record["video_encoding_events"].append({
            "name": event,
            "frames": int(frames),
            "duration_sec": _round_sec(seconds),
        })
        record["video_encoding_sec"] = _round_sec(record["video_encoding_sec"] + float(seconds))
        self._merge_peak(record, peak)

    def finish_item(self) -> None:
        if not self.enabled:
            return
        item_total = time.perf_counter() - self._item_start_sec if self._item_start_sec is not None else 0.0
        for record in self.records:
            record["item_total_sec"] = _round_sec(item_total)
            # Shared stages set after the last chunk record was created (e.g.
            # the pre-decode DiT offload or the overlap drain, which happen
            # after inference) would otherwise be lost; backfill them here.
            for name, stage in self._shared_stages.items():
                shared_key = f"{name}_once_per_item_sec"
                if shared_key not in record:
                    record[shared_key] = stage["seconds"]
                    record[f"{name}_shared_sec"] = stage["seconds"]
            record["dit_block_group_total_sec"] = _round_sec(
                sum(float(x["duration_sec"]) for x in record["dit_block_groups"])
            )
            record["total_profiled_sec"] = _round_sec(
                float(record.get("text_encoding_sec", 0.0))
                + float(record.get("image_vae_encode_sec", 0.0))
                + float(record.get("cpu_gpu_transfer_sec", 0.0))
                + float(record.get("denoising_total_sec", 0.0))
                + float(record.get("kv_cache_update_sec", 0.0))
                + float(record.get("vae_decode_sec", 0.0))
                + float(record.get("postprocess_sec", 0.0))
                + float(record.get("video_encoding_sec", 0.0))
            )

    def write_records(self, path: Optional[str] = None) -> int:
        if not self.enabled:
            return 0
        output_path = path or self.path
        if not output_path:
            return 0
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        count = 0
        with open(output_path, "a", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        return count

    def chunk_index_for_latent_frame(self, latent_frame: int) -> Optional[int]:
        for record in self.records:
            if (
                int(record["latent_start_frame"])
                <= int(latent_frame)
                < int(record["latent_end_frame_exclusive"])
            ):
                return int(record["chunk_index"])
        return None

    def _record_or_pending(self, chunk_index: Optional[int]):
        if chunk_index is None:
            if self.records:
                return self.records[0]
            return None
        return self._records_by_chunk.get(int(chunk_index))

    def _record_or_first(self, chunk_index: Optional[int]):
        if chunk_index is not None:
            return self._records_by_chunk.get(int(chunk_index))
        return self.records[0] if self.records else None

    def _append_transfer(self, record: Dict[str, Any], event: Dict[str, Any]) -> None:
        record["transfer_events"].append(event)
        record["cpu_gpu_transfer_sec"] = _round_sec(
            float(record["cpu_gpu_transfer_sec"]) + float(event["duration_sec"])
        )
        if "peak_vram_allocated_gib" in event:
            record["peak_vram_allocated_gib"] = _round_gib(
                max(
                    float(record.get("peak_vram_allocated_gib", 0.0)),
                    float(event.get("peak_vram_allocated_gib", 0.0)),
                )
            )
        if "peak_vram_reserved_gib" in event:
            record["peak_vram_reserved_gib"] = _round_gib(
                max(
                    float(record.get("peak_vram_reserved_gib", 0.0)),
                    float(event.get("peak_vram_reserved_gib", 0.0)),
                )
            )

    def _merge_peak(self, record: Dict[str, Any], peak: Optional[Dict[str, float]]) -> None:
        if not peak:
            return
        record["peak_vram_allocated_gib"] = _round_gib(
            max(float(record.get("peak_vram_allocated_gib", 0.0)), float(peak.get("peak_allocated_gib", 0.0)))
        )
        record["peak_vram_reserved_gib"] = _round_gib(
            max(float(record.get("peak_vram_reserved_gib", 0.0)), float(peak.get("peak_reserved_gib", 0.0)))
        )

    def _prefixed_peak(self, peak: Dict[str, float]) -> Dict[str, float]:
        return {
            "peak_vram_allocated_gib": _round_gib(float(peak.get("peak_allocated_gib", 0.0))),
            "peak_vram_reserved_gib": _round_gib(float(peak.get("peak_reserved_gib", 0.0))),
        }

    def _synchronize(self) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        device = self.device
        if device is not None and getattr(device, "type", None) != "cuda":
            return
        torch.cuda.synchronize(device)

    def _reset_peak_memory(self) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        device = self.device
        if device is not None and getattr(device, "type", None) != "cuda":
            return
        torch.cuda.reset_peak_memory_stats(device)

    def _memory_snapshot(self) -> Dict[str, float]:
        if not self.enabled or not torch.cuda.is_available():
            return {}
        device = self.device
        if device is not None and getattr(device, "type", None) != "cuda":
            return {}
        return {
            "peak_allocated_gib": _round_gib(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
            "peak_reserved_gib": _round_gib(torch.cuda.max_memory_reserved(device) / (1024 ** 3)),
        }
