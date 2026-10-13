"""Multi-GPU height-tiled Wan VAE decode helpers.

This module keeps parallel VAE decode opt-in and independent from DiT
generation. Each worker rank owns a deterministic subset of height tiles,
decodes full temporal chunks for those tiles, and returns CPU tile tensors for
ordered merge in the parent process.
"""

from __future__ import annotations

import os
import queue
import tempfile
import time
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.multiprocessing as mp

from utils.wan_wrapper import WanVAEWrapper


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.float32:
        return "float32"
    raise ValueError(f"Unsupported dtype: {dtype}")


def assign_tile_indices(num_tiles: int, vae_ranks: int) -> List[List[int]]:
    if int(num_tiles) < 1:
        raise ValueError("num_tiles must be >= 1")
    if int(vae_ranks) < 1:
        raise ValueError("vae_ranks must be >= 1")
    assignments = [[] for _ in range(int(vae_ranks))]
    for tile_index in range(int(num_tiles)):
        assignments[tile_index % int(vae_ranks)].append(tile_index)
    return assignments


def validate_parallel_tiled_config(
    latent_height: int,
    tile_height: int,
    tile_overlap: int,
    tile_blend_mode: str,
    vae_ranks: int,
):
    if int(vae_ranks) < 1:
        raise ValueError("vae_ranks must be >= 1")
    if int(tile_height) < 1:
        raise ValueError("tile_height must be >= 1")
    if int(tile_height) >= int(latent_height):
        raise ValueError("parallel tiled decode requires tile_height < latent height")
    if int(tile_overlap) < 0:
        raise ValueError("tile_overlap must be >= 0")
    if int(tile_overlap) >= int(tile_height):
        raise ValueError("tile_overlap must be smaller than tile_height")
    blend_mode = str(tile_blend_mode or "linear").strip().lower()
    if blend_mode not in {"crop", "linear"}:
        raise ValueError("tile_blend_mode must be 'crop' or 'linear'")


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    }


def _tile_records_by_index(decoded_tiles: Iterable[Dict]) -> Dict[int, Dict]:
    by_index = {}
    for record in decoded_tiles:
        tile_index = int(record["tile_index"])
        if tile_index in by_index:
            raise ValueError(f"duplicate decoded tile index: {tile_index}")
        by_index[tile_index] = record
    return by_index


def merge_decoded_height_tiles(
    decoded_tiles: Iterable[Dict],
    tile_ranges: Sequence[Tuple[int, int]],
    latent_height: int,
    tile_blend_mode: str = "linear",
) -> torch.Tensor:
    """Merge decoded CPU tile tensors into [B, T, C, H, W] order."""
    tile_blend_mode = str(tile_blend_mode or "linear").strip().lower()
    if tile_blend_mode not in {"crop", "linear"}:
        raise ValueError("tile_blend_mode must be 'crop' or 'linear'")
    if not tile_ranges:
        raise ValueError("tile_ranges must not be empty")

    by_index = _tile_records_by_index(decoded_tiles)
    missing = sorted(set(range(len(tile_ranges))) - set(by_index))
    if missing:
        raise ValueError(f"missing decoded tile indices: {missing}")

    first = by_index[0]["decoded"]
    if first.ndim != 5:
        raise ValueError("decoded tile tensors must have shape [B, T, C, H, W]")
    latent_start, latent_end = tile_ranges[0]
    scale_h = int(first.shape[-2]) // max(1, int(latent_end) - int(latent_start))
    if first.shape[-2] != (int(latent_end) - int(latent_start)) * scale_h:
        raise RuntimeError("parallel tiled decode produced a non-integral height scale")
    full_pixel_height = int(latent_height) * scale_h

    if tile_blend_mode == "crop":
        output = first.new_zeros(*first.shape[:-2], full_pixel_height, first.shape[-1])
        for tile_index in range(len(tile_ranges)):
            decoded = by_index[tile_index]["decoded"]
            latent_start, latent_end = tile_ranges[tile_index]
            tile_scale_h = int(decoded.shape[-2]) // max(1, int(latent_end) - int(latent_start))
            if tile_scale_h != scale_h:
                raise RuntimeError("parallel tiled decode produced inconsistent height scales")
            pixel_start = int(latent_start) * scale_h
            own_start, own_end = WanVAEWrapper._owned_pixel_range(
                tile_ranges, tile_index, int(latent_height), scale_h)
            local_start = own_start - pixel_start
            local_end = own_end - pixel_start
            output[..., own_start:own_end, :] = decoded[..., local_start:local_end, :]
        return output

    accum = first.float().new_zeros(*first.shape[:-2], full_pixel_height, first.shape[-1])
    weight = first.float().new_zeros(1, 1, 1, full_pixel_height, 1)
    for tile_index in range(len(tile_ranges)):
        decoded = by_index[tile_index]["decoded"].float()
        latent_start, latent_end = tile_ranges[tile_index]
        tile_scale_h = int(decoded.shape[-2]) // max(1, int(latent_end) - int(latent_start))
        if tile_scale_h != scale_h:
            raise RuntimeError("parallel tiled decode produced inconsistent height scales")
        pixel_start = int(latent_start) * scale_h
        pixel_end = pixel_start + int(decoded.shape[-2])
        tile_weight = WanVAEWrapper._tile_blend_weights(
            tile_ranges,
            tile_index,
            int(decoded.shape[-2]),
            scale_h,
            device=decoded.device,
            dtype=decoded.dtype,
        )
        accum[..., pixel_start:pixel_end, :] += decoded * tile_weight
        weight[..., pixel_start:pixel_end, :] += tile_weight

    return accum / weight.clamp_min(1.0e-6)


def _parallel_vae_worker(
    rank: int,
    device_id: int,
    task_queue,
    result_queue,
    latents_path: str,
    output_dir: str,
    vae_path: str,
    model_root_path: str,
    dtype_label: str,
    tile_ranges: Sequence[Tuple[int, int]],
):
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("parallel VAE decode requires CUDA")
        device = torch.device("cuda", int(device_id))
        torch.cuda.set_device(device)
        dtype = dtype_from_name(dtype_label)

        load_start = time.perf_counter()
        latents_cpu = torch.load(latents_path, map_location="cpu")
        if not torch.is_tensor(latents_cpu):
            raise RuntimeError("latents_path must contain a torch tensor")
        vae = WanVAEWrapper(model_root_path=model_root_path, vae_path=vae_path)
        vae = vae.to(device=device, dtype=dtype).eval()
        _sync(device)
        load_sec = time.perf_counter() - load_start
        result_queue.put({
            "type": "ready",
            "rank": int(rank),
            "device_id": int(device_id),
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "load_sec": float(load_sec),
        })

        while True:
            task = task_queue.get()
            if task.get("type") == "stop":
                break
            if task.get("type") != "decode":
                raise RuntimeError(f"unknown worker task: {task}")

            assigned_tile_indices = [int(x) for x in task["tile_indices"]]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            rank_decode_start = time.perf_counter()
            tile_results = []

            for tile_index in assigned_tile_indices:
                latent_start, latent_end = tile_ranges[tile_index]
                h2d_start = time.perf_counter()
                tile_latents = latents_cpu[..., int(latent_start):int(latent_end), :].contiguous()
                tile_latents = tile_latents.to(device=device, dtype=dtype)
                _sync(device)
                h2d_sec = time.perf_counter() - h2d_start

                decode_start = time.perf_counter()
                with torch.no_grad():
                    decoded = vae.decode_to_pixel(tile_latents)
                _sync(device)
                decode_sec = time.perf_counter() - decode_start

                d2h_start = time.perf_counter()
                decoded_cpu = decoded.detach().cpu()
                _sync(device)
                d2h_sec = time.perf_counter() - d2h_start

                tile_path = os.path.join(output_dir, f"rank{rank}_tile{tile_index}.pt")
                save_start = time.perf_counter()
                torch.save(decoded_cpu, tile_path)
                save_sec = time.perf_counter() - save_start
                tile_results.append({
                    "tile_index": int(tile_index),
                    "latent_start": int(latent_start),
                    "latent_end": int(latent_end),
                    "decoded_shape": list(decoded_cpu.shape),
                    "path": tile_path,
                    "h2d_sec": float(h2d_sec),
                    "decode_sec": float(decode_sec),
                    "d2h_sec": float(d2h_sec),
                    "save_sec": float(save_sec),
                })

                del tile_latents, decoded, decoded_cpu
                torch.cuda.empty_cache()

            _sync(device)
            rank_decode_sec = time.perf_counter() - rank_decode_start
            result_queue.put({
                "type": "done",
                "rank": int(rank),
                "device_id": int(device_id),
                "assigned_tile_indices": assigned_tile_indices,
                "rank_decode_sec": float(rank_decode_sec),
                "memory": _memory_snapshot(device),
                "tiles": tile_results,
            })
    except Exception as exc:
        result_queue.put({
            "type": "error",
            "rank": int(rank),
            "device_id": int(device_id),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


def _collect_messages(result_queue, expected_types, expected_count: int, timeout_sec: float):
    messages = []
    deadline = time.perf_counter() + float(timeout_sec)
    expected_types = set(expected_types)
    while len(messages) < int(expected_count):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {expected_types} messages")
        try:
            message = result_queue.get(timeout=min(remaining, 5.0))
        except queue.Empty:
            continue
        if message.get("type") == "error":
            raise RuntimeError(
                f"parallel VAE worker {message.get('rank')} failed: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )
        if message.get("type") in expected_types:
            messages.append(message)
    return messages


def decode_parallel_tiled_vae(
    latents: torch.Tensor,
    vae_path: str,
    vae_ranks: int,
    tile_height: int,
    tile_overlap: int,
    tile_blend_mode: str = "linear",
    dtype: torch.dtype = torch.bfloat16,
    device_ids: Optional[Sequence[int]] = None,
    model_root_path: str = "./",
    temp_dir: Optional[str] = None,
    timeout_sec: float = 900.0,
):
    """Decode a latent video with one process per VAE worker rank.

    Returns:
        tuple(decoded, metadata), where decoded has shape [B, T, C, H, W].
    """
    if not torch.is_tensor(latents):
        raise TypeError("latents must be a torch.Tensor")
    if latents.ndim != 5:
        raise ValueError("latents must have shape [B, T, C, H, W]")
    if not torch.cuda.is_available():
        raise RuntimeError("parallel VAE decode requires CUDA")

    latent_height = int(latents.shape[-2])
    validate_parallel_tiled_config(
        latent_height=latent_height,
        tile_height=int(tile_height),
        tile_overlap=int(tile_overlap),
        tile_blend_mode=tile_blend_mode,
        vae_ranks=int(vae_ranks),
    )
    available_gpus = torch.cuda.device_count()
    if device_ids is None:
        device_ids = list(range(int(vae_ranks)))
    else:
        device_ids = [int(x) for x in device_ids]
    if len(device_ids) < int(vae_ranks):
        raise ValueError("device_ids must contain at least vae_ranks entries")
    if max(device_ids[:int(vae_ranks)]) >= available_gpus:
        raise RuntimeError(
            f"requested CUDA device id {max(device_ids[:int(vae_ranks)])}, "
            f"but only {available_gpus} visible CUDA devices are available"
        )

    tile_blend_mode = str(tile_blend_mode or "linear").strip().lower()
    tile_ranges = WanVAEWrapper._height_tile_ranges(
        latent_height, int(tile_height), int(tile_overlap))
    assignments = assign_tile_indices(len(tile_ranges), int(vae_ranks))
    dtype_label = dtype_name(dtype)

    temp_context = None
    if temp_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="parallel_tiled_vae_")
        working_dir = temp_context.name
    else:
        working_dir = temp_dir
        os.makedirs(working_dir, exist_ok=True)

    processes = []
    task_queues = []
    try:
        latents_path = os.path.join(working_dir, "latents.pt")
        torch.save(latents.detach().cpu(), latents_path)

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        for rank in range(int(vae_ranks)):
            task_queue = ctx.Queue()
            process = ctx.Process(
                target=_parallel_vae_worker,
                args=(
                    rank,
                    int(device_ids[rank]),
                    task_queue,
                    result_queue,
                    latents_path,
                    working_dir,
                    vae_path,
                    model_root_path,
                    dtype_label,
                    tile_ranges,
                ),
            )
            process.start()
            processes.append(process)
            task_queues.append(task_queue)

        ready_messages = _collect_messages(
            result_queue, expected_types={"ready"}, expected_count=int(vae_ranks), timeout_sec=timeout_sec)
        ready_messages = sorted(ready_messages, key=lambda x: int(x["rank"]))

        dispatch_start = time.perf_counter()
        for rank, task_queue in enumerate(task_queues):
            task_queue.put({
                "type": "decode",
                "tile_indices": assignments[rank],
            })
        done_messages = _collect_messages(
            result_queue, expected_types={"done"}, expected_count=int(vae_ranks), timeout_sec=timeout_sec)
        parallel_decode_wall_sec = time.perf_counter() - dispatch_start
        done_messages = sorted(done_messages, key=lambda x: int(x["rank"]))

        tile_records = []
        load_tile_start = time.perf_counter()
        for rank_message in done_messages:
            for tile in rank_message["tiles"]:
                decoded = torch.load(tile["path"], map_location="cpu")
                tile_records.append({
                    "tile_index": int(tile["tile_index"]),
                    "decoded": decoded,
                })
        load_tile_sec = time.perf_counter() - load_tile_start

        merge_start = time.perf_counter()
        decoded = merge_decoded_height_tiles(
            tile_records,
            tile_ranges=tile_ranges,
            latent_height=latent_height,
            tile_blend_mode=tile_blend_mode,
        )
        merge_sec = time.perf_counter() - merge_start

        for task_queue in task_queues:
            task_queue.put({"type": "stop"})
        for process in processes:
            process.join(timeout=30)

        rank_peak_allocated = [
            float(message.get("memory", {}).get("peak_allocated_gib", 0.0))
            for message in done_messages
        ]
        rank_peak_reserved = [
            float(message.get("memory", {}).get("peak_reserved_gib", 0.0))
            for message in done_messages
        ]
        metadata = {
            "vae_ranks": int(vae_ranks),
            "device_ids": [int(x) for x in device_ids[:int(vae_ranks)]],
            "dtype": dtype_label,
            "latent_shape": list(latents.shape),
            "decoded_shape": list(decoded.shape),
            "tile_height": int(tile_height),
            "tile_overlap": int(tile_overlap),
            "tile_blend_mode": tile_blend_mode,
            "tile_ranges": [list(x) for x in tile_ranges],
            "tile_assignments": [[int(x) for x in row] for row in assignments],
            "ready": ready_messages,
            "workers": done_messages,
            "parallel_decode_wall_sec": float(parallel_decode_wall_sec),
            "tile_load_sec": float(load_tile_sec),
            "merge_sec": float(merge_sec),
            "total_parallel_decode_sec": float(parallel_decode_wall_sec + load_tile_sec + merge_sec),
            "max_rank_peak_allocated_gib": max(rank_peak_allocated) if rank_peak_allocated else 0.0,
            "sum_rank_peak_allocated_gib": sum(rank_peak_allocated),
            "max_rank_peak_reserved_gib": max(rank_peak_reserved) if rank_peak_reserved else 0.0,
            "sum_rank_peak_reserved_gib": sum(rank_peak_reserved),
        }
        return decoded, metadata
    finally:
        for task_queue in task_queues:
            try:
                task_queue.put({"type": "stop"})
            except Exception:
                pass
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        if temp_context is not None:
            temp_context.cleanup()
