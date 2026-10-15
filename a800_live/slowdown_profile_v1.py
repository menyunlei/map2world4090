"""Bounded live-chunk tracing; never changes model math or synchronizes a GPU.

Place GPU marks on the existing decode stream before wait_event, after
wait_event, after P2P, after VAE, and after uint8 conversion. Resolve after the
existing CPU readbacks/session join, using completed events only. CPU spans are
separate: GPU elapsed time is not described as pure VAE unless bounded by
after_p2p/after_vae. No CUDA imports occur in offline analysis.
"""
from collections import Counter
import json
from pathlib import Path
import statistics
import time


class ChunkTrace:
    GPU_ORDER = ("before_source_wait", "after_source_wait", "after_p2p",
                 "after_vae", "after_uint8")

    def __init__(self, latent_start, stream, *, max_events=8):
        self.latent_start = int(latent_start)
        self.stream = stream
        self.max_events = int(max_events)
        self.events = {}
        self.host = {"dequeue_epoch": time.time()}

    def gpu_mark(self, name):
        if name not in self.GPU_ORDER or name in self.events:
            raise ValueError("Unexpected or duplicate GPU phase")
        if len(self.events) >= self.max_events:
            raise ValueError("Bounded trace event budget exceeded")
        import torch
        event = torch.cuda.Event(enable_timing=True)
        event.record(self.stream)
        self.events[name] = event

    def host_mark(self, name):
        if name in self.host:
            raise ValueError("Duplicate host phase")
        self.host[name] = time.time()

    def resolve(self):
        """Query only; unresolved work is explicitly marked and never waited on."""
        ready = {name: event.query() for name, event in self.events.items()}
        spans = {}
        for before, after, label in (
            ("before_source_wait", "after_source_wait", "source_wait_gpu_ms"),
            ("after_source_wait", "after_p2p", "p2p_gpu_ms"),
            ("after_p2p", "after_vae", "vae_gpu_ms"),
            ("after_vae", "after_uint8", "rgb_convert_gpu_ms"),
        ):
            if before in self.events and after in self.events:
                spans[label] = (self.events[before].elapsed_time(self.events[after])
                                if ready[before] and ready[after] else None)
        return {"latent_start": self.latent_start, "host": dict(self.host),
                "gpu_ms": spans, "gpu_events_completed": ready,
                "synchronize_called_by_profiler": False}


class SessionTrace:
    def __init__(self, *, max_chunks=128):
        if not 1 <= max_chunks <= 512:
            raise ValueError("Bounded session trace required")
        self.max_chunks = int(max_chunks)
        self.traces = []

    def chunk(self, latent_start, stream):
        if len(self.traces) >= self.max_chunks:
            raise ValueError("Bounded trace chunk budget exceeded")
        trace = ChunkTrace(latent_start, stream)
        self.traces.append(trace)
        return trace

    def save(self, path, *, extra=None):
        path = Path(path)
        value = {"version": "slowdown_profile_v1", "bounded_max_chunks": self.max_chunks,
                 "traces": [trace.resolve() for trace in self.traces],
                 "weights_sampler_precision_unchanged": True,
                 "GPU_synchronize_added": False, "extra": dict(extra or {})}
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)
        return value


def vae_residency(vae):
    """Metadata only, called while idle; does not copy tensor values or run VAE."""
    import torch
    tensors = []
    for name, tensor in list(vae.named_parameters()) + list(vae.named_buffers()):
        tensors.append((name, tensor))
    for name in ("mean", "std", "scale"):
        value = getattr(vae, name, None)
        if isinstance(value, torch.Tensor):
            tensors.append((name, value))
        elif isinstance(value, (list, tuple)):
            tensors.extend((f"{name}.{i}", item) for i, item in enumerate(value)
                           if isinstance(item, torch.Tensor))
    model = getattr(vae, "model", vae)
    for name in ("_feat_map", "_enc_feat_map"):
        tensors.extend((f"cache.{name}.{i}", tensor)
                       for i, tensor in enumerate(getattr(model, name, []))
                       if isinstance(tensor, torch.Tensor))
    groups = Counter((str(t.device), str(t.dtype)) for _, t in tensors)
    return {"groups": [{"device": d, "dtype": dtype, "tensors": n}
                       for (d, dtype), n in sorted(groups.items())],
            "cache_entries": [{"name": name, "shape": list(t.shape),
                               "device": str(t.device), "dtype": str(t.dtype)}
                              for name, t in tensors if name.startswith("cache.")],
            "grad_enabled": torch.is_grad_enabled(),
            "metadata_only": True}


def _summary(values):
    values = list(values)
    if not values:
        return None
    n = min(10, len(values))
    return {"count": len(values), "mean": statistics.mean(values),
            "min": min(values), "max": max(values),
            "first10_mean": statistics.mean(values[:n]),
            "last10_mean": statistics.mean(values[-n:])}


def analyze_chunk_records(records):
    """Offline causal bounds, including old records missing producer timings."""
    records = sorted(records, key=lambda row: row["latent_start"])
    answer = {"chunks": len(records), "cause": "UNRESOLVED",
              "pure_vae_time_available": all("vae_gpu_ms" in row for row in records)
                                         and bool(records)}
    for key in ("sampled_epoch", "model_callback_epoch", "rgb_ready_epoch", "jpeg_ready_epoch"):
        if records and all(key in row for row in records):
            answer[key + "_interval_seconds"] = _summary(
                b[key] - a[key] for a, b in zip(records, records[1:]))
    for a, b, key in (
        ("sampled_epoch", "rgb_ready_epoch", "sample_to_rgb_seconds"),
        ("model_callback_epoch", "decode_start_epoch", "decode_queue_seconds"),
        ("rgb_ready_epoch", "jpeg_ready_epoch", "jpeg_and_host_seconds"),
        ("sampled_epoch", "model_callback_epoch", "camera_to_model_callback_seconds"),
    ):
        if records and all(a in row and b in row for row in records):
            answer[key] = _summary(row[b] - row[a] for row in records)
    if "decode_queue_seconds" in answer:
        answer["decoder_backlog_observed"] = answer["decode_queue_seconds"]["max"] > .1
    else:
        answer["decoder_backlog_observed"] = None
    answer["limitation"] = ("Camera-to-RGB includes model, queue, transfers, VAE, CPU readback "
                            "and hashing. Source-event wait must be excluded before attributing VAE.")
    return answer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks_json", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = {str(path): analyze_chunk_records(json.loads(path.read_text()))
               for path in args.chunks_json}
    text = json.dumps(reports, indent=2)
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
