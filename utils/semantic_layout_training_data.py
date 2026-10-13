"""Strict cached-tensor manifest for semantic-layout adapter training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .map_utils import LABEL_NAMES, NUM_LABELS
except ImportError:  # Support standalone loading in focused CPU tests.
    from map_utils import LABEL_NAMES, NUM_LABELS


MANIFEST_SCHEMA = "dreamx.semantic_layout_training"
MANIFEST_VERSION = 1
_REQUIRED_HEADER = {
    "type",
    "schema",
    "version",
    "prediction_type",
    "latent_layout",
    "prompt_layout",
    "label_layout",
    "label_names",
    "num_train_timesteps",
    "timestep_shift",
}
_REQUIRED_SAMPLE = {"type", "id", "latent", "prompt_embedding", "labels", "frame_progress"}
_REQUIRED_ARTIFACT = {"path", "sha256", "dtype", "shape"}
_CAMERA_ARTIFACTS = ("camera_viewmats", "camera_intrinsics")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_DTYPES = {np.dtype("float16"), np.dtype("float32")}
_LABEL_DTYPES = {
    np.dtype("uint8"),
    np.dtype("int8"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
}


class SemanticLayoutTrainingDataset(Dataset):
    """Load validated ``.npy`` latents, prompt embeddings, and layout labels."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        records = _load_jsonl(self.manifest_path)
        if len(records) < 2:
            raise ValueError("training manifest must contain one header and at least one sample")
        self.header = _validate_header(records[0])
        samples = [_validate_sample_record(record) for record in records[1:]]
        sample_ids = [record["id"] for record in samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("training manifest contains duplicate sample ids")
        self.samples = sorted(samples, key=lambda record: record["id"])
        self.manifest_digest = canonical_manifest_digest(self.header, self.samples)
        self.root = self.manifest_path.parent
        for record in self.samples:
            self._validate_artifacts(record)
        camera_presence = [all(field in record for field in _CAMERA_ARTIFACTS) for record in self.samples]
        if any(camera_presence) and not all(camera_presence):
            raise ValueError("camera artifacts must be present for every sample or for none")
        fixed_shape_fields = ["latent", "prompt_embedding", "labels"]
        if all(camera_presence):
            fixed_shape_fields.extend(_CAMERA_ARTIFACTS)
        for field in fixed_shape_fields:
            shapes = {tuple(record[field]["shape"]) for record in self.samples}
            if len(shapes) != 1:
                raise ValueError(f"all {field} artifacts must have one fixed shape per manifest")
        self.has_camera_conditioning = all(camera_presence)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[int(index)]
        latent = _load_array(self._artifact_path(record["latent"])).astype(np.float32, copy=True)
        prompt = _load_array(self._artifact_path(record["prompt_embedding"])).astype(np.float32, copy=True)
        labels = _load_array(self._artifact_path(record["labels"])).astype(np.int64, copy=True)
        item = {
            "id": record["id"],
            "clean_latents": torch.from_numpy(latent),
            "prompt_embeddings": torch.from_numpy(prompt),
            "semantic_layout_labels": torch.from_numpy(labels),
            "semantic_layout_progress": torch.tensor(record["frame_progress"], dtype=torch.float32),
        }
        if self.has_camera_conditioning:
            viewmats = _load_array(self._artifact_path(record["camera_viewmats"])).astype(np.float32, copy=True)
            intrinsics = _load_array(self._artifact_path(record["camera_intrinsics"])).astype(np.float32, copy=True)
            item["camera_viewmats"] = torch.from_numpy(viewmats)
            item["camera_intrinsics"] = torch.from_numpy(intrinsics)
        return item

    def _artifact_path(self, descriptor: Mapping[str, Any]) -> Path:
        relative = Path(descriptor["path"])
        if relative.is_absolute():
            raise ValueError("artifact paths must be relative to the manifest")
        resolved = (self.root / relative).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"artifact path escapes the manifest directory: {relative}")
        return resolved

    def _validate_artifacts(self, record: Mapping[str, Any]) -> None:
        arrays: dict[str, np.ndarray] = {}
        artifact_fields = ["latent", "prompt_embedding", "labels"]
        artifact_fields.extend(field for field in _CAMERA_ARTIFACTS if field in record)
        for field in artifact_fields:
            descriptor = record[field]
            path = self._artifact_path(descriptor)
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256_file(path) != descriptor["sha256"]:
                raise ValueError(f"SHA-256 mismatch for sample {record['id']!r} field {field!r}")
            array = _load_array(path, mmap_mode="r")
            if str(array.dtype) != descriptor["dtype"]:
                raise ValueError(f"dtype mismatch for sample {record['id']!r} field {field!r}")
            if list(array.shape) != descriptor["shape"]:
                raise ValueError(f"shape mismatch for sample {record['id']!r} field {field!r}")
            arrays[field] = array

        latent = arrays["latent"]
        prompt = arrays["prompt_embedding"]
        labels = arrays["labels"]
        if latent.dtype not in _FLOAT_DTYPES or latent.ndim != 4:
            raise ValueError("latent arrays must be float16/float32 with layout [C,F,H,W]")
        if prompt.dtype not in _FLOAT_DTYPES or prompt.ndim != 2:
            raise ValueError("prompt embeddings must be float16/float32 with layout [L,D]")
        if labels.dtype not in _LABEL_DTYPES or labels.ndim != 3:
            raise ValueError("label arrays must be integer with layout [F,H,W]")
        if int(latent.shape[1]) != int(labels.shape[0]):
            raise ValueError("latent and semantic-layout frame counts must match")
        if not bool(np.isfinite(latent).all()) or not bool(np.isfinite(prompt).all()):
            raise ValueError("latent and prompt arrays must contain only finite values")
        if "camera_viewmats" in arrays:
            viewmats = arrays["camera_viewmats"]
            intrinsics = arrays["camera_intrinsics"]
            if viewmats.dtype not in _FLOAT_DTYPES or viewmats.ndim != 3 or viewmats.shape[-2:] != (4, 4):
                raise ValueError("camera_viewmats must be float16/float32 with layout [F,4,4]")
            if intrinsics.dtype not in _FLOAT_DTYPES or intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
                raise ValueError("camera_intrinsics must be float16/float32 with layout [F,3,3]")
            if viewmats.shape[0] != intrinsics.shape[0]:
                raise ValueError("camera viewmat and intrinsics frame counts must match")
            if not bool(np.isfinite(viewmats).all()) or not bool(np.isfinite(intrinsics).all()):
                raise ValueError("camera arrays must contain only finite values")
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= NUM_LABELS):
            raise ValueError(f"semantic label ids must lie in [0, {NUM_LABELS - 1}]")
        if len(record["frame_progress"]) != int(labels.shape[0]):
            raise ValueError("frame_progress length must match the semantic-layout frame count")


def canonical_manifest_digest(
    header: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> str:
    """Return a stable SHA-256 over a validated, path-preserving manifest."""

    records = [dict(header)] + [dict(record) for record in sorted(samples, key=lambda value: value["id"])]
    canonical = "\n".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for record in records
    ) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {value}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid JSON on manifest line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            records.append(record)
    return records


def _validate_header(record: Mapping[str, Any]) -> dict[str, Any]:
    if not _REQUIRED_HEADER.issubset(record):
        raise ValueError(f"manifest header is missing fields: {sorted(_REQUIRED_HEADER - set(record))}")
    if record["type"] != "manifest" or record["schema"] != MANIFEST_SCHEMA:
        raise ValueError("unsupported semantic-layout training manifest schema")
    if record["version"] != MANIFEST_VERSION:
        raise ValueError("unsupported semantic-layout training manifest version")
    expected_layouts = {
        "prediction_type": "flow",
        "latent_layout": "C,F,H,W",
        "prompt_layout": "L,D",
        "label_layout": "F,H,W",
    }
    for field, expected in expected_layouts.items():
        if record[field] != expected:
            raise ValueError(f"manifest {field} must be {expected!r}")
    if list(record["label_names"]) != list(LABEL_NAMES):
        raise ValueError("manifest label_names do not match the canonical label schema")
    if not isinstance(record["num_train_timesteps"], int) or record["num_train_timesteps"] < 2:
        raise ValueError("num_train_timesteps must be an integer greater than one")
    shift = float(record["timestep_shift"])
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("timestep_shift must be finite and positive")
    return dict(record)


def _validate_sample_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not _REQUIRED_SAMPLE.issubset(record):
        raise ValueError(f"sample is missing fields: {sorted(_REQUIRED_SAMPLE - set(record))}")
    if record["type"] != "sample" or not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("each sample requires a non-empty string id")
    camera_fields = [field for field in _CAMERA_ARTIFACTS if field in record]
    if len(camera_fields) not in {0, len(_CAMERA_ARTIFACTS)}:
        raise ValueError("camera_viewmats and camera_intrinsics must be provided together")
    artifact_fields = ["latent", "prompt_embedding", "labels", *camera_fields]
    for field in artifact_fields:
        descriptor = record[field]
        if not isinstance(descriptor, Mapping) or set(descriptor) != _REQUIRED_ARTIFACT:
            raise ValueError(f"sample artifact {field!r} must contain exactly {sorted(_REQUIRED_ARTIFACT)}")
        if not isinstance(descriptor["path"], str) or not descriptor["path"]:
            raise ValueError("artifact path must be a non-empty string")
        if not isinstance(descriptor["sha256"], str) or not _SHA256_RE.fullmatch(descriptor["sha256"]):
            raise ValueError("artifact sha256 must be 64 lowercase hexadecimal characters")
        if not isinstance(descriptor["dtype"], str):
            raise ValueError("artifact dtype must be a string")
        if not isinstance(descriptor["shape"], list) or not descriptor["shape"] or any(
            not isinstance(value, int) or value < 1 for value in descriptor["shape"]
        ):
            raise ValueError("artifact shape must be a non-empty list of positive integers")
    progress = np.asarray(record["frame_progress"], dtype=np.float64)
    if progress.ndim != 1 or progress.size < 1 or not bool(np.isfinite(progress).all()):
        raise ValueError("frame_progress must be a non-empty finite one-dimensional list")
    if bool(((progress < 0) | (progress > 1)).any()) or bool((np.diff(progress) < 0).any()):
        raise ValueError("frame_progress must be nondecreasing within [0, 1]")
    normalized = dict(record)
    normalized["frame_progress"] = progress.tolist()
    return normalized


def _load_array(path: Path, *, mmap_mode: str | None = None) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    except (OSError, ValueError) as error:
        raise ValueError(f"failed to load safe NumPy array: {path}") from error
    if not isinstance(array, np.ndarray):
        raise ValueError(f"artifact must be a single .npy array, not an archive: {path}")
    return array


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
