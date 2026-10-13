"""Content identities for a transformer config and its weight files."""

from __future__ import annotations

import hashlib
from pathlib import Path


_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}


def compute_checkpoint_identity(checkpoint_dir: str | Path) -> dict[str, object]:
    """Hash model config/index files and every weight file deterministically."""

    root = Path(checkpoint_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = []
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    files.append(config)
    files.extend(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in _WEIGHT_SUFFIXES
            or path.name.endswith(".index.json")
        )
    )
    files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    weight_files = [path for path in files if path.suffix.lower() in _WEIGHT_SUFFIXES]
    if not weight_files:
        raise ValueError(f"checkpoint contains no recognized weight files: {root}")

    file_hashes = {}
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest = _sha256_file(path)
        file_hashes[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "sha256": aggregate.hexdigest(),
        "files": file_hashes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
