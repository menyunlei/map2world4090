"""Standalone trainable semantic-layout adapter and strict checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


_CHECKPOINT_FORMAT = "dreamx.semantic_layout_adapter"
_CHECKPOINT_VERSION = 1
_CHECKPOINT_KEYS = {"format", "version", "architecture", "metadata", "state_dict"}


class SemanticLabelEncoder(nn.Module):
    """Encode discrete ``[B,F,H,W]`` layouts as a sequence of learned tokens."""

    def __init__(self, *, num_labels: int, embedding_dim: int, output_dim: int):
        super().__init__()
        if int(num_labels) < 1 or int(embedding_dim) < 1 or int(output_dim) < 1:
            raise ValueError("num_labels, embedding_dim, and output_dim must be positive")
        self.num_labels = int(num_labels)
        self.embedding_dim = int(embedding_dim)
        self.output_dim = int(output_dim)
        self.label_embedding = nn.Embedding(self.num_labels, self.embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim + 3, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(
        self,
        labels: torch.Tensor,
        *,
        frame_progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        labels = torch.as_tensor(labels, device=self.label_embedding.weight.device)
        if labels.ndim != 4:
            raise ValueError(f"labels must have shape [B,F,H,W], got {tuple(labels.shape)}")
        if labels.dtype.is_floating_point or labels.dtype == torch.bool:
            raise TypeError("labels must have an integer dtype")
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= self.num_labels):
            raise ValueError(f"labels must lie in [0, {self.num_labels - 1}]")
        batch, frames, height, width = [int(value) for value in labels.shape]
        if frame_progress is None:
            progress = torch.linspace(0, 1, frames, device=labels.device, dtype=torch.float32)[None].expand(batch, -1)
        else:
            progress = torch.as_tensor(frame_progress, device=labels.device, dtype=torch.float32)
            if progress.ndim == 1:
                progress = progress.unsqueeze(0)
            if tuple(progress.shape) == (1, frames) and batch > 1:
                progress = progress.expand(batch, -1)
            if tuple(progress.shape) != (batch, frames):
                raise ValueError(f"frame_progress must have shape [{batch},{frames}], got {tuple(progress.shape)}")
            if not bool(torch.isfinite(progress).all()):
                raise ValueError("frame_progress must contain only finite values")

        dtype = self.label_embedding.weight.dtype
        y = torch.linspace(-1, 1, height, device=labels.device, dtype=dtype)
        x = torch.linspace(-1, 1, width, device=labels.device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1)[None, None].expand(batch, frames, -1, -1, -1)
        progress_feature = progress.to(dtype=dtype)[:, :, None, None, None].expand(-1, -1, height, width, -1)
        embedded = self.label_embedding(labels.to(torch.long))
        features = torch.cat((embedded, coordinates, progress_feature), dim=-1)
        return self.projection(features).reshape(batch, frames * height * width, self.output_dim)


class ProjectedSemanticEncoder(nn.Module):
    """Encode image-projected semantic occupancy without losing overlaps."""

    def __init__(self, *, input_channels: int, embedding_dim: int, output_dim: int):
        super().__init__()
        if min(int(input_channels), int(embedding_dim), int(output_dim)) < 1:
            raise ValueError("input_channels, embedding_dim, and output_dim must be positive")
        self.input_channels = int(input_channels)
        self.embedding_dim = int(embedding_dim)
        self.output_dim = int(output_dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(self.input_channels, self.embedding_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.embedding_dim, self.embedding_dim, 3, padding=1),
        )
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim + 3, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        frame_progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = torch.as_tensor(features, device=self.spatial[0].weight.device)
        if features.ndim != 5:
            raise ValueError(f"features must have shape [B,F,C,H,W], got {tuple(features.shape)}")
        batch, frames, channels, height, width = [int(value) for value in features.shape]
        if channels != self.input_channels:
            raise ValueError(f"expected {self.input_channels} semantic channels, got {channels}")
        if not features.dtype.is_floating_point:
            raise TypeError("projected semantic features must use a floating dtype")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("projected semantic features must contain only finite values")
        if frame_progress is None:
            progress = torch.linspace(0, 1, frames, device=features.device, dtype=torch.float32)[None].expand(batch, -1)
        else:
            progress = torch.as_tensor(frame_progress, device=features.device, dtype=torch.float32)
            if progress.ndim == 1:
                progress = progress.unsqueeze(0)
            if tuple(progress.shape) == (1, frames) and batch > 1:
                progress = progress.expand(batch, -1)
            if tuple(progress.shape) != (batch, frames):
                raise ValueError(f"frame_progress must have shape [{batch},{frames}], got {tuple(progress.shape)}")
            if not bool(torch.isfinite(progress).all()):
                raise ValueError("frame_progress must contain only finite values")

        dtype = self.spatial[0].weight.dtype
        encoded = self.spatial(features.reshape(batch * frames, channels, height, width).to(dtype=dtype))
        encoded = encoded.reshape(batch, frames, self.embedding_dim, height, width).permute(0, 1, 3, 4, 2)
        y = torch.linspace(-1, 1, height, device=features.device, dtype=dtype)
        x = torch.linspace(-1, 1, width, device=features.device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1)[None, None].expand(batch, frames, -1, -1, -1)
        progress_feature = progress.to(dtype=dtype)[:, :, None, None, None].expand(-1, -1, height, width, -1)
        tokens = torch.cat((encoded, coordinates, progress_feature), dim=-1)
        return self.projection(tokens).reshape(batch, frames * height * width, self.output_dim)


class NonCausalSemanticCrossAttention(nn.Module):
    """Attend every video token to every layout token and return a residual.

    The output projection and scalar gate are zero-initialized, making a newly
    constructed branch exactly neutral when added to existing ``[B,L,C]``
    video tokens.
    """

    def __init__(self, *, token_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        if min(int(token_dim), int(context_dim), int(num_heads)) < 1:
            raise ValueError("token_dim, context_dim, and num_heads must be positive")
        if int(token_dim) % int(num_heads) != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = int(token_dim)
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)
        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.k_proj = nn.Linear(self.context_dim, self.token_dim)
        self.v_proj = nn.Linear(self.context_dim, self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.zero_init_output()

    def zero_init_output(self) -> None:
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.residual_gate)

    def forward(
        self,
        video_tokens: torch.Tensor,
        layout_tokens: torch.Tensor,
        *,
        layout_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if video_tokens.ndim != 3:
            raise ValueError(f"video_tokens must have shape [B,L,C], got {tuple(video_tokens.shape)}")
        if layout_tokens.ndim != 3:
            raise ValueError(f"layout_tokens must have shape [B,N,C], got {tuple(layout_tokens.shape)}")
        batch, token_count, token_dim = video_tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(f"expected video token width {self.token_dim}, got {token_dim}")
        if int(layout_tokens.shape[0]) != batch or int(layout_tokens.shape[-1]) != self.context_dim:
            raise ValueError("layout token batch/width is incompatible with video tokens")
        context_count = int(layout_tokens.shape[1])
        if context_count < 1:
            raise ValueError("layout_tokens must contain at least one token")

        valid = None
        if layout_mask is not None:
            valid = torch.as_tensor(layout_mask, device=video_tokens.device, dtype=torch.bool)
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if tuple(valid.shape) == (1, context_count) and batch > 1:
                valid = valid.expand(batch, -1)
            if tuple(valid.shape) != (batch, context_count):
                raise ValueError(f"layout_mask must have shape [{batch},{context_count}], got {tuple(valid.shape)}")
            if not bool(valid.any(dim=1).all()):
                raise ValueError("each batch item must keep at least one layout token")

        q = self.q_proj(self.query_norm(video_tokens)).view(batch, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        context = self.context_norm(layout_tokens)
        k = self.k_proj(context).view(batch, context_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch, context_count, self.num_heads, self.head_dim).transpose(1, 2)
        attention_mask = None if valid is None else valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attended = attended.transpose(1, 2).reshape(batch, token_count, self.token_dim)
        return self.out_proj(attended) * (1.0 + self.residual_gate)


class SpatiallyAlignedSemanticResidual(nn.Module):
    """Local map residual with a one-to-one video/map token correspondence.

    ``video_tokens[:, i]`` only sees ``layout_tokens[:, i]``.  This is the
    essential geometry constraint missing from global cross-attention: a map
    flip therefore changes the residual at the corresponding image location.
    The final projection is zero initialized so an untrained branch is an
    exact no-op while its upstream map encoder remains trainable.
    """

    def __init__(self, *, token_dim: int, context_dim: int, rank: int = 64):
        super().__init__()
        if min(int(token_dim), int(context_dim), int(rank)) < 1:
            raise ValueError("token_dim, context_dim, and rank must be positive")
        self.token_dim = int(token_dim)
        self.context_dim = int(context_dim)
        self.rank = int(rank)
        self.video_norm = nn.LayerNorm(self.token_dim)
        self.map_norm = nn.LayerNorm(self.context_dim)
        self.map_down = nn.Linear(self.context_dim, self.rank)
        self.video_down = nn.Linear(self.token_dim, self.rank)
        self.film = nn.Linear(self.rank, self.rank * 2)
        self.out_proj = nn.Linear(self.rank, self.token_dim)
        self.residual_gate = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def zero_init_output(self) -> None:
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.residual_gate)

    def forward(self, video_tokens: torch.Tensor, layout_tokens: torch.Tensor) -> torch.Tensor:
        if video_tokens.ndim != 3 or layout_tokens.ndim != 3:
            raise ValueError("video_tokens and layout_tokens must have shape [B,L,C]")
        if video_tokens.shape[0] != layout_tokens.shape[0] or video_tokens.shape[1] != layout_tokens.shape[1]:
            raise ValueError(
                "spatial semantic layout requires one map token per video token: "
                f"video={tuple(video_tokens.shape[:2])}, map={tuple(layout_tokens.shape[:2])}"
            )
        if video_tokens.shape[-1] != self.token_dim or layout_tokens.shape[-1] != self.context_dim:
            raise ValueError("spatial semantic token widths do not match the adapter")
        v = self.video_down(self.video_norm(video_tokens))
        m = self.map_down(self.map_norm(layout_tokens))
        scale, shift = self.film(m).chunk(2, dim=-1)
        hidden = v * (1.0 + torch.tanh(scale)) + shift
        hidden = F.silu(hidden)
        return self.out_proj(hidden) * (1.0 + self.residual_gate)


class SemanticLayoutAdapter(nn.Module):
    """Compose label encoding with non-causal video-to-layout attention."""

    def __init__(
        self,
        *,
        num_labels: int,
        token_dim: int,
        label_embedding_dim: int,
        context_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.token_dim = int(token_dim)
        self.label_embedding_dim = int(label_embedding_dim)
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.label_encoder = SemanticLabelEncoder(
            num_labels=self.num_labels,
            embedding_dim=self.label_embedding_dim,
            output_dim=self.context_dim,
        )
        self.cross_attention = NonCausalSemanticCrossAttention(
            token_dim=self.token_dim,
            context_dim=self.context_dim,
            num_heads=self.num_heads,
        )

    def architecture_metadata(self) -> dict[str, int | str]:
        return {
            "module": f"{type(self).__module__}.{type(self).__qualname__}",
            "num_labels": self.num_labels,
            "token_dim": self.token_dim,
            "label_embedding_dim": self.label_embedding_dim,
            "context_dim": self.context_dim,
            "num_heads": self.num_heads,
        }

    def forward(
        self,
        video_tokens: torch.Tensor,
        labels: torch.Tensor,
        *,
        frame_progress: torch.Tensor | None = None,
        layout_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout_tokens = self.label_encoder(labels, frame_progress=frame_progress)
        return self.cross_attention(video_tokens, layout_tokens, layout_mask=layout_mask)


def save_semantic_layout_adapter_checkpoint(
    path: str | os.PathLike[str],
    adapter: SemanticLayoutAdapter,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save adapter-only weights with validated JSON metadata."""

    if not isinstance(adapter, SemanticLayoutAdapter):
        raise TypeError("adapter must be a SemanticLayoutAdapter")
    canonical_metadata = canonicalize_semantic_layout_metadata(metadata or {})
    payload = {
        "format": _CHECKPOINT_FORMAT,
        "version": _CHECKPOINT_VERSION,
        "architecture": adapter.architecture_metadata(),
        "metadata": canonical_metadata,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in adapter.state_dict().items()},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_semantic_layout_adapter_checkpoint(
    path: str | os.PathLike[str],
    adapter: SemanticLayoutAdapter,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate and load adapter-only weights, or raise without loading."""

    if not isinstance(adapter, SemanticLayoutAdapter):
        raise TypeError("adapter must be a SemanticLayoutAdapter")
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError(f"invalid checkpoint fields; expected exactly {sorted(_CHECKPOINT_KEYS)}")
    if payload["format"] != _CHECKPOINT_FORMAT or payload["version"] != _CHECKPOINT_VERSION:
        raise ValueError("unsupported semantic layout adapter checkpoint format/version")
    if payload["architecture"] != adapter.architecture_metadata():
        raise ValueError("checkpoint architecture metadata does not match the adapter")
    saved_metadata = canonicalize_semantic_layout_metadata(payload["metadata"])
    if expected_metadata is not None:
        required_metadata = canonicalize_semantic_layout_metadata(expected_metadata)
        if saved_metadata != required_metadata:
            raise ValueError("checkpoint metadata does not exactly match expected_metadata")

    state = payload["state_dict"]
    expected_state = adapter.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected_state):
        raise ValueError("checkpoint state_dict keys do not exactly match the adapter")
    for name, expected in expected_state.items():
        value = state[name]
        if not torch.is_tensor(value):
            raise ValueError(f"checkpoint value {name!r} is not a tensor")
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(
                f"checkpoint tensor {name!r} mismatch: expected {tuple(expected.shape)} {expected.dtype}, "
                f"got {tuple(value.shape)} {value.dtype}"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"checkpoint tensor {name!r} contains non-finite values")
    adapter.load_state_dict(state, strict=True)
    return saved_metadata


def canonicalize_semantic_layout_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    try:
        encoded = json.dumps(dict(metadata), sort_keys=True, allow_nan=False, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain only finite JSON values") from error
    if not isinstance(decoded, dict):
        raise ValueError("metadata must encode a JSON object")
    return decoded
