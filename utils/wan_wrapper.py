import gc
import types
from typing import List, Optional
import torch
import os

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.t5 import umt5_xxl
from wan.modules.vae_2_2 import _video_vae as _video_vae_2_2
from wan.modules.attention import (
    assert_sage_attention_available,
    attention_backend_status,
    set_attention_backend,
)
from wan.modules.fp8_ffn import assert_fp8_ffn_available, fp8_ffn_status
from wan.modules.map_adapter import MAP_ADAPTER_FEATURE_DIM


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.2-TI2V-5B-Camera", model_root_path: str = "",
                 text_encoder_path: str = None, tokenizer_path: str = None):
        super().__init__()
        if text_encoder_path is None:
            text_encoder_path = os.path.join(model_root_path, f"wan_models/{model_name}/models_t5_umt5-xxl-enc-bf16.pth")
        if tokenizer_path is None:
            tokenizer_path = os.path.join(model_root_path, f"wan_models/{model_name}/google/umt5-xxl/")

        self.text_encoder = umt5_xxl(
            encoder_only=True, return_tokenizer=False,
            dtype=torch.bfloat16, device=torch.device('cpu')
        ).eval().requires_grad_(False)
        _sd = torch.load(text_encoder_path, map_location='cpu', weights_only=False)
        self.text_encoder.load_state_dict(_sd, assign=True)
        del _sd
        gc.collect()
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace')

    @property
    def device(self):
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        return {"prompt_embeds": context}


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_root_path="", vae_path: str = None):
        super().__init__()
        mean = [
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
            -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
            -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
            0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
        ]
        std = [
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        if vae_path is None:
            vae_path = os.path.join(model_root_path, "wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
        self.model = _video_vae_2_2(
            pretrained_path=vae_path,
            z_dim=48, temperal_downsample=[False, True, True]
        ).eval().requires_grad_(False)
        self._tiled_decode_caches = None
        self._tiled_decode_cache_key = None

    def clear_tiled_decode_cache(self):
        self._tiled_decode_caches = None
        self._tiled_decode_cache_key = None

    def clear_cache(self):
        self.clear_tiled_decode_cache()
        self.model.clear_cache()

    def enable_channels_last_3d(self):
        """Convert decoder conv weights to channels_last_3d.

        Measured on RTX 4090 (704x1280, 3-latent window, use_cache=True):
        2.345 -> 2.220 s/window (-5.3%); the decode is compute-bound so this is
        the only software lever found (cudnn.benchmark and torch.compile both
        gave 0% and bit-identical output). uint8 parity vs NCHW: max_diff=1 on
        4.7e-06 of voxels — the same tolerance bar as the vectorized color
        correction. Enable on EVERY decode path in a process (serial streaming
        and overlap VAEs alike) so cross-path comparisons stay meaningful.
        """
        converted = 0
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5:
                m.weight.data = m.weight.data.contiguous(
                    memory_format=torch.channels_last_3d)
                converted += 1
        return converted

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(
        self,
        latent: torch.Tensor,
        use_cache: bool = False,
        tile_height: int = 0,
        tile_overlap: int = 0,
        tile_blend_mode: str = "linear",
    ) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        if int(tile_height or 0) > 0 and int(tile_height) < int(zs.shape[-2]):
            return self._decode_to_pixel_tiled(
                zs,
                scale,
                use_cache=use_cache,
                tile_height=int(tile_height),
                tile_overlap=int(tile_overlap or 0),
                tile_blend_mode=tile_blend_mode,
            )
        if use_cache and self._tiled_decode_caches is not None:
            self.clear_cache()
        if not use_cache:
            self.clear_tiled_decode_cache()
        decode_fn = self.model.cached_decode if use_cache else self.model.decode
        output = []
        with torch.autocast(device_type=device.type, dtype=dtype):
            for u in zs:
                output.append(decode_fn(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)

    def _decode_to_pixel_tiled(
        self,
        zs: torch.Tensor,
        scale,
        use_cache: bool,
        tile_height: int,
        tile_overlap: int,
        tile_blend_mode: str,
    ) -> torch.Tensor:
        if tile_height < 1:
            raise ValueError("tile_height must be >= 1")
        if tile_overlap < 0:
            raise ValueError("tile_overlap must be >= 0")
        if tile_overlap >= tile_height:
            raise ValueError("tile_overlap must be smaller than tile_height")
        tile_blend_mode = str(tile_blend_mode or "linear").strip().lower()
        if tile_blend_mode not in {"crop", "linear"}:
            raise ValueError("tile_blend_mode must be 'crop' or 'linear'")

        latent_height = int(zs.shape[-2])
        tile_ranges = self._height_tile_ranges(latent_height, tile_height, tile_overlap)
        if use_cache:
            cache_key = (
                latent_height,
                int(zs.shape[-1]),
                tile_height,
                tile_overlap,
                tile_blend_mode,
                len(tile_ranges),
            )
            if self._tiled_decode_cache_key != cache_key or self._tiled_decode_caches is None:
                self.model.clear_cache()
                self._tiled_decode_caches = [
                    [None] * int(self.model._conv_num) for _ in tile_ranges
                ]
                self._tiled_decode_cache_key = cache_key
        else:
            self.clear_tiled_decode_cache()

        output = []
        with torch.autocast(device_type=zs.device.type, dtype=zs.dtype):
            for batch_index, u in enumerate(zs):
                output.append(
                    self._decode_single_tiled(
                        u.unsqueeze(0),
                        scale,
                        tile_ranges,
                        use_cache=use_cache,
                        tile_blend_mode=tile_blend_mode,
                        batch_index=batch_index,
                    ).squeeze(0)
                )
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)

    def _decode_single_tiled(
        self,
        z: torch.Tensor,
        scale,
        tile_ranges,
        use_cache: bool,
        tile_blend_mode: str,
        batch_index: int,
    ) -> torch.Tensor:
        latent_height = int(z.shape[-2])
        original_feat_map = getattr(self.model, "_feat_map", None)
        original_conv_idx = getattr(self.model, "_conv_idx", None)
        accum = None
        weight = None
        output = None

        try:
            for tile_index, (latent_start, latent_end) in enumerate(tile_ranges):
                z_tile = z[:, :, :, latent_start:latent_end, :]
                if use_cache:
                    self.model._feat_map = self._tiled_decode_caches[tile_index]
                    self.model._conv_idx = [0]
                    decoded = self.model.cached_decode(z_tile, scale).float().clamp_(-1, 1)
                    self._tiled_decode_caches[tile_index] = self.model._feat_map
                else:
                    decoded = self.model.decode(z_tile, scale).float().clamp_(-1, 1)

                tile_pixel_height = int(decoded.shape[-2])
                scale_h = tile_pixel_height // max(1, latent_end - latent_start)
                if tile_pixel_height != (latent_end - latent_start) * scale_h:
                    raise RuntimeError("Tiled VAE decode produced a non-integral height scale")
                pixel_start = latent_start * scale_h
                pixel_end = pixel_start + tile_pixel_height
                full_pixel_height = latent_height * scale_h

                if tile_blend_mode == "crop":
                    if output is None:
                        output = decoded.new_zeros(*decoded.shape[:-2], full_pixel_height, decoded.shape[-1])
                    own_start, own_end = self._owned_pixel_range(
                        tile_ranges, tile_index, latent_height, scale_h)
                    local_start = own_start - pixel_start
                    local_end = own_end - pixel_start
                    output[..., own_start:own_end, :] = decoded[..., local_start:local_end, :]
                else:
                    if accum is None:
                        accum = decoded.float().new_zeros(
                            *decoded.shape[:-2], full_pixel_height, decoded.shape[-1])
                        weight = decoded.float().new_zeros(1, 1, 1, full_pixel_height, 1)
                    tile_weight = self._tile_blend_weights(
                        tile_ranges, tile_index, tile_pixel_height, scale_h,
                        device=decoded.device, dtype=decoded.dtype)
                    accum[..., pixel_start:pixel_end, :] += decoded.float() * tile_weight
                    weight[..., pixel_start:pixel_end, :] += tile_weight

                del decoded, z_tile
        finally:
            if original_feat_map is not None:
                self.model._feat_map = original_feat_map
            if original_conv_idx is not None:
                self.model._conv_idx = original_conv_idx

        if tile_blend_mode == "crop":
            return output
        return (accum / weight.clamp_min(1.0e-6)).to(dtype=accum.dtype)

    @staticmethod
    def _height_tile_ranges(height: int, tile_height: int, tile_overlap: int):
        if tile_height >= height:
            return [(0, height)]
        stride = tile_height - tile_overlap
        ranges = []
        start = 0
        while True:
            end = min(start + tile_height, height)
            ranges.append((start, end))
            if end >= height:
                break
            next_start = min(start + stride, height - tile_height)
            if next_start <= start:
                break
            start = next_start
        return ranges

    @staticmethod
    def _owned_pixel_range(tile_ranges, tile_index: int, latent_height: int, scale_h: int):
        latent_start, latent_end = tile_ranges[tile_index]
        if tile_index == 0:
            owned_start = 0
        else:
            prev_end = tile_ranges[tile_index - 1][1]
            owned_start = int(round(((prev_end + latent_start) * scale_h) / 2.0))
        if tile_index == len(tile_ranges) - 1:
            owned_end = latent_height * scale_h
        else:
            next_start = tile_ranges[tile_index + 1][0]
            owned_end = int(round(((latent_end + next_start) * scale_h) / 2.0))
        return owned_start, owned_end

    @staticmethod
    def _tile_blend_weights(tile_ranges, tile_index: int, tile_pixel_height: int, scale_h: int, device, dtype):
        latent_start, latent_end = tile_ranges[tile_index]
        weights = torch.ones(tile_pixel_height, device=device, dtype=dtype)
        if tile_index > 0:
            prev_end = tile_ranges[tile_index - 1][1]
            overlap = max(0, prev_end - latent_start) * scale_h
            if overlap > 1:
                weights[:overlap] = torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)
            elif overlap == 1:
                weights[:overlap] = 0.5
        if tile_index < len(tile_ranges) - 1:
            next_start = tile_ranges[tile_index + 1][0]
            overlap = max(0, latent_end - next_start) * scale_h
            if overlap > 1:
                weights[-overlap:] = torch.linspace(1.0, 0.0, overlap, device=device, dtype=dtype)
            elif overlap == 1:
                weights[-overlap:] = 0.5
        return weights.view(1, 1, 1, tile_pixel_height, 1)


class WanDiffusionCameraWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.2-TI2V-5B-Camera",
            timestep_shift=5.0,
            is_causal=True,
            local_attn_size=12,
            sink_size=3,
            model_root_path="",
            model_config_path: str = None,
            **kwargs,
    ):
        super().__init__()

        from wan.modules.causal_camera_model_2_2_prope_infinity import CausalWanModel

        num_output_frames = kwargs.get('num_output_frames', 21)
        eprope = kwargs.get('eprope', False)
        attn_compress = 4 if eprope else 1
        attention_backend = kwargs.get("attention_backend", "auto")
        sage_smooth_k = kwargs.get("sage_smooth_k", True)
        sage_strict = kwargs.get("sage_strict", False)
        ffn_backend = str(kwargs.get("ffn_backend", "bf16")).strip().lower()
        set_attention_backend(
            attention_backend,
            sage_smooth_k=sage_smooth_k,
            strict=sage_strict,
        )
        if str(attention_backend).strip().lower() in ("sage", "sageattention", "sage-attention"):
            assert_sage_attention_available(torch.cuda.current_device() if torch.cuda.is_available() else None)
        if ffn_backend == "fp8":
            assert_fp8_ffn_available(torch.cuda.current_device() if torch.cuda.is_available() else None)

        if model_config_path is None:
            model_config_path = os.path.join(model_root_path, f"wan_models/{model_name}/config.json")
        self.model = CausalWanModel.from_config(
            model_config_path,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            attn_compress=attn_compress,
            ffn_backend=ffn_backend,
            ffn_fp8_dtype=kwargs.get("ffn_fp8_dtype", "e4m3fn"),
            ffn_fp8_output_dtype=kwargs.get("ffn_fp8_output_dtype", "bfloat16"),
            ffn_fp8_use_fast_accum=kwargs.get("ffn_fp8_use_fast_accum", False),
            ffn_fp8_detect_anomaly=kwargs.get("ffn_fp8_detect_anomaly", True),
            map_sparse_ffn=kwargs.get("map_sparse_ffn", False),
            map_sparse_attention=kwargs.get("map_sparse_attention", False),
            map_sparse_attention_camera_motion_threshold=kwargs.get(
                "map_sparse_attention_camera_motion_threshold", float("inf")),
            emapctrl_enabled=kwargs.get("emapctrl_enabled", False),
            emapctrl_map_channels=kwargs.get("emapctrl_map_channels", 9),
            emapctrl_attn_compress=kwargs.get("emapctrl_attn_compress", 4),
            emapctrl_downsample=kwargs.get("emapctrl_downsample", 2),
            map_adapter_enabled=kwargs.get("map_adapter_enabled", False),
            map_adapter_feature_dim=kwargs.get("map_adapter_feature_dim", MAP_ADAPTER_FEATURE_DIM),
            map_adapter_hidden_dim=kwargs.get("map_adapter_hidden_dim", 512),
            map_adapter_progress_sigma=kwargs.get("map_adapter_progress_sigma", 0.12),
        )
        self.model.eval()
        print(f"Attention backend: {attention_backend_status()}")
        print(f"FFN backend: {ffn_backend}; FP8 status: {fp8_ffn_status()}")
        print(f"Map sparse FFN: {bool(getattr(self.model, 'map_sparse_ffn', False))}")
        print(f"Map sparse attention: {bool(getattr(self.model, 'map_sparse_attention', False))}")
        print(f"E-MapCtrl: {bool(getattr(self.model, 'emapctrl_enabled', False))}")
        print(f"Map adapter: {bool(getattr(self.model, 'map_adapter_enabled', False))}")

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 880 * num_output_frames
        self._bind_scheduler_methods()

    def _bind_scheduler_methods(self):
        self.scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, self.scheduler)
        self.scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, self.scheduler)
        self.scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, self.scheduler)

    def get_scheduler(self) -> SchedulerInterface:
        return self.scheduler

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps])
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (xt - sigma_t * flow_pred).to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        y_camera,
        timestep: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        cache_update_policy: str = "commit_detached",
        runtime_profiler=None,
        profile_chunk_index=None,
        profile_step_index=None,
        profile_phase="denoising",
        map_sparse_ffn_plan=None,
        map_sparse_attention_plan=None,
        emapctrl_condition=None,
        map_adapter_condition=None,
        ray_grounded_condition=None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        skip_length = noisy_image_or_video.shape[-1] * noisy_image_or_video.shape[-2] // 4
        original_timestep = timestep[:, ::skip_length]

        y_camera_input = y_camera if (y_camera is None or isinstance(y_camera, dict)) else y_camera.permute(0, 2, 1, 3, 4)

        model_kwargs = {}
        if ray_grounded_condition is not None:
            model_kwargs["ray_grounded_condition"] = ray_grounded_condition

        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep,
            context=prompt_embeds,
            y=y.permute(0, 2, 1, 3, 4) if y is not None else None,
            y_camera=y_camera_input,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            cache_update_policy=cache_update_policy,
            runtime_profiler=runtime_profiler,
            profile_chunk_index=profile_chunk_index,
            profile_step_index=profile_step_index,
            profile_phase=profile_phase,
            map_sparse_ffn_plan=map_sparse_ffn_plan,
            map_sparse_attention_plan=map_sparse_attention_plan,
            emapctrl_condition=emapctrl_condition,
            map_adapter_condition=map_adapter_condition,
            **model_kwargs,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            original_timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def freeze_base_for_emapctrl_training(self):
        if hasattr(self.model, "freeze_base_for_emapctrl_training"):
            self.model.freeze_base_for_emapctrl_training()
        return self

    def freeze_base_for_map_adapter_training(self):
        if hasattr(self.model, "freeze_base_for_map_adapter_training"):
            self.model.freeze_base_for_map_adapter_training()
        return self

    def freeze_base_for_ray_grounded_training(self):
        if hasattr(self.model, "freeze_base_for_ray_grounded_training"):
            self.model.freeze_base_for_ray_grounded_training()
        return self
