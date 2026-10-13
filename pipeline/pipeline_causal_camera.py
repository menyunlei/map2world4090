from typing import List, Optional
import gc
import time
import torch
import tqdm

from utils.wan_wrapper import WanDiffusionCameraWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import cpu, gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from utils.residual_reuse import normalize_disabled_chunks, residual_relative_l1_change, residual_reuse_decision, chunk_difficulty_classes
from wan.modules.map_sparse_attention import build_map_sparse_attention_plan_from_router_result
from wan.modules.map_sparse_ffn import build_map_sparse_ffn_plan_from_router_result
from wan.modules.emapctrl_branch import (
    build_emapctrl_condition_from_router_result,
    slice_emapctrl_condition,
)
from wan.modules.map_adapter import slice_map_adapter_condition


class CausalCameraInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None,
                 model_config_path=None, text_encoder_path=None, tokenizer_path=None, vae_path=None,
                 **kwargs):
        super().__init__()
        model_kwargs = getattr(args, "model_kwargs", {})
        model_name = model_kwargs["model_name"]
        model_root_path = model_kwargs["model_root_path"]

        self.generator = WanDiffusionCameraWrapper(
            **model_kwargs, is_causal=True, model_config_path=model_config_path,
            **kwargs) if generator is None else generator
        self.text_encoder = WanTextEncoder(
            model_name=model_name, model_root_path=model_root_path,
            text_encoder_path=text_encoder_path, tokenizer_path=tokenizer_path,
        ) if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper(
            model_root_path=model_root_path, vae_path=vae_path,
        ) if vae is None else vae

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        self.local_attn_size = self.generator.model.local_attn_size
        self.sequential_offload = False
        self.act_probe = None  # optional tools/probe_act_stats.py probe
        self.latent_seam_blend_alpha = float(getattr(args, "latent_seam_blend_alpha", 0.0))
        self.latent_seam_blend_frames = int(getattr(args, "latent_seam_blend_frames", 1))
        self.latent_seam_velocity_scale = float(getattr(args, "latent_seam_velocity_scale", 0.5))

        print(f"KV inference with {self.num_frame_per_block} frames per block")
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.map_adaptive_denoising_log = []
        self.last_inference_wall_profile = {}
        self.residual_reuse_enabled = False
        self.residual_reuse_threshold = 0.0
        self.residual_reuse_disabled_chunks = set()
        self.residual_reuse_discontinuity_events = []
        self.residual_reuse_log = []

    def set_map_adaptive_denoising_schedule(
        self,
        denoising_step_list,
        *,
        chunk_id=None,
        hardness_score=None,
        hardness_class=None,
        role=None,
    ):
        """Set the active denoising timestep list for a map-scheduled chunk."""
        if denoising_step_list is None:
            raise ValueError("denoising_step_list is required")
        raw_steps = [int(value) for value in denoising_step_list]
        if not raw_steps:
            raise ValueError("denoising_step_list must not be empty")
        if min(raw_steps) < 1 or max(raw_steps) > 1000:
            raise ValueError("denoising steps must be in [1, 1000]")
        self.args.denoising_step_list = raw_steps
        step_tensor = torch.tensor(raw_steps, dtype=torch.long)
        if self.args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            step_tensor = timesteps[1000 - step_tensor]
        self.denoising_step_list = step_tensor
        effective_steps = [
            float(value) if step_tensor.dtype.is_floating_point else int(value)
            for value in step_tensor.cpu().tolist()
        ]
        event = {
            "chunk_id": chunk_id,
            "hardness_score": hardness_score,
            "hardness_class": hardness_class,
            "role": role,
            "step_count": len(raw_steps),
            "raw_denoising_step_list": raw_steps,
            "effective_denoising_step_list": effective_steps,
        }
        self.map_adaptive_denoising_log.append(event)
        print(
            "[map_adaptive_denoising] "
            f"chunk={chunk_id} role={role} hardness={hardness_class} "
            f"step_count={len(raw_steps)} steps={effective_steps}"
        )
        return event

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        y: torch.Tensor,
        y_camera: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = True,
        low_memory: bool = False,
        decode_video: bool = True,
        runtime_profiler=None,
        residual_reuse_enabled: Optional[bool] = None,
        residual_reuse_threshold: Optional[float] = None,
        residual_reuse_disabled_chunks=None,
        residual_reuse_discontinuity_events=None,
        residual_reuse_record_steps: bool = False,
        vae_tile_height: int = 0,
        vae_tile_overlap: int = 0,
        vae_tile_blend_mode: str = "linear",
        map_sparse_ffn_router_result=None,
        emapctrl_condition=None,
        map_adapter_condition=None,
        per_chunk_step_lists=None,
        per_chunk_difficulty_classes=None,
        per_chunk_captions: Optional[List[str]] = None,
        per_chunk_caption_extrapolation: float = 0.0,
        ar_boundary_renoise_strength: float = 0.0,
        per_chunk_anchor_latents: Optional[torch.Tensor] = None,
        per_chunk_anchor_strength: float = 1.0,
        per_chunk_anchor_mode: str = "hard",
        per_chunk_anchor_soft_timestep: int = 250,
        chunk_callback=None,
    ) -> torch.Tensor:
        inference_wall_start = time.perf_counter()
        self.last_inference_wall_profile = {}
        adaptive_chunk_steps_enabled = bool(per_chunk_step_lists)
        native_denoising_step_list = self.denoising_step_list
        native_raw_step_list = list(self.args.denoising_step_list)
        reuse_enabled = bool(self.residual_reuse_enabled if residual_reuse_enabled is None else residual_reuse_enabled)
        reuse_threshold = float(
            self.residual_reuse_threshold if residual_reuse_threshold is None else residual_reuse_threshold
        )
        reuse_disabled_chunks = normalize_disabled_chunks(
            self.residual_reuse_disabled_chunks if residual_reuse_disabled_chunks is None else residual_reuse_disabled_chunks
        )
        self.residual_reuse_discontinuity_events = list(
            self.residual_reuse_discontinuity_events
            if residual_reuse_discontinuity_events is None
            else residual_reuse_discontinuity_events
        )
        self.residual_reuse_log = []
        reuse_record_steps = bool(residual_reuse_record_steps or reuse_enabled)
        reuse_collect_state = bool(reuse_enabled or reuse_record_steps)
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        conditioning_wall_start = time.perf_counter()
        if per_chunk_captions is not None:
            per_chunk_captions = [str(caption) for caption in per_chunk_captions]
            if len(per_chunk_captions) < num_blocks:
                raise ValueError(
                    f"per_chunk_captions needs at least {num_blocks} captions "
                    f"for {num_blocks} blocks, got {len(per_chunk_captions)}")
        per_chunk_prompt_embeds = None
        caption_extrapolation = float(per_chunk_caption_extrapolation)
        if per_chunk_captions is not None and caption_extrapolation != 0.0:
            # A2 extrapolation: keep the global caption as the anchor and push
            # each chunk embedding s times along (chunk - global):
            #     emb'_i = emb_g + s * (emb_i - emb_g)
            # s=1 recovers plain per-chunk captions; s>1 amplifies the
            # chunk-specific direction while staying anchored to the global
            # scene. The global caption is encoded in the same batched T5 pass
            # (row 0) so anchor and chunks share one tokenizer/encoder state.
            encode_prompts = list(text_prompts) + list(per_chunk_captions)
        else:
            encode_prompts = per_chunk_captions if per_chunk_captions is not None else text_prompts
        if self.sequential_offload:
            if runtime_profiler is not None:
                _, transfer_sec, transfer_peak = runtime_profiler.measure(
                    lambda: self.text_encoder.to(gpu), reset_peak=True)
                runtime_profiler.add_pending_transfer_event("text_encoder_cpu_to_gpu", transfer_sec, transfer_peak)
            else:
                self.text_encoder.to(gpu)
        if runtime_profiler is not None:
            conditional_dict, text_encode_sec, text_encode_peak = runtime_profiler.measure(
                lambda: self.text_encoder(text_prompts=encode_prompts), reset_peak=True)
            runtime_profiler.set_shared_stage("text_encoding", text_encode_sec, text_encode_peak)
        else:
            conditional_dict = self.text_encoder(text_prompts=encode_prompts)
        if per_chunk_captions is not None:
            # One batched T5 pass over all chunk captions (same wall cost as a
            # single caption). Slice 0 conditions the init pass / chunk 0; the
            # chunk loop swaps in slice i and invalidates the cross-attn cache
            # whenever the caption changes between consecutive chunks.
            per_chunk_prompt_embeds = conditional_dict["prompt_embeds"]
            if caption_extrapolation != 0.0:
                # Row 0 of the batched pass is the global caption (the anchor);
                # rows 1.. are the chunk captions. Extrapolate each chunk away
                # from the anchor and re-anchor chunk 0 on the extrapolated
                # embedding so the init pass and chunk loop agree.
                if per_chunk_prompt_embeds.shape[0] < num_blocks + 1:
                    raise ValueError(
                        f"expected at least {num_blocks + 1} prompt-embed rows "
                        f"(global + {num_blocks} chunks), got "
                        f"{per_chunk_prompt_embeds.shape[0]}")
                emb_g = per_chunk_prompt_embeds[0:1]
                emb_c = per_chunk_prompt_embeds[1:]
                per_chunk_prompt_embeds = emb_g + caption_extrapolation * (emb_c - emb_g)
                print(f"[per_chunk_captions] A2 extrapolation s={caption_extrapolation}: "
                      f"chunk embeds pushed s x along (chunk - global); "
                      f"global anchor {int(emb_g.shape[1])} tokens")
            if per_chunk_prompt_embeds.shape[0] < num_blocks:
                raise ValueError(
                    f"expected at least {num_blocks} prompt-embed rows, got "
                    f"{per_chunk_prompt_embeds.shape[0]}")
            conditional_dict = {"prompt_embeds": per_chunk_prompt_embeds[0:1].contiguous()}
        if self.sequential_offload:
            if runtime_profiler is not None:
                _, transfer_sec, transfer_peak = runtime_profiler.measure(
                    lambda: self.text_encoder.to(cpu), reset_peak=True)
                runtime_profiler.add_pending_transfer_event("text_encoder_gpu_to_cpu", transfer_sec, transfer_peak)
            else:
                self.text_encoder.to(cpu)
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[1/3 text encoder] done. Free VRAM: {get_cuda_free_memory_gb(gpu):.1f} GB")

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
        conditioning_wall_sec = time.perf_counter() - conditioning_wall_start

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        if self.sequential_offload:
            if runtime_profiler is not None:
                _, transfer_sec, transfer_peak = runtime_profiler.measure(
                    lambda: self.generator.to(gpu), reset_peak=True)
                runtime_profiler.add_pending_transfer_event("generator_cpu_to_gpu", transfer_sec, transfer_peak)
            else:
                self.generator.to(gpu)
            print(f"[2/3 diffusion] generator loaded. Free VRAM: {get_cuda_free_memory_gb(gpu):.1f} GB")

        # Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size, noise.dtype, noise.device, num_frames)
            self._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        # Cache initial latent frames
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                initial_emapctrl_condition = self._emapctrl_condition_for_range(
                    emapctrl_condition=emapctrl_condition,
                    router_result=map_sparse_ffn_router_result,
                    latent_start_frame=0,
                    latent_end_frame=1,
                    device=noise.device,
                    dtype=noise.dtype,
                )
                initial_map_adapter_condition = self._map_adapter_condition_for_range(
                    map_adapter_condition=map_adapter_condition,
                    latent_start_frame=0,
                    latent_end_frame=1,
                    device=noise.device,
                    dtype=noise.dtype,
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    y=y[:, :1] if y is not None else None,
                    y_camera=y_camera if isinstance(y_camera, dict) else y_camera[:, :1],
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    emapctrl_condition=initial_emapctrl_condition,
                    map_adapter_condition=initial_map_adapter_condition,
                )
                current_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                y_latents = y[:, current_start_frame:current_start_frame + self.num_frame_per_block] if y is not None else None
                y_camera_latents = y_camera if isinstance(y_camera, dict) else y_camera[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                initial_emapctrl_condition = self._emapctrl_condition_for_range(
                    emapctrl_condition=emapctrl_condition,
                    router_result=map_sparse_ffn_router_result,
                    latent_start_frame=current_start_frame,
                    latent_end_frame=current_start_frame + self.num_frame_per_block,
                    device=noise.device,
                    dtype=noise.dtype,
                )
                initial_map_adapter_condition = self._map_adapter_condition_for_range(
                    map_adapter_condition=map_adapter_condition,
                    latent_start_frame=current_start_frame,
                    latent_end_frame=current_start_frame + self.num_frame_per_block,
                    device=noise.device,
                    dtype=noise.dtype,
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    y=y_latents, y_camera=y_camera_latents,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    emapctrl_condition=initial_emapctrl_condition,
                    map_adapter_condition=initial_map_adapter_condition,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        first_frame_mask = torch.zeros_like(noise)
        first_frame_mask[:, 1:] = 1

        diffusion_loop_wall_start = time.perf_counter()
        latent_seam_blend_event_count = 0
        # A1 SDEdit boundary re-noising: seed chunks i>0 from the previous
        # chunk's final latents re-noised to a chosen timestep t_seed, blended
        # with the chunk's pure noise:
        #     latents = (1-strength)*pure_noise + strength*add_noise(prev_final, eps, t_seed)
        # strength=0 (default) keeps the exact pure-noise behavior.
        # t_seed=0 selects the chunk's first denoise timestep (full-noise
        # start: prev structure is masked entirely at warped t~63; what the
        # blend then does is damp the chunk-start noise magnitude).
        # t_seed>0 (raw units, pre-warp) is TRUE SDEdit: re-noise to a LOWER
        # timestep so prev structure survives, and start this chunk's step
        # list at that timestep instead of the top (shorter trajectory =
        # cheaper and keeps the seeded content).
        ar_boundary_renoise = float(ar_boundary_renoise_strength)
        ar_boundary_renoise_t = int(getattr(self.args, "ar_boundary_renoise_timestep", 0))
        renoise_rng_state = None
        if ar_boundary_renoise > 0.0:
            renoise_rng_state = torch.get_rng_state()
        prev_chunk_latents = None
        # Per-chunk anchor re-conditioning: for chunk i>0, optionally replace
        # the chunk's FIRST latent frame (the anchor slot the AR window inherits
        # from the previous chunk's last denoised frame) with a supplied anchor
        # latent, scaled by per_chunk_anchor_strength (1 = hard replace, 0.5 =
        # blend with the previous chunk's final latent). This is the per-chunk
        # generalization of the frame-0 image anchor: the round-4 factorial
        # showed the anchor latent is the dominant content conditioner, so
        # swapping it per chunk is how scene semantics can follow the map
        # region-by-region. Strength 0 on any chunk leaves it untouched.
        per_chunk_anchor_strength = float(per_chunk_anchor_strength)
        per_chunk_anchor_mode = str(per_chunk_anchor_mode)
        # Mode may be a single value broadcast to every chunk, or a comma-
        # separated per-chunk list aligned with the anchor tokens (chunk i>0
        # consumes entry i-1, mirroring per_chunk_anchor_latents rows). "none"
        # entries leave that chunk unpinned even when a non-zero anchor latent
        # was supplied for it.
        anchor_mode_seq = [t.strip().lower() for t in per_chunk_anchor_mode.split(",") if t.strip()]
        if not anchor_mode_seq or any(m not in ("hard", "soft", "none") for m in anchor_mode_seq):
            raise ValueError(
                f"per_chunk_anchor_mode entries must be 'hard', 'soft' or 'none', "
                f"got {per_chunk_anchor_mode!r}")
        if per_chunk_anchor_latents is not None:
            per_chunk_anchor_latents = per_chunk_anchor_latents.to(
                device=noise.device, dtype=noise.dtype)
            if per_chunk_anchor_latents.dim() != 5 or per_chunk_anchor_latents.shape[2:] != noise.shape[2:]:
                raise ValueError(
                    f"per_chunk_anchor_latents must be [N, 1, C, H, W] matching "
                    f"noise channels/spatial ({noise.shape[2:]}), got "
                    f"{tuple(per_chunk_anchor_latents.shape)}")
            n_anchor_tokens = per_chunk_anchor_latents.shape[0]
            if len(anchor_mode_seq) > 1 and len(anchor_mode_seq) != n_anchor_tokens:
                raise ValueError(
                    f"per-chunk mode list has {len(anchor_mode_seq)} entries but "
                    f"{n_anchor_tokens} anchor tokens were supplied")
            mode_label = anchor_mode_seq[0] if len(anchor_mode_seq) == 1 else (
                f"list[{len(anchor_mode_seq)}] hard={anchor_mode_seq.count('hard')} "
                f"soft={anchor_mode_seq.count('soft')} none={anchor_mode_seq.count('none')}")
            print(f"[per_chunk_anchor] enabled: {per_chunk_anchor_latents.shape[0]} anchor(s), "
                  f"strength={per_chunk_anchor_strength}, mode={mode_label}"
                  + (f", soft_timestep={int(per_chunk_anchor_soft_timestep)}"
                     if "soft" in anchor_mode_seq else ""))
        anchor_soft_shift = None
        for i, current_num_frames in tqdm.tqdm(enumerate(all_num_frames)):
            if self.act_probe is not None:
                self.act_probe.note_chunk(int(i))
            cached_residual = None
            previous_step_latents = None
            anchor_soft_shift = None
            chunk_reuse_disabled = reuse_enabled and int(i) in reuse_disabled_chunks
            if adaptive_chunk_steps_enabled and int(i) in per_chunk_step_lists:
                chunk_step_list = per_chunk_step_lists[int(i)]
                if chunk_step_list is not None and list(chunk_step_list) != list(native_raw_step_list):
                    self.args.denoising_step_list = [int(v) for v in chunk_step_list]
                    step_tensor = torch.tensor([int(v) for v in chunk_step_list], dtype=torch.long)
                    if self.args.warp_denoising_step:
                        timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                        step_tensor = timesteps[1000 - step_tensor]
                    self.denoising_step_list = step_tensor
                    difficulty = None
                    if per_chunk_difficulty_classes and int(i) in per_chunk_difficulty_classes:
                        difficulty = per_chunk_difficulty_classes[int(i)]
                    self.map_adaptive_denoising_log.append({
                        "chunk_id": int(i),
                        "hardness_class": (difficulty or {}).get("reason", "reduced_steps"),
                        "hardness_score": (difficulty or {}).get("difficulty"),
                        "step_count": len(chunk_step_list),
                        "raw_denoising_step_list": [int(v) for v in chunk_step_list],
                    })
            elif adaptive_chunk_steps_enabled:
                self.args.denoising_step_list = native_raw_step_list
                self.denoising_step_list = native_denoising_step_list
            if per_chunk_prompt_embeds is not None:
                # Per-chunk text conditioning: switch the active caption and
                # invalidate the cross-attention K/V cache so the model
                # re-encodes the (new) context instead of reusing chunk-0's
                # frozen text K/V (WanCrossAttention only recomputes k/v when
                # crossattn_cache["is_init"] is False). Same caption as the
                # previous chunk -> keep the cache to preserve exact behavior.
                if int(i) == 0:
                    conditional_dict = {"prompt_embeds": per_chunk_prompt_embeds[0:1]}
                else:
                    caption_changed = bool(
                        per_chunk_captions[int(i)] != per_chunk_captions[int(i) - 1])
                    if caption_changed:
                        for cache in self.crossattn_cache:
                            cache["is_init"] = False
                        print(f"[per_chunk_captions] chunk {int(i)}: switching caption "
                              f"({len(per_chunk_captions[int(i)])} chars), "
                              f"cross-attn cache invalidated")
                    conditional_dict = {
                        "prompt_embeds": per_chunk_prompt_embeds[int(i):int(i) + 1]}
            if profile:
                block_start.record()
            chunk_wall_timer = runtime_profiler.start_timer(reset_peak=True) if runtime_profiler is not None else None
            chunk_latent_start = current_start_frame - num_input_frames
            chunk_latent_end = chunk_latent_start + current_num_frames
            if runtime_profiler is not None:
                step_values = [
                    float(value) if hasattr(value, "item") else float(value)
                    for value in self.denoising_step_list.detach().cpu().tolist()
                ]
                runtime_profiler.start_chunk(
                    chunk_index=i,
                    latent_start_frame=chunk_latent_start,
                    latent_end_frame=chunk_latent_end,
                    total_chunks=len(all_num_frames),
                    denoising_step_count=len(self.denoising_step_list),
                    denoising_timesteps=step_values,
                )

            noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input
            if (per_chunk_anchor_latents is not None and int(i) > 0
                    and int(i) - 1 < per_chunk_anchor_latents.shape[0]
                    and per_chunk_anchor_strength > 0.0):
                # For chunks i>0 the window is 3 pure-noise frames with an
                # all-ones first_frame_mask: local frame 0 is the chunk's
                # first OUTPUT frame, not an AR context slot, so a one-shot
                # write into it is OOD input for one model call whose effect
                # is then overwritten (round-5 failure mode).
                anchor_row = int(i) - 1
                anchor_lat = per_chunk_anchor_latents[anchor_row:anchor_row + 1].to(
                    device=noise.device, dtype=noise.dtype)
                if float(anchor_lat.abs().sum().item()) > 0.0:
                    chunk_anchor_mode = (anchor_mode_seq[int(i) - 1]
                                         if len(anchor_mode_seq) > 1 else anchor_mode_seq[0])
                    if chunk_anchor_mode == "none":
                        pass
                    elif chunk_anchor_mode == "soft":
                        # Soft SDEdit mode: instead of hard-pinning the anchor
                        # clean at t=0 (a visible content cut), seed the chunk's
                        # first output frame with the anchor noised to an
                        # intermediate timestep and let it re-render on the TAIL
                        # of the denoise schedule: its per-step model timestep
                        # is the step list shifted so it starts at the first
                        # grid step <= the seed timestep. Content stays
                        # anchor-biased without an exact-match keyframe. A
                        # seed below the grid bottom (e.g. raw 125 with list
                        # [1000,750,500,250]) keeps the run timestep at the
                        # bottom step while seeding with MORE surviving anchor
                        # structure than that step declares — a stronger pull
                        # that is still re-rendered, not copied.
                        if bool(getattr(self.args, "warp_denoising_step", False)):
                            ts_table = torch.cat((
                                self.scheduler.timesteps.cpu(),
                                torch.tensor([0], dtype=torch.float32)))
                            t_seed_warped = float(ts_table[1000 - int(per_chunk_anchor_soft_timestep)])
                        else:
                            t_seed_warped = float(per_chunk_anchor_soft_timestep)
                        raw_steps = [int(v) for v in self.args.denoising_step_list]
                        start_idx = next(
                            (j for j, s in enumerate(raw_steps)
                             if s <= int(per_chunk_anchor_soft_timestep)),
                            len(raw_steps) - 1)
                        anchor_eps = torch.randn_like(anchor_lat)
                        noised_anchor = self.scheduler.add_noise(
                            anchor_lat.flatten(0, 1),
                            anchor_eps.flatten(0, 1),
                            torch.ones(anchor_lat.shape[1], device=noise.device,
                                       dtype=torch.float32) * t_seed_warped,
                        ).unflatten(0, anchor_lat.shape[:2]).to(noisy_input.dtype)
                        noisy_input = noisy_input.clone()
                        noisy_input[:, :1] = noised_anchor
                        latents = noisy_input
                        anchor_soft_shift = start_idx
                        print(f"[per_chunk_anchor] chunk {int(i)}: soft SDEdit anchor row "
                              f"{anchor_row} seeded at t={t_seed_warped:.1f} (raw "
                              f"{int(per_chunk_anchor_soft_timestep)}), schedule shift={start_idx}")
                    elif per_chunk_anchor_strength >= 1.0:
                        # Mask mode: replicate chunk 0's I2V pattern — pin
                        # local frame 0 to the clean anchor latent and zero
                        # its global first_frame_mask entry. The step loop
                        # then automatically runs its tokens at t=0
                        # (temp_ts), re-imposes the anchor after every step,
                        # writes it to output[:, current_start_frame] as the
                        # chunk's first rendered keyframe, and the KV warm
                        # pass commits its keys/values for later chunks.
                        anchor_slot = current_start_frame - num_input_frames
                        noisy_input = noisy_input.clone()
                        noisy_input[:, :1] = anchor_lat
                        first_frame_mask[:, anchor_slot] = 0.0
                        latents = noisy_input
                        print(f"[per_chunk_anchor] chunk {int(i)}: mask-mode anchor row "
                              f"{anchor_row} pinned to window frame 0 (global frame "
                              f"{int(anchor_slot)}, t=0)")
                    else:
                        # Blend mode (0 < strength < 1): legacy one-shot
                        # perturbation of the chunk's first output-frame noise.
                        base_ctx = noisy_input[:, :1]
                        blend = per_chunk_anchor_strength * anchor_lat + \
                            (1.0 - per_chunk_anchor_strength) * base_ctx
                        noisy_input = noisy_input.clone()
                        noisy_input[:, :1] = blend.to(noisy_input.dtype)
                        latents = noisy_input
                        print(f"[per_chunk_anchor] chunk {int(i)}: blend anchor row {anchor_row} "
                              f"into first output-frame noise, strength={per_chunk_anchor_strength}")
                else:
                    print(f"[per_chunk_anchor] chunk {int(i)}: anchor row {anchor_row} is an "
                          f"all-zeros sentinel -> skipped (chunk runs like base)")
            chunk_seed_timestep = None
            if ar_boundary_renoise > 0.0 and prev_chunk_latents is not None and int(i) > 0:
                # A1 SDEdit boundary re-noising: seed chunk i's NEW frames from
                # the previous chunk's final latent frame, re-noised to t_seed,
                # and blend with the chunk's pure noise. Anchor input frames
                # keep their original noise (the first_frame_mask enforces
                # them after every model call anyway). The saved/restored RNG
                # state makes the re-noise epsilon deterministic regardless of
                # how many chunks ran, and independent of the global stream.
                # t_seed=0 means "the chunk's first denoise timestep" (warped
                # top step; prev structure fully masked — pure noise damping).
                # t_seed>0 is TRUE SDEdit in raw pre-warp units: re-noise to a
                # lower timestep so prev structure survives (sigma<1), and
                # truncate the chunk's step list to start at t_seed.
                if ar_boundary_renoise_t > 0:
                    ts_table = None
                    if self.args.warp_denoising_step:
                        ts_table = torch.cat((self.scheduler.timesteps.cpu(),
                                              torch.tensor([0], dtype=torch.float32)))
                        chunk_seed_timestep = float(ts_table[1000 - int(ar_boundary_renoise_t)])
                    else:
                        chunk_seed_timestep = float(ar_boundary_renoise_t)
                    # The chunk's effective step list was already set above
                    # (adaptive per-chunk branch or native). Truncate it to
                    # start at the seed timestep so the model denoises FROM
                    # the re-noised state instead of the full trajectory.
                    # args.denoising_step_list is restored afterwards so the
                    # next chunk's adaptive branch reads the untruncated list.
                    raw_steps = [int(v) for v in self.args.denoising_step_list]
                    kept = [s for s in raw_steps if s <= int(ar_boundary_renoise_t)]
                    if len(kept) < 1:
                        kept = [min(raw_steps)]
                    if list(kept) != raw_steps:
                        step_tensor = torch.tensor(kept, dtype=torch.long)
                        if ts_table is not None:
                            step_tensor = ts_table[1000 - step_tensor]
                        self.denoising_step_list = step_tensor
                        if not adaptive_chunk_steps_enabled:
                            self.args.denoising_step_list = native_raw_step_list
                        print(f"[ar_boundary_renoise] chunk {int(i)}: step list truncated to "
                              f"raw {kept} (starts at seed timestep)")
                else:
                    chunk_seed_timestep = float(self.denoising_step_list[0].item())
                n_new = int(current_num_frames)
                prev_final = prev_chunk_latents[:, -1:].to(noisy_input.dtype)
                if prev_final.shape[1] == 1 and n_new > 1:
                    prev_final = prev_final.repeat(1, n_new, 1, 1, 1)
                saved_rng = torch.get_rng_state()
                if renoise_rng_state is not None:
                    torch.set_rng_state(renoise_rng_state)
                renoised = self.scheduler.add_noise(
                    prev_final.flatten(0, 1),
                    torch.randn_like(prev_final).flatten(0, 1),
                    (torch.ones(n_new, device=noise.device, dtype=torch.float32)
                     * float(chunk_seed_timestep)),
                ).unflatten(0, prev_final.shape[:2]).to(noisy_input.dtype)
                renoise_rng_state = torch.get_rng_state()
                torch.set_rng_state(saved_rng)
                latents = noisy_input.clone()
                latents[:, noisy_input.shape[1] - n_new:] = (
                    (1.0 - ar_boundary_renoise) * noisy_input[:, noisy_input.shape[1] - n_new:]
                    + ar_boundary_renoise * renoised).to(noisy_input.dtype)
                print(f"[ar_boundary_renoise] chunk {int(i)}: seeded {n_new} new frames from "
                      f"prev final latent, t_seed={chunk_seed_timestep:.1f} "
                      f"(raw {ar_boundary_renoise_t or int(self.denoising_step_list[0].item())}), "
                      f"strength={ar_boundary_renoise}, "
                      f"steps={len(self.denoising_step_list)}")
            y_latents = y[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames] if y is not None else None

            if isinstance(y_camera, dict):
                start_index = (current_start_frame - num_input_frames) * self.frame_seq_length
                end_index = start_index + self.frame_seq_length * current_num_frames
                y_camera_latents = {
                    'viewmats': y_camera['viewmats'][:, start_index:end_index],
                    'K': y_camera['K'][:, start_index:end_index],
                }
            else:
                y_camera_latents = y_camera[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            first_frame_mask_block = first_frame_mask[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            map_sparse_ffn_plan = None
            map_sparse_attention_plan = None
            emapctrl_condition_latents = None
            if (
                map_sparse_ffn_router_result is not None
                and bool(getattr(getattr(self.generator, "model", None), "map_sparse_ffn", False))
            ):
                map_sparse_ffn_plan = build_map_sparse_ffn_plan_from_router_result(
                    map_sparse_ffn_router_result,
                    latent_start_frame=chunk_latent_start,
                    latent_end_frame=chunk_latent_end,
                    frame_seq_length=self.frame_seq_length,
                    semantic_map_version=str(
                        getattr(self.args, "map_sparse_ffn_semantic_map_version", "default")
                    ),
                )
            if (
                map_sparse_ffn_router_result is not None
                and bool(getattr(getattr(self.generator, "model", None), "map_sparse_attention", False))
            ):
                model_kwargs = getattr(self.args, "model_kwargs", {})
                map_sparse_attention_plan = build_map_sparse_attention_plan_from_router_result(
                    map_sparse_ffn_router_result,
                    latent_start_frame=chunk_latent_start,
                    latent_end_frame=chunk_latent_end,
                    frame_seq_length=self.frame_seq_length,
                    semantic_map_version=str(
                        getattr(self.args, "map_sparse_attention_semantic_map_version", "default")
                    ),
                    camera_motion_threshold=float(
                        getattr(
                            model_kwargs,
                            "map_sparse_attention_camera_motion_threshold",
                            float("inf"),
                        )
                    ),
                )
            emapctrl_condition_latents = self._emapctrl_condition_for_range(
                emapctrl_condition=emapctrl_condition,
                router_result=map_sparse_ffn_router_result,
                latent_start_frame=chunk_latent_start,
                latent_end_frame=chunk_latent_end,
                device=noise.device,
                dtype=noise.dtype,
            )
            map_adapter_condition_latents = self._map_adapter_condition_for_range(
                map_adapter_condition=map_adapter_condition,
                latent_start_frame=chunk_latent_start,
                latent_end_frame=chunk_latent_end,
                device=noise.device,
                dtype=noise.dtype,
            )

            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                step_wall_timer = runtime_profiler.start_timer(reset_peak=True) if runtime_profiler is not None else None
                step_input_latents = latents
                temp_ts = ((first_frame_mask_block[0, :, 0, ::2, ::2]) * current_timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(self.frame_seq_length * current_num_frames - temp_ts.size(0)) * current_timestep
                ])
                if anchor_soft_shift is not None:
                    # Soft-mode anchor frame: run its tokens on the tail of the
                    # schedule (list[index + shift]) instead of the chunk's
                    # current step. The mask grid is frame-major, so the first
                    # (H//2)*(W//2) entries are exactly local frame 0's tokens.
                    soft_steps = self.denoising_step_list
                    anchor_ts = soft_steps[min(index + anchor_soft_shift,
                                               len(soft_steps) - 1)]
                    soft_hw = (first_frame_mask_block.shape[3] // 2) * \
                              (first_frame_mask_block.shape[4] // 2)
                    temp_ts[:soft_hw] = anchor_ts
                timestep = temp_ts.unsqueeze(0).expand(batch_size, temp_ts.size(0))
                reuse_metric = None
                if reuse_collect_state and previous_step_latents is not None:
                    reuse_metric = residual_relative_l1_change(step_input_latents, previous_step_latents)
                reuse_step, reuse_reason = residual_reuse_decision(
                    enabled=reuse_enabled,
                    threshold=reuse_threshold,
                    chunk_disabled=chunk_reuse_disabled,
                    step_index=index,
                    step_count=len(self.denoising_step_list),
                    has_cached_residual=cached_residual is not None,
                    metric=reuse_metric,
                )

                if index < len(self.denoising_step_list) - 1:
                    if reuse_step:
                        denoised_pred = step_input_latents + cached_residual.to(
                            device=step_input_latents.device,
                            dtype=step_input_latents.dtype,
                        )
                        model_sec, model_peak = 0.0, {}
                    else:
                        model_timer = runtime_profiler.start_timer(reset_peak=True) if runtime_profiler is not None else None
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=latents,
                            conditional_dict=conditional_dict,
                            y=y_latents, y_camera=y_camera_latents,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            runtime_profiler=runtime_profiler,
                            profile_chunk_index=i,
                            profile_step_index=index,
                            profile_phase="denoising",
                            map_sparse_ffn_plan=map_sparse_ffn_plan,
                            map_sparse_attention_plan=map_sparse_attention_plan,
                            emapctrl_condition=emapctrl_condition_latents,
                            map_adapter_condition=map_adapter_condition_latents)
                        model_sec, model_peak = runtime_profiler.stop_timer(model_timer) if runtime_profiler is not None else (0.0, {})
                        if reuse_enabled:
                            cached_residual = (denoised_pred - step_input_latents).detach()
                    scheduler_timer = runtime_profiler.start_timer(reset_peak=True) if runtime_profiler is not None else None
                    next_timestep = self.denoising_step_list[index + 1]
                    next_timestep = next_timestep * torch.ones(
                        [batch_size, current_num_frames], device=noise.device, dtype=torch.long)
                    if anchor_soft_shift is not None:
                        # Scheduler side of the soft-mode tail: re-noise the
                        # anchor frame to its own next step (clamped at the
                        # bottom), so it iterates at the seed/fixed-point
                        # timestep instead of descending with frames 1..n.
                        next_timestep[:, 0] = int(
                            self.denoising_step_list[
                                min(index + anchor_soft_shift + 1,
                                    len(self.denoising_step_list) - 1)].item())
                    if i == 0:
                        next_timestep[:, 0] = 0
                    scheduler_noise_reference = step_input_latents if reuse_step else denoised_pred
                    latents = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(scheduler_noise_reference.flatten(0, 1)),
                        next_timestep.flatten()
                    ).unflatten(0, denoised_pred.shape[:2])
                    latents = latents * first_frame_mask_block + noisy_input * (1 - first_frame_mask_block)
                    scheduler_sec, scheduler_peak = runtime_profiler.stop_timer(scheduler_timer) if runtime_profiler is not None else (0.0, {})
                else:
                    model_timer = runtime_profiler.start_timer(reset_peak=True) if runtime_profiler is not None else None
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=latents,
                        conditional_dict=conditional_dict,
                        y=y_latents, y_camera=y_camera_latents,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        runtime_profiler=runtime_profiler,
                        profile_chunk_index=i,
                        profile_step_index=index,
                        profile_phase="denoising",
                        map_sparse_ffn_plan=map_sparse_ffn_plan,
                        map_sparse_attention_plan=map_sparse_attention_plan,
                        emapctrl_condition=emapctrl_condition_latents,
                        map_adapter_condition=map_adapter_condition_latents)
                    model_sec, model_peak = runtime_profiler.stop_timer(model_timer) if runtime_profiler is not None else (0.0, {})
                    if reuse_enabled:
                        cached_residual = (denoised_pred - step_input_latents).detach()
                    denoised_pred = denoised_pred * first_frame_mask_block + noisy_input * (1 - first_frame_mask_block)
                    scheduler_sec, scheduler_peak = 0.0, {}
                if reuse_record_steps:
                    self.residual_reuse_log.append({
                        "chunk_index": int(i),
                        "step_index": int(index),
                        "timestep": float(current_timestep.item() if hasattr(current_timestep, "item") else current_timestep),
                        "decision": "reused" if reuse_step else "computed",
                        "reason": reuse_reason,
                        "metric": None if reuse_metric is None else float(reuse_metric),
                        "threshold": float(reuse_threshold),
                        "camera_discontinuity_disabled": bool(chunk_reuse_disabled),
                    })
                if reuse_collect_state:
                    previous_step_latents = step_input_latents.detach()
                if runtime_profiler is not None:
                    step_sec, step_peak = runtime_profiler.stop_timer(step_wall_timer)
                    combined_peak = {
                        "peak_allocated_gib": max(
                            float(model_peak.get("peak_allocated_gib", 0.0)),
                            float(scheduler_peak.get("peak_allocated_gib", 0.0)),
                            float(step_peak.get("peak_allocated_gib", 0.0)),
                        ),
                        "peak_reserved_gib": max(
                            float(model_peak.get("peak_reserved_gib", 0.0)),
                            float(scheduler_peak.get("peak_reserved_gib", 0.0)),
                            float(step_peak.get("peak_reserved_gib", 0.0)),
                        ),
                    }
                    runtime_profiler.add_denoising_step(
                        chunk_index=i,
                        step_index=index,
                        timestep=float(current_timestep.item() if hasattr(current_timestep, "item") else current_timestep),
                        model_forward_sec=model_sec,
                        scheduler_update_sec=scheduler_sec,
                        total_sec=step_sec,
                        peak=combined_peak,
                        residual_reuse_decision="reused" if reuse_step else "computed",
                        residual_reuse_reason=reuse_reason,
                        residual_reuse_metric=reuse_metric,
                        residual_reuse_threshold=reuse_threshold,
                        camera_discontinuity_disabled=chunk_reuse_disabled,
                    )

            denoised_pred, seam_events = self._apply_latent_seam_blend(
                output=output,
                denoised_pred=denoised_pred,
                current_start_frame=current_start_frame,
            )
            latent_seam_blend_event_count += seam_events
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            if ar_boundary_renoise > 0.0:
                prev_chunk_latents = denoised_pred.detach()

            # Rerun with context noise to update KV cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            kv_timer = runtime_profiler.start_timer(reset_peak=True) if runtime_profiler is not None else None
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                y=y_latents, y_camera=y_camera_latents,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                runtime_profiler=runtime_profiler,
                profile_chunk_index=i,
                profile_step_index=None,
                profile_phase="kv_cache_update",
                emapctrl_condition=emapctrl_condition_latents,
                map_adapter_condition=map_adapter_condition_latents,
            )
            if runtime_profiler is not None:
                kv_sec, kv_peak = runtime_profiler.stop_timer(kv_timer)
                runtime_profiler.add_kv_cache_update(i, kv_sec, kv_peak)

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_times.append(block_start.elapsed_time(block_end))
            if runtime_profiler is not None:
                chunk_sec, chunk_peak = runtime_profiler.stop_timer(chunk_wall_timer)
                runtime_profiler.finish_dit_chunk(i, chunk_sec, chunk_peak)

            if chunk_callback is not None:
                # The latent frames of this chunk are final here: seam blending
                # only reads previous frames and later chunks never overwrite
                # this range. Hand a CPU copy to the callback so the DiT can
                # immediately continue with the next chunk.
                chunk_latent_frames = output[
                    :, current_start_frame:current_start_frame + current_num_frames
                ].detach().to(device="cpu", copy=True)
                chunk_callback(
                    chunk_index=int(i),
                    latent_start_frame=int(current_start_frame),
                    latent_frame_count=int(current_num_frames),
                    latent_chunk=chunk_latent_frames,
                )

            current_start_frame += current_num_frames
        diffusion_loop_wall_sec = time.perf_counter() - diffusion_loop_wall_start

        if adaptive_chunk_steps_enabled:
            self.args.denoising_step_list = native_raw_step_list
            self.denoising_step_list = native_denoising_step_list

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)

        if self.sequential_offload:
            self._clear_caches()
            if runtime_profiler is not None:
                _, transfer_sec, transfer_peak = runtime_profiler.measure(
                    lambda: self.generator.to(cpu), reset_peak=True)
                runtime_profiler.add_transfer_event(None, "generator_gpu_to_cpu", transfer_sec, transfer_peak)
            else:
                self.generator.to(cpu)
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[2/3 diffusion] done. Free VRAM: {get_cuda_free_memory_gb(gpu):.1f} GB")
        elif not decode_video:
            # Streaming-decode path: latent-only return. The generator and VAE
            # stay resident (the caller reuses them for chunked VAE decode and
            # for subsequent items); only the KV/cross-attn caches are freed to
            # release the per-generation cache memory before VAE decode.
            self._clear_caches()
            torch.cuda.empty_cache()
            gc.collect()

        if not decode_video:
            total_wall_sec = time.perf_counter() - inference_wall_start
            self.last_inference_wall_profile = {
                "conditioning_setup_sec": conditioning_wall_sec,
                "diffusion_loop_sec": diffusion_loop_wall_sec,
                "total_inference_sec": total_wall_sec,
                "other_inference_overhead_sec": max(0.0, total_wall_sec - conditioning_wall_sec - diffusion_loop_wall_sec),
                "denoising_step_count": len(self.denoising_step_list),
                "latent_seam_blend_events": latent_seam_blend_event_count,
            }
            if profile:
                total_time = init_time + diffusion_time
                print(f"Profiling: init={init_time:.0f}ms, diffusion={diffusion_time:.0f}ms, total={total_time:.0f}ms")
                if latent_seam_blend_event_count:
                    print(f"  Latent seam blend events: {latent_seam_blend_event_count}")
                for i, bt in enumerate(block_times):
                    print(f"  Block {i}: {bt:.0f}ms")
            return (None, output) if return_latents else output

        if profile:
            vae_start.record()

        if self.sequential_offload:
            if runtime_profiler is not None:
                _, transfer_sec, transfer_peak = runtime_profiler.measure(
                    lambda: self.vae.to(gpu), reset_peak=True)
                runtime_profiler.add_transfer_event(0, "vae_cpu_to_gpu_for_decode", transfer_sec, transfer_peak)
            else:
                self.vae.to(gpu)
            print(f"[3/3 VAE decode] vae loaded. Free VRAM: {get_cuda_free_memory_gb(gpu):.1f} GB")

        if runtime_profiler is not None:
            video, vae_decode_sec, vae_decode_peak = runtime_profiler.measure(
                lambda: self.vae.decode_to_pixel(
                    output,
                    tile_height=vae_tile_height,
                    tile_overlap=vae_tile_overlap,
                    tile_blend_mode=vae_tile_blend_mode,
                ),
                reset_peak=True)
            runtime_profiler.add_vae_decode_chunk(0, 0, output.shape[1], vae_decode_sec, vae_decode_peak)
        else:
            video = self.vae.decode_to_pixel(
                output,
                tile_height=vae_tile_height,
                tile_overlap=vae_tile_overlap,
                tile_blend_mode=vae_tile_blend_mode,
            )
        video = (video * 0.5 + 0.5).clamp(0, 1)
        total_wall_sec = time.perf_counter() - inference_wall_start
        self.last_inference_wall_profile = {
            "conditioning_setup_sec": conditioning_wall_sec,
            "diffusion_loop_sec": diffusion_loop_wall_sec,
            "total_inference_sec": total_wall_sec,
            "other_inference_overhead_sec": max(0.0, total_wall_sec - conditioning_wall_sec - diffusion_loop_wall_sec),
            "denoising_step_count": len(self.denoising_step_list),
            "latent_seam_blend_events": latent_seam_blend_event_count,
        }

        if self.sequential_offload:
            if runtime_profiler is not None:
                _, transfer_sec, transfer_peak = runtime_profiler.measure(
                    lambda: self.vae.to(cpu), reset_peak=True)
                runtime_profiler.add_transfer_event(0, "vae_gpu_to_cpu_after_decode", transfer_sec, transfer_peak)
            else:
                self.vae.to(cpu)
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[3/3 VAE decode] done. Free VRAM: {get_cuda_free_memory_gb(gpu):.1f} GB")

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            print(f"Profiling: init={init_time:.0f}ms, diffusion={diffusion_time:.0f}ms, vae={vae_time:.0f}ms, total={total_time:.0f}ms")
            if latent_seam_blend_event_count:
                print(f"  Latent seam blend events: {latent_seam_blend_event_count}")
            for i, bt in enumerate(block_times):
                print(f"  Block {i}: {bt:.0f}ms")

        return (video, output) if return_latents else video

    def _apply_latent_seam_blend(self, output, denoised_pred, current_start_frame):
        """Blend the first latent frame of an AR block with a velocity prediction.

        Causal generation advances in blocks of ``num_frame_per_block`` latent
        frames. A small seam correction before the KV-cache update reduces the
        visible 12-pixel-frame cadence without changing the model weights.
        """
        alpha = max(0.0, min(1.0, float(getattr(self, "latent_seam_blend_alpha", 0.0))))
        blend_frames = max(0, int(getattr(self, "latent_seam_blend_frames", 1)))
        if alpha <= 0.0 or blend_frames <= 0:
            return denoised_pred, 0
        if int(current_start_frame) <= 0 or int(denoised_pred.shape[1]) <= 0:
            return denoised_pred, 0

        blend_frames = min(blend_frames, int(denoised_pred.shape[1]))
        velocity_scale = float(getattr(self, "latent_seam_velocity_scale", 0.5))
        velocity_scale = max(0.0, min(1.5, velocity_scale))

        corrected = denoised_pred.clone()
        events = 0
        for local_idx in range(blend_frames):
            prev_idx = int(current_start_frame) + local_idx - 1
            if prev_idx < 0:
                continue

            prev = output[:, prev_idx]
            if prev_idx >= 1:
                prev2 = output[:, prev_idx - 1]
                target = prev + velocity_scale * (prev - prev2)
            else:
                target = prev

            local_alpha = alpha / float(local_idx + 1)
            corrected[:, local_idx] = (
                (1.0 - local_alpha) * corrected[:, local_idx]
                + local_alpha * target
            )
            events += 1
        return corrected, events

    def _emapctrl_condition_for_range(
        self,
        *,
        emapctrl_condition,
        router_result,
        latent_start_frame: int,
        latent_end_frame: int,
        device,
        dtype,
    ):
        if not bool(getattr(getattr(self.generator, "model", None), "emapctrl_enabled", False)):
            return None
        if emapctrl_condition is not None:
            return slice_emapctrl_condition(
                emapctrl_condition,
                latent_start_frame=latent_start_frame,
                latent_end_frame=latent_end_frame,
                device=device,
                dtype=dtype,
            )
        if router_result is None:
            return None
        return build_emapctrl_condition_from_router_result(
            router_result,
            latent_start_frame=latent_start_frame,
            latent_end_frame=latent_end_frame,
            frame_seq_length=self.frame_seq_length,
            device=device,
            dtype=dtype,
        )

    def _map_adapter_condition_for_range(
        self,
        *,
        map_adapter_condition,
        latent_start_frame: int,
        latent_end_frame: int,
        device,
        dtype,
    ):
        if not bool(getattr(getattr(self.generator, "model", None), "map_adapter_enabled", False)):
            return None
        if map_adapter_condition is None:
            return None
        return slice_map_adapter_condition(
            map_adapter_condition,
            latent_start_frame=latent_start_frame,
            latent_end_frame=latent_end_frame,
            device=device,
            dtype=dtype,
        )

    def _initialize_kv_cache(self, batch_size, dtype, device, num_frames=21):
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 18480

        num_head = 24
        self.kv_cache1 = [
            {
                "k": torch.zeros([batch_size, kv_cache_size, num_head, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_head, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(self.num_transformer_blocks)
        ]

    def _clear_caches(self):
        if self.kv_cache1 is not None:
            for cache in self.kv_cache1:
                for key in list(cache.keys()):
                    if isinstance(cache[key], torch.Tensor):
                        cache[key] = None
            self.kv_cache1 = None

        if self.crossattn_cache is not None:
            for cache in self.crossattn_cache:
                for key in list(cache.keys()):
                    if isinstance(cache[key], torch.Tensor):
                        cache[key] = None
            self.crossattn_cache = None

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        num_head = 24
        self.crossattn_cache = [
            {
                "k": torch.zeros([batch_size, 512, num_head, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_head, 128], dtype=dtype, device=device),
                "is_init": False,
            }
            for _ in range(self.num_transformer_blocks)
        ]
