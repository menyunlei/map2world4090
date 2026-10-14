"""Load the exact official AR weights without initializing large CPU weights.

Native model/wrapper/pipeline classes, scheduler and tokenizer are retained.
Only weight construction/loading changes: meta parameters, strict assign,
then the same native BF16 cast. This module launches no GPU jobs by itself.
"""
import hashlib
from pathlib import Path
import time


def _sample_bits(tensor, count=64):
    import torch
    values = tensor.detach().reshape(-1)[:count].to(device="cpu", dtype=torch.bfloat16).contiguous()
    return {"numel_sampled": values.numel(),
            "bf16_sha256": hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "first_values_float32": values.float()[:8].tolist()}


def _meta_tensors(module):
    import torch
    bad = [f"parameter:{name}" for name, value in module.named_parameters() if value.is_meta]
    bad += [f"buffer:{name}" for name, value in module.named_buffers() if value.is_meta]
    for module_name, child in module.named_modules():
        for name, value in vars(child).items():
            if isinstance(value, torch.Tensor) and value.is_meta:
                bad.append(f"attribute:{module_name}.{name}")
    return sorted(set(bad))


def _state_report(module, incompatibility, source_keys, expected_sample, loaded_sample_key):
    state = module.state_dict()
    loaded_sample = _sample_bits(state[loaded_sample_key])
    missing = list(incompatibility.missing_keys)
    unexpected = list(incompatibility.unexpected_keys)
    meta = _meta_tensors(module)
    result = {"source_key_count": len(source_keys), "loaded_key_count": len(state),
              "source_keys": sorted(source_keys), "missing": missing, "unexpected": unexpected,
              "strict_assign": True, "meta_tensors_remaining": meta,
              "parameters": sum(parameter.numel() for parameter in module.parameters()),
              "dtypes": sorted({str(parameter.dtype) for parameter in module.parameters()}),
              "sample_key": loaded_sample_key, "checkpoint_bf16_sample": expected_sample,
              "loaded_bf16_sample": loaded_sample,
              "sample_exact": expected_sample == loaded_sample}
    if missing or unexpected or meta or not result["sample_exact"]:
        raise RuntimeError(f"Native weight loading contract failed: {result}")
    return result


def load_native_pipeline(config, *, checkpoint_path, model_config_path, device,
                         num_output_frames, model_name=None, text_encoder_path=None,
                         tokenizer_path=None, vae_path=None, height=384, width=640):
    """Return (native pipeline, strict loader report); all inference stays native.

    The caller must choose an authorized GPU/current device before calling.
    No distributed group or sequence-parallel adapter is initialized here.
    """
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from wan.modules.causal_camera_model_2_2_prope_infinity import CausalWanModel
    from wan.modules.t5 import umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer
    from utils.scheduler import FlowMatchScheduler
    from utils.wan_wrapper import WanDiffusionCameraWrapper, WanTextEncoder, WanVAEWrapper
    from pipeline.pipeline_causal_camera import CausalCameraInferencePipeline

    started = time.perf_counter()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("This bounded loader is for an explicitly selected authorized CUDA GPU")
    if height % 32 or width % 32:
        raise ValueError("Native latent/patch grid must divide spatial dimensions by 32 exactly")
    frame_tokens = (height // 32) * (width // 32)
    if num_output_frames < 3 or num_output_frames % 3:
        raise ValueError("Retain the native three-latent-frame block contract")
    if model_name is not None:
        model_name = Path(model_name)
        text_encoder_path = text_encoder_path or model_name / "models_t5_umt5-xxl-enc-bf16.pth"
        tokenizer_path = tokenizer_path or model_name / "google/umt5-xxl"
        vae_path = vae_path or model_name / "Wan2.2_VAE.pth"
    if any(value is None for value in (text_encoder_path, tokenizer_path, vae_path)):
        raise ValueError("Supply native Wan model_name or all three native component paths")

    kwargs = getattr(config, "model_kwargs", {})
    local_window = int(kwargs.get("local_attn_size", 12))
    sink_frames = int(kwargs.get("sink_size", 3))
    shift = float(kwargs.get("timestep_shift", 5.0))
    if local_window != 12 or sink_frames != 3 or shift != 5.0:
        raise ValueError("V1 keeps the verified native window12/sink3/shift5 contract")
    generator = WanDiffusionCameraWrapper.__new__(WanDiffusionCameraWrapper)
    nn.Module.__init__(generator)
    phase_start = time.perf_counter()
    # Do not make unregistered, deterministic RoPE frequencies meta: the native
    # forward moves those real CPU complex tensors to its execution device.
    with init_empty_weights(include_buffers=False):
        model = CausalWanModel.from_config(str(model_config_path),
                                          local_attn_size=local_window, sink_size=sink_frames)
    if model.freqs.is_meta:
        raise RuntimeError("Deterministic native RoPE frequencies must be materialized")
    ar_state = load_file(str(checkpoint_path), device="cpu")
    ar_keys = list(ar_state)
    expected_ar_sample = _sample_bits(ar_state["patch_embedding.weight"])
    ar_load = model.load_state_dict(ar_state, strict=True, assign=True)
    del ar_state
    model = model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    ar_report = _state_report(model, ar_load, ar_keys, expected_ar_sample, "patch_embedding.weight")
    ar_report["rope_freqs_materialized"] = not model.freqs.is_meta
    ar_report["rope_freqs_dtype"] = str(model.freqs.dtype)
    ar_report["load_seconds"] = time.perf_counter() - phase_start
    generator.model = model
    generator.scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    generator.scheduler.set_timesteps(1000, training=True)
    generator.seq_len = frame_tokens * num_output_frames
    generator._bind_scheduler_methods()
    if generator.scheduler.sigmas.is_meta or generator.scheduler.timesteps.is_meta:
        raise RuntimeError("Native scheduler tensors must remain materialized")

    phase_start = time.perf_counter()
    text = WanTextEncoder.__new__(WanTextEncoder)
    nn.Module.__init__(text)
    text_model = umt5_xxl(encoder_only=True, return_tokenizer=False,
                          dtype=torch.float32, device=torch.device("meta"))
    text_state = torch.load(str(text_encoder_path), map_location="cpu", weights_only=False)
    text_keys = list(text_state)
    expected_text_sample = _sample_bits(text_state["token_embedding.weight"])
    text_load = text_model.load_state_dict(text_state, strict=True, assign=True)
    del text_state
    text.text_encoder = text_model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    text.tokenizer = HuggingfaceTokenizer(name=str(tokenizer_path), seq_len=512, clean="whitespace")
    text_report = _state_report(text.text_encoder, text_load, text_keys, expected_text_sample,
                                "token_embedding.weight")
    text_report["load_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    # Official VAE construction already performs strict meta+assign loading.
    # Keep that implementation and its mean/std and cached decode interfaces.
    vae = WanVAEWrapper(model_root_path=str(kwargs.get("model_root_path", "")),
                        vae_path=str(vae_path)).eval().requires_grad_(False)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    vae_meta = _meta_tensors(vae)
    if vae_meta or vae.mean.is_meta or vae.std.is_meta:
        raise RuntimeError(f"Native VAE has unresolved meta tensors: {vae_meta}")
    vae_report = {"native_meta_assign_constructor_unchanged": True,
                  "keys": len(vae.model.state_dict()), "meta_tensors_remaining": vae_meta,
                  "dtypes": sorted({str(parameter.dtype) for parameter in vae.parameters()}),
                  "parameters": sum(parameter.numel() for parameter in vae.parameters()),
                  "load_seconds": time.perf_counter() - phase_start}

    pipeline = CausalCameraInferencePipeline(config, device=device, generator=generator,
                                             text_encoder=text, vae=vae,
                                             num_output_frames=num_output_frames)
    pipeline.frame_seq_length = frame_tokens
    for block in model.blocks:
        for attention in [block.self_attn] + ([block.cam_self_attn] if hasattr(block, "cam_self_attn") else []):
            attention.max_attention_size = local_window * frame_tokens
    pipeline = pipeline.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    unresolved = _meta_tensors(pipeline)
    if unresolved:
        raise RuntimeError(f"Native injected pipeline has unresolved meta tensors: {unresolved}")
    torch.cuda.synchronize(device)
    report = {"status": "PASSED", "loader_version": "native_loader_v1",
              "model_class": type(model).__name__, "native_classes_retained": True,
              "random_large_weight_initialization_skipped": True,
              "additional_adapters": False, "training": False, "precision": "bfloat16",
              "device": str(device), "height": height, "width": width,
              "frame_sequence_length": frame_tokens, "latent_frames": num_output_frames,
              "generator": ar_report, "text_encoder": text_report, "vae": vae_report,
              "scheduler": {"class": type(generator.scheduler).__name__, "shift": shift,
                            "sigma_min": 0.0, "extra_one_step": True,
                            "set_timesteps": 1000, "training_schedule": True,
                            "native_conversion_methods_bound": True},
              "all_meta_tensors_resolved": True, "total_load_seconds": time.perf_counter() - started,
              "paths": {"checkpoint": str(checkpoint_path), "model_config": str(model_config_path),
                        "text_encoder": str(text_encoder_path), "tokenizer": str(tokenizer_path),
                        "vae": str(vae_path)},
              "checkpoint_file_bytes": Path(checkpoint_path).stat().st_size,
              "text_encoder_file_bytes": Path(text_encoder_path).stat().st_size}
    return pipeline, report
