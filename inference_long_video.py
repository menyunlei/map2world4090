"""
Long video generation via multi-segment inference.
Each segment uses the last frame of the previous segment as the start image.
Segments are concatenated with ffmpeg into a single long video.
"""
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import math
import random
import shutil
import subprocess
import tempfile
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Compatibility patch: transformers 5.x removed FLAX_WEIGHTS_NAME but diffusers 0.34.0 still imports it.
# Apply patch before any diffusers import to avoid ImportError.
import transformers.utils as _transformers_utils
if not hasattr(_transformers_utils, 'FLAX_WEIGHTS_NAME'):
    _transformers_utils.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'
if not hasattr(_transformers_utils, 'FLAX_WEIGHTS_INDEX_NAME'):
    _transformers_utils.FLAX_WEIGHTS_INDEX_NAME = 'flax_model.msgpack.index.json'

from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from transformers import AutoTokenizer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from dist import set_multi_gpus_devices, shard_model
from models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer,
                    WanT5EncoderModel)
from models import Wan2_2Transformer3DModel
from pipeline.pipeline_dreamxworld import Wan2_2_CameraPipeline
from utils.checkpoint_identity import compute_checkpoint_identity
from utils.fp8_optimization import (convert_model_weight_to_float8,
                                    convert_weight_dtype_wrapper,
                                    replace_parameters_by_name)
from utils.lora_utils import merge_lora, unmerge_lora
from utils.utils import filter_kwargs, get_image_to_video_latent, save_videos_grid
from utils.fm_solvers import FlowDPMSolverMultistepScheduler
from utils.fm_solvers import get_sampling_sigmas, retrieve_timesteps as fm_retrieve_timesteps
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.inference_utils import ActionToPoseFromID, GetPoseEmbedsFromPosesPrope
from utils.route_to_actions import (
    ACTION_FORWARD,
    action_speeds_from_turn_angles,
    evaluate_native_action_plan,
    fit_actions_global_motion_primitives,
    route_to_actions as route_to_actions_converter,
    route_to_actions_by_heading_delta,
    route_to_actions_by_heading_delta_subsegment,
    load_route_polyline_from_poses_csv,
)

# MapCtrl: optional Block-0 transformer attention intervention (lazy import so the
# baseline path has no hard dependency on the mapctrl_benchmark experiment package).
def _load_mapctrl_wrapper():
    from experiments.mapctrl_benchmark.mapctrl_inference_wrapper import MapCtrlInferenceWrapper
    return MapCtrlInferenceWrapper

# All available single-step actions
ALL_ACTIONS = ["w", "a", "d", "s", "j", "l", "i", "k",
               "wj", "wl", "wi", "wk", "dj", "sj", "sl"]


def load_map_utils():
    from utils.map_utils import semantic_map_to_rgb
    from utils.reactive_nav import load_label_map
    from utils.semantic_layout_condition import build_ego_semantic_layout
    return load_label_map, semantic_map_to_rgb, build_ego_semantic_layout


def load_reactive_nav_utils():
    from utils.reactive_nav import (
        DEFAULT_OBSTACLE_LABELS,
        DEFAULT_WALKABLE_LABELS,
        choose_reactive_actions,
        load_label_map,
        make_reactive_nav_state,
        parse_label_set,
    )
    return {
        "DEFAULT_OBSTACLE_LABELS": DEFAULT_OBSTACLE_LABELS,
        "DEFAULT_WALKABLE_LABELS": DEFAULT_WALKABLE_LABELS,
        "choose_reactive_actions": choose_reactive_actions,
        "load_label_map": load_label_map,
        "make_reactive_nav_state": make_reactive_nav_state,
        "parse_label_set": parse_label_set,
    }


def load_vlm_nav_utils():
    from utils.vlm_nav import (
        action_from_vlm_result,
        query_vlm_nav,
        vlm_nav_error_record,
    )
    return action_from_vlm_result, query_vlm_nav, vlm_nav_error_record


def print_info(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Long video generation via multi-segment inference")
    parser.add_argument("--config_path", type=str, default="configs/wan2.2/wan_ti2v_5b.yaml")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to eval.json; only the first entry is used")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="long_video.mp4")
    parser.add_argument("--target_duration", type=float, default=150.0,
                        help="Target duration in seconds")
    parser.add_argument("--sample_size", type=int, nargs=2, default=[704, 1280])
    parser.add_argument("--video_length", type=int, default=121,
                        help="Frames per segment (must satisfy 1+4k, e.g. 81, 121)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", type=str,
                        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    parser.add_argument("--cam_method", type=str, default="prope")
    parser.add_argument("--add_control_adapter", action="store_true", default=True)
    parser.add_argument("--sampler_name", type=str, default="Flow",
                        choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--weight_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--GPU_memory_mode", type=str, default=None)
    parser.add_argument("--cuda_mem_fraction", type=float, default=0.0,
                        help="If >0, clamp this process to that fraction of "
                             "GPU memory (simulates a smaller card, e.g. 0.27 "
                             "on 80GB ≈ a 24GB 4090).")
    parser.add_argument("--ground_lock_camera", action="store_true",
                        help="Keep camera motion on the horizontal ground plane and clamp pitch/roll drift.")
    parser.add_argument("--camera_pitch_min", type=float, default=-1.0,
                        help="Minimum camera pitch in degrees when --ground_lock_camera is enabled.")
    parser.add_argument("--camera_pitch_max", type=float, default=1.0,
                        help="Maximum camera pitch in degrees when --ground_lock_camera is enabled.")
    parser.add_argument("--map_path", type=str, default=None,
                        help="Fallback semantic label map PNG for every segment.")
    parser.add_argument("--map_projected_occupancy", type=str, default=None,
                        help="Image-projected semantic occupancy .npy [T,4,H,W] for the aligned map adapter.")
    parser.add_argument("--map_strength", type=float, default=0.0,
                        help="Map conditioning strength. 0 disables semantic map conditioning.")
    parser.add_argument("--map_layers", type=str, default="mid",
                        help="Map-conditioned transformer layers: all, early, mid, or comma-separated indices.")
    parser.add_argument("--map_inject_path", type=str, default=None,
                        help="Path to map image for VAE->DiT cross-attention injection (map-to-latent).")
    parser.add_argument("--map_inject_strength", type=float, default=1.0,
                        help="Strength scaling for map-to-latent injection tokens.")
    parser.add_argument("--semantic_map_baseline", type=str, default=None,
                        help="Path to .occ.npy semantic guide [F,4,44,80] for the CNN-encoder baseline "
                             "(direct structured map injection into cross-attention). ICLR comparison arm.")
    parser.add_argument("--map_guidance_path", type=str, default=None,
                        help="Path to map image for Spatial Map Guidance (SMG) — training-free multiplicative gating.")
    parser.add_argument("--map_guidance_layers", type=int, default=10,
                        help="Number of early DiT layers to apply SMG (default 10).")
    parser.add_argument("--map_guidance_strength", type=float, default=0.1,
                        help="Maximum SMG modulation strength (default 0.1 = 10%%).")
    parser.add_argument("--map_latent_path", type=str, default=None,
                        help="Path to map image for Semantic Map Latent (SML) — training-free additive vector conditioning.")
    parser.add_argument("--map_latent_layers", type=int, default=10,
                        help="Number of early DiT layers to apply SML (default 10).")
    parser.add_argument("--map_latent_strength", type=float, default=0.05,
                        help="Maximum SML additive strength (default 0.05 = 5%%).")

    # MapState adapter (dense_residual): trained residual injection via
    # MapStateAdapter (root mapstate.py). Consumed through the SAME map_latent
    # kwarg as SML — the two are mutually exclusive.
    parser.add_argument("--mapstate_path", type=str, default=None,
                        help="Path to the map PNG for MapState adapter conditioning (6-channel parse).")
    parser.add_argument("--mapstate_ckpt", type=str, default=None,
                        help="Path to MapState adapter checkpoint (e.g. experiments/mapstate/overfit_ckpts/mapstate_step_000500.pt).")
    parser.add_argument("--mapstate_strength", type=float, default=1.0,
                        help="MapState residual strength multiplier (default 1.0 = trained weights as-is).")
    parser.add_argument("--mapstate_crop_extent_px", type=float, default=128.0,
                        help="Ego crop extent in map pixels for MapState readout (default 128).")
    parser.add_argument("--mapstate_representation", type=str, default="s-f",
                        choices=["s-f", "s-l", "p-f", "p-l"],
                        help="Map representation: s-f=semantic-full(6ch), s-l=semantic-light(3ch), "
                             "p-f=proxy-full(3ch RGB), p-l=proxy-light(4ch RGB+depth). "
                             "Overrides ckpt config when set explicitly.")

    # MKI: Map-Key Injection — map-aware attention
    parser.add_argument("--map_kv_path", type=str, default=None,
                        help="Path to semantic map PNG for MKI (map-aware attention).")
    parser.add_argument("--map_kv_layers", type=int, default=10,
                        help="Number of early DiT layers for MKI injection (default 10).")
    parser.add_argument("--map_kv_key_strength", type=float, default=0.05,
                        help="Strength of map-key injection: changes attention pattern (default 0.05).")
    parser.add_argument("--map_kv_value_strength", type=float, default=0.05,
                        help="Strength of map-value injection: changes aggregated content (default 0.05).")
    parser.add_argument("--map_kv_segment_npys", type=str, default=None,
                        help="Comma-separated per-segment topology raster .npy paths for MKI. When set, "
                             "segment k uses entry k (last entry reused if fewer) so e.g. a route-visible "
                             "channel can advance with generation progress. Overrides --map_kv_path per segment.")
    # Layout guidance (Route B): sampling-time screen-occupancy steering via critic E
    parser.add_argument("--layout_guidance_occ", type=str, default=None,
                        help="Occupancy guide .npy [T,4,44,80] (channels road,water,building,vegetation). "
                             "Latent frames are taken as occ[0::4]; 0 disables guidance. "
                             "Comma-separated list: entry k becomes segment k's guide "
                             "(last entry reused if fewer) so chained segments each steer "
                             "toward their own progress slice of the route.")
    parser.add_argument("--layout_guidance_lambda", type=float, default=0.0,
                        help="Layout guidance strength: fraction of velocity RMS per denoise step (RMS-normalized).")
    parser.add_argument("--layout_guidance_critic", type=str, default=None,
                        help="Layout critic E checkpoint (experiments/mapstate/layout_critic.py LayoutCritic).")
    parser.add_argument("--layout_guidance_channel_map", type=str, default="0,2,3",
                        help="Comma-separated guide-npy channel indices feeding E channels "
                             "[road,building,vegetation]; sky is unclaimed and masked out of the loss.")
    parser.add_argument("--layout_guidance_start_step", type=int, default=0,
                        help="First denoise step (0-indexed) at which layout guidance applies.")
    parser.add_argument("--layout_guidance_end_step", type=int, default=-1,
                        help="Denoise step index AFTER the last step at which layout guidance applies; -1 = through the final step.")
    # ---- World canvas (read-before-write world memory, VML RQ6 revisit) ----
    # Re-indexes appearance memory from TIME (chained last frames) to WORLD
    # coordinates: after each segment, frozen SegFormer semantics of the saved
    # frames are splatted onto a persistent BEV grid; before the next segment,
    # observed canvas cells are projected into the next segment's frusta as a
    # screen-occupancy guide consumed by the same Route-B critic hook above.
    parser.add_argument("--world_canvas", action="store_true",
                        help="Enable the world-canvas read-before-write world memory. OFF (default) keeps "
                             "the pure time-chained baseline: identical seeds/config, no canvas read/write.")
    parser.add_argument("--world_canvas_dir", type=str, default=None,
                        help="Directory for world-canvas state (guide_<seg>.npz, canvas.npz, audit.jsonl). "
                             "Defaults to <output_dir>/world_canvas.")
    parser.add_argument("--world_canvas_lambda", type=float, default=0.2,
                        help="Canvas guide strength (RMS-normalized fraction of velocity), M1E optimum.")
    parser.add_argument("--world_canvas_start_step", type=int, default=0,
                        help="First denoise step (0-indexed) at which canvas guidance applies.")
    parser.add_argument("--world_canvas_end_step", type=int, default=8,
                        help="Denoise step index AFTER the last guided step; M1E early-window optimum 8.")
    parser.add_argument("--world_canvas_write_every", type=int, default=8,
                        help="Write path: extract semantics every N video frames per segment.")
    parser.add_argument("--world_canvas_memory", type=str, default="union",
                        choices=["union", "first", "latest"],
                        help="Canvas memory fusion: 'union' keeps every class ever observed at a cell "
                             "(frozen A/B behaviour); 'first' keeps the first visit's layout forever; "
                             "'latest' lets the most recent visit overwrite the cell.")
    parser.add_argument("--world_canvas_critic", type=str,
                        default="outputs/geomap/mapstate/critic_E/last.pt",
                        help="Layout critic E checkpoint for the canvas READ guide.")
    parser.add_argument("--map_adapter_path", type=str, default=None,
                        help="Trained spatially aligned semantic-layout adapter checkpoint. Required when map_strength > 0.")
    parser.add_argument("--map_layout_crop_size", type=int, nargs=2, default=(16, 16),
                        metavar=("HEIGHT", "WIDTH"), help="Ego semantic-layout crop token grid.")
    parser.add_argument("--map_layout_lateral_extent", type=float, default=96.0,
                        help="Half-width of the ego map crop in source-map pixels.")
    parser.add_argument("--map_layout_lookahead", type=float, default=160.0,
                        help="Forward extent of the ego map crop in source-map pixels.")
    parser.add_argument("--map_layout_rear_extent", type=float, default=24.0,
                        help="Rear extent of the ego map crop in source-map pixels.")
    parser.add_argument("--use_segment_context", action="store_true",
                        help="Append per-segment route/node context from input_json trajectory_segments.")
    parser.add_argument("--use_segment_start_overrides", action="store_true",
                        help="Use input_json segment_start_image_overrides for hard spatial anchor resets.")
    # ---- MapCtrl: Block-0 transformer attention intervention (opt-in) ----
    parser.add_argument("--mapctrl", action="store_true",
                        help="Enable MapCtrl Block-0 attention intervention guided by a binary road/obstacle map tensor.")
    parser.add_argument("--mapctrl_tensor", type=str, default="dynamic_custom_30s.npy",
                        help="Path to map tensor .npy [T,1,H,W] binary {0,1} road/obstacle mask.")
    parser.add_argument("--mapctrl_poses_csv", type=str, default="dynamic_custom_30s_poses.csv",
                        help="Path to poses CSV (frame,x,y,theta_rad,...) matching the map tensor length.")
    parser.add_argument("--mapctrl_total_video_frames", type=int, default=480,
                        help="Total video frames the map tensor spans (scaling reference for latent->map frame interp).")
    parser.add_argument("--mapctrl_intervention_mode", type=str, default="token_bias",
                        choices=["token_bias", "attention_penalty", "none"],
                        help="MapCtrl intervention mode. token_bias = contrastive main-token bias (default, gentle). "
                             "attention_penalty = injects road/obstacle bias BEFORE softmax (stronger spatial control).")
    parser.add_argument("--mapctrl_lambda_road", type=float, default=0.02,
                        help="Boost applied to road-masked tokens in token_bias mode.")
    parser.add_argument("--mapctrl_lambda_obs", type=float, default=0.05,
                        help="Penalty applied to obstacle/off-road tokens in token_bias mode.")
    parser.add_argument("--mapctrl_obstacle_penalty", type=float, default=-10.0,
                        help="attention_penalty mode: penalty added to attention logits when a road query "
                             "attends to an off-road key. More negative = stronger spatial separation. Try -5 to -20.")
    parser.add_argument("--mapctrl_attention_backend", type=str, default="manual",
                        choices=["manual", "sdpa"],
                        help="attention_penalty mode: backend for computing attention with the map bias.")
    parser.add_argument("--mapctrl_warmup_ratio", type=float, default=0.25,
                        help="Fraction of denoising steps (from high-noise start) during which MapCtrl intervenes (gamma(t)).")
    parser.add_argument("--mapctrl_boundary_ema_alpha", type=float, default=0.0,
                        help="Boundary EMA: blend each AR block's first-frame hidden state with the "
                             "previous block's cached tail (0=disabled). Smooths segment-to-segment "
                             "roadside building jumps in latent space. Try 0.2-0.4.")
    parser.add_argument("--mapctrl_boundary_ema_frames", type=int, default=1,
                        help="Number of frames at each AR block boundary to blend via boundary_ema_alpha.")
    parser.add_argument("--mapctrl_temporal_anneal_gain", type=float, default=0.0,
                        help="Temporal anneal: lambda_obs grows with frame index by this factor, "
                             "strengthening off-road suppression for later frames (keeps distant "
                             "buildings pinned to the map). 0=disabled. Try 0.5-1.0.")
    parser.add_argument("--mapctrl_mask_spatial_blur_kernel", type=int, default=0,
                        help="Spatial blur kernel for the road mask; softens the road/roadside "
                             "boundary so building suppression transitions gradually instead of a "
                             "hard cut at the road edge. 0=disabled. Try 3-5.")
    # ---- PSCAR: Pose-Projected Semantic Cross-Attention Routing ------------
    # Multi-class generalization of the POI cross-attn path. Each class gets its
    # own text-encoder context; image tokens are blended across class contexts
    # via softmax-normalized occupancy weights. Training-free: the frozen DiT
    # q/k/v/o are reused (see mapctrl_inference_wrapper.set_psc_route_context).
    parser.add_argument("--psc_route", action="store_true",
                        help="Enable PSCAR multi-class spatial routing. Implies --mapctrl. "
                             "Per-class uMT5 contexts are blended per image-token using the "
                             "occupancy weights from --psc_occupancy.")
    parser.add_argument("--psc_occupancy", type=str, default=None,
                        help="Path to the per-class image-plane occupancy .npy [T,C,H,W] in [0,1] "
                             "produced by tools/build_psc_condition.py.")
    parser.add_argument("--psc_class_meta", type=str, default=None,
                        help="Path to the psc_class_meta.json written by tools/build_psc_condition.py. "
                             "Provides class_names, class_texts (encoded to per-class contexts), and the "
                             "image-grid geometry.")
    parser.add_argument("--psc_target_blocks", type=str, default="0",
                        help="Comma-separated DiT block indices on which to install the PSCAR "
                             "cross-attention hook (e.g. '0' or '0,3,6').")
    parser.add_argument("--psc_tau", type=float, default=0.1,
                        help="Softmax temperature for the per-token class-mixing weights.")
    parser.add_argument("--psc_base_floor", type=float, default=0.05,
                        help="Minimum (pre-softmax) weight for the base class so empty cells fall "
                             "back to the base prompt context.")
    parser.add_argument("--psc_layer_strengths", type=str, default=None,
                        help="Optional comma-separated per-block interpolation strengths (one per "
                             "target block). 0 = no routing, 1 = full routed output. Defaults to 1.0 "
                             "for every target block.")
    parser.add_argument("--start_image_lock_frames", type=int, default=1,
                        help="Number of latent frames locked to the start_image at the head of "
                             "each segment (seg 0 excluded). Frame 0 is fully locked; subsequent "
                             "frames decay linearly to free. Suppresses cross-segment building "
                             "jumps. 1=original behavior (hard lock first frame only). Try 3-5.")
    parser.add_argument("--proxy_layout_guidance", action="store_true",
                        help="Opt-in training-free layout proxy latent guidance. Segment proxy frames are read from input_json proxy_layout_paths.")
    parser.add_argument("--proxy_layout_beta_schedule", type=str, default="0.25,0.15,0.08",
                        help="Comma-separated low-frequency residual strengths for --proxy_layout_guidance.")
    parser.add_argument("--proxy_layout_step_indices", type=str, default="2,6,10",
                        help="Comma-separated zero-based denoising step indices for --proxy_layout_guidance.")
    parser.add_argument("--proxy_layout_kernel_size", type=int, default=7,
                        help="Odd spatial low-pass kernel size used by --proxy_layout_guidance.")
    parser.add_argument("--scaffold_warm_start", action="store_true",
                        help="SDEdit warm start from the per-segment proxy_layout_paths video: "
                             "VAE-encode the proxy video and initialize latents at "
                             "--world_warm_step_index from its noised latents (full-coverage mask), "
                             "without requiring utils/world_warm_start. Mutually exclusive with "
                             "--proxy_layout_guidance (use one training-free mechanism per run).")
    parser.add_argument("--map_driven_camera", action="store_true",
                        help="Map-driven camera mode: derive the per-segment action_seq from the map route "
                             "polyline (loaded from --mapctrl_poses_csv or item['route_polyline']) instead of "
                             "using a manually-configured action_seq. Turn direction is read from the signed "
                             "angle between consecutive route segments, then handed to the existing "
                             "ActionToPoseFromID -> PRoPE pathway. Mutually exclusive with manual route "
                             "action_seq / segment_action_sequences (leakage guard).")
    parser.add_argument("--map_driven_route_polyline", type=str, default=None,
                        help="Optional explicit route polyline source for --map_driven_camera: a JSON file "
                             "holding {\"polyline\": [[x,y],...], \"progress\": [p0,...]} or a poses CSV. "
                             "Defaults to --mapctrl_poses_csv.")
    parser.add_argument("--map_driven_metadata", type=str, default=None,
                        help="Optional map trajectory metadata JSON. In global_fit mode this passes "
                             "event-remapped target frame progress and route event anchors into the "
                             "native motion-primitive fitter.")
    parser.add_argument("--map_driven_action_method", type=str, default="global_fit",
                        choices=["global_fit", "heading_delta", "nearest_vertex"],
                        help="How --map_driven_camera quantizes the map route into native DreamX actions. "
                             "global_fit uses beam search over native motion primitives; heading_delta is "
                             "the previous local accumulated-yaw quantizer.")
    parser.add_argument("--map_driven_beam_width", type=int, default=96,
                        help="Beam width for --map_driven_action_method global_fit.")
    parser.add_argument("--map_driven_turn_threshold_deg", type=float, default=10.0,
                        help="Minimum absolute per-segment yaw change, in degrees, required to emit wj/wl.")
    parser.add_argument("--map_driven_min_turn_speed", type=float, default=2.0,
                        help="Minimum DreamX action speed for map-derived turn segments.")
    parser.add_argument("--map_driven_max_turn_speed", type=float, default=5.2,
                        help="Maximum DreamX action speed for map-derived turn segments.")
    parser.add_argument("--map_driven_sub_segments", type=int, default=0,
                        help="Sub-segments per 5s diffusion segment for heading_delta (scheme A+B). "
                             "0 = off (one token per segment, current behavior). >0 emits that many "
                             "~1s sub-tokens per segment with real turn angles, so the pose model "
                             "resolves turns at finer time granularity. Only affects heading_delta.")
    parser.add_argument("--resume_existing_segments", action="store_true",
                        help="Skip existing segment mp4/last-frame pairs and continue the chain.")
    # ---- R1b world warm start (replay-instead-of-regenerate, segment granularity) ----
    parser.add_argument("--world_warm_start", action="store_true",
                        help="R1b: maintain a world index of earlier segments' rendered frames and "
                             "SDEdit-warm-start segments whose ground band is >= min_coverage covered, "
                             "running only the tail of the denoise schedule. Requires the planned "
                             "action chain to execute verbatim.")
    parser.add_argument("--world_warm_step_index", type=int, default=15,
                        help="Denoise step index s0 warm segments start from, chosen by sigma at "
                             "shift=3.0 (s0=15 -> sigma=0.500, 5 steps; s0=13 -> sigma=0.618, 7 steps).")
    parser.add_argument("--world_warm_tile_m", type=float, default=16.0,
                        help="World tile edge (metres) for the warm index keys.")
    parser.add_argument("--world_warm_min_coverage", type=float, default=0.5,
                        help="Mean ground-band coverage over anchors 1..30 required to warm a segment.")
    parser.add_argument("--world_warm_max_candidates", type=int, default=8,
                        help="Max source anchors (newest first) warped per target anchor.")
    parser.add_argument("--world_warm_camera_height_m", type=float, default=1.55)
    parser.add_argument("--world_warm_camera_pitch_deg", type=float, default=-5.0)
    parser.add_argument("--world_warm_anchor_release", type=float, default=0.0,
                        help="Anchor-release: stop the per-step re-lock once post-step sigma "
                             "drops below this (0 = lock all steps = prior clone at the "
                             "endpoint). ~0.2 leaves the final low-sigma steps free to "
                             "repaint texture over the anchored layout.")
    parser.add_argument("--world_warm_lowpass", type=int, default=4,
                        help="Baked-prior loader lowpass factor (area down/up before VAE "
                             "encode). 4 = area-average dust; 8 = aggressive de-speckle.")
    parser.add_argument("--world_warm_cold_control", action="store_true",
                        help="Knee attribution control: with --world_warm_start, zero the prior "
                             "weight so warm segments still run the truncated schedule "
                             "(timesteps[s0:]) but blend reduces to latents = sigma*eps. "
                             "Isolates schedule-length effects from prior-content effects.")
    parser.add_argument("--world_warm_render3d", type=str, default=None,
                        help="3D pivot brick 1: path to a fused voxel map npz "
                             "(tools/fuse_voxel_map.py over this route's bank anchors + COLMAP "
                             "known-pose MVS depth). The warm index then renders full-frame "
                             "geometrically-true novel-view priors (z-buffered voxel splat) "
                             "instead of the ground-plane warp composite. Content is seg-gated "
                             "(min contributing segment < current segment) for strictly-earlier "
                             "parity with the 2.5D arms.")
    parser.add_argument("--world_warm_anchor", action="store_true",
                        help="anchor10: per-step hard re-lock of covered pixels onto the prior's "
                             "flow-matching trajectory. After each scheduler.step, overwrite "
                             "covered pixels to (1-sigma_t)*prior + sigma_t*eps_fix, generalizing "
                             "the frame-0 start-image lock (measured perfect: 1022.7 vs bank "
                             "1018.4) to all covered pixels at every denoise step. Knee experiment "
                             "showed the prior is a RESCUER (warm10 >> cold10); anchoring prevents "
                             "short-schedule drift from destroying the rescued sharpness.")
    parser.add_argument("--world_latent_cache", action="store_true",
                        help="R1c-latent: maintain a world index of earlier segments' clean VAE "
                             "LATENTS (not pixels) keyed by (tile, view bin). On revisit, retrieve "
                             "the exact latent losslessly — no pixel warp, no VAE re-encode — "
                             "bypassing the 13-17 dB bilinear-warp ceiling that closed R1b. "
                             "Coverage is per-anchor binary (full-frame or nothing). Uses the same "
                             "tile/view-bin/s0 controls as --world_warm_start.")
    parser.add_argument("--save_segment_latents", type=str, default=None,
                        help="R1c pregeneration: save each segment's clean post-denoise latent to "
                             "this directory (seg_####.pt + poses.npz). Enables decode-only replay "
                             "via tools/replay_segment_latents.py at zero DiT cost.")
    # ---- Warp+Fill prior (MiDaS depth forward-warp of start_image) ----
    parser.add_argument("--warp_fill", action="store_true",
                        help="Warp+Fill: forward-warp each segment's start_image to all anchor "
                             "frames using MiDaS metric depth + the segment pose chain, producing "
                             "a full-coverage structural prior for SDEdit-style truncated DiT. "
                             "Works on ANY map/route (no town.pt needed). Combines with "
                             "--world_warm_step_index to control denoise truncation.")
    parser.add_argument("--warp_fill_step_index", type=int, default=15,
                        help="Denoise step index s0 for warp+fill warm start (default 15 = "
                             "sigma 0.500, 5 steps). Lower = more DiT correction of warp artifacts; "
                             "higher = faster but artifacts may persist.")
    parser.add_argument("--warp_fill_min_d", type=float, default=0.5,
                        help="Minimum metric depth (m) for warp+fill depth normalization.")
    parser.add_argument("--warp_fill_max_d", type=float, default=80.0,
                        help="Maximum metric depth (m) for warp+fill depth normalization.")
    parser.add_argument("--warp_fill_mode", type=str, default="chain",
                        choices=["chain", "single"],
                        help="Warp mode: 'chain' (autoregressive, re-estimate MiDaS depth at each "
                             "anchor; valid to ~2m+ forward motion) or 'single' (warp from "
                             "start_image only; valid to ~0.9m). Default: chain.")
    parser.add_argument("--warp_fill_min_seg_coverage", type=float, default=0.10,
                        help="Minimum mean valid-pixel fraction (excl. frame 0) for a segment to "
                             "use warp+fill warm-start. Below this, the segment runs full denoise "
                             "(no warm-start). Prevents truncated-denoise-with-invalid-prior failure.")
    parser.add_argument("--warp_fill_min_anchor_coverage", type=float, default=0.02,
                        help="Minimum valid-pixel fraction for an individual anchor to receive "
                             "warm-start weight. Below this, the anchor's coverage mask is zeroed "
                             "(runs cold). Prevents blending black/invalid priors at full weight.")
    parser.add_argument("--warp_fill_midas_skip", type=int, default=1,
                        help="In chain mode, re-estimate MiDaS depth every Nth anchor instead of "
                             "every anchor (default 1 = every anchor). N=3 cuts MiDaS calls by ~3× "
                             "with minimal quality impact (depth changes slowly over ~0.8m steps).")
    parser.add_argument("--warp_fill_no_cuda_warp", action="store_true",
                        help="Disable CUDA-accelerated forward warp, use numpy fallback. "
                             "CUDA warp is ~10-20× faster but may produce slightly different "
                             "results due to float32 vs float64.")
    parser.add_argument("--segment_range", type=str, default=None,
                        help="Two space-separated ints 'lo hi' (hi exclusive). When set, only "
                             "segments with seg_idx in [lo, hi) are generated; all others are "
                             "skipped. Used to shard a long run across parallel GPU groups.")
    parser.add_argument("--reactive_nav", action="store_true",
                        help="Rewrite segment actions from semantic maps so forward motion slows/turns before walls and stays over walkable ground.")
    parser.add_argument("--reactive_walkable_labels", type=str, default="2,5",
                        help="Comma-separated semantic labels treated as walkable for --reactive_nav. Default: road,plaza.")
    parser.add_argument("--reactive_obstacle_labels", type=str, default="1,3,4",
                        help="Comma-separated semantic labels treated as obstacles for --reactive_nav. Default: building,green,water.")
    parser.add_argument("--reactive_forward_clearance", type=float, default=0.48,
                        help="Minimum walkable ratio required in the forward look-ahead corridor.")
    parser.add_argument("--reactive_ground_clearance", type=float, default=0.62,
                        help="Minimum walkable ratio required in the lower ground contact window.")
    parser.add_argument("--reactive_turn_speed", type=float, default=2.8,
                        help="Speed used for avoidance turn actions. Larger means stronger yaw correction.")
    parser.add_argument("--reactive_creep_speed", type=float, default=0.32,
                        help="Slow forward speed used when the next path is marginal but not blocked.")
    parser.add_argument("--vlm_nav", action="store_true",
                        help="Use a local VLM to inspect the current frame and override the next segment action.")
    parser.add_argument("--vlm_url", type=str,
                        default="http://127.0.0.1:8071/v1/chat/completions",
                        help="OpenAI-compatible VLM chat completions URL for --vlm_nav.")
    parser.add_argument("--vlm_model", type=str,
                        default="/hpc2hdd/home/xhe989/Documents/ExGRPO/qwen/Qwen2.5-VL-72B-Instruct",
                        help="VLM model id for --vlm_nav.")
    parser.add_argument("--vlm_timeout", type=int, default=120,
                        help="Timeout in seconds for each VLM navigation call.")
    parser.add_argument("--vlm_min_confidence", type=float, default=0.25,
                        help="Keep the planned action when VLM confidence is below this threshold.")
    parser.add_argument("--vlm_audit_path", type=str, default=None,
                        help="JSONL path for VLM navigation decisions. Defaults to output_dir/vlm_nav_feedback.jsonl.")
    parser.add_argument("--vlm_nav_prompt", type=str, default="",
                        help="Optional custom prompt for VLM navigation frame checks.")
    parser.add_argument("--concat_skip_first_frames", type=int, default=0,
                        help="When stitching, drop this many initial frames from every segment after segment 0 to hide segment restart flashes.")
    parser.add_argument("--cache_prompt_embeds", action="store_true",
                        help="Cache repeated positive/negative T5 embeddings across segments.")
    parser.add_argument("--free_text_encoder", action="store_true",
                        help="Pre-warm the T5 prompt cache then move the text encoder to CPU, "
                             "freeing ~9 GB VRAM so fp8 DiT + VAE + 704x1280 activations fit "
                             "on a single 24 GB GPU with model_full_load_and_qfloat8. Requires "
                             "--cache_prompt_embeds and that all segments share one prompt "
                             "(no per-segment prefixes/suffixes, no --mapstate language arm, "
                             "no --psc_route).")
    parser.add_argument("--async_segment_write", action="store_true",
                        help="Write segment mp4 files on a rank-0 background thread while the next segment runs.")
    # ---- MapState: trained-adapter token residual hook (opt-in) --------------
    # Adds a per-token residual (MapStateV1 raster adapter or its LanguageStateV1
    # twin) inside one DiT attention block. Eval counterpart of the twin-adapter
    # training in experiments/mapstate/: map content enters ONLY through this
    # hook, so the caption can stay neutral. Single-process sequence layout only
    # (see --mapstate_adapter_path help).
    parser.add_argument("--mapstate_adapter_path", type=str, default=None,
                        help="Path to a trained adapter checkpoint (torch.save dict with "
                             "adapter_state_dict/format/step, as written by train_mapstate.py). "
                             "Enables the MapState residual hook. Requires single-process "
                             "sequence layout (--ulysses_degree 1 --ring_degree 1, one GPU): "
                             "sequence-parallel token shards are not supported and the run "
                             "aborts rather than silently degrading to the baseline.")
    parser.add_argument("--mapstate_adapter_format", type=str, default=None,
                        choices=["raster", "language"],
                        help="Adapter arm. Defaults to the checkpoint's recorded format.")
    parser.add_argument("--mapstate_map_png", type=str, default=None,
                        help="Illustrated map PNG for the route (raster masks and, for BOTH "
                             "arms, the camera pose track via build_map_pose_tracks).")
    parser.add_argument("--mapstate_map_meta", type=str, default=None,
                        help="Map metadata JSON sidecar for --mapstate_map_png.")
    parser.add_argument("--mapstate_text_file", type=str, default=None,
                        help="Plain-text file with the map-fact caption (language arm input). "
                             "Encoded with the pipeline's tokenizer/text_encoder exactly like "
                             "contract_dataset.encode_text.")
    parser.add_argument("--mapstate_block", type=int, default=0,
                        help="DiT block index whose forward gets the residual (default 0, "
                             "matching the training hook and PSCAR's target block).")
    # Multi-block injection: a second adapter on a different block. Each adapter
    # was trained independently at its own block; installing both combines their
    # complementary spatial effects (e.g. block 15 → chirality, block 20 → Fréchet).
    parser.add_argument("--mapstate_adapter_path_2", type=str, default=None,
                        help="Optional second adapter checkpoint for multi-block injection. "
                             "Installed on --mapstate_block_2. Shares the same map content "
                             "and pose track as the primary adapter.")
    parser.add_argument("--mapstate_block_2", type=int, default=None,
                        help="DiT block index for the second adapter (multi-block injection).")
    parser.add_argument("--mapstate_report", type=str, default=None,
                        help="Where rank 0 dumps the hook report JSON (applied_calls + events) "
                             "after generation. Default: <output_dir>/mapstate_report.json.")
    return parser.parse_args()


def resolve_map_layers(map_layers_str: str, num_layers: int = 30):
    s = map_layers_str.strip().lower()
    if s == "all":
        return list(range(num_layers))
    if s == "early":
        return list(range(0, num_layers // 2))
    if s == "mid":
        return list(range(8, min(24, num_layers)))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def setup_models(args, enable_map_conditioning=False, map_self_attn_layers=None):
    device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    if getattr(args, "cuda_mem_fraction", 0.0) > 0:
        # Simulate a smaller GPU (e.g. 0.27 on 80GB A800 ≈ a 24GB RTX
        # 4090): the process OOMs instead of borrowing past the budget, so a
        # clean run proves the configuration fits the target card.
        # Cap every visible device: under sequence parallelism the per-rank
        # current device is still cuda:0 at this point (NCCL init does not
        # set it), so capping only the current device would leave the other
        # ranks' GPUs uncapped and the simulation invalid.
        for dev_idx in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(args.cuda_mem_fraction, dev_idx)
        budget = (torch.cuda.get_device_properties(0).total_memory
                  * args.cuda_mem_fraction / 2**30)
        print_info(f" CUDA memory fraction {args.cuda_mem_fraction} "
                   f"(budget {budget:.1f} GiB per visible device)")
    config = OmegaConf.load(args.config_path)

    weight_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        args.weight_dtype, torch.float32)

    boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)
    config['transformer_additional_kwargs']['cam_method'] = args.cam_method
    config['transformer_additional_kwargs']['add_control_adapter'] = args.add_control_adapter
    config['transformer_additional_kwargs']['add_map_conditioning'] = enable_map_conditioning
    if enable_map_conditioning:
        config['transformer_additional_kwargs']['map_self_attn_layers'] = map_self_attn_layers
        print_info(f"🗺  Map conditioning enabled: strength={args.map_strength}, layers={map_self_attn_layers}")

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        args.transformer_path,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=weight_dtype,
    )
    if enable_map_conditioning:
        if not args.map_adapter_path:
            raise ValueError("--map_adapter_path is required when --map_strength is positive")
        metadata = transformer.load_semantic_layout_adapter(args.map_adapter_path)
        if dist.is_available() and dist.is_initialized():
            identity_holder = [
                compute_checkpoint_identity(args.transformer_path)
                if dist.get_rank() == 0
                else None
            ]
            dist.broadcast_object_list(identity_holder, src=0)
            base_checkpoint_identity = identity_holder[0]
        else:
            base_checkpoint_identity = compute_checkpoint_identity(args.transformer_path)
        if metadata.get("base_checkpoint_identity") != base_checkpoint_identity:
            raise ValueError(
                "semantic layout adapter was trained against a different base transformer checkpoint"
            )
        print_info(f"🗺  Loaded semantic layout adapter: {args.map_adapter_path} metadata={metadata}")

    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]

    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    Chosen_Scheduler = scheduler_dict[args.sampler_name]
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    pipeline = Wan2_2_CameraPipeline(
        transformer=transformer,
        transformer_2=None,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    if args.ulysses_degree > 1 or args.ring_degree > 1:
        transformer.enable_multi_gpus_inference()

    if args.GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if getattr(args, "free_text_encoder", False):
            # Don't load anything to GPU yet — T5 pre-warm needs the full VRAM
            # (T5 9.4GB + activations 6GB = 15.4GB, well within 24GB alone).
            # main() will pre-warm T5 on GPU, free it, then load DiT+VAE.
            pipeline.text_encoder.to(device)
        else:
            pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    return pipeline, device, weight_dtype, config, vae, boundary


def get_camera_sequence(action_seq, action_speed_list, video_length, sample_size,
                        ground_lock=False, pitch_min=-1.0, pitch_max=1.0,
                        yaw_deltas=None):
    duration = math.ceil(video_length / len(action_seq))
    pose_kwargs = {"duration": duration}
    signature = inspect.signature(ActionToPoseFromID)
    if "ground_lock" in signature.parameters:
        pose_kwargs.update({
            "ground_lock": ground_lock,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
        })
    if yaw_deltas is not None:
        pose_kwargs["yaw_deltas"] = yaw_deltas
    total_pose = ActionToPoseFromID(action_seq, action_speed_list, **pose_kwargs)
    total_pose = total_pose[:video_length]
    control_camera_video, _ = GetPoseEmbedsFromPosesPrope(
        total_pose, sample_size[0], sample_size[1], len(total_pose), False, 0
    )
    return control_camera_video


class MapDrivenCameraConflict(Exception):
    """Raised when map-driven camera mode is enabled together with a manually
    supplied route action sequence (leakage guard).

    Allowing both would silently let a hand-written action_seq encode the
    route turns, re-introducing the route-information-leakage confounder
    documented in ROUTE_INFORMATION_LEAKAGE.md. The guard fails loudly instead.
    """


def _enforce_map_driven_leakage_guard(args, item, segment_action_sequences):
    """Fail clearly if map-driven mode is on while manual route actions exist."""
    if not getattr(args, "map_driven_camera", False):
        return
    offenders = []
    manual_top = item.get("action_seq")
    if manual_top:
        offenders.append(f"item['action_seq']={manual_top}")
    if segment_action_sequences:
        offenders.append(f"segment_action_sequences={segment_action_sequences}")
    if offenders:
        raise MapDrivenCameraConflict(
            "Map-driven camera mode (--map_driven_camera) is enabled but a "
            "manual route action sequence was also supplied: "
            + "; ".join(offenders)
            + ". This would re-introduce the route-information-leakage "
            "confounder (ROUTE_INFORMATION_LEAKAGE.md): turns must be derived "
            "from the map route, not hand-authored. Remove the manual "
            "action_seq / segment_action_sequences from the input JSON to use "
            "map-driven camera control."
        )


def _load_route_polyline_for_map_driven(args, item):
    """Load the route polyline (and optional progress) for map-driven mode.

    Priority: explicit --map_driven_route_polyline file (JSON or CSV), then the
    input item's 'route_polyline' field, finally --mapctrl_poses_csv.
    """
    explicit = getattr(args, "map_driven_route_polyline", None)
    # An input-JSON field can name the polyline source file (JSON or CSV).
    item_source = item.get("route_polyline_source")
    if item_source and not explicit:
        explicit = item_source
    sources = []
    if explicit:
        sources.append(explicit)
    if item.get("route_polyline"):
        sources.append(None)  # handled inline below
    sources.append(args.mapctrl_poses_csv)

    # 1) explicit JSON file: {"polyline": [[x,y],...], "progress": [...]}
    if explicit and (explicit.endswith(".json") or explicit.endswith(".jsonl")):
        with open(explicit, "r") as f:
            data = json.load(f)
        poly = np.asarray(data["polyline"], dtype=np.float64)
        prog = data.get("progress")
        prog = np.asarray(prog, dtype=np.float64) if prog is not None else None
        return poly, prog

    # 2) item['route_polyline'] (inline coords)
    if item.get("route_polyline"):
        rp = item["route_polyline"]
        poly = np.asarray(rp, dtype=np.float64) if not isinstance(rp, dict) \
            else np.asarray(rp.get("polyline", rp), dtype=np.float64)
        prog = None
        if isinstance(rp, dict) and rp.get("progress") is not None:
            prog = np.asarray(rp["progress"], dtype=np.float64)
        return poly, prog

    # 3) poses CSV (default)
    csv_path = explicit if explicit else args.mapctrl_poses_csv
    return load_route_polyline_from_poses_csv(csv_path)


def build_map_driven_action_plan(args, item, num_segments, is_rank0_fn=None):
    """Derive the per-segment action plan from the map route polyline.

    Returns:
        (planned_actions, planned_speeds, route_result_dict)
        where planned_actions is a list of DreamX action tokens (one per
        segment) and planned_speeds reuses the configured action_speed_list
        (cycled to match). route_result_dict holds the logged turn sequence +
        normalized turn progress for auditing.
    """
    poly, prog = _load_route_polyline_for_map_driven(args, item)
    method = getattr(args, "map_driven_action_method", "global_fit")
    target_frame_progress = None
    route_events = None
    route_length_units = None
    if method == "global_fit" and getattr(args, "map_driven_metadata", None):
        from utils.map_trajectory import build_map_trajectory, load_metadata, settings_from_metadata
        from utils.route_vectorizer import build_route_polyline

        metadata = load_metadata(args.map_driven_metadata)
        route_mask = np.zeros(
            (
                int(metadata.get("route_image_height", 1)),
                int(metadata.get("route_image_width", 1)),
            ),
            dtype=bool,
        )
        if item.get("route_image_size"):
            route_mask = np.zeros(
                (
                    int(item["route_image_size"][1]),
                    int(item["route_image_size"][0]),
                ),
                dtype=bool,
            )
        route = build_route_polyline(
            poly,
            image_size=(
                int(item.get("route_image_size", [route_mask.shape[1], route_mask.shape[0]])[0]),
                int(item.get("route_image_size", [route_mask.shape[1], route_mask.shape[0]])[1]),
            ),
            route_mask=route_mask,
        )
        settings = settings_from_metadata(metadata)
        traj = build_map_trajectory(route, metadata, settings=settings)
        prog = route.progress
        target_frame_progress = traj.route_progress
        route_events = traj.events
        route_length_units = (
            int(num_segments)
            * float((item.get("action_speed_list") or [4.0])[0])
            * (int(getattr(args, "video_length", 81)) - 1)
            / float(getattr(args, "video_length", 81))
        )
    if method == "nearest_vertex":
        result = route_to_actions_converter(
            poly,
            num_segments,
            progress=prog,
            straight_threshold_rad=math.radians(float(args.map_driven_turn_threshold_deg)),
        )
    elif method == "heading_delta":
        sub_n = int(getattr(args, "map_driven_sub_segments", 0) or 0)
        if sub_n > 0:
            # Scheme A+B: finer within-segment time resolution + real angles.
            # Returns flat sub-token lists grouped per segment (list-of-lists).
            sub_actions, sub_angles, seg_offsets = route_to_actions_by_heading_delta_subsegment(
                poly,
                num_segments,
                sub_segments_per_segment=sub_n,
                progress=prog,
                straight_threshold_rad=math.radians(float(args.map_driven_turn_threshold_deg)),
            )
            # Group per segment -> list-of-lists.
            planned_actions = [
                list(sub_actions[seg_offsets[s]:seg_offsets[s + 1]])
                for s in range(num_segments)
            ]
            per_seg_angles = [
                list(sub_angles[seg_offsets[s]:seg_offsets[s + 1]])
                for s in range(num_segments)
            ]
            # Per-sub-token speeds via the same speed policy (real-angle magnitude).
            planned_speeds = [
                action_speeds_from_turn_angles(
                    planned_actions[s],
                    per_seg_angles[s],
                    straight_speed=float((item.get("action_speed_list") or [4.0])[0]),
                    min_turn_speed=float(args.map_driven_min_turn_speed),
                    max_turn_speed=float(args.map_driven_max_turn_speed),
                )
                for s in range(num_segments)
            ]
            if is_rank0_fn is None or is_rank0_fn():
                flat_actions = [t for seg in planned_actions for t in seg]
                flat_angles = [a for seg in per_seg_angles for a in seg]
                print_info(
                    f"🧭 Map-driven camera: derived {num_segments} segments x "
                    f"{sub_n} sub-segments ({len(flat_actions)} sub-tokens) from "
                    f"route polyline ({poly.shape[0]} pts) using heading_delta[A+B]:\n"
                    f"   actions={flat_actions}\n"
                    f"   turn_angles_deg={[round(a, 1) for a in flat_angles]}"
                )
            route_result_dict = {
                "mode": "map_driven",
                "method": "heading_delta",
                "sub_segments_per_segment": sub_n,
                "n_segments": num_segments,
                "actions_flat": [t for seg in planned_actions for t in seg],
                "turn_angles_deg_flat": [round(a, 4) for seg in per_seg_angles for a in seg],
                "segment_actions": planned_actions,
                "segment_speeds": [[round(float(s), 4) for s in seg] for seg in planned_speeds],
                "segment_turn_angles_deg": [[round(a, 4) for a in seg] for seg in per_seg_angles],
            }
            return planned_actions, planned_speeds, route_result_dict
        result = route_to_actions_by_heading_delta(
            poly,
            num_segments,
            progress=prog,
            straight_threshold_rad=math.radians(float(args.map_driven_turn_threshold_deg)),
        )
    else:
        # When called from inference, explicit target per-frame progress is not
        # available in the input JSON, so the fitter uses uniform progress over
        # the supplied route polyline. Offline evaluation tools pass the full
        # event-remapped progress sequence.
        result = fit_actions_global_motion_primitives(
            poly,
            num_segments,
            progress=prog,
            target_frame_progress=target_frame_progress,
            events=route_events,
            frames_per_segment=int(getattr(args, "video_length", 81)),
            straight_speed=float((item.get("action_speed_list") or [4.0])[0]),
            route_length_units=route_length_units,
            beam_width=int(getattr(args, "map_driven_beam_width", 96)),
        )

    base_speeds = item.get("action_speed_list", [4.0])
    straight_speed = float(base_speeds[0]) if base_speeds else 4.0
    min_turn_speed = float(args.map_driven_min_turn_speed)
    max_turn_speed = float(args.map_driven_max_turn_speed)
    if result.action_speed_list:
        planned_speeds = [float(speed) for speed in result.action_speed_list]
    else:
        planned_speeds = action_speeds_from_turn_angles(
            result.actions,
            result.turn_angles_deg,
            straight_speed=straight_speed,
            min_turn_speed=min_turn_speed,
            max_turn_speed=max_turn_speed,
        )
    if is_rank0_fn is None or is_rank0_fn():
        if method == "global_fit":
            metrics = result.metrics
            if not metrics and result.action_speed_list:
                metrics = evaluate_native_action_plan(
                    result.actions,
                    result.action_speed_list,
                    poly,
                    progress=prog,
                    target_frame_progress=target_frame_progress,
                    events=route_events,
                    frames_per_segment=int(getattr(args, "video_length", 81)),
                    route_length_units=route_length_units,
                )
                for key in list(metrics.keys()):
                    if key.startswith("_"):
                        metrics.pop(key, None)
                result.metrics.update(metrics)
        print_info(
            f"🧭 Map-driven camera: derived {result.n_segments} actions from "
            f"route polyline ({poly.shape[0]} pts) using "
            f"{method}: {result.actions}\n"
            f"   speeds={[round(float(s), 2) for s in planned_speeds]}\n"
            f"   turn_sequence={result.turn_sequence}\n"
            f"   turn_angles_deg={[round(a, 1) for a in result.turn_angles_deg]}\n"
            f"   normalized_progress={[round(p, 3) for p in result.progress]}"
        )
    return list(result.actions), planned_speeds, result.to_dict()


def build_segment_semantic_layout(
    map_path,
    sample_size,
    route_polyline,
    seg_idx,
    num_segments,
    num_latent_frames,
    *,
    route_vertex_progress=None,
    crop_size=(16, 16),
    lateral_extent=96.0,
    lookahead=160.0,
    rear_extent=24.0,
):
    load_label_map, _, build_ego_semantic_layout = load_map_utils()
    label_map = load_label_map(map_path, sample_size)
    start = float(seg_idx) / float(num_segments)
    end = float(seg_idx + 1) / float(num_segments)
    progress = torch.linspace(start, end, int(num_latent_frames), dtype=torch.float32)
    condition = build_ego_semantic_layout(
        torch.from_numpy(label_map),
        torch.as_tensor(route_polyline, dtype=torch.float32),
        progress,
        route_vertex_progress=(
            None
            if route_vertex_progress is None
            else torch.as_tensor(route_vertex_progress, dtype=torch.float32)
        ),
        crop_size=crop_size,
        lateral_extent=float(lateral_extent),
        lookahead=float(lookahead),
        rear_extent=float(rear_extent),
    )
    return {
        "labels": condition.labels.cpu(),
        "progress": condition.route_progress.cpu(),
    }


def build_segment_projected_layout(path, seg_idx, video_length, num_latent_frames, sample_size):
    from utils.semantic_layout_condition import load_projected_semantic_occupancy

    patch_grid = (int(sample_size[0]) // 16 // 2, int(sample_size[1]) // 16 // 2)
    return load_projected_semantic_occupancy(
        path,
        segment_index=seg_idx,
        pixel_frames_per_segment=video_length,
        num_latent_frames=num_latent_frames,
        target_size=patch_grid,
    )


def normalize_indexed_paths(value):
    if not value:
        return {}
    if isinstance(value, dict):
        return {int(k): v for k, v in value.items()}
    if isinstance(value, list):
        return {idx: path for idx, path in enumerate(value) if path}
    raise TypeError("Expected indexed paths as dict or list")


def parse_number_list(value, *, cast=float):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def spatial_lowpass_avg(latent, *, kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError(f"proxy layout kernel size must be a positive odd integer, got {kernel_size}")
    original_shape = tuple(latent.shape)
    height, width = int(latent.shape[-2]), int(latent.shape[-1])
    work = latent.reshape(-1, 1, height, width)
    pad = kernel_size // 2
    if min(height, width) > pad:
        work = F.pad(work, (pad, pad, pad, pad), mode="reflect")
        out = F.avg_pool2d(work, kernel_size=kernel_size, stride=1)
    else:
        out = F.avg_pool2d(work, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False)
    return out.reshape(original_shape).to(device=latent.device, dtype=latent.dtype)


def prepare_proxy_layout_latents(pipeline, proxy_layout_path, args, video_length, device, weight_dtype, generator):
    _proxy_path = str(proxy_layout_path)
    if os.path.splitext(_proxy_path)[1].lower() in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
        # get_image_to_video_latent only opens stills (PIL); for a per-segment
        # proxy VIDEO, decode frames here into the same [1,C,F,H,W] float 0..1
        # tensor that the still path returns downstream (the mask is unused).
        import imageio.v2 as imageio
        _reader = imageio.get_reader(_proxy_path)
        _frames = [_f for _f in _reader]
        _reader.close()
        if not _frames:
            raise ValueError(f"proxy layout video has no decodable frames: {_proxy_path}")
        _idx = np.linspace(0, len(_frames) - 1, video_length).round().astype(np.int64)
        _picked = np.stack([np.asarray(_frames[int(_i)]) for _i in _idx])  # [F,H,W,3] uint8
        proxy_video = torch.from_numpy(_picked).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    else:
        proxy_video, _, _ = get_image_to_video_latent(
            _proxy_path,
            None,
            video_length=video_length,
            sample_size=args.sample_size,
        )
    proxy_video = pipeline.image_processor.preprocess(
        rearrange(proxy_video, "b c f h w -> (b f) c h w"),
        height=args.sample_size[0],
        width=args.sample_size[1],
    )
    proxy_video = proxy_video.to(dtype=torch.float32)
    proxy_video = rearrange(proxy_video, "(b f) c h w -> b c f h w", f=video_length)
    proxy_latents = pipeline.prepare_control_latents(
        None,
        proxy_video,
        batch_size=1,
        height=args.sample_size[0],
        width=args.sample_size[1],
        dtype=weight_dtype,
        device=device,
        generator=generator,
        do_classifier_free_guidance=args.guidance_scale > 1.0,
    )[1]
    return proxy_latents.detach()


def resolve_pipeline_execution_device(pipeline):
    device = getattr(pipeline, "_execution_device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)
    for module_name in ("vae", "transformer", "text_encoder"):
        module = getattr(pipeline, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            if str(param.device) != "meta":
                return param.device
    return torch.device("cpu")


def save_last_frame(video_tensor, path):
    """Save last frame of video tensor [B, C, T, H, W] as PNG file, return path."""
    frame = video_tensor[0, :, -1, :, :]  # [C, H, W]
    frame = (frame * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(frame).save(path)
    return path


def is_rank0():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def image_file_is_readable(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def distributed_barrier(device=None):
    if not (dist.is_available() and dist.is_initialized()):
        return
    if isinstance(device, torch.device) and device.type == "cuda":
        dist.barrier(device_ids=[device.index or 0])
    else:
        dist.barrier()


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_json_with_wait(path, timeout=600.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {path}")


def apply_vlm_nav_feedback(args, current_start_image, seg_idx,
                           action_seq, action_speed_list, audit_path, decision_dir):
    action_from_vlm_result, query_vlm_nav, vlm_nav_error_record = load_vlm_nav_utils()
    planned_actions = list(action_seq)
    planned_speeds = list(action_speed_list)
    decision_path = os.path.join(decision_dir, f"vlm_decision_{seg_idx:04d}.json")

    if is_rank0():
        record = {
            "segment": seg_idx,
            "image_path": str(current_start_image),
            "planned_actions": planned_actions,
            "planned_speeds": planned_speeds,
            "actions": planned_actions,
            "speeds": planned_speeds,
            "decision": "vlm_not_run",
        }
        actions = planned_actions
        speeds = planned_speeds

        if not isinstance(current_start_image, str) or not os.path.exists(current_start_image):
            record["decision"] = "vlm_skipped_missing_image"
        else:
            try:
                vlm_result = query_vlm_nav(
                    current_start_image,
                    url=args.vlm_url,
                    model=args.vlm_model,
                    prompt=args.vlm_nav_prompt or None,
                    timeout=args.vlm_timeout,
                )
            except Exception as exc:
                vlm_result = vlm_nav_error_record(current_start_image, exc)

            record["vlm"] = {
                key: value for key, value in vlm_result.items()
                if key != "_raw"
            }
            if vlm_result.get("error"):
                record["decision"] = "vlm_error_keep_planned"
            elif float(vlm_result.get("confidence", 0.0)) < float(args.vlm_min_confidence):
                record["decision"] = "vlm_low_confidence_keep_planned"
            else:
                base_speed = float(planned_speeds[0]) if planned_speeds else float(args.reactive_creep_speed)
                actions, speeds, decision = action_from_vlm_result(
                    vlm_result,
                    base_speed=base_speed,
                    creep_speed=args.reactive_creep_speed,
                    turn_speed=args.reactive_turn_speed,
                )
                record.update({
                    "actions": actions,
                    "speeds": speeds,
                    "decision": decision,
                })

        append_jsonl(audit_path, record)
        print_info(
            "  vlm_nav: "
            f"decision={record['decision']} "
            f"actions={record['actions']} "
            f"ground={record.get('vlm', {}).get('lower_half_ground_ratio', '?')} "
            f"wall={record.get('vlm', {}).get('wall_or_building_too_close', '?')} "
            f"fly={record.get('vlm', {}).get('camera_elevated_or_flying', '?')}"
        )
        with open(decision_path, "w") as f:
            json.dump(record, f, indent=2)

    record = read_json_with_wait(decision_path)
    return record["actions"], record["speeds"], record


def compute_scheduler_timesteps(pipeline, num_inference_steps, shift, device):
    """Mirror pipeline_dreamxworld.Wan2_2_CameraPipeline.__call__'s timestep prep so
    MapCtrl can register the real denoising schedule for gamma(t) time-truncation.
    Returns a 1-D float tensor of timesteps (length == num_inference_steps)."""
    scheduler = pipeline.scheduler
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        timesteps, _ = fm_retrieve_timesteps(scheduler, num_inference_steps, device, None, mu=1)
    elif isinstance(scheduler, FlowUniPCMultistepScheduler):
        scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps
    elif isinstance(scheduler, FlowDPMSolverMultistepScheduler):
        sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
        timesteps, _ = fm_retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
    else:
        timesteps, _ = fm_retrieve_timesteps(scheduler, num_inference_steps, device, None)
    return torch.as_tensor(timesteps, device=device, dtype=torch.float32).flatten()


def run_segment(pipeline, args, prompt, start_image, action_seq, action_speed_list,
                vae, boundary, seg_idx, control_map=None, mapctrl=None,
                video_window=None, latent_frames=None, proxy_layout_path=None,
                yaw_deltas=None, warm_start_pack=None, warm_seg_info=None):
    video_length = args.video_length
    video_length = (
        int((video_length - 1) // vae.config.temporal_compression_ratio
            * vae.config.temporal_compression_ratio) + 1
        if video_length != 1 else 1
    )

    control_camera_video = get_camera_sequence(
        action_seq,
        action_speed_list,
        video_length,
        args.sample_size,
        ground_lock=args.ground_lock_camera,
        pitch_min=args.camera_pitch_min,
        pitch_max=args.camera_pitch_max,
        yaw_deltas=yaw_deltas,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed + seg_idx)

    with torch.no_grad():
        callback_chain = []
        pipeline_kwargs = {
            "num_frames": video_length,
            "negative_prompt": args.negative_prompt,
            "height": args.sample_size[0],
            "width": args.sample_size[1],
            "generator": generator,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "start_image": start_image,
            "control_camera_video": control_camera_video,
            "boundary": boundary,
            "shift": args.shift,
            "cache_prompt_embeds": args.cache_prompt_embeds,
        }
        if control_map is not None:
            if "features" in control_map:
                pipeline_kwargs["semantic_layout_features"] = control_map["features"]
            else:
                pipeline_kwargs["semantic_layout_labels"] = control_map["labels"]
            pipeline_kwargs["semantic_layout_progress"] = control_map["progress"]
            pipeline_kwargs["map_strength"] = args.map_strength
        # Only segments chained from a previous tail frame (seg_idx > 0) benefit from
        # multi-frame start_image locking; seg 0 uses the original 1-frame lock.
        if seg_idx > 0 and args.start_image_lock_frames > 1:
            pipeline_kwargs["start_image_lock_frames"] = args.start_image_lock_frames

        # R1c-latent: prior is already in latent space (no VAE encode needed).
        # The LC pack has key "warm_latent" [1,C,T_lat,H_lat,W_lat] + binary
        # "coverage_mask" [1,1,T_lat,H_lat,W_lat].  Pass directly to pipeline.
        if warm_start_pack is not None and "warm_latent" in warm_start_pack:
            _ws_dev = resolve_pipeline_execution_device(pipeline)
            _ws_dtype = pipeline.text_encoder.dtype
            _warm_lat = warm_start_pack["warm_latent"].to(
                device=_ws_dev, dtype=_ws_dtype)
            _wmask = warm_start_pack["coverage_mask"].to(
                device=_ws_dev, dtype=_ws_dtype)
            if getattr(args, "world_warm_cold_control", False):
                _wmask = torch.zeros_like(_wmask)
            pipeline_kwargs["warm_start_latents"] = _warm_lat
            pipeline_kwargs["warm_start_mask"] = _wmask
            pipeline_kwargs["warm_start_step_index"] = int(args.world_warm_step_index)
            pipeline_kwargs["warm_start_info"] = {
                "seg_idx": int(seg_idx),
                "coverage": float(warm_start_pack["coverage"]),
            }
            if getattr(args, "world_warm_anchor", False):
                pipeline_kwargs["world_warm_anchor"] = True
                setattr(pipeline, "_warm_anchor_release",
                        float(getattr(args, "world_warm_anchor_release", 0.0)))
            print_info(f"  [world_latent] seg {seg_idx}: warm-latent inject "
                       f"(coverage {float(warm_start_pack['coverage']):.3f}, "
                       f"s0={int(args.world_warm_step_index)})")
        # R1b world warm start: expand the 31 latent-anchor composite frames to
        # the causal VAE pixel layout [a0, a1 x4, a2 x4, ...] and encode with the
        # SAME convention as start_image_latentes (prepare_control_latents[1],
        # input range [-1, 1], vae.encode().mode()). Uncovered ground pixels are
        # black; the latent coverage mask zeroes them inside the pipeline blend.
        elif warm_start_pack is not None:
            _ws_t0 = time.time()
            _ws_dev = resolve_pipeline_execution_device(pipeline)
            _ws_dtype = pipeline.text_encoder.dtype
            _anchors = torch.from_numpy(
                np.ascontiguousarray(warm_start_pack["anchors"])).float() / 255.0  # [F_a,H,W,3] in [0,1]
            _rep = [0] + [g for g in range(1, _anchors.shape[0]) for _ in range(4)]
            _comp = _anchors[_rep]  # [T,H,W,3] causal layout, T = 4*(F_a-1)+1
            _comp = rearrange(_comp, "f h w c -> 1 c f h w") * 2.0 - 1.0
            _warm_lat = pipeline.prepare_control_latents(
                None, _comp, 1, args.sample_size[0], args.sample_size[1],
                _ws_dtype, _ws_dev, generator, True,
            )[1]
            _wmask = torch.from_numpy(
                np.ascontiguousarray(warm_start_pack["coverage_latent"]))
            _wmask = _wmask.to(device=_ws_dev, dtype=_ws_dtype).unsqueeze(0).unsqueeze(0)
            if getattr(args, "world_warm_cold_control", False):
                # Knee control arm: keep the truncated schedule (pipeline slices
                # timesteps[s0:]) but zero the prior weight, so the SDEdit blend
                # in the pipeline reduces to latents = sigma * eps.
                _wmask = torch.zeros_like(_wmask)
            pipeline_kwargs["warm_start_latents"] = _warm_lat
            pipeline_kwargs["warm_start_mask"] = _wmask
            pipeline_kwargs["warm_start_step_index"] = int(args.world_warm_step_index)
            pipeline_kwargs["warm_start_info"] = {
                "seg_idx": int(seg_idx),
                "coverage": float(warm_start_pack["coverage"]),
            }
            if getattr(args, "world_warm_anchor", False):
                pipeline_kwargs["world_warm_anchor"] = True
                setattr(pipeline, "_warm_anchor_release",
                        float(getattr(args, "world_warm_anchor_release", 0.0)))
            print_info(f"  [world_warm] seg {seg_idx}: warm encode {time.time()-_ws_t0:.1f}s "
                       f"(coverage {float(warm_start_pack['coverage']):.3f}, "
                       f"s0={int(args.world_warm_step_index)})")
        elif warm_seg_info is not None:
            # Cold segment under --world_warm_start: pass seg metadata anyway so
            # [world_warm_timing] logs the true segment index instead of -1.
            pipeline_kwargs["warm_start_info"] = warm_seg_info

        if getattr(args, "proxy_layout_guidance", False) and proxy_layout_path:
            proxy_path = str(proxy_layout_path)
            proxy_device = resolve_pipeline_execution_device(pipeline)
            proxy_dtype = next(pipeline.vae.parameters()).dtype
            proxy_latents = prepare_proxy_layout_latents(
                pipeline,
                proxy_path,
                args,
                video_length,
                proxy_device,
                proxy_dtype,
                generator,
            )
            beta_values = parse_number_list(args.proxy_layout_beta_schedule, cast=float)
            step_indices = parse_number_list(args.proxy_layout_step_indices, cast=int)
            if len(beta_values) != len(step_indices):
                raise ValueError(
                    "--proxy_layout_beta_schedule and --proxy_layout_step_indices must have the same length"
                )
            beta_by_step = {int(step): float(beta) for step, beta in zip(step_indices, beta_values)}
            kernel_size = int(args.proxy_layout_kernel_size)

            def _proxy_layout_step_cb(_pipe, step_index, _t, _kwargs):
                if int(step_index) not in beta_by_step:
                    return _kwargs
                latents = _kwargs.get("latents")
                if latents is None:
                    return _kwargs
                proxy = proxy_latents.to(device=latents.device, dtype=latents.dtype)
                if tuple(proxy.shape) != tuple(latents.shape):
                    raise ValueError(
                        f"proxy layout latent shape mismatch: proxy={tuple(proxy.shape)} latents={tuple(latents.shape)}"
                    )
                residual = spatial_lowpass_avg(proxy - latents, kernel_size=kernel_size)
                denoise_mask = torch.ones_like(latents)
                denoise_mask[:, :, 0:1] = 0
                guided = latents + beta_by_step[int(step_index)] * residual * denoise_mask
                if not torch.isfinite(guided).all():
                    raise ValueError("proxy layout guidance produced NaN/Inf latents")
                _kwargs["latents"] = guided.to(device=latents.device, dtype=latents.dtype)
                return _kwargs

            callback_chain.append(_proxy_layout_step_cb)
            print_info(
                f"  proxy_layout_guidance: {proxy_path} "
                f"steps={step_indices} beta={beta_values} kernel={kernel_size}"
            )

        if mapctrl is not None and video_window is not None:
            # Map this segment's latent frames onto a slice of the map tensor timeline.
            mapctrl.set_active_video_window(
                video_start=int(video_window[0]),
                video_end=int(video_window[1]),
                latent_frames=int(latent_frames if latent_frames is not None else video_length),
            )
            # Register the real denoising schedule so gamma(t) time-truncation works.
            # Under sequential_cpu_offload the transformer params are meta tensors, so we
            # build the schedule on CPU (the *values* are all gamma needs). This is a
            # best-effort probe; the callback below drives the ratio directly from the
            # diffusers step_index, which is exact and never touches a tensor device.
            try:
                seg_timesteps = compute_scheduler_timesteps(
                    pipeline, args.num_inference_steps, args.shift, torch.device("cpu")
                )
                mapctrl.set_active_denoising_schedule(seg_timesteps)
            except Exception as exc:  # keep generation running if schedule probing fails
                print_info(f"  ⚠️  MapCtrl schedule probe failed ({exc}); gamma(t) will be driven by step_index.")
            # The callback receives (pipeline, step_index, t, kwargs) per diffusers convention.
            # step_index is a plain int (0-based, 0 = highest noise); use it to set the
            # denoising step ratio exactly, independent of any (possibly meta) tensor device.
            _mapctrl_total_steps = int(args.num_inference_steps)

            def _mapctrl_step_cb(_pipe, _step_index, t, _kwargs):
                try:
                    mapctrl.set_denoising_step_by_index(int(_step_index), _mapctrl_total_steps)
                except Exception:
                    mapctrl._update_denoising_state_from_timestep(t)
                return _kwargs
            callback_chain.append(_mapctrl_step_cb)

        if callback_chain:
            def _combined_step_cb(_pipe, _step_index, t, _kwargs):
                for callback in callback_chain:
                    _kwargs = callback(_pipe, _step_index, t, _kwargs)
                return _kwargs

            pipeline_kwargs["callback_on_step_end"] = _combined_step_cb

        if mapctrl is not None and video_window is not None:
            with mapctrl.installed():
                sample = pipeline(prompt, **pipeline_kwargs).videos
        else:
            sample = pipeline(prompt, **pipeline_kwargs).videos

    return sample  # [B, C, T, H, W] in [0,1]


def concat_videos_ffmpeg(segment_paths, output_path, fps, skip_first_frames=0):
    """Concatenate video files. Uses ffmpeg if available, falls back to imageio."""
    skip_first_frames = max(0, int(skip_first_frames))
    if skip_first_frames > 0:
        return concat_videos_trimmed(segment_paths, output_path, fps, skip_first_frames)

    # Try ffmpeg first (lossless copy, fast)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        # Search common conda/system locations
        for candidate in [
            "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg",
            os.path.expanduser("~/miniconda3/envs/wonderjourney/bin/ffmpeg"),
            os.path.expanduser("~/miniconda3/envs/wonderjourney_clean/bin/ffmpeg"),
            os.path.expanduser("~/miniconda3/envs/dreamx/bin/ffmpeg"),
            os.path.expanduser("~/miniconda3/bin/ffmpeg"),
        ]:
            if os.path.isfile(candidate):
                ffmpeg_bin = candidate
                break

    if ffmpeg_bin:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for p in segment_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
            list_file = f.name
        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
                 "-i", list_file, "-c", "copy", output_path],
                check=True
            )
            return
        except subprocess.CalledProcessError:
            pass
        finally:
            os.unlink(list_file)

    # Fallback: frame-by-frame via imageio (slower but always works)
    print("ffmpeg not found, using imageio fallback for concatenation...")
    import imageio
    writer = None
    for seg in segment_paths:
        reader = imageio.get_reader(seg)
        if writer is None:
            writer = imageio.get_writer(output_path, fps=fps, quality=8,
                                        macro_block_size=None)
        for frame in reader:
            writer.append_data(frame)
        reader.close()
    if writer:
        writer.close()


def concat_videos_trimmed(segment_paths, output_path, fps, skip_first_frames):
    """Concatenate by re-encoding while dropping restart frames after segment 0."""
    import imageio

    writer = imageio.get_writer(output_path, fps=fps, quality=8, macro_block_size=None)
    try:
        for seg_idx, seg in enumerate(segment_paths):
            reader = imageio.get_reader(seg)
            try:
                for frame_idx, frame in enumerate(reader):
                    if seg_idx > 0 and frame_idx < skip_first_frames:
                        continue
                    writer.append_data(frame)
            finally:
                reader.close()
    finally:
        writer.close()


def _world_canvas_write_segment(args, canvas, extractor, seg_idx, seg_path,
                                global_c2w, audit_path):
    """Rank-0 WRITE path: frozen SegFormer on the saved segment -> canvas splat.

    Returns the write audit record (or a skip record when the extractor or the
    video is unavailable). The canvas is then persisted so --resume_existing_segments
    and post-run analysis both see the same world state.
    """
    import json as _json

    try:
        frame_semantics = extractor.segment_id_maps(
            seg_path, every=int(args.world_canvas_write_every)
        )
        rec = canvas.write_segment(seg_idx, global_c2w, frame_semantics)
    except Exception as exc:  # extractor failure must not kill the run
        print_info(f"  [canvas] write FAILED for segment {seg_idx}: {exc}")
        rec = {"seg_idx": int(seg_idx), "error": repr(exc)}
    canvas_path = os.path.join(_world_canvas_dir_of(args), "canvas.npz")
    canvas.save(canvas_path)
    try:
        with open(audit_path, "a") as f:
            f.write(_json.dumps({"event": "write", **rec}) + "\n")
    except Exception:
        pass
    return rec


def _world_canvas_dir_of(args):
    base = getattr(args, "world_canvas_dir", None) or os.path.join(args.output_dir, "world_canvas")
    os.makedirs(base, exist_ok=True)
    return base


def main():
    args = parse_args()

    # --psc_route implies --mapctrl (PSCAR rides on the MapCtrl wrapper's hook
    # installation machinery; the wrapper API set_psc_route_context is what
    # distinguishes PSCAR from the plain token-bias path).
    if args.psc_route and not args.mapctrl:
        args.mapctrl = True
        print_info("🧭 --psc_route set: implicitly enabling --mapctrl (PSCAR rides on the MapCtrl wrapper).")

    print_info("=" * 80)
    print_info("🎬 Long Video Generation (Multi-Segment)")
    print_info("=" * 80)

    with open(args.input_json, 'r') as f:
        items = json.load(f)

    item = items[0]
    start_image_path = item['image_path']
    prompt = item['caption'].replace('\n', ' ').replace('\r', ' ').strip()
    original_action_seq = item.get('action_seq', ['w'])
    original_action_speed_list = item.get('action_speed_list', [4])
    trajectory_segments = item.get('trajectory_segments', [])
    segment_map_paths = normalize_indexed_paths(item.get('segment_map_paths'))
    proxy_layout_paths = normalize_indexed_paths(item.get('proxy_layout_paths'))
    segment_start_overrides = normalize_indexed_paths(item.get('segment_start_image_overrides'))
    segment_action_sequences = item.get('segment_action_sequences') or {}
    segment_action_speed_lists = item.get('segment_action_speed_lists') or {}
    if isinstance(segment_action_sequences, list):
        segment_action_sequences = {idx: seq for idx, seq in enumerate(segment_action_sequences) if seq}
    else:
        segment_action_sequences = {int(k): v for k, v in segment_action_sequences.items()} if segment_action_sequences else {}
    if isinstance(segment_action_speed_lists, list):
        segment_action_speed_lists = {idx: speeds for idx, speeds in enumerate(segment_action_speed_lists) if speeds}
    else:
        segment_action_speed_lists = {int(k): v for k, v in segment_action_speed_lists.items()} if segment_action_speed_lists else {}
    segment_prompt_suffixes = item.get('segment_prompt_suffixes', {})
    if isinstance(segment_prompt_suffixes, list):
        segment_prompt_suffixes = {idx: suffix for idx, suffix in enumerate(segment_prompt_suffixes) if suffix}
    else:
        segment_prompt_suffixes = {int(k): v for k, v in segment_prompt_suffixes.items()} if segment_prompt_suffixes else {}
    segment_prompt_prefixes = item.get('segment_prompt_prefixes', {})
    if isinstance(segment_prompt_prefixes, list):
        segment_prompt_prefixes = {idx: pfx for idx, pfx in enumerate(segment_prompt_prefixes) if pfx}
    else:
        segment_prompt_prefixes = {int(k): v for k, v in segment_prompt_prefixes.items()} if segment_prompt_prefixes else {}

    segment_duration = (args.video_length - 1) / args.fps  # seconds per segment
    num_segments = math.ceil(args.target_duration / segment_duration)
    map_path = args.map_path or item.get('map_path')
    map_projected_occupancy = args.map_projected_occupancy or item.get('map_projected_occupancy')
    map_strength = float(args.map_strength)
    if not math.isfinite(map_strength) or map_strength < 0:
        raise ValueError("--map_strength must be a finite non-negative value")
    has_map_source = bool(map_path or segment_map_paths or map_projected_occupancy)
    if map_strength > 0 and not has_map_source:
        raise ValueError("--map_strength > 0 requires a semantic map or --map_projected_occupancy")
    enable_map = map_strength > 0
    if enable_map:
        missing_map_segments = [] if map_projected_occupancy else [
            seg_idx
            for seg_idx in range(num_segments)
            if not segment_map_paths.get(seg_idx, map_path)
        ]
        if missing_map_segments:
            raise ValueError(
                "semantic layout conditioning has no map for segments "
                + ", ".join(str(seg_idx) for seg_idx in missing_map_segments)
            )
    load_label_map = load_map_utils()[0] if (args.reactive_nav or enable_map) else None
    reactive_utils = load_reactive_nav_utils() if args.reactive_nav else None
    if args.reactive_nav:
        parse_label_set = reactive_utils["parse_label_set"]
        reactive_walkable_labels = parse_label_set(
            args.reactive_walkable_labels,
            reactive_utils["DEFAULT_WALKABLE_LABELS"],
        )
        reactive_obstacle_labels = parse_label_set(
            args.reactive_obstacle_labels,
            reactive_utils["DEFAULT_OBSTACLE_LABELS"],
        )
        reactive_nav_state = reactive_utils["make_reactive_nav_state"](item)
        choose_reactive_actions = reactive_utils["choose_reactive_actions"]
    else:
        reactive_walkable_labels = set()
        reactive_obstacle_labels = set()
        reactive_nav_state = {}
        choose_reactive_actions = None

    print_info(f"📋 Input image: {start_image_path}")
    print_info(f"⏱  Target duration: {args.target_duration}s")
    print_info(f"🎞  Segment duration: {segment_duration:.2f}s ({args.video_length} frames @ {args.fps}fps)")
    print_info(f"🔢 Number of segments: {num_segments}")
    print_info(f"📂 Output: {os.path.join(args.output_dir, args.output_name)}")
    if enable_map:
        print_info(f"🗺  Map condition: {map_projected_occupancy or map_path or '<segment-specific only>'}")
    if args.proxy_layout_guidance:
        print_info(f"🧱 Proxy layout guidance enabled for {len(proxy_layout_paths)} segment(s)")
    if args.reactive_nav:
        print_info(
            "🧭 Reactive navigation enabled: "
            f"walkable={sorted(reactive_walkable_labels)}, "
            f"obstacles={sorted(reactive_obstacle_labels)}"
        )
    if args.vlm_nav:
        print_info(f"👁  VLM navigation enabled: {args.vlm_url}")

    print_info("\n🔧 Loading models...")
    map_layers = resolve_map_layers(args.map_layers) if enable_map else None
    pipeline, device, weight_dtype, config, vae, boundary = setup_models(
        args,
        enable_map_conditioning=enable_map,
        map_self_attn_layers=map_layers,
    )
    print_info("✅ Models loaded\n")

    # === MAP-TO-LATENT INJECTION (training-free) ===
    if getattr(args, "map_inject_path", None):
        from map_injector import encode_map_to_dit_tokens
        print_info(f"Map-to-latent injection: {args.map_inject_path} (strength={args.map_inject_strength})")
        map_tokens = encode_map_to_dit_tokens(
            map_path=args.map_inject_path,
            vae=vae,
            patch_embedding=pipeline.transformer.patch_embedding,
            target_size=tuple(args.sample_size[::-1]),
            device=device,
            dtype=weight_dtype,
            map_strength=args.map_inject_strength,
        )
        pipeline._map_tokens = map_tokens
        print_info(f"Map tokens injected: shape={map_tokens.shape}")
    elif getattr(args, "semantic_map_baseline", None):
        from semantic_map_baseline import encode_semantic_map_to_tokens
        print_info(f"Semantic map baseline (CNN encoder): {args.semantic_map_baseline}")
        map_tokens = encode_semantic_map_to_tokens(
            npy_path=args.semantic_map_baseline,
            d_model=pipeline.transformer.config.dim,
            n_tokens=64,
            device=device,
            dtype=weight_dtype,
            map_strength=1.0,
        )
        pipeline._map_tokens = map_tokens
        print_info(f"Semantic map baseline tokens: shape={map_tokens.shape}")
    else:
        pipeline._map_tokens = None

    # === SPATIAL MAP GUIDANCE (SMG) — training-free multiplicative gating ===
    pipeline._map_guidance = None
    if getattr(args, "map_guidance_path", None):
        from map_guidance import prepare_map_guidance
        print_info(f"SMG: {args.map_guidance_path} (layers={args.map_guidance_layers}, strength={args.map_guidance_strength})")
        # We prepare guidance per-segment (grid_sizes change), so store params and
        # compute lazily in the pipeline call. For now, store a callable.
        import math as _math
        _smg_cfg = {
            'map_path': args.map_guidance_path,
            'inject_layers': args.map_guidance_layers,
            'max_strength': args.map_guidance_strength,
        }
        pipeline._smg_config = _smg_cfg
        print_info(f"SMG configured: will compute per-segment spatial gates")
    else:
        pipeline._smg_config = None

    # === SEMANTIC MAP LATENT (SML) — training-free additive vector conditioning ===
    pipeline._sml_config = None
    if getattr(args, "map_latent_path", None):
        from map_latent import prepare_map_latent
        print_info(f"SML: {args.map_latent_path} (layers={args.map_latent_layers}, strength={args.map_latent_strength})")
        _sml_cfg = {
            'map_path': args.map_latent_path,
            'inject_layers': args.map_latent_layers,
            'max_strength': args.map_latent_strength,
        }
        pipeline._sml_config = _sml_cfg
        print_info(f"SML configured: will compute per-segment semantic latents")
    else:
        pipeline._sml_config = None

    # === MAPSTATE ADAPTER (dense_residual) — trained residual injection ===
    # Uses MapStateAdapter from root mapstate.py. The signal flows through the
    # SAME map_latent kwarg as SML (same list-of-tensors format), so the two
    # are mutually exclusive: MapState takes precedence when active.
    pipeline._mapstate_config = None
    pipeline._mapstate_route_progress = (0.0, 1.0)  # updated per-segment below
    if getattr(args, "mapstate_path", None) and getattr(args, "mapstate_ckpt", None):
        if pipeline._sml_config is not None:
            print_info("WARNING: both --map_latent_path and --mapstate_path are set; "
                       "MapState takes precedence (SML will be ignored).")
        print_info(f"MapState: {args.mapstate_path} (ckpt={args.mapstate_ckpt}, "
                   f"strength={args.mapstate_strength}, crop={args.mapstate_crop_extent_px})")
        pipeline._mapstate_config = {
            'map_path': args.mapstate_path,
            'ckpt': args.mapstate_ckpt,
            'strength': args.mapstate_strength,
            'crop_extent_px': args.mapstate_crop_extent_px,
            'representation': args.mapstate_representation,
            'route_polyline': None,  # set below once layout_route_polyline is known
        }
        print_info(f"MapState configured: will compute per-segment residual signals")

    # === MKI: Map-Key Injection — map-aware attention ===
    pipeline._mki_config = None
    if getattr(args, "map_kv_path", None):
        from map_key_injection import prepare_map_key_value
        print_info(f"MKI: {args.map_kv_path} (layers={args.map_kv_layers}, key={args.map_kv_key_strength}, val={args.map_kv_value_strength})")
        pipeline._mki_config = {
            'map_path': args.map_kv_path,
            'inject_layers': args.map_kv_layers,
            'key_strength': args.map_kv_key_strength,
            'value_strength': args.map_kv_value_strength,
        }
        print_info(f"MKI configured: map-aware attention via key/value injection")

    # Per-segment topology rasters (route-visible channel advances with generation).
    mki_segment_maps = None
    if getattr(args, "map_kv_segment_npys", None):
        mki_segment_maps = [p.strip() for p in args.map_kv_segment_npys.split(",") if p.strip()]
        for p in mki_segment_maps:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"--map_kv_segment_npys entry not found: {p}")
        if pipeline._mki_config is None:
            pipeline._mki_config = {
                'map_path': mki_segment_maps[0],
                'inject_layers': args.map_kv_layers,
                'key_strength': args.map_kv_key_strength,
                'value_strength': args.map_kv_value_strength,
            }
        print_info(f"MKI segment rasters: {len(mki_segment_maps)} entries; segment k uses entry min(k, {len(mki_segment_maps)-1})")

    # === Layout guidance (Route B): sampling-time occupancy steering via critic E ===
    pipeline._layout_guidance_config = None
    lg_segment_guides = None
    if getattr(args, "layout_guidance_occ", None) and getattr(args, "layout_guidance_lambda", 0.0) > 0:
        import torch as _torch
        # remap guide channels -> E channels [road, building, vegetation, sky]
        ch_map = [int(c) for c in args.layout_guidance_channel_map.split(",") if c.strip() != ""]
        assert len(ch_map) == 3, "--layout_guidance_channel_map needs exactly 3 indices (road,building,veg)"

        def _load_guide_e(occ_path):
            occ = np.load(occ_path)                            # [T, 4, 44, 80] road,water,building,vegetation
            occ = np.asarray(occ, dtype=np.float32)
            guide_frames = occ[0::4]                           # latent frames: video frame 4f
            assert guide_frames.shape[1:] == (4, 44, 80), f"unexpected guide shape {guide_frames.shape}"
            n_f = guide_frames.shape[0]
            guide_e = np.zeros((n_f, 4, 44, 80), dtype=np.float32)
            for e_ch, g_ch in enumerate(ch_map):
                guide_e[:, e_ch] = guide_frames[:, g_ch]
            return torch.from_numpy(guide_e)

        occ_paths = [p.strip() for p in args.layout_guidance_occ.split(",")]
        # empty slot (or "none") = no guidance for that segment (pipeline skips
        # when guide is None) — keeps positions so segment k maps to entry k.
        # NOTE: no trailing-None trimming — segments past the last entry reuse
        # the final entry, so the caller must list all N segment slots.
        occ_paths = [p if p and p.lower() != "none" else None for p in occ_paths]
        if len(occ_paths) > 1:
            lg_segment_guides = [(_load_guide_e(p) if p is not None else None)
                                 for p in occ_paths]
            guide_e = next(g for g in lg_segment_guides if g is not None)
            n_none = sum(1 for g in lg_segment_guides if g is None)
            print_info(f"LG: {len(lg_segment_guides)} per-segment guides ({n_none} slots off); "
                       f"segment k uses entry min(k, {len(lg_segment_guides)-1}) — first={occ_paths[0]!r}")
        else:
            guide_e = _load_guide_e(occ_paths[0])
        n_f = guide_e.shape[0]
        # channel weights: claimed channels 1, sky 0 (masked out of the loss)
        cw = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "mapstate"))
        from layout_critic import LayoutCritic as _LayoutCritic
        ckpt = _torch.load(args.layout_guidance_critic, map_location="cpu")
        cfg = ckpt.get("config", {})
        critic = _LayoutCritic(**cfg).float().eval()
        critic.load_state_dict(ckpt["state_dict"])
        critic.to(device=device, dtype=torch.float32)
        for p_ in critic.parameters():
            p_.requires_grad_(False)
        pipeline._layout_guidance_config = {
            "critic": critic,
            "guide": guide_e,
            "lambda": float(args.layout_guidance_lambda),
            "channel_weights": torch.from_numpy(cw),
            "start_step": int(args.layout_guidance_start_step),
            "end_step": int(args.layout_guidance_end_step),
            "_logged_once": False,
        }
        print_info(f"LG: occ={args.layout_guidance_occ} frames={n_f} lambda={args.layout_guidance_lambda} "
                   f"critic={args.layout_guidance_critic} ch_map={ch_map}")

    # ---- World canvas (read-before-write world memory) -----------------------
    # ON:  each segment k>0 reads the canvas (rank0 paints guide npz, all ranks
    #      load identical bytes) through the SAME Route-B critic hook above;
    #      each finished segment is written back from its saved frames.
    # OFF: nothing below ever runs; the run is the pure time-chained baseline.
    world_canvas_obj = None
    world_canvas_extractor = None
    world_canvas_prev_end_c2w = None       # global end pose of segment k-1
    world_canvas_critic = None
    world_canvas_audit_path = None
    if getattr(args, "world_canvas", False):
        from utils.world_canvas import WorldCanvas, WorldCanvasConfig

        world_canvas_obj = WorldCanvas(config=WorldCanvasConfig(memory_mode=args.world_canvas_memory))
        world_canvas_dir = _world_canvas_dir_of(args)
        world_canvas_audit_path = os.path.join(world_canvas_dir, "audit.jsonl")
        canvas_state_path = os.path.join(world_canvas_dir, "canvas.npz")
        if os.path.exists(canvas_state_path) and not args.resume_existing_segments:
            # Every rank must raise: a rank0-only exit would hang the others
            # at the next distributed barrier.
            raise SystemExit(
                f"[canvas] refusing to reuse existing canvas state without "
                f"--resume_existing_segments: {canvas_state_path}. Delete it or "
                f"resume — deterministic A/B arms must start from an empty canvas."
            )
        if os.path.exists(canvas_state_path) and args.resume_existing_segments:
            world_canvas_obj = WorldCanvas.load(canvas_state_path)
            if world_canvas_obj.config.memory_mode != args.world_canvas_memory:
                # A resumed canvas must keep fusing the way it was built;
                # silently switching mid-run would produce a third, unlabelled
                # hybrid arm.
                raise SystemExit(
                    f"[canvas] resumed state was built with memory_mode="
                    f"{world_canvas_obj.config.memory_mode!r} but --world_canvas_memory="
                    f"{args.world_canvas_memory!r}: refusing to mix modes mid-run."
                )
            print_info(f"[canvas] resumed world canvas from {canvas_state_path} "
                       f"({len(world_canvas_obj.segments_written)} segments written)")
        # Critic E for the READ guide (same load path as Route-B LG above).
        import torch as _torch
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "mapstate"))
        from layout_critic import LayoutCritic as _LayoutCritic
        _wc_ckpt = _torch.load(args.world_canvas_critic, map_location="cpu")
        _wc_cfg = _wc_ckpt.get("config", {})
        world_canvas_critic = _LayoutCritic(**_wc_cfg).float().eval()
        world_canvas_critic.load_state_dict(_wc_ckpt["state_dict"])
        world_canvas_critic.to(device=device, dtype=torch.float32)
        for _p in world_canvas_critic.parameters():
            _p.requires_grad_(False)
        if is_rank0():
            from utils.world_canvas import FrozenLayoutExtractor

            world_canvas_extractor = FrozenLayoutExtractor(device=device)
        # The canvas owns the LG hook for the whole run: a plain --layout_guidance_occ
        # guide would be overwritten segment-by-segment anyway, so forbid the mix.
        if pipeline._layout_guidance_config is not None:
            raise SystemExit("[canvas] --world_canvas and --layout_guidance_occ are mutually exclusive: "
                             "the canvas installs its own per-segment guide through the same hook.")
        cw_canvas = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        pipeline._layout_guidance_config = {
            "critic": world_canvas_critic,
            "guide": torch.zeros(1),          # replaced per segment (rank-synchronized below)
            "guide_mask": None,               # replaced per segment
            "lambda": float(args.world_canvas_lambda),
            "channel_weights": torch.from_numpy(cw_canvas),
            "start_step": int(args.world_canvas_start_step),
            "end_step": int(args.world_canvas_end_step),
            "_logged_once": False,
        }
        print_info(f"[canvas] world canvas ON: dir={world_canvas_dir} "
                   f"lambda={args.world_canvas_lambda} memory={args.world_canvas_memory} "
                   f"steps[{args.world_canvas_start_step},"
                   f"{args.world_canvas_end_step}) write_every={args.world_canvas_write_every} "
                   f"critic={args.world_canvas_critic}")

    # ---- MapCtrl: construct once on the transformer (non-causal Wan2_2 path) ----
    mapctrl = None
    if args.mapctrl:
        MapCtrlInferenceWrapper = _load_mapctrl_wrapper()
        # Pass pipeline.transformer directly: Wan2_2_CameraPipeline has no .generator,
        # and _resolve_dit_model does not check .transformer, so resolving from the
        # pipeline would fail. The transformer is a Wan2_2Transformer3DModel with .blocks.
        mapctrl = MapCtrlInferenceWrapper(
            pipeline.transformer,
            map_tensor_path=args.mapctrl_tensor,
            poses_csv=args.mapctrl_poses_csv,
            device=device,
            total_video_frames=args.mapctrl_total_video_frames,
            intervention_mode=args.mapctrl_intervention_mode,
            lambda_road=args.mapctrl_lambda_road,
            lambda_obs=args.mapctrl_lambda_obs,
            obstacle_penalty=args.mapctrl_obstacle_penalty,
            attention_backend=args.mapctrl_attention_backend,
            mapctrl_warmup_ratio=args.mapctrl_warmup_ratio,
            boundary_ema_alpha=args.mapctrl_boundary_ema_alpha,
            boundary_ema_frames=args.mapctrl_boundary_ema_frames,
            temporal_anneal_gain=args.mapctrl_temporal_anneal_gain,
            mask_spatial_blur_kernel=args.mapctrl_mask_spatial_blur_kernel,
            verbose=is_rank0(),
        )
        print_info(
            f"🧩 MapCtrl enabled: mode={args.mapctrl_intervention_mode} "
            f"lambda_road={args.mapctrl_lambda_road} lambda_obs={args.mapctrl_lambda_obs} "
            f"warmup_ratio={args.mapctrl_warmup_ratio} tensor={args.mapctrl_tensor} "
            f"(adaptive per-window step counts disabled in v1; fixed {args.num_inference_steps} steps)"
        )

        # ---- PSCAR: register multi-class routed cross-attention context ----
        # Reuses the MapCtrl wrapper's hook machinery. Per-class uMT5 contexts
        # are encoded here (raw 4096-d text_encoder output) and projected
        # through model.text_embedding inside the wrapper's forward path,
        # exactly like the POI cross-attn route. Frozen q/k/v/o are reused.
        if args.psc_route:
            if not args.psc_occupancy or not args.psc_class_meta:
                raise RuntimeError("--psc_route requires --psc_occupancy and --psc_class_meta.")
            with open(args.psc_class_meta, 'r') as _f:
                psc_meta = json.load(_f)
            psc_class_names = tuple(psc_meta.get("class_names", []))
            psc_class_texts = psc_meta.get("class_texts", {})
            if not psc_class_names:
                raise RuntimeError(f"--psc_class_meta has no class_names: {args.psc_class_meta}")
            occ = np.load(args.psc_occupancy)  # [T,C,H,W] in [0,1]
            if occ.ndim != 4:
                raise RuntimeError(f"--psc_occupancy must be [T,C,H,W], got shape {occ.shape}")
            if int(occ.shape[1]) != len(psc_class_names):
                raise RuntimeError(
                    f"PSCAR occupancy has {int(occ.shape[1])} class channels but class_meta names "
                    f"{len(psc_class_names)} classes: {psc_class_names}"
                )
            target_blocks = tuple(int(b) for b in str(args.psc_target_blocks).split(",") if str(b).strip())
            if args.psc_layer_strengths:
                layer_strengths = tuple(float(s) for s in str(args.psc_layer_strengths).split(","))
                if len(layer_strengths) != len(target_blocks):
                    raise RuntimeError(
                        f"--psc_layer_strengths has {len(layer_strengths)} values but "
                        f"--psc_target_blocks has {len(target_blocks)} blocks."
                    )
            else:
                layer_strengths = tuple(1.0 for _ in target_blocks)

            # Encode each class text to raw uMT5 hidden states (CPU/GPU as the
            # pipeline text_encoder lives). The wrapper projects through
            # model.text_embedding at call time, so raw 4096-d output is right.
            psc_class_contexts: dict[str, torch.Tensor] = {}
            for _cname in psc_class_names:
                _ctext = psc_class_texts.get(_cname, _cname)
                _embeds = pipeline._get_t5_prompt_embeds(
                    prompt=[_ctext],
                    num_videos_per_prompt=1,
                    max_sequence_length=512,
                    device=device,
                    dtype=pipeline.text_encoder.dtype,
                )
                # _get_t5_prompt_embeds returns a list of per-prompt trimmed tensors.
                psc_class_contexts[_cname] = _embeds[0].unsqueeze(0).to(device=device)

            mapctrl.set_psc_route_context(
                class_contexts=psc_class_contexts,
                class_occupancy=args.psc_occupancy,  # path; wrapper loads + normalizes
                class_names=psc_class_names,
            )
            mapctrl.psc_target_blocks = target_blocks
            mapctrl.psc_tau = float(args.psc_tau)
            mapctrl.psc_base_floor = float(args.psc_base_floor)
            mapctrl.psc_layer_strengths = layer_strengths
            print_info(
                f"🧭 PSCAR enabled: classes={list(psc_class_names)} "
                f"shape={list(occ.shape)} blocks={list(target_blocks)} "
                f"tau={args.psc_tau} base_floor={args.psc_base_floor} "
                f"layer_strengths={list(layer_strengths)}"
            )

    # ---- MapState: trained-adapter token residual hook (opt-in) ----
    # Eval counterpart of experiments/mapstate/train_mapstate.py: the adapter
    # adds a per-token residual inside one DiT block, so map content enters
    # ONLY through the hook and the caption can stay neutral.
    mapstate_hook = None
    if getattr(args, "mapstate_adapter_path", None):
        if args.ulysses_degree > 1 or args.ring_degree > 1:
            raise RuntimeError(
                "--mapstate_adapter_path requires a single-process sequence layout "
                "(--ulysses_degree 1 --ring_degree 1, one GPU per clip): the hook "
                "skips sequence-parallel token shards and the run would silently "
                "degenerate to the baseline."
            )

        from experiments.mapstate.mapstate_hook import MapStateInferenceHook
        from experiments.mapstate.mapstate_v1 import (MapStateGeometry, MapStateV1,
                                                      build_map_pose_tracks)
        from experiments.mapstate.language_state import LanguageStateV1

        ckpt = torch.load(args.mapstate_adapter_path, map_location="cpu")
        ckpt_format = ckpt.get("format")
        ms_format = args.mapstate_adapter_format or ckpt_format
        if ckpt_format is not None and ms_format != ckpt_format:
            raise RuntimeError(
                f"--mapstate_adapter_format={ms_format} contradicts the checkpoint's "
                f"recorded format={ckpt_format}"
            )
        if ms_format not in ("raster", "language"):
            raise RuntimeError(
                f"cannot resolve MapState adapter format (flag="
                f"{args.mapstate_adapter_format}, checkpoint={ckpt_format}); "
                "pass --mapstate_adapter_format explicitly"
            )
        if ms_format == "raster" and not (args.mapstate_map_png and args.mapstate_map_meta):
            raise RuntimeError("raster adapter requires --mapstate_map_png and --mapstate_map_meta")
        if ms_format == "language" and not args.mapstate_text_file:
            raise RuntimeError("language adapter requires --mapstate_text_file")
        block = pipeline.transformer.blocks[args.mapstate_block]
        d_dit = int(pipeline.transformer.dim)

        # Token grid from the live config (VAE 16x spatial, patch [1,2,2]):
        # 704x1280 -> latent 44x80 -> grid 22x40.
        vae_spatial = int(config['vae_kwargs'].get('spatial_compression_ratio', 16))
        patch = pipeline.transformer.config.patch_size
        ms_grid_h = (args.sample_size[0] // vae_spatial) // int(patch[1])
        ms_grid_w = (args.sample_size[1] // vae_spatial) // int(patch[2])

        if ms_format == "raster":
            num_heads = int(pipeline.transformer.num_heads)
            adapter = MapStateV1(
                in_channels=9, hidden_dim=64, d_map=64, d_dit=d_dit,
                num_heads=num_heads, image_grid_h=ms_grid_h, image_grid_w=ms_grid_w)
        else:
            adapter = LanguageStateV1(d_map=64, d_dit=d_dit)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        adapter.to(device=device).eval()
        n_params = sum(p.numel() for p in adapter.parameters())

        # Pose track from the map route (shared axis: both arms use it).
        if not (args.mapstate_map_png and args.mapstate_map_meta):
            raise RuntimeError(
                "--mapstate_adapter_path requires --mapstate_map_png/--mapstate_map_meta "
                "(the pose track is built from the map route for both arms)"
            )
        tracks = build_map_pose_tracks(args.mapstate_map_png, args.mapstate_map_meta)
        ms_latent_frames = (args.video_length - 1) // int(
            config['vae_kwargs'].get('temporal_compression_ratio', 4)) + 1
        latent_poses = tracks.latent_poses(
            range(ms_latent_frames), total_latent_frames=ms_latent_frames)
        latent_poses = latent_poses.to(device=device, dtype=torch.float32)

        geometry = MapStateGeometry(
            meters_per_pixel=float(tracks.meters_per_pixel), field_stride=4)
        mapstate_hook = MapStateInferenceHook(
            adapter, geometry, block,
            image_grid_h=ms_grid_h, image_grid_w=ms_grid_w)

        if ms_format == "raster":
            content = tracks.masks.to(device=device, dtype=torch.float32)
        else:
            # Mirror contract_dataset.encode_text (max_length 512, trimmed to
            # the true token length, bf16-quantized like the cached training
            # text) so the language arm's input distribution matches training.
            ms_text = open(args.mapstate_text_file, "r", encoding="utf-8").read()
            _ms_inputs = pipeline.tokenizer(
                ms_text, padding="max_length", max_length=512, truncation=True,
                add_special_tokens=True, return_tensors="pt", return_attention_mask=True,
            )
            _ms_len = int(_ms_inputs.attention_mask.sum().item())
            _ms_emb = pipeline.text_encoder(
                _ms_inputs.input_ids.to(device),
                attention_mask=_ms_inputs.attention_mask.to(device),
            )[0]
            content = _ms_emb[0][:_ms_len].to(dtype=torch.bfloat16).float()

        mapstate_hook.set_segment(
            content, latent_poses, latent_frame_offset=0, device=device)
        mapstate_hook.install()
        _ms_ev = mapstate_hook.events[-1]
        print_info(
            f"🗺  MapState hook enabled: format={ms_format} block={args.mapstate_block} "
            f"params={n_params} grid={ms_grid_h}x{ms_grid_w} latent_frames={ms_latent_frames} "
            f"step={ckpt.get('step')} tokens_abs_max={_ms_ev.get('tokens_abs_max', _ms_ev.get('residual_abs_max', 0.0)):.4e}"
        )

        # ---- Multi-block injection: second adapter on a different block ----
        if getattr(args, "mapstate_adapter_path_2", None):
            if args.mapstate_block_2 is None:
                raise RuntimeError(
                    "--mapstate_adapter_path_2 requires --mapstate_block_2")
            if args.mapstate_block_2 == args.mapstate_block:
                raise RuntimeError(
                    "--mapstate_block_2 must differ from --mapstate_block")
            ckpt2 = torch.load(args.mapstate_adapter_path_2, map_location="cpu")
            fmt2 = ckpt2.get("format", ms_format)
            if fmt2 != ms_format:
                raise RuntimeError(
                    f"second adapter format={fmt2} != primary={ms_format}; "
                    "multi-block injection requires same format")
            block2 = pipeline.transformer.blocks[args.mapstate_block_2]
            if ms_format == "raster":
                adapter2 = MapStateV1(
                    in_channels=9, hidden_dim=64, d_map=64, d_dit=d_dit,
                    num_heads=num_heads, image_grid_h=ms_grid_h,
                    image_grid_w=ms_grid_w)
            else:
                adapter2 = LanguageStateV1(d_map=64, d_dit=d_dit)
            adapter2.load_state_dict(ckpt2["adapter_state_dict"])
            adapter2.to(device=device).eval()
            n_params2 = sum(p.numel() for p in adapter2.parameters())
            mapstate_hook2 = MapStateInferenceHook(
                adapter2, geometry, block2,
                image_grid_h=ms_grid_h, image_grid_w=ms_grid_w)
            mapstate_hook2.set_segment(
                content, latent_poses, latent_frame_offset=0, device=device)
            mapstate_hook2.install()
            _ms_ev2 = mapstate_hook2.events[-1]
            print_info(
                f"🗺  MapState hook-2 enabled: format={fmt2} block={args.mapstate_block_2} "
                f"params={n_params2} step={ckpt2.get('step')} "
                f"tokens_abs_max={_ms_ev2.get('tokens_abs_max', _ms_ev2.get('residual_abs_max', 0.0)):.4e}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    # Segment intermediates must be per-clip: concurrent processes writing the
    # same {output_dir}/segments/seg_0000.mp4 corrupt each other's concat reads
    # (intermittent late rc=1 crashes when runner instances share a condition dir).
    seg_dir = os.path.join(
        args.output_dir, "segments",
        os.path.splitext(os.path.basename(args.output_name))[0])
    os.makedirs(seg_dir, exist_ok=True)
    vlm_audit_path = args.vlm_audit_path or os.path.join(
        args.output_dir,
        "vlm_nav_feedback.jsonl",
    )
    vlm_decision_dir = os.path.join(args.output_dir, "vlm_nav_decisions")
    if args.vlm_nav and is_rank0() and not args.resume_existing_segments:
        os.makedirs(os.path.dirname(vlm_audit_path), exist_ok=True)
        os.makedirs(vlm_decision_dir, exist_ok=True)
        for name in os.listdir(vlm_decision_dir):
            if name.startswith("vlm_decision_") and name.endswith(".json"):
                os.unlink(os.path.join(vlm_decision_dir, name))
        open(vlm_audit_path, "w").close()
    distributed_barrier(device)

    # ---- Free the T5 text encoder after pre-warming the prompt cache ----
    # With model_full_load_and_qfloat8, T5 (9.4 GB bf16) + fp8 DiT + VAE +
    # 704x1280 activations exceeds 24 GB. --cache_prompt_embeds means the T5
    # output is cached after the first encode_prompt call and never re-invoked.
    # Pre-warm the cache with the single shared prompt here, then move T5 to
    # CPU, reclaiming ~9 GB VRAM so the no-offload path fits on a 4090.
    if getattr(args, "free_text_encoder", False):
        if not args.cache_prompt_embeds:
            raise SystemExit("--free_text_encoder requires --cache_prompt_embeds")
        if segment_prompt_suffixes or segment_prompt_prefixes:
            raise SystemExit(
                "--free_text_encoder requires all segments to share one prompt "
                "(no segment_prompt_prefixes/suffixes in input_json)")
        if getattr(args, "psc_route", False):
            raise SystemExit("--free_text_encoder is incompatible with --psc_route")
        if getattr(args, "mapstate_adapter_path", None):
            raise SystemExit("--free_text_encoder is incompatible with --mapstate_adapter_path")
        if args.reactive_nav or args.vlm_nav:
            raise SystemExit("--free_text_encoder is incompatible with reactive/vlm nav")
        _fte_t0 = time.time()
        # With model_full_load_and_qfloat8, setup_models() loaded ONLY T5 to GPU
        # (DiT+VAE are still on CPU). Pre-warm the prompt cache now, then free T5
        # and load DiT+VAE to GPU — this avoids the 17 GB T5+DiT+VAE co-residence
        # that would OOM during T5 forward.
        # Pre-warm the positive + negative prompt caches on GPU.
        # CRITICAL: must run under torch.no_grad() — encode_prompt is NOT
        # decorated (only __call__ is). Without no_grad, the T5 forward builds
        # an autograd graph that retains ~15 GB of activations on GPU, which
        # can't be freed even after T5 deletion + gc + empty_cache.
        with torch.no_grad():
            pipeline.encode_prompt(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                do_classifier_free_guidance=args.guidance_scale > 1.0,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=device,
                dtype=pipeline.text_encoder.dtype,
                cache_prompt_embeds=True,
            )
        _cache_n = len(getattr(pipeline, "_positive_prompt_embed_cache", {})) \
            + len(getattr(pipeline, "_negative_prompt_embed_cache", {}))
        # Delete T5 entirely and replace with a lightweight stub.
        # .to("cpu") copies params to CPU but the GPU caching allocator retains
        # the freed blocks. Fully deleting the module + gc.collect() is the only
        # reliable way to return the VRAM. The prompt cache is pre-warmed, so
        # _get_t5_prompt_embeds (the only path that invokes T5 / tokenizer) will
        # never run again. The stub provides .dtype for __call__ (line 633) and
        # _cached_t5 (line 291), which reference self.text_encoder.dtype on
        # every segment.
        _t5_dtype = pipeline.text_encoder.dtype

        class _FreedT5Stub:
            """Stand-in for the deleted T5. Provides .dtype for pipeline code
            paths that reference it on every segment (weight_dtype, cache key).
            Not an nn.Module so diffusers' _execution_device skips it."""
            dtype = _t5_dtype
            def to(self, *args, **kwargs):
                return self
            def parameters(self):
                return iter(())
            def modules(self):
                return iter(())
            def named_parameters(self, *a, **kw):
                return iter(())
            def named_modules(self, *a, **kw):
                return iter(())

        _t5_ref = pipeline.text_encoder
        # Step 1: move T5 params to CPU. This replaces GPU tensors with CPU
        # tensors in-place (param.data = cpu_tensor), making the GPU tensor
        # blocks unreferenced by the module.
        _t5_ref.to("cpu")
        # Step 2: replace with stub and drop the reference.
        object.__setattr__(pipeline, "text_encoder", _FreedT5Stub())
        del _t5_ref
        import gc
        # Step 3: gc collects the now-unreferenced GPU tensor objects, then
        # empty_cache returns the freed blocks to the driver.
        gc.collect()
        torch.cuda.empty_cache()
        _mem_free, _mem_total = torch.cuda.mem_get_info(device)
        print_info(f"🔥 Pre-warmed T5 prompt cache ({_cache_n} entries) in "
                   f"{time.time() - _fte_t0:.1f}s; T5 deleted, "
                   f"GPU free: {_mem_free / 1e9:.1f} / {_mem_total / 1e9:.1f} GB; "
                   f"loading DiT+VAE to GPU...")
        pipeline.transformer.to(device)
        pipeline.vae.to(device)
        torch.cuda.empty_cache()
        # _execution_device defaults to the first component's parameter device.
        # The stub is not an nn.Module so diffusers skips it, falling through to
        # the transformer (now on GPU). Patch it explicitly as a guarantee.
        _fte_gpu = device
        type(pipeline)._execution_device = property(lambda self: _fte_gpu)
        _mem_free2, _ = torch.cuda.mem_get_info(device)
        print_info(f"✅ DiT+VAE on GPU, GPU free: {_mem_free2 / 1e9:.1f} GB. "
                   f"Total free_text_encoder setup: {time.time() - _fte_t0:.1f}s")

    current_start_image = start_image_path
    segment_paths = []
    segment_write_executor = None
    segment_write_futures = []
    if is_rank0() and args.async_segment_write:
        segment_write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segment-writer")
        print_info("⚡ Async segment mp4 writing enabled (single rank-0 writer).")
    encoded_maps = {}
    reactive_label_maps = {}
    reactive_nav_records = []

    if enable_map and map_path:
        _, semantic_map_to_rgb, _ = load_map_utils()
        label_map = load_label_map(map_path, args.sample_size)
        semantic_map_to_rgb(label_map, use_ade20k=False).save(
            os.path.join(args.output_dir, "semantic_map.png")
        )
        semantic_map_to_rgb(label_map, use_ade20k=True).save(
            os.path.join(args.output_dir, "semantic_map_ade20k.png")
        )

    # Build per-segment action plan.
    # Map-driven camera mode: derive the action plan from the map route polyline
    # and forbid any manually-supplied route action sequence (leakage guard).
    _enforce_map_driven_leakage_guard(args, item, segment_action_sequences)
    map_driven_route_result = None
    if getattr(args, "map_driven_camera", False):
        planned_actions, planned_speeds, map_driven_route_result = build_map_driven_action_plan(
            args, item, num_segments, is_rank0_fn=is_rank0,
        )
        # Persist the derived plan + turn audit next to the output.
        if is_rank0():
            _map_driven_log_path = os.path.join(args.output_dir, "map_driven_camera_plan.json")
            with open(_map_driven_log_path, "w") as _f:
                json.dump(
                    {
                        "mode": "map_driven",
                        "n_segments": num_segments,
                        "planned_actions": planned_actions,
                        "planned_speeds": planned_speeds,
                        "route": map_driven_route_result,
                    },
                    _f, indent=2,
                )
    else:
        planned_actions = original_action_seq
        planned_speeds = original_action_speed_list

    layout_route_polyline = None
    layout_route_progress = None
    if enable_map or pipeline._mapstate_config is not None:
        layout_route_polyline, layout_route_progress = _load_route_polyline_for_map_driven(args, item)
        if layout_route_polyline is None or len(layout_route_polyline) < 2:
            _need = "semantic layout conditioning" if enable_map else "MapState adapter"
            raise ValueError(f"{_need} requires a route polyline with at least two points")
        # Feed the route polyline into the MapState config (needed for per-frame
        # egocentric readout via poses_from_route).
        if pipeline._mapstate_config is not None:
            pipeline._mapstate_config['route_polyline'] = layout_route_polyline
            print_info(f"MapState: route polyline loaded ({len(layout_route_polyline)} points)")

    # Sub-segment (scheme A+B) plans are list-of-lists: planned_actions[idx] is
    # already a list of sub-tokens for segment idx. Flat plans are list[str]:
    # planned_actions[idx] is a single token that must be wrapped. Detect once.
    subsegment_plan = bool(planned_actions) and isinstance(planned_actions[0], list)
    # A+B ideal: per-sub-token REAL yaw deltas (degrees), grouped per segment.
    # Only present for the sub-segment heading_delta path; None -> legacy pose.
    seg_yaw_angles = None
    if subsegment_plan and isinstance(map_driven_route_result, dict):
        seg_yaw_angles = map_driven_route_result.get("segment_turn_angles_deg")

    # ---- World canvas: pre-compute the EXECUTED global pose chain ------------
    # The canvas indexes memory by the trajectory the DiT actually executes
    # (the same ActionToPoseFromID rows the PRoPE branch consumes), not by map
    # pixels. Chaining is exact SE(3) composition: segment k's local frame
    # coincides with segment k-1's end camera frame. Pre-computed once so both
    # the READ (before segment k) and WRITE (after segment k) sides see the
    # same world coordinates, and so --segment_range / --resume_existing_segments
    # still know where skipped segments ended up.
    world_canvas_global_c2w = {}
    if world_canvas_obj is not None:
        # The pre-computed chain assumes the plan executes verbatim. Runtime
        # action rewriters (reactive/vlm/manual per-segment sequences) would
        # desync the world frame -> fail loudly instead of writing the canvas
        # at wrong coordinates.
        if args.reactive_nav or args.vlm_nav or segment_action_sequences:
            raise SystemExit("[canvas] --world_canvas requires the planned action "
                             "chain to execute verbatim (no --reactive_nav, no "
                             "--vlm_nav, no per-segment action overrides).")
        from utils.world_canvas import build_segment_local_poses, chain_c2w

        _wc_vlen = int(args.video_length)
        _wc_vlen = (
            int((_wc_vlen - 1) // vae.config.temporal_compression_ratio
                * vae.config.temporal_compression_ratio) + 1
            if _wc_vlen != 1 else 1
        )
        _wc_latent_f = (_wc_vlen - 1) // vae.config.temporal_compression_ratio + 1
        _wc_prev = world_canvas_prev_end_c2w
        for _k in range(num_segments):
            if subsegment_plan:
                _a = list(planned_actions[_k % len(planned_actions)])
                _s = list(planned_speeds[_k % len(planned_speeds)])
                _y = seg_yaw_angles[_k % len(seg_yaw_angles)] if seg_yaw_angles is not None else None
            else:
                _a = [planned_actions[_k % len(planned_actions)]]
                _s = [planned_speeds[_k % len(planned_speeds)]]
                _y = None
            _local, _ = build_segment_local_poses(_a, _s, _wc_vlen, yaw_deltas=_y)
            _glob = chain_c2w(_local, _wc_prev)
            world_canvas_global_c2w[_k] = _glob
            _wc_prev = _glob[-1]
        if is_rank0():
            _end_pos = _wc_prev[:3, 3]
            _end_x, _, _end_z = _end_pos
            print_info(f"[canvas] executed chain: {num_segments} segments, "
                       f"end pose xyz=({_end_x:.1f}, {float(_end_pos[1]):.1f}, {_end_z:.1f}) m "
                       f"(loop closure gap to origin = {math.hypot(_end_x, _end_z):.1f} m)")

    # ---- R1b world warm start: executed pose chain + rendered-frame index ----
    # Same exact-SE(3)-composition chain the canvas uses: the warm index must
    # key content by the trajectory the DiT actually executes. The plan is an
    # INPUT (PRoPE enforces it), so the chain is known before any generation —
    # the composite for segment k can be built the moment segment k-1 lands.
    world_warm_index = None
    world_latent_cache = None
    world_warm_poses = {}
    if (getattr(args, "world_warm_start", False)
            or getattr(args, "world_latent_cache", False)
            or getattr(args, "save_segment_latents", None)):
        if args.reactive_nav or args.vlm_nav or segment_action_sequences:
            raise SystemExit("[world_warm] --world_warm_start/--world_latent_cache/--save_segment_latents requires the "
                             "planned action chain to execute verbatim (no --reactive_nav, no "
                             "--vlm_nav, no per-segment action overrides).")
        from utils.world_canvas import build_segment_local_poses, chain_c2w

        _ww_vlen = int(args.video_length)
        _ww_vlen = (
            int((_ww_vlen - 1) // vae.config.temporal_compression_ratio
                * vae.config.temporal_compression_ratio) + 1
            if _ww_vlen != 1 else 1
        )
        _ww_prev = None
        for _k in range(num_segments):
            if subsegment_plan:
                _a = list(planned_actions[_k % len(planned_actions)])
                _s = list(planned_speeds[_k % len(planned_speeds)])
                _y = seg_yaw_angles[_k % len(seg_yaw_angles)] if seg_yaw_angles is not None else None
            else:
                _a = [planned_actions[_k % len(planned_actions)]]
                _s = [planned_speeds[_k % len(planned_speeds)]]
                _y = None
            _local, _ = build_segment_local_poses(_a, _s, _ww_vlen, yaw_deltas=_y)
            _glob = chain_c2w(_local, _ww_prev)
            world_warm_poses[_k] = _glob
            _ww_prev = _glob[-1]
        if getattr(args, "world_warm_start", False):
            if getattr(args, "world_warm_render3d", None):
                from utils.world_3d_map import Render3DPriorIndex, BakedPriorIndex
                if os.path.isdir(args.world_warm_render3d):
                    world_warm_index = BakedPriorIndex(
                        args.world_warm_render3d,
                        latent_stride=int(getattr(vae, "spatial_compression_ratio", 16)),
                        lowpass=int(getattr(args, "world_warm_lowpass", 4)))
                else:
                    world_warm_index = Render3DPriorIndex(
                        args.world_warm_render3d,
                        latent_stride=int(getattr(vae, "spatial_compression_ratio", 16)),
                    )
                if is_rank0():
                    _m = getattr(world_warm_index, "map", None)
                    _desc = (f"{_m.xyz.shape[0]/1e6:.1f}M voxels kept" if _m is not None
                             else "baked composites")
                    print_info(f"[world_warm] render3d prior: {args.world_warm_render3d} ({_desc})")
            else:
                from utils.world_warm_start import WorldWarmStartIndex
                world_warm_index = WorldWarmStartIndex(
                    tile_m=args.world_warm_tile_m,
                    height_m=args.world_warm_camera_height_m,
                    pitch_deg=args.world_warm_camera_pitch_deg,
                    height_px=args.sample_size[0],
                    width_px=args.sample_size[1],
                    latent_stride=int(getattr(vae, "spatial_compression_ratio", 16)),
                    max_candidates=args.world_warm_max_candidates,
                )
        if getattr(args, "world_latent_cache", False):
            from utils.world_latent_cache import WorldLatentCache
            world_latent_cache = WorldLatentCache(
                tile_m=args.world_warm_tile_m,
                height_m=args.world_warm_camera_height_m,
                pitch_deg=args.world_warm_camera_pitch_deg,
                height_px=args.sample_size[0],
                width_px=args.sample_size[1],
                latent_stride=int(getattr(vae, "spatial_compression_ratio", 16)),
                max_candidates=args.world_warm_max_candidates,
            )
        if is_rank0():
            _end_pos = _ww_prev[:3, 3]
            _idx_tag = ("warm+latent" if (world_warm_index is not None
                         and world_latent_cache is not None)
                         else "latent" if world_latent_cache is not None
                         else "warm")
            print_info(f"[world_{_idx_tag}] executed chain: {num_segments} segments, "
                       f"end pose xyz=({float(_end_pos[0]):.1f}, {float(_end_pos[1]):.1f}, "
                       f"{float(_end_pos[2]):.1f}) m; tile={args.world_warm_tile_m}m; "
                       f"s0={args.world_warm_step_index}; "
                       f"min_coverage={args.world_warm_min_coverage}; "
                       f"camera h={args.world_warm_camera_height_m}m "
                       f"pitch={args.world_warm_camera_pitch_deg}deg")

    # ---- Warp+Fill prior: MiDaS forward-warp of each segment's start_image ----
    # Computes the same executed pose chain as world_warm (needed for warp), and
    # builds a WarpFillPriorIndex that warps start_image → anchor frames. Reuses
    # the existing warm_start_pack → run_segment → pipeline SDEdit blend path.
    warp_fill_index = None
    if getattr(args, "warp_fill", False):
        if args.reactive_nav or args.vlm_nav or segment_action_sequences:
            raise SystemExit("[warp_fill] --warp_fill requires the planned "
                             "action chain to execute verbatim (no --reactive_nav, no "
                             "--vlm_nav, no per-segment action overrides).")
        from utils.world_canvas import build_segment_local_poses, chain_c2w
        from utils.warp_fill_prior import WarpFillPriorIndex
        from utils.map_trajectory import default_prope_K

        # Compute poses if world_warm didn't already
        if not world_warm_poses:
            _wf_vlen = int(args.video_length)
            _wf_vlen = (
                int((_wf_vlen - 1) // vae.config.temporal_compression_ratio
                    * vae.config.temporal_compression_ratio) + 1
                if _wf_vlen != 1 else 1
            )
            _wf_prev = None
            for _k in range(num_segments):
                if subsegment_plan:
                    _a = list(planned_actions[_k % len(planned_actions)])
                    _s = list(planned_speeds[_k % len(planned_speeds)])
                    _y = seg_yaw_angles[_k % len(seg_yaw_angles)] if seg_yaw_angles is not None else None
                else:
                    _a = [planned_actions[_k % len(planned_actions)]]
                    _s = [planned_speeds[_k % len(planned_speeds)]]
                    _y = None
                _local, _ = build_segment_local_poses(_a, _s, _wf_vlen, yaw_deltas=_y)
                _glob = chain_c2w(_local, _wf_prev)
                world_warm_poses[_k] = _glob
                _wf_prev = _glob[-1]

        _wf_K = torch.from_numpy(
            default_prope_K(args.video_length)).float()[0].numpy()
        warp_fill_index = WarpFillPriorIndex(
            H=args.sample_size[0], W=args.sample_size[1], K_np=_wf_K,
            min_d=args.warp_fill_min_d, max_d=args.warp_fill_max_d,
            cam_h=args.world_warm_camera_height_m,
            latent_stride=int(getattr(vae, "spatial_compression_ratio", 16)),
            warp_mode=args.warp_fill_mode,
            min_seg_coverage=args.warp_fill_min_seg_coverage,
            min_anchor_coverage=args.warp_fill_min_anchor_coverage,
            midas_skip=int(getattr(args, "warp_fill_midas_skip", 1)),
            use_cuda_warp=not bool(getattr(args, "warp_fill_no_cuda_warp", False)),
        )
        if is_rank0():
            _end_pos = _wf_prev[:3, 3] if world_warm_poses else None
            print_info(f"[warp_fill] enabled: {num_segments} segments, "
                       f"mode={args.warp_fill_mode}, "
                       f"step_index={args.warp_fill_step_index}, "
                       f"depth=[{args.warp_fill_min_d}, {args.warp_fill_max_d}]m, "
                       f"cam_h={args.world_warm_camera_height_m}m, "
                       f"seg_cov>={args.warp_fill_min_seg_coverage}, "
                       f"anchor_cov>={args.warp_fill_min_anchor_coverage}")

    # Optional segment-range sharding: when set, only segments in [lo, hi) run.
    # Used to split a long run across parallel GPU groups (each group writes its
    # own output_dir + concatenates only the segments it generated).
    seg_range_lo, seg_range_hi = 0, num_segments
    if getattr(args, "segment_range", None):
        try:
            _lo, _hi = args.segment_range.replace("-", " ").split()
            seg_range_lo, seg_range_hi = int(_lo), int(_hi)
            seg_range_hi = min(seg_range_hi, num_segments)
            print_info(f"🔪 Segment range: only generating segments [{seg_range_lo}, {seg_range_hi})")
        except Exception as exc:
            print_info(f"⚠️  Invalid --segment_range '{args.segment_range}' ({exc}); ignoring (full run).")

    for seg_idx in range(num_segments):
        if seg_idx < seg_range_lo or seg_idx >= seg_range_hi:
            continue
        # MapState per-segment route progress: this segment covers arc-length
        # window [seg_idx/n, (seg_idx+1)/n] of the route — same convention as
        # build_segment_semantic_layout (line 910-911).
        if pipeline._mapstate_config is not None:
            _ms_prog_lo = float(seg_idx) / float(num_segments)
            _ms_prog_hi = float(seg_idx + 1) / float(num_segments)
            pipeline._mapstate_route_progress = (_ms_prog_lo, _ms_prog_hi)
        seg_yaw_deltas = None
        if seg_idx in segment_action_sequences:
            action_seq = segment_action_sequences[seg_idx]
            spd_entry = planned_speeds[seg_idx % len(planned_speeds)]
            if subsegment_plan and isinstance(spd_entry, list):
                base = spd_entry
            else:
                base = [spd_entry] * len(action_seq)
            action_speed_list = segment_action_speed_lists.get(seg_idx, base)
        else:
            # Cycle through planned actions
            idx = seg_idx % len(planned_actions)
            if subsegment_plan:
                action_seq = list(planned_actions[idx])
                action_speed_list = list(planned_speeds[idx])
                # A+B ideal: drive yaw with the route's real per-sub-segment angles.
                if seg_yaw_angles is not None and idx < len(seg_yaw_angles):
                    seg_yaw_deltas = list(seg_yaw_angles[idx])
            else:
                action_seq = [planned_actions[idx]]
                action_speed_list = [planned_speeds[idx]]
        active_map_path = segment_map_paths.get(seg_idx, map_path)
        if mki_segment_maps and pipeline._mki_config is not None:
            pipeline._mki_config['map_path'] = mki_segment_maps[min(seg_idx, len(mki_segment_maps) - 1)]
        if lg_segment_guides is not None and pipeline._layout_guidance_config is not None:
            # Multi-segment LG: segment k steers toward guide entry k (last reused),
            # each rendered over the SAME uniform progress slice [k/n, (k+1)/n] the
            # map-driven camera planner uses, and each with the _F=31 latent frames
            # the pipeline's step callback expects.
            pipeline._layout_guidance_config["guide"] = lg_segment_guides[min(seg_idx, len(lg_segment_guides) - 1)]
            pipeline._layout_guidance_config["_logged_once"] = False

        # ---- World canvas READ: rank0 paints the guide, every rank loads it ----
        # First visit (empty canvas / nothing in view) -> guidance OFF for this
        # segment (mask empty), so canvas-ON's first segment is bit-comparable
        # to the baseline. Revisits read what was written at first visit.
        if world_canvas_obj is not None:
            from utils.world_canvas import load_guide_npz, save_guide_npz

            _wc_guide_path = os.path.join(_world_canvas_dir_of(args), f"guide_{seg_idx:04d}.npz")
            if is_rank0():
                _wc_result = world_canvas_obj.read_guide(
                    world_canvas_global_c2w[seg_idx], _wc_latent_f
                )
                if _wc_result is None:
                    # First visit: a correctly-shaped all-zero guide/mask keeps the
                    # masked loss exactly 0 (gradient exactly 0), so this segment is
                    # bit-identical to canvas-OFF instead of erroring on a shape
                    # mismatch in the pipeline's LG block.
                    save_guide_npz(
                        _wc_guide_path,
                        {"guide": np.zeros((_wc_latent_f, 4, 44, 80), np.float32),
                         "mask": np.zeros((_wc_latent_f, 1, 44, 80), np.float32),
                         "coverage": np.zeros(_wc_latent_f, np.float32)},
                        seg_idx=seg_idx, meta={"hit": False, "memory_mode": args.world_canvas_memory},
                    )
                    print_info(f"  [canvas] segment {seg_idx}: first visit — no canvas hit, guidance off")
                else:
                    save_guide_npz(_wc_guide_path, _wc_result, seg_idx=seg_idx,
                                   meta={"hit": True, "memory_mode": args.world_canvas_memory})
                    _cov = float(np.asarray(_wc_result["coverage"]).mean())
                    print_info(f"  [canvas] segment {seg_idx}: canvas hit — mean screen coverage {_cov:.3f}")
            distributed_barrier(device)
            _wc_payload = load_guide_npz(_wc_guide_path)
            pipeline._layout_guidance_config["guide"] = torch.from_numpy(_wc_payload["guide"])
            pipeline._layout_guidance_config["guide_mask"] = torch.from_numpy(_wc_payload["mask"])
            pipeline._layout_guidance_config["_logged_once"] = False

        proxy_layout_path = proxy_layout_paths.get(seg_idx)
        seg_path = os.path.join(seg_dir, f"seg_{seg_idx:04d}.mp4")
        last_frame_path = os.path.join(seg_dir, f"last_frame_{seg_idx:04d}.png")

        if args.resume_existing_segments and os.path.exists(seg_path) and image_file_is_readable(last_frame_path):
            if args.reactive_nav and active_map_path:
                if active_map_path not in reactive_label_maps:
                    reactive_label_maps[active_map_path] = load_label_map(
                        active_map_path,
                        args.sample_size,
                    )
                action_seq, action_speed_list, _ = choose_reactive_actions(
                    action_seq,
                    action_speed_list,
                    reactive_label_maps[active_map_path],
                    reactive_nav_state,
                    args,
                    reactive_walkable_labels,
                    reactive_obstacle_labels,
                )
            if is_rank0():
                print_info(f"\n[{seg_idx+1}/{num_segments}] resume: reusing {seg_path}")
                segment_paths.append(seg_path)
                if world_warm_index is not None:
                    print_info(f"  [world_warm] WARNING: resumed segment {seg_idx} is NOT in "
                               "the warm index; later segments cannot source content from it "
                               "(coverage will be understated).")
            current_start_image = last_frame_path
            distributed_barrier(device)
            continue

        if args.use_segment_start_overrides and seg_idx in segment_start_overrides:
            current_start_image = segment_start_overrides[seg_idx]
            print_info(f"  anchor reset: {current_start_image}")

        segment_prompt = prompt
        if seg_idx in segment_prompt_prefixes:
            # map facts go FIRST — earlier tokens carry more weight in the caption
            segment_prompt = f"{segment_prompt_prefixes[seg_idx]} {segment_prompt}"
        if args.use_segment_context and seg_idx < len(trajectory_segments):
            seg_meta = trajectory_segments[seg_idx]
            seg_context = " ".join(
                str(seg_meta.get(key, ""))
                for key in ("node", "tile", "map_intent")
                if seg_meta.get(key)
            ).strip()
            if seg_context:
                segment_prompt = f"{segment_prompt} Segment route context: {seg_context}."
        if seg_idx in segment_prompt_suffixes:
            segment_prompt = f"{segment_prompt} {segment_prompt_suffixes[seg_idx]}"

        if args.reactive_nav:
            if active_map_path:
                if active_map_path not in reactive_label_maps:
                    reactive_label_maps[active_map_path] = load_label_map(
                        active_map_path,
                        args.sample_size,
                    )
                planned_action_seq, planned_action_speeds = action_seq, action_speed_list
                action_seq, action_speed_list, reactive_metrics = choose_reactive_actions(
                    action_seq,
                    action_speed_list,
                    reactive_label_maps[active_map_path],
                    reactive_nav_state,
                    args,
                    reactive_walkable_labels,
                    reactive_obstacle_labels,
                )
                reactive_record = {
                    "segment": seg_idx,
                    "map_path": active_map_path,
                    "planned_actions": planned_action_seq,
                    "planned_speeds": planned_action_speeds,
                    "actions": action_seq,
                    "speeds": action_speed_list,
                    "metrics": {
                        key: round(value, 4) if isinstance(value, float) else value
                        for key, value in reactive_metrics.items()
                    },
                }
                reactive_nav_records.append(reactive_record)
                print_info(
                    "  reactive_nav: "
                    f"front={reactive_metrics['front_walkable']:.2f}/"
                    f"{reactive_metrics['local_front_walkable']:.2f} "
                    f"ground={reactive_metrics['ground_walkable']:.2f} "
                    f"L={reactive_metrics['left_walkable']:.2f} "
                    f"R={reactive_metrics['right_walkable']:.2f} "
                    f"decision={reactive_metrics['decision']}"
                )
            else:
                print_info("  reactive_nav: no semantic map available; keeping planned action")

        if args.vlm_nav:
            action_seq, action_speed_list, _ = apply_vlm_nav_feedback(
                args,
                current_start_image,
                seg_idx,
                action_seq,
                action_speed_list,
                vlm_audit_path,
                vlm_decision_dir,
            )

        control_map = None
        if enable_map and (active_map_path or map_projected_occupancy):
            cache_key = (map_projected_occupancy or active_map_path, seg_idx)
            if cache_key not in encoded_maps:
                num_latent_frames = (
                    (args.video_length - 1) // vae.config.temporal_compression_ratio + 1
                )
                if map_projected_occupancy:
                    print_info(f"  loading projected semantic layout: {map_projected_occupancy}")
                    encoded_maps[cache_key] = build_segment_projected_layout(
                        map_projected_occupancy, seg_idx, args.video_length,
                        num_latent_frames, args.sample_size,
                    )
                else:
                    print_info(f"  building ego semantic layout: {active_map_path}")
                    encoded_maps[cache_key] = build_segment_semantic_layout(
                        active_map_path,
                        args.sample_size,
                        layout_route_polyline,
                        seg_idx,
                        num_segments,
                        num_latent_frames,
                        route_vertex_progress=layout_route_progress,
                        crop_size=tuple(args.map_layout_crop_size),
                        lateral_extent=args.map_layout_lateral_extent,
                        lookahead=args.map_layout_lookahead,
                        rear_extent=args.map_layout_rear_extent,
                    )
            control_map = encoded_maps[cache_key]

        print_info(f"\n[{seg_idx+1}/{num_segments}] actions={action_seq} speeds={action_speed_list}")
        print_info(f"  start_image: {current_start_image if isinstance(current_start_image, str) else '<PIL Image>'}")
        if control_map is not None:
            print_info(f"  map_strength: {args.map_strength}")
        if args.proxy_layout_guidance:
            if proxy_layout_path:
                print_info(f"  proxy_layout: {proxy_layout_path}")
            else:
                print_info("  proxy_layout: none for this segment")

        # MapCtrl per-segment video window: map this segment's frames onto a slice of the
        # map tensor timeline. Window spans [seg_idx, seg_idx+1) * video_length frames, then
        # wraps modulo the map tensor's OWN frame count (map_frame_count) so a 480-frame map
        # covers an arbitrary-length run. We must wrap on map_frame_count, NOT on
        # mapctrl_total_video_frames: the range check in set_active_video_window validates
        # against map_tensor.shape[0], so using a larger total_video_frames here (e.g. 721
        # for a 30s run) would compute v_start beyond the 480-frame tensor and raise
        # "video_end out of range".
        video_window = None
        mapctrl_latent_frames = None
        if mapctrl is not None:
            seg_vlen = args.video_length
            map_fc = int(mapctrl.map_frame_count)
            v_start = (seg_idx * seg_vlen) % map_fc
            v_end = (v_start + seg_vlen) % map_fc
            if v_end <= v_start:
                v_end = map_fc  # don't wrap mid-segment; clamp to end
            video_window = (v_start, v_end)
            mapctrl_latent_frames = (seg_vlen - 1) // 4 + 1

        # ---- R1b world warm start: build the cached-world composite for this
        # segment from STRICTLY EARLIER segments' rendered frames, then apply
        # the pre-registered coverage threshold (one place; printed either way
        # so the log shows every warm/cold decision).
        warm_start_pack = None
        # Scaffold SDEdit warm start (training-free): VAE-encode this segment's
        # proxy layout video and hand it to the pipeline's warm_start path
        # (latents = (1-sigma_s0)*warm + sigma_s0*eps, timesteps truncated at s0,
        # frame 0 masked out by the pipeline itself). Reuses the same proxy
        # layout video loader as --proxy_layout_guidance; the two mechanisms are
        # alternatives, not a stack.
        if (getattr(args, "scaffold_warm_start", False) and world_warm_index is None):
            _sws_path = proxy_layout_paths.get(seg_idx)
            if _sws_path:
                _sws_vl = args.video_length
                _sws_vl = (int((_sws_vl - 1) // vae.config.temporal_compression_ratio
                               * vae.config.temporal_compression_ratio) + 1
                           if _sws_vl != 1 else 1)
                _sws_warm = prepare_proxy_layout_latents(
                    pipeline, _sws_path, args, _sws_vl, device, weight_dtype, None)
                warm_start_pack = {
                    "warm_latent": _sws_warm,
                    "coverage_mask": torch.ones_like(_sws_warm[:, :1]),
                    "coverage": 1.0,
                }
                if is_rank0():
                    print_info(f"  [scaffold_warm_start] seg {seg_idx}: warm prior from "
                               f"{_sws_path} (s0={int(args.world_warm_step_index)})")
            elif is_rank0():
                print_info(f"  [scaffold_warm_start] seg {seg_idx}: no proxy layout path; cold")
        if world_warm_index is not None:
            _ww_t0 = time.time()
            _ww_pack = world_warm_index.build_warm_composite(
                world_warm_poses[seg_idx], min_coverage=args.world_warm_min_coverage)
            _ww_sec = time.time() - _ww_t0
            if _ww_pack is not None and _ww_pack["coverage"] >= args.world_warm_min_coverage:
                warm_start_pack = _ww_pack
                _nc = _ww_pack["n_candidates"]
                _nc_used = [int(c) for c in _nc[1:] if c > 0]
                print_info(f"  [world_warm] seg {seg_idx}: WARM coverage="
                           f"{_ww_pack['coverage']:.3f} anchors_hit="
                           f"{_ww_pack['n_anchors_hit']}/{_ww_pack['n_anchors_total']-1} "
                           f"cands_mean={float(np.mean(_nc_used)) if _nc_used else 0.0:.1f} "
                           f"composite_sec={_ww_sec:.1f}")
            else:
                _ww_cov = float(_ww_pack["coverage"]) if _ww_pack is not None else 0.0
                print_info(f"  [world_warm] seg {seg_idx}: cold "
                           f"(coverage={_ww_cov:.3f} < {args.world_warm_min_coverage}, "
                           f"index_anchors={len(world_warm_index._anchors)}), "
                           f"composite_sec={_ww_sec:.1f}")

        # ---- R1c-latent: lossless latent retrieval (bypasses pixel warp).
        # Same index keys as R1b but stores the clean VAE latent, not pixels.
        # On a warm hit, the prior is the EXACT generated latent — no warp, no
        # re-encode, no 13-17 dB ceiling.  Coverage is per-anchor binary.
        # Mutually exclusive with world_warm_index for this segment: LC takes
        # priority because its prior is strictly sharper when available.
        if world_latent_cache is not None and warm_start_pack is None:
            _lc_t0 = time.time()
            _lc_pack = world_latent_cache.build_warm_latent(
                world_warm_poses[seg_idx], min_coverage=args.world_warm_min_coverage)
            _lc_sec = time.time() - _lc_t0
            if _lc_pack is not None and _lc_pack["coverage"] >= args.world_warm_min_coverage:
                warm_start_pack = _lc_pack
                _lc_nc = _lc_pack["n_candidates"]
                _lc_used = [int(c) for c in _lc_nc[1:] if c > 0]
                print_info(f"  [world_latent] seg {seg_idx}: WARM-LATENT coverage="
                           f"{_lc_pack['coverage']:.3f} anchors_hit="
                           f"{_lc_pack['n_anchors_hit']}/{_lc_pack['n_anchors_total']-1} "
                           f"cands_mean={float(np.mean(_lc_used)) if _lc_used else 0.0:.1f} "
                           f"retrieval_sec={_lc_sec:.1f}")
            else:
                _lc_cov = float(_lc_pack["coverage"]) if _lc_pack is not None else 0.0
                print_info(f"  [world_latent] seg {seg_idx}: cold "
                           f"(coverage={_lc_cov:.3f} < {args.world_warm_min_coverage}, "
                           f"index_anchors={len(world_latent_cache._anchors)}), "
                           f"retrieval_sec={_lc_sec:.1f}")

        # ---- Warp+Fill prior: forward-warp this segment's start_image to all
        # anchor frames using MiDaS depth + the pose chain. Full coverage (no
        # cross-segment dependency), works on segment 0. Only fires when
        # world_warm didn't already produce a pack for this segment (the two
        # are complementary: world_warm reuses earlier content, warp_fill
        # extrapolates from the current start_image).
        if warp_fill_index is not None and warm_start_pack is None:
            _wf_t0 = time.time()
            _wf_pack = warp_fill_index.build_warm_composite(
                world_warm_poses[seg_idx], current_start_image)
            _wf_sec = time.time() - _wf_t0
            if _wf_pack is None:
                # Coverage gate rejected this segment — run full denoise.
                warm_start_pack = None
                print_info(f"  [warp_fill] seg {seg_idx}: COLD (coverage below "
                           f"{args.warp_fill_min_seg_coverage}, running full denoise), "
                           f"sec={_wf_sec:.1f}")
            else:
                warm_start_pack = _wf_pack
                _wf_cov = float(_wf_pack["coverage"])
                print_info(f"  [warp_fill] seg {seg_idx}: prior built "
                           f"(valid_pix={_wf_cov:.3f}, "
                           f"anchors_hit={_wf_pack['n_anchors_hit']}/"
                           f"{_wf_pack['n_anchors_total']-1}, "
                           f"sec={_wf_sec:.1f})")
                # Override step index for warp_fill (separate control from world_warm)
                args.world_warm_step_index = int(args.warp_fill_step_index)

        sample = run_segment(
            pipeline, args, segment_prompt, current_start_image,
            action_seq, action_speed_list, vae, boundary, seg_idx,
            control_map=control_map,
            mapctrl=mapctrl,
            video_window=video_window,
            latent_frames=mapctrl_latent_frames,
            proxy_layout_path=proxy_layout_path,
            yaw_deltas=seg_yaw_deltas,
            warm_start_pack=warm_start_pack,
            warm_seg_info=({"seg_idx": int(seg_idx), "coverage": 0.0}
                           if (world_warm_index is not None
                               or world_latent_cache is not None
                               or warp_fill_index is not None) else None),
        )

        # R1b index maintenance: the decoded sample is IDENTICAL on every SP
        # rank (transformer head all-gathers), so each rank maintains its own
        # index from the same frames — zero communication.
        if world_warm_index is not None:
            world_warm_index.add_segment(seg_idx, sample, world_warm_poses[seg_idx])

        # R1c-latent + R1c pregeneration: the clean post-denoise latent is
        # captured by the pipeline into pipeline._last_clean_latent (CPU).
        # Both consumers read the SAME tensor (no double-copy): R1c save is
        # rank0-only disk write; LC cache-add is per-rank (zero-comm, same as
        # R1b).  Order: save first (rank0), then add to cache, then free.
        if getattr(args, "save_segment_latents", None) and is_rank0():
            _clean_lat = getattr(pipeline, "_last_clean_latent", None)
            if _clean_lat is not None:
                _r1c_dir = args.save_segment_latents
                os.makedirs(_r1c_dir, exist_ok=True)
                _lat_path = os.path.join(_r1c_dir, f"seg_{seg_idx:04d}.pt")
                _poses_path = os.path.join(_r1c_dir, f"poses_seg_{seg_idx:04d}.npy")
                torch.save(_clean_lat.clone().cpu(), _lat_path)
                np.save(_poses_path, np.asarray(world_warm_poses[seg_idx]))
                print_info(f"  [r1c_save] seg {seg_idx}: latent saved to {_lat_path}")

        if world_latent_cache is not None:
            _clean_lat = getattr(pipeline, "_last_clean_latent", None)
            if _clean_lat is not None:
                # Cache-poisoning guard: only index COLD-segment latents.
                # Warm segments run SDEdit (25% warm + 75% noise at s0=10) and
                # their output is a denoised artifact, not ground truth.  If
                # cached, it pollutes the index and later revisits to the same
                # tile retrieve the degraded latent, producing near-flat
                # outputs (observed: lc10/lc5 segs 15-25 → std≈1.5 pixel).
                _was_warm = (warm_start_pack is not None
                             and "warm_latent" in warm_start_pack)
                if _was_warm:
                    print_info(f"  [world_latent_cache] seg {seg_idx}: skip cache "
                               f"(warm output — not ground truth)")
                else:
                    world_latent_cache.add_segment(
                        seg_idx, _clean_lat, world_warm_poses[seg_idx])
        pipeline._last_clean_latent = None  # free reference after both consumers

        if is_rank0():
            # The next segment depends only on this PNG, so keep it synchronous.
            save_last_frame(sample, last_frame_path)
            segment_paths.append(seg_path)
            if segment_write_executor is not None:
                for future in segment_write_futures:
                    if future.done():
                        future.result()
                segment_write_futures = [future for future in segment_write_futures if not future.done()]
                segment_write_futures.append(
                    segment_write_executor.submit(save_videos_grid, sample, seg_path, fps=args.fps)
                )
                print_info(f"  ⚡ Queued async write: {seg_path}")
            else:
                save_videos_grid(sample, seg_path, fps=args.fps)
                print_info(f"  ✅ Saved: {seg_path}")

            # ---- World canvas WRITE: rank0 splats this segment's frames ----
            # Must run on the SAVED mp4 (not the in-memory sample), so with
            # --async_segment_write we wait for this segment's write future to
            # finish first — the canvas is only ever written from bytes on disk.
            if world_canvas_obj is not None:
                if segment_write_executor is not None:
                    segment_write_futures[-1].result()
                _wc_rec = _world_canvas_write_segment(
                    args, world_canvas_obj, world_canvas_extractor, seg_idx,
                    seg_path, world_canvas_global_c2w[seg_idx], world_canvas_audit_path,
                )
                print_info(f"  [canvas] write segment {seg_idx}: "
                           f"frames={_wc_rec.get('frames_written')} "
                           f"cells={_wc_rec.get('cells_written')} "
                           f"total_obs={_wc_rec.get('observed_cells_total')}")

        distributed_barrier(device)
        current_start_image = last_frame_path

    if is_rank0() and segment_write_executor is not None:
        segment_write_executor.shutdown(wait=True)
        for future in segment_write_futures:
            future.result()
        print_info("  ✅ All async segment writes completed.")

    # Concatenate all segments
    if is_rank0():
        output_path = os.path.join(args.output_dir, args.output_name)
        print_info(f"\n🔗 Concatenating {len(segment_paths)} segments...")
        if args.concat_skip_first_frames > 0:
            print_info(f"   Dropping first {args.concat_skip_first_frames} frames from every segment after segment 0")
        concat_videos_ffmpeg(
            segment_paths,
            output_path,
            args.fps,
            skip_first_frames=args.concat_skip_first_frames,
        )
        total_duration = len(segment_paths) * segment_duration
        if args.concat_skip_first_frames > 0 and len(segment_paths) > 1:
            total_duration -= (len(segment_paths) - 1) * args.concat_skip_first_frames / args.fps
        print_info(f"✅ Done! Output: {output_path}")
        print_info(f"   Total duration: {total_duration:.1f}s ({len(segment_paths)} segments)")
        if reactive_nav_records:
            reactive_path = os.path.join(args.output_dir, "reactive_nav_plan.json")
            with open(reactive_path, "w") as f:
                json.dump(reactive_nav_records, f, indent=2)
            print_info(f"   Reactive nav plan: {reactive_path}")

    # ---- MapState: hook report (rank 0) ----
    if mapstate_hook is not None and is_rank0():
        report = {
            "adapter_path": args.mapstate_adapter_path,
            "adapter_format": args.mapstate_adapter_format,
            "block": args.mapstate_block,
            "applied_calls": mapstate_hook.applied_calls,
            "events": mapstate_hook.events,
        }
        report_path = args.mapstate_report or os.path.join(
            args.output_dir, "mapstate_report.json")
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        if mapstate_hook.applied_calls == 0:
            print_info(
                "⚠️  MapState hook applied ZERO denoise calls — the adapter never "
                f"perturbed the model. Report: {report_path} (check events for "
                "grid_mismatch_skip / zero residual)."
            )
        else:
            print_info(
                f"🗺  MapState report: applied_calls={mapstate_hook.applied_calls} "
                f"-> {report_path}"
            )
        mapstate_hook.restore()


if __name__ == "__main__":
    main()

# MKI inference patch applied
