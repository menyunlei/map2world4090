"""Portable launcher for the recorded map10 left/right turn event."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=["left", "right"], default="right")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--model-dir", default="models/Wan2.2-TI2V-5B")
    parser.add_argument("--camera-model-dir", default="models/DreamX-World-5B-Cam")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    resolve = lambda value: str((root / value).resolve())
    model, camera = resolve(args.model_dir), resolve(args.camera_model_dir)
    if not Path(model).exists() or not Path(camera).exists():
        parser.error("Download the base and camera checkpoints first; see README.md.")
    out = Path(resolve(args.output_dir or f"outputs/map10_{args.direction}_turn"))
    out.mkdir(parents=True, exist_ok=True)
    route = resolve(f"examples/{args.direction}/clip_route.json")
    item = json.loads((root / f"examples/{args.direction}/input.json").read_text())[0]
    item["image_path"] = resolve("inputs/map10/junction_start.png")
    item["route_polyline_source"] = route
    config = out / "input.json"
    config.write_text(json.dumps([item], indent=2))
    speed = "1.6692156929483608"
    cmd = [sys.executable, "inference_long_video.py",
        "--input_json", str(config), "--output_dir", str(out),
        "--output_name", f"map10_{args.direction}_turn_event.mp4",
        "--config_path", "configs/wan2.2/wan_ti2v_5b.yaml",
        "--model_name", model, "--transformer_path", camera,
        "--target_duration", "12", "--sample_size", "384", "640",
        "--video_length", "193", "--fps", "16",
        "--guidance_scale", "3.0", "--num_inference_steps", "15", "--seed", "42",
        "--weight_dtype", "bfloat16", "--ulysses_degree", "1", "--ring_degree", "1",
        "--cam_method", "prope", "--add_control_adapter", "--sampler_name", "Flow_DPM++",
        "--GPU_memory_mode", "model_cpu_offload", "--cuda_mem_fraction", "0.90",
        "--ground_lock_camera", "--start_image_lock_frames", "1", "--cache_prompt_embeds",
        "--map_driven_camera", "--map_driven_route_polyline", route,
        "--map_driven_action_method", "heading_delta", "--map_driven_sub_segments", "12",
        "--map_driven_turn_threshold_deg", "10",
        "--map_driven_min_turn_speed", speed, "--map_driven_max_turn_speed", speed,
        "--map_inject_path", resolve("inputs/map_build/map/10.jpg"),
        "--map_inject_strength", "0.6", "--negative_prompt", item["negative_prompt_extra"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    raise SystemExit(subprocess.call(cmd, cwd=root, env=env))

if __name__ == "__main__":
    main()

