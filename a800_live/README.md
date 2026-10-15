# A800 map-conditioned interactive native generation

This is the current deployed-source snapshot, updated 2026-10-16. It extends the original offline map2world4090 experiment with continuous keyboard-controlled DreamX AR generation, whole-map switching/upload, automatic cruise, prompt controls, optional pose-indexed 2D context retrieval, and a lower-latency native decoder pipeline. It does not introduce new trained weights or establish map accuracy, persistent landmark identity, or game-level responsiveness.

The earlier implementation remains available in `live_service_v5.py`, `live_index_v10.html`, and `live_index_v12.html`. See `LEGACY_LIVE_README.md` for that historical deployment. The new entry point is **`service_v3.py`**, serving **`interactive_index_v3.html`**.

## Current execution

- Native DreamX causal AR checkpoint, BF16, four denoising steps plus native clean-context refresh; native Wan VAE.
- Three resident A800 components: visible CUDA0 = DiT, CUDA1 = video decoder, CUDA2 = separate map/reference encoder. This is component separation, not four-way DiT acceleration. An additional free GPU was used for an isolated VAE parity check and then released.
- Native resolutions: 640x384 and 768x448. Playback target: **16 fps**. A 24 fps / 30 fps mode is not established.
- Native three-latent chunks with bounded rolling KV, decoder and archive queues. The old approximately 75-second allocation cap is removed. All native raw latent/RGB chunks remain on disk; RAM buffers are bounded. Disk archives still grow with session duration.
- Map changes are idle-only transactions. The map remains 16 globally pooled conditioning tokens; geographic alignment is unverified. Changing a map retains the reference image unless that input is separately changed.
- Optional memory retrieves same-map/session camera-near latent context and refreshes only native sink KV. It generates fresh RGB rather than playing old frames. Revisit identity remains unaccepted.

## Low-latency implementation

`NativePixelParts.pixels()` preserves the original whole-chunk normalization and `conv2`, then publishes each native latent's RGB as the unchanged cached decoder produces it. The first chunk emits 1/4/4 frames; subsequent chunks emit 4/4/4. A separate JPEG thread prevents encoding all 12 frames from blocking the next VAE decode. The browser uses a bounded three-frame display queue and a smaller startup buffer. No RGB interpolation, frame repetition, geometry correction, or new weights are used.

A completed matched run used the same 24 chunks, input noise, camera poses and checkpoint at 768x448. **All 285 raw RGB frames were exactly equal** before and after the pipeline change. Full job time including native decode/archive fell from 19.2960s to 15.7298s: 284 fresh frames / elapsed time = **14.7181 to 18.0549 fps**. This is one paired engineering measurement, not a statistical speed benchmark or a visual-quality improvement.

An actual interactive run saved and independently decoded **3069 frames / 191.8125s**. Its latest periodic browser report covered 3040 displayed frames: mean interval62.50ms, P9567.5ms, max100.1ms, dropped0. The last partial report may be absent. Fresh-input marker to displayed-frame acknowledgements were **1.026–1.591s** (cold start1.395s). These are metadata acknowledgements, not independent RGB-motion onset measurements. The interface still does **not** provide immediate game-like control. Very short press/release events between chunk samples can be missed; event-time integration is not implemented in this snapshot.

## Deployment contract

This is an **existing-installation snapshot, not a standalone installer**. It uses the verified upstream AR revision `a1f4c6e5e45600718e5236955f2e0702e53fc275`. Model weights and runtime artifacts are excluded. Preserve upstream license/model terms and use the prepared native environment.

The prepared experiment root must contain:

- `repo_sp_v1/`: the compatible upstream AR code/configs;
- `inputs/input.json`: local `map_path`, `image_path`, `caption`;
- `inputs/initial_noise.pt`, `inputs/cameras.pt`;
- `outputs/single_map_bulk/raw_latent.pt` and `rgb_uint8.pt` for the native startup parity check;
- `low_latency_parts_preflight_v3/COMPLETE.json`: an actually completed native-VAE parity result, not a fabricated flag;
- `interactive_prompt_quality_v2/PROMPT_PRESETS_V2.json` and completed `quality_comparison/neutral_scene/CHUNKS.json` from the matched fixed route. Prompt-review MP4s are optional for the service, but required to display the old comparison videos.

`Runtime` resolves models relative to the experiment root's grandparent: `models/DreamX-World-5B/model.safetensors` and `models/Wan2.2-TI2V-5B/` including native VAE, UMT5 and tokenizer assets. Adapt local deployment configuration explicitly if your layout differs.

Copy this source set into a **fresh isolated code directory**. Create a private `RESTORE_SOURCE_V3.json` there using `RESTORE_SOURCE_V3.example.json`, filling in the approved map path, its actual SHA256, source dimensions and desired prompt/resolution. The service validates the original-caption contract and completed matching reference data at startup. It writes a new `pipeline_parity_v3/` and verifies all285 raw RGB hashes against the prior neutral fixed route before becoming ready. Do not rerun an already COMPLETE parity job or overwrite a previous code/run directory.

Select three authorized, verified-free A800 UUIDs in the visible order DiT / decoder / map encoder. Verify that no existing owner is using the same experiment or port. On Linux, a CPU ownership lock can wrap startup:

```bash
export CUDA_VISIBLE_DEVICES="$DIT_GPU_UUID,$DECODE_GPU_UUID,$MAP_GPU_UUID"
export DREAMX_MAPS_ROOT=/path/to/approved/maps
flock -n "$CODE/interactive.owner.lock" \
  python "$CODE/service_v3.py" --root "$EXPERIMENT_ROOT" --code "$CODE" --port 18767
```

`CODE`, `EXPERIMENT_ROOT` and the three UUID variables are deployment inputs, not included secrets. Never launch a second owner alongside healthy existing workers. The original machine-specific launcher and private restore file are deliberately excluded. Startup/benchmark records must come from real execution; this snapshot does not generate or fake prerequisites.

The service binds loopback18767. Use your existing SSH login/forward local18765 -> remote18767. Open `http://127.0.0.1:18765/`. The new integrated `/maps` catalog reads `DREAMX_MAPS_ROOT` (default `inputs/map_build/map`); the browser submits opaque IDs, not server paths. Stop saves at a native chunk boundary. “End/release GPU” closes the owner. Idle shutdown is eight hours. This commit neither starts nor restarts any GPU task.

## Code and functions

| File | Main responsibility |
| --- | --- |
| `service_v3.py: main`, nested `State.post`, `worker` | Loopback owner, idle-only settings/map transactions, startup parity, session lifecycle |
| `runtime_v1.py: Runtime.configure/reset/startup_parity` | Persistent component placement, map/reference encoding, native parity |
| `native_stepper_v1.py: build_step_kernel`, `NativeStepper.step` | Guarded native AST extraction; persistent three-latent generation |
| `session_v3.py: LiveCamera`, `run_session` | Camera controls, bounded decode/JPEG/archive pipeline, per-frame delivery and evidence |
| `native_pixel_parts_v3.py: NativePixelParts.pixels` | Native decoder output in early parts, identical RGB |
| `spatial_memory_v1.py: SpatialMemoryBank`, `NativeSinkMemoryAdapter.recache` | Pose-indexed disk/SQLite history; scoped sink-context refresh |
| `memory_selection_v1.py: PoseMemoryIndex.query` | Fixed distance/yaw/map/session/recent-history eligibility |
| `prompt_policy_v2.py: prompt_record/settings_job` | Prompt validation and idle-only settings |
| `map_revision_v2.py: request/apply_job` | Validated map transactions and rollback |
| `native_execution_cache_v1.py: install_execution_cache` | Bounded native camera/attention metadata caches |
| `interactive_index_v3.html` | Keyboard/cruise, maps, prompts, bounded native frame pacing, three-stage command indicators |
| `quality_review_v2.py: serve_review` | Existing same-condition prompt review, with no new inference |

Run CPU-only checks from this directory in the prepared environment:

```bash
python -m unittest test_memory_selection_v1 test_prompt_policy_v2 test_map_revision_v2
```

`SOURCE_MANIFEST_PUBLIC_V3.json` covers the exact public payload. `PERFORMANCE_SUMMARY_V3.json` contains sanitized measured results. Credentials, internal hostnames/paths, GPU assignments, model weights, satellite maps, videos, session archives and personal runtime logs are not published. No training, unified2D/3D success, scientific PASS, stable return identity, or photorealistic-town claim is made.
