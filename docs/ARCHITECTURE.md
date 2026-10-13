# DreamX map10 turn runs: implementation-grounded architecture

This diagram describes the successful left/right turn inference runs on 2026-10-13, not every optional branch in the repository. No new training or shared inference-code edits were performed for this diagram or these two turn runs.

## What is reused, and what changes relative to the Wan base

| Component | Actual behavior |
|---|---|
| Base assets | Wan2.2-TI2V-5B VAE and uMT5 text encoder |
| Video Transformer weights | DreamX-World-5B-Cam; 30 blocks, width 3072, 24 heads |
| Camera conditioning | A parallel PRoPE self-attention branch in every active block, with weights loaded from the existing DreamX checkpoint |
| Map conditioning | Training-free map-to-token path, reusing the same VAE and Transformer patch embedding; prepend map tokens to projected text context |
| Sampling | Existing flow-matching inference machinery, Flow_DPM++, 15 steps, CFG 3.0; re-lock the first latent frame after each update |
| Output | 193 newly generated RGB frames at 640×384, exported at 16 fps; the 17.0625-second preview also includes a reused approach segment |

The class name “Transformer3D” refers here to spatiotemporal processing of a tensor with time, height and width axes. This enabled path does not construct an explicit XYZ city representation.

## 1. VAE: unchanged model, reused for multiple inputs

The chosen class is AutoencoderKLWan3_8, loaded from the Wan2.2_VAE.pth base asset with the run configuration overriding spatial_compression_ratio to 16 and temporal_compression_ratio to 4. The actual latent channel count is 48.

For this run, the video diffusion state is:

    z_sigma: [B, 48, 49, 24, 40]
    49 = 1 + (193 - 1) / 4
    24 = 384 / 16
    40 = 640 / 16

The VAE encoder also encodes the starting RGB input after it has been prepared as a video-sized input. Only its first latent temporal slice, z_start[:, :, 0:1], is retained as the hard start constraint in this run; the full prepared encoding can have multiple temporal slices. After the final sampling update, the VAE decoder turns the clean latent into RGB video.

The map image is resized to the actual video size, 640×384, and encoded using the same VAE. Its logged latent shape is [1,48,1,24,40]. Some legacy comments in map_injector.py mention 16 latent channels and other resolutions; those are not the measured dimensions in these runs.

Sources:
- ../inference_long_video.py:645, model/VAE loading
- ../configs/wan2.2/wan_ti2v_5b.yaml, actual compression-ratio configuration
- ../pipeline/pipeline_dreamxworld.py:350, latent shape
- ../pipeline/pipeline_dreamxworld.py:774, starting-image encoding and initial mask
- ../pipeline/pipeline_dreamxworld.py:446, final VAE decoding
- ../examples/left/measured_run.json and the original run log, actual map latent shape

## 2. Map injection: additional cross-attention context

The active function is encode_map_to_dit_tokens, not the spatial-K/V or map-noise alternatives in the same source file.

    M_raw = VAE.encode(resize(map_RGB)).latent_dist.mode()
    M_norm = LayerNorm(M_raw)
    M_pool = AdaptiveAvgPool3d(M_norm, output_size=(1,8,8))
    M_embed = existing_DiT_patch_embedding(M_pool)
    M = flatten(M_embed)                         # [1,16,3072]
    M_scaled = 0.6 * M

The function contains channel-compatibility padding/truncation, but the actual 48-channel VAE output already matches the 48-channel patch-embedding input. No 16→48 padding is needed in this run.

The Conv3D patch embedding has patch/stride [1,2,2]. Thus the 8×8 map grid becomes a 4×4 grid: 16 map tokens.

The uMT5 text context has 512 padded slots of width 4096. The existing Transformer text projection maps it to width 3072. The Transformer prepends the 16 map tokens after this text projection:

    C = concat(M_scaled, projected_text, dim=sequence)
      = [B,528,3072]

Every Transformer block receives the same extended context. In cross-attention, queries come from video tokens; keys and values come from C through the existing attention projections.

This creates a soft conditioning signal. It does not establish a one-to-one correspondence between a particular map building and a particular perspective-image location, and there is no explicit world-to-image geometric projection in this active map path.

Sources:
- ../map_injector.py:34, encode_map_to_dit_tokens
- ../inference_long_video.py:1723, map encoder call with actual sample size
- ../models/wan_transformer3d.py:1130, projected text context
- ../models/wan_transformer3d.py:1189, map/text context concatenation
- ../models/wan_transformer3d.py:383, cross-attention Q/K/V

## 3. Camera control: parallel attention rather than a 3D renderer

The designed map polyline is compiled into signed heading changes and native movement actions. ActionToPoseFromID builds camera pose sequences. GetPoseEmbedsFromPosesPrope aligns the 193 camera poses to the VAE's 49 temporal positions and returns relative view matrices and normalized default camera intrinsics K.

The matrices are relative camera controls; the starting RGB frame has not been independently registered to a precise satellite-map camera pose.

The active block computes:

    h = norm_and_timestep_modulation(x)
    y_rope  = original_self_attention(h)
    y_cam   = camera_self_attention_PRoPE(h, viewmats, K)
    y       = y_rope + y_cam
    x       = x + timestep_gate * y
    x       = x + cross_attention(norm(x), C)
    x       = x + timestep_gate * FFN(modulated_norm(x))

The camera branch has independent Q/K/V and output projections. It applies prope_qkv with viewmats and K, computes attention, applies the corresponding output transform, then projects back to the model width.

The original self-attention branch remains. Camera self-attention is a parallel residual contribution, not a replacement for the ordinary spatiotemporal RoPE branch. Both left/right runs also change the direction words in the prompt; they are not a prompt-controlled-camera ablation.

Sources:
- ../utils/inference_utils.py:346, GetPoseEmbedsFromPosesPrope
- ../models/wan_transformer3d.py:210, PropeSelfAttention
- ../models/wan_transformer3d.py:461, camera-branch construction
- ../models/wan_transformer3d.py:528, parallel camera residual
- ../examples/left/CAMERA_AUDIT.json

## 4. Sampling: same flow framework, additional conditions

The video patch embedding converts the latent state into:

    grid = [49,12,20]
    tokens = [B,11760,3072]

The DiT output head and unpatchify operation produce the flow prediction with the original latent shape. The code names this output noise_pred, but it is consumed by a flow-matching scheduler.

CFG uses guidance_scale=3.0:

    v_cfg = v_negative + 3.0 * (v_positive - v_negative)

The implementation duplicates the camera controls and expands the same map context across both CFG branches. The positive and negative text contexts differ.

For each of the 15 Flow_DPM++ steps:

    z_next = scheduler.step(v_cfg, timestep, z_current)
    z_next = (1 - mask) * z_start + mask * z_next

With start_image_lock_frames=1, mask is zero only at the first latent temporal slice and one at later slices. The first latent slice is also locked before sampling. VAE decoding happens after the loop, not at every denoising step.

Sources:
- ../pipeline/pipeline_dreamxworld.py:924, duplicated camera controls
- ../pipeline/pipeline_dreamxworld.py:1100, Transformer call with map_tokens and y_camera
- ../pipeline/pipeline_dreamxworld.py:1118, classifier-free guidance
- ../pipeline/pipeline_dreamxworld.py:1178, scheduler update
- ../pipeline/pipeline_dreamxworld.py:1202, first-frame latent re-lock

## Provenance and scope

The current copied inference_long_video.py must match the recorded hash from the successful turn run:
b7933348bf03af4094ee4f2cd13667ba12e52e9116fbc1cac803951289370921.

Other implementation files were copied from the authorized server for this explanation. Their hashes are recorded in SOURCE_PROVENANCE.json. They are current inspection snapshots; historical per-file hashes were not recorded during the turn run.

The active model load log reports 0 missing and 0 unexpected Transformer keys, and 6132.802752 million parameters for the loaded Transformer. “5B” is the checkpoint family name.

The original ground-level first image came from Codex's built-in ImageGen with explicit map-layout instructions. A generated approach segment then supplied the shared junction frame used by these turn runs. No specific underlying ImageGen model version is asserted by these records.

The generated architecture image is explanatory artwork. Numeric dimensions and interface placement were checked against source and logs; detailed implementation evidence is this document and the copied source files.
