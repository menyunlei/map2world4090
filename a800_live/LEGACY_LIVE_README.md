# A800 native live demo and server map picker

This directory preserves the deployed A800 live service and its latest map chooser. It supplements the original offline 4090 demo; it does not replace its model or claims.

Features: keyboard control, automatic cruise, paced native RGB stream, smooth camera controls, whole-map upload, and selection of existing server maps through a read-only CPU catalog. Click a thumbnail, preview, explicitly Apply, then Start. Applying during generation is rejected. Existing video outputs and map revisions remain intact.

## Existing A800 installation

These files are a deployment snapshot, not a standalone installer. The GPU service expects the prepared experiment root: `repo_sp_v1/` containing the official AR code, `inputs/input.json`, `inputs/initial_noise.pt`, `inputs/cameras.pt`, the original reference assets, and `outputs/single_map_bulk/{raw_latent.pt,rgb_uint8.pt}` for native warmup parity. It also requires the original checkpoint paths specified in the input/config. Weights and these large runtime artifacts are deliberately excluded. Use the verified upstream AR revision `a1f4c6e5e45600718e5236955f2e0702e53fc275` and the existing prepared environment. Do not launch a second GPU owner on an active installation.

The example configuration serves the GPU service on loopback18767 and uses a local SSH forward18765 for the browser. `live_service_v5.py --root <prepared-experiment-root> --port 18767` starts a resident model/VAE pair only after ownership and free GPUs have been verified externally. Use `CUDA_VISIBLE_DEVICES` with the authorized UUIDs; no UUID allocation is hard-coded in this service. Preserve the service's owner records and never restart it just to replace static HTML.

The service reads `live_index_v10.html`; its contents here are the latest V12 UI. Copying only that static frontend into an existing compatible root does not require restarting the GPU owner.

## Server maps

`server_map_catalog_v1.py` is CPU-only and read-only. Set the trusted startup environment variable `DREAMX_MAPS_ROOT` to your approved map directory (default: relative `inputs/map_build/map`). The HTTP API never accepts filesystem paths. No map image assets are added by this commit.

Copy the catalog and launcher beside the service files in the experiment root, then run `DREAMX_MAPS_ROOT=/path/to/approved/maps python launch_map_catalog_v1.py` once. A process lock and owner audit prevent duplicate catalog owners. It binds loopback18769. Forward local18769 to remote18769 alongside the existing18765→18767 SSH tunnel. The frontend uses `http://127.0.0.1:18769`.

API: GET `/maps` returns opaque IDs/names/dimensions; GET `/maps/thumbnail?id=...` serves previews; GET `/maps/image?id=...` serves original images. Direct eligible images only, escaping symlinks excluded, unknown IDs404. Exact CORS origins are `http://127.0.0.1:18765` and `http://localhost:18765`, allowing canvas use for explicit Apply. Update both frontend API URL and CORS allowlist if changing ports.

## Validation and limits

The five full-image and five thumbnail requests were checked, as were unknown-ID/path rejection and foreign-Origin rejection. Actual browser selection and Apply of12.jpg succeeded and produced new native map tokens while the existing GPU owner stayed unchanged. The map revision CPU tests cover size bounds, busy-state exclusion and rollback. No additional model inference was required for the gallery change.

Native generation is640×384 at16fps with a four-step AR sampler and native cached VAE, using one A800 for model and one for decoding. Previously measured fresh-frame generation was approximately16.65fps; this is not zero-latency input response. Each native session remains capped at roughly75seconds. Changing the map does not change the original starting reference image/caption. Map geographic fidelity and revisit identity are unverified. No training, interpolation, new model weights, or explicit3D-world guarantees are introduced.
