"""Query live input once per native 3-latent chunk; no native math changes."""
import ast
import copy
import hashlib


def build_live_inference(pipeline):
    from streaming_pipeline_v1 import build_stream_inference
    from cpu_cache_indices_v1 import install_cpu_cache_indices
    from pipeline.pipeline_causal_camera import CausalCameraInferencePipeline
    original = build_stream_inference(CausalCameraInferencePipeline)
    install_cpu_cache_indices(pipeline, original)
    tree = ast.parse(original.transformed_source)
    # Match exactly the already-output-verified CPU scalar placement.
    reset_changes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1: continue
        target = ast.unparse(node.targets[0])
        if target.endswith("['global_end_index']") or target.endswith("['local_end_index']"):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'torch.tensor':
                for keyword in node.value.keywords:
                    if keyword.arg == 'device':
                        assert ast.unparse(keyword.value) == 'noise.device'
                        keyword.value = ast.Constant(value='cpu'); reset_changes.append(target)
    assert len(reset_changes) == 2
    base = copy.deepcopy(tree)
    fn = tree.body[0]
    fn.args.kwonlyargs.append(ast.arg(arg='camera_callback'))
    fn.args.kw_defaults.append(ast.Constant(value=None))
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For) and
             ast.unparse(n.target) == '(i, current_num_frames)']
    assert len(loops) == 1
    loop = loops[0]
    candidates = [(i, n) for i, n in enumerate(loop.body) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'isinstance(y_camera, dict)']
    assert len(candidates) == 1
    index, camera_if = candidates[0]
    wrapper = ast.parse('if camera_callback is not None:\n    y_camera_latents = camera_callback(current_start_frame, current_num_frames)\n').body[0]
    wrapper.orelse = [camera_if]
    loop.body[index] = wrapper
    restored = copy.deepcopy(tree)
    restored.body[0].args = copy.deepcopy(base.body[0].args)
    restored_loop = [n for n in ast.walk(restored) if isinstance(n, ast.For) and
                     ast.unparse(n.target) == '(i, current_num_frames)'][0]
    restored_loop.body[index] = copy.deepcopy(camera_if)
    assert ast.dump(restored, include_attributes=False) == ast.dump(base, include_attributes=False)
    ast.fix_missing_locations(tree)
    namespace = dict(CausalCameraInferencePipeline.inference.__globals__)
    exec(compile(tree, '<native_ar_live_camera_v1>', 'exec'), namespace)
    function = namespace['inference_stream_v1']
    source = ast.unparse(tree)
    function.live_audit = {'native_source_sha256': original.native_source_sha256,
        'live_transform_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'camera_provider_only_change_after_verified_CPU_cache_placement': True,
        'sampler_noise_I0_KV_RoPE_context_refresh_and_codec_unchanged': True,
        'camera_query': 'latest input at start of each native3-latent chunk',
        'new_weights_or_training': False}
    return function
