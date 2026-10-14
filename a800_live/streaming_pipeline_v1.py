"""Add an output callback to the official AR inference method by checked AST.

No source files, native classes, model weights, denoising schedules, noise draws
or cache policies are modified. The returned function must be bound explicitly
to the caller's pipeline instance. This module launches no jobs.
"""
import ast
import copy
import hashlib
import inspect
import textwrap


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _temporal_loop(function):
    candidates = []
    for node in ast.walk(function):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Tuple):
            continue
        names = [item.id if isinstance(item, ast.Name) else None for item in node.target.elts]
        if names == ["i", "current_num_frames"]:
            candidates.append(node)
    if len(candidates) != 1:
        raise ValueError(f"Expected one native temporal chunk loop, found{len(candidates)}")
    return candidates[0]


def _chunk_increment(node):
    return (
        isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name) and node.target.id == "current_start_frame"
        and isinstance(node.op, ast.Add)
        and isinstance(node.value, ast.Name) and node.value.id == "current_num_frames"
    )


def _context_refresh(node):
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    return (
        isinstance(call.func, ast.Attribute) and call.func.attr == "generator"
        and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
        and any(keyword.arg == "timestep" and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "context_timestep" for keyword in call.keywords)
    )


def _final_decode(node):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    if not isinstance(node.targets[0], ast.Name) or node.targets[0].id != "video":
        return False
    call = node.value
    return (
        isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        and call.func.attr == "decode_to_pixel"
        and isinstance(call.func.value, ast.Attribute) and call.func.value.attr == "vae"
        and isinstance(call.func.value.value, ast.Name) and call.func.value.value.id == "self"
        and len(call.args) == 1 and isinstance(call.args[0], ast.Name) and call.args[0].id == "output"
    )


def build_stream_inference(native_pipeline_class=None):
    """Return a dynamically compiled inference_stream_v1 function.

    Additional keyword-only arguments are chunk_callback=None and
    decode_final=False. The callback receives (detached chunk, global latent
    start, latent frame count), after the native context refresh/cache commit
    and before incrementing the native cursor. It must preserve diffusion RNG
    state; the intended callback only queues native VAE decoding.

    Function attributes: native_source_sha256, transform_metadata,
    transformed_source. With return_latents=True/decode_final=False, the return
    value is (None, raw native output latent). Native profiling is unsupported
    when final decode is skipped; the worker uses profile=False and records
    streaming measurements separately.
    """
    if native_pipeline_class is None:
        from pipeline.pipeline_causal_camera import CausalCameraInferencePipeline
        native_pipeline_class = CausalCameraInferencePipeline
    native = native_pipeline_class.inference
    source = textwrap.dedent(inspect.getsource(native))
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "inference":
        raise ValueError("The native source must contain exactly one inference function")
    original = copy.deepcopy(functions[0])
    transformed = copy.deepcopy(original)
    transformed.name = "inference_stream_v1"
    parameter_names = {arg.arg for arg in transformed.args.posonlyargs + transformed.args.args + transformed.args.kwonlyargs}
    if parameter_names.intersection({"chunk_callback", "decode_final"}):
        raise ValueError("Refusing to transform an already extended native function")
    transformed.args.kwonlyargs.extend([ast.arg(arg="chunk_callback"), ast.arg(arg="decode_final")])
    transformed.args.kw_defaults.extend([ast.Constant(value=None), ast.Constant(value=False)])

    loop = _temporal_loop(transformed)
    refreshes = [i for i, statement in enumerate(loop.body) if _context_refresh(statement)]
    increments = [i for i, statement in enumerate(loop.body) if _chunk_increment(statement)]
    if len(refreshes) != 1 or len(increments) != 1 or refreshes[0] >= increments[0]:
        raise ValueError("Native context refresh and cursor increment anchors are not unique/in order")
    callback = ast.parse(
        "if chunk_callback is not None:\n"
        "    chunk_callback(denoised_pred.detach(), current_start_frame, current_num_frames)\n"
    ).body[0]
    loop.body.insert(increments[0], callback)

    decodes = [i for i, statement in enumerate(transformed.body) if _final_decode(statement)]
    if len(decodes) != 1:
        raise ValueError(f"Expected one native final decode, found{len(decodes)}")
    decode_index = decodes[0]
    decode_statements = transformed.body[decode_index:decode_index + 2]
    expected_mapping = ast.parse("video = (video * 0.5 + 0.5).clamp(0, 1)").body[0]
    if len(decode_statements) != 2 or _dump(decode_statements[1]) != _dump(expected_mapping):
        raise ValueError("Native final RGB mapping differs from the frozen official pattern")
    decode_wrapper = ast.If(test=ast.Name(id="decode_final", ctx=ast.Load()),
                            body=decode_statements,
                            orelse=ast.parse("video = None").body)
    transformed.body[decode_index:decode_index + 2] = [decode_wrapper]
    guard = ast.parse(
        "if profile and not decode_final:\n"
        "    raise ValueError('Native profile=True requires decode_final=True; streaming worker uses profile=False')\n"
    ).body[0]
    # Keep the docstring in its normal position if the native method has one.
    guard_index = 1 if transformed.body and isinstance(transformed.body[0], ast.Expr) and isinstance(transformed.body[0].value, ast.Constant) and isinstance(transformed.body[0].value.value, str) else 0
    transformed.body.insert(guard_index, guard)

    # Undo only the specified additions in a copy and require the ENTIRE native
    # function AST to match. This guards against an accidental change to its
    # sampler/noise/I0 mask/cache/context logic, including all native branches.
    restored = copy.deepcopy(transformed)
    restored.name = original.name
    restored.args = copy.deepcopy(original.args)
    restored.body.pop(guard_index)
    restored_loop = _temporal_loop(restored)
    restored_callbacks = [i for i, statement in enumerate(restored_loop.body) if _dump(statement) == _dump(callback)]
    if len(restored_callbacks) != 1:
        raise AssertionError("Callback insertion is not exactly reversible")
    restored_loop.body.pop(restored_callbacks[0])
    wrappers = [i for i, statement in enumerate(restored.body) if _dump(statement) == _dump(decode_wrapper)]
    if len(wrappers) != 1:
        raise AssertionError("Final decode wrapper is not exactly reversible")
    wrapper_index = wrappers[0]
    restored.body[wrapper_index:wrapper_index + 1] = copy.deepcopy(restored.body[wrapper_index].body)
    if _dump(restored) != _dump(original):
        raise AssertionError("Transform changed native semantics beyond the declared callback/decode switch")

    module = ast.fix_missing_locations(ast.Module(body=[transformed], type_ignores=[]))
    transformed_source = ast.unparse(module) + "\n"
    namespace = dict(native.__globals__)
    compile_name = inspect.getsourcefile(native) or "<official_ar_inference>"
    exec(compile(module, compile_name + ":streaming_v1", "exec"), namespace)
    function = namespace["inference_stream_v1"]
    function.native_source_sha256 = _sha(source)
    function.transformed_source = transformed_source
    function.transform_metadata = {
        "native_source_file": inspect.getsourcefile(native),
        "native_source_sha256": _sha(source),
        "native_function_ast_sha256": _sha(_dump(original)),
        "transformed_source_sha256": _sha(transformed_source),
        "restored_original_ast_equal": True,
        "temporal_loop_count": 1,
        "context_refresh_anchor_count": 1,
        "cursor_increment_anchor_count": 1,
        "final_decode_anchor_count": 1,
        "callback_location": "after native context refresh/cache update, before current_start_frame increment",
        "callback_arguments": ["denoised_pred.detach()", "current_start_frame", "current_num_frames"],
        "new_keyword_only_parameters": {"chunk_callback": None, "decode_final": False},
        "no_final_decode_return_with_return_latents": "(None, output)",
        "sampler_noise_I0_mask_context_refresh_cache_logic_preserved": True,
        "native_class_and_source_files_modified": False,
        "callback_must_not_change_diffusion_rng": True,
        "streamed_decoder": "provided separately by caller via official native cached_decode API",
        "native_full_vs_cached_decode_bitwise_parity_claimed": False,
        "new_trainable_parameters": 0,
    }
    return function
