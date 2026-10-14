"""Keep native KV cursor metadata on CPU, avoiding CUDA scalar synchronizations.

K/V values, model operations, cache policy and sampler remain native. This is
an execution-only optimization; acceptance requires actual raw-latent parity.
"""
import ast
import copy
import types


def install_cpu_cache_indices(pipeline, stream_function):
    import torch
    from pipeline.pipeline_causal_camera import CausalCameraInferencePipeline
    tree = ast.parse(stream_function.transformed_source)
    original = copy.deepcopy(tree)
    changes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1: continue
        target = ast.unparse(node.targets[0])
        if not any(target.endswith(f"['{name}']") for name in ['global_end_index','local_end_index']):continue
        if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != 'torch.tensor':continue
        for keyword in node.value.keywords:
            if keyword.arg == 'device':
                assert ast.unparse(keyword.value) == 'noise.device'
                keyword.value = ast.Constant(value='cpu'); changes.append(target)
    assert len(changes) == 2, changes
    # Reverse the two location changes and compare the full inference AST.
    restored = copy.deepcopy(tree)
    for node in ast.walk(restored):
        if isinstance(node, ast.Assign) and len(node.targets)==1 and ast.unparse(node.targets[0]) in changes:
            for keyword in node.value.keywords:
                if keyword.arg == 'device':keyword.value=ast.parse('noise.device',mode='eval').body
    assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
    ast.fix_missing_locations(tree)
    namespace = dict(CausalCameraInferencePipeline.inference.__globals__)
    exec(compile(tree, '<native_stream_cpu_indices_v1>', 'exec'), namespace)
    function = namespace['inference_stream_v1']
    allocate = pipeline._initialize_kv_cache
    def initialize(self, *args, **kwargs):
        allocate(*args, **kwargs)
        for cache in self.kv_cache1:
            for key in ['global_end_index','local_end_index']:
                assert cache[key].numel()==1 and cache[key].item()==0
                cache[key] = torch.zeros(1, dtype=torch.long, device='cpu')
    pipeline._initialize_kv_cache = types.MethodType(initialize,pipeline)
    function.cpu_index_audit = {'reset_assignments':changes,'all_other_stream_ast_exact':True,
                               'native_kv_values_and_updates_unchanged':True}
    return function
