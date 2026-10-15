"""Bounded, exact execution caches for the native DreamX AR inference path.

No model source, weights, kernel, precision, sampler, or cache policy changes.
Acceptance requires an independent native latent/RGB parity benchmark.  This
module only avoids repeating camera-only preparation and integer FA metadata.
It imports no CUDA modules and launches no work until the installer is called.
"""
import ast
from collections import OrderedDict
import copy
import hashlib
import importlib
import inspect
import textwrap


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class NativeExecutionCache:
    def __init__(self, camera_module, attention_module, max_camera_entries=4,
                 max_attention_entries=32):
        if not 1 <= max_camera_entries <= 8 or not 1 <= max_attention_entries <= 128:
            raise ValueError("Execution caches must remain small and bounded")
        self.camera_module = camera_module
        self.attention_module = attention_module
        self.max_camera_entries = int(max_camera_entries)
        self.max_attention_entries = int(max_attention_entries)
        self.camera = OrderedDict()
        self.lengths = OrderedDict()
        self.bindings = []
        self.counters = dict(camera_hits=0, camera_misses=0, camera_bypasses=0,
                             camera_evictions=0, attention_hits=0,
                             attention_misses=0, attention_evictions=0)
        self.camera_original = camera_module._prepare_apply_fns_all_dim
        self.attention_original = attention_module.flash_attention
        self.contract = {}

    @staticmethod
    def _tensor_key(tensor):
        if tensor is None:
            return None
        # Strong references in each entry prevent allocator pointer reuse.  The
        # version protects a camera tensor modified in place between chunks.
        # Inference tensors without version counters are not cacheable.
        return (id(tensor), tensor.data_ptr(), int(tensor._version),
                tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype),
                str(tensor.device))

    @staticmethod
    def _stream_record(device):
        if device.type != "cuda":
            return None
        import torch
        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)
        return (int(stream.cuda_stream), event)

    @staticmethod
    def _wait_if_needed(entry, device):
        if entry is None:
            return
        import torch
        stream = torch.cuda.current_stream(device)
        if int(stream.cuda_stream) != entry[0]:
            stream.wait_event(entry[1])

    def prepare_camera(self, head_dim, viewmats, Ks, patches_x, patches_y,
                       image_width, image_height, coeffs_x=None, coeffs_y=None):
        original = self.camera_original
        # The actual native path uses coeffs=None.  Avoid assuming anything
        # about mutable optional coefficient objects in other callers.
        if coeffs_x is not None or coeffs_y is not None:
            self.counters["camera_bypasses"] += 1
            return original(head_dim, viewmats, Ks, patches_x, patches_y,
                            image_width, image_height, coeffs_x, coeffs_y)
        try:
            key = (int(head_dim), self._tensor_key(viewmats), self._tensor_key(Ks),
                   patches_x, patches_y, image_width, image_height)
        except RuntimeError:
            self.counters["camera_bypasses"] += 1
            return original(head_dim, viewmats, Ks, patches_x, patches_y,
                            image_width, image_height, coeffs_x, coeffs_y)
        entry = self.camera.get(key)
        if entry is not None:
            self.counters["camera_hits"] += 1
            self.camera.move_to_end(key)
            self._wait_if_needed(entry[2], viewmats.device)
            return entry[0]
        self.counters["camera_misses"] += 1
        result = original(head_dim, viewmats, Ks, patches_x, patches_y,
                          image_width, image_height, coeffs_x, coeffs_y)
        # Keep the exact original partial functions and their original tensor
        # inputs alive; no transform is recomputed in a different dtype.
        self.camera[key] = (result, (viewmats, Ks), self._stream_record(viewmats.device))
        if len(self.camera) > self.max_camera_entries:
            self.camera.popitem(last=False)
            self.counters["camera_evictions"] += 1
        return result

    def _attention_entry(self, tensor, batch, length):
        import torch
        device = tensor.device
        key = (device.type, device.index, int(batch), int(length))
        entry = self.lengths.get(key)
        if entry is not None:
            self.counters["attention_hits"] += 1
            self.lengths.move_to_end(key)
            self._wait_if_needed(entry[2], device)
            return entry
        self.counters["attention_misses"] += 1
        # Identical integer construction and cumulative-sum operations as the
        # native function, performed once for each finite sequence shape.
        lengths = torch.tensor([length] * batch, dtype=torch.int32).to(
            device=device, non_blocking=True)
        cumulative = torch.cat([lengths.new_zeros([1]), lengths]).cumsum(
            0, dtype=torch.int32).to(device=device, non_blocking=True)
        entry = (lengths, cumulative, self._stream_record(device))
        self.lengths[key] = entry
        if len(self.lengths) > self.max_attention_entries:
            self.lengths.popitem(last=False)
            self.counters["attention_evictions"] += 1
        return entry

    def _length_tensor(self, tensor, batch, length):
        return self._attention_entry(tensor, batch, length)[0]

    def _cumulative_tensor(self, tensor, batch, length):
        return self._attention_entry(tensor, batch, length)[1]

    def _build_attention(self):
        native = self.attention_original
        source = textwrap.dedent(inspect.getsource(native))
        tree = ast.parse(source)
        original = copy.deepcopy(tree)
        replacements = []
        # Full AST guards mean only the two native unmasked length builders
        # and four FA2/FA3 cumulative metadata expressions can change.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            name = ast.unparse(node.targets[0])
            if name not in ("q_lens", "k_lens"):
                continue
            expected = ast.parse(
                f"torch.tensor([{'lq' if name == 'q_lens' else 'lk'}] * b, "
                "dtype=torch.int32).to(device="
                f"{'q' if name == 'q_lens' else 'k'}.device, non_blocking=True)",
                mode="eval").body
            if _dump(node.value) == _dump(expected):
                before = copy.deepcopy(node.value)
                tensor, length = ("q", "lq") if name == "q_lens" else ("k", "lk")
                node.value = ast.parse(
                    f"_native_execution_length({tensor}, b, {length})", mode="eval").body
                replacements.append((node, before, copy.deepcopy(node.value)))
        if len(replacements) != 2:
            raise ValueError("Native FlashAttention unmasked length anchors changed")
        cumulative_count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg not in ("cu_seqlens_q", "cu_seqlens_k"):
                continue
            side = "q" if node.arg.endswith("_q") else "k"
            length = "lq" if side == "q" else "lk"
            expected = ast.parse(
                f"torch.cat([{side}_lens.new_zeros([1]), {side}_lens]).cumsum("
                "0, dtype=torch.int32).to(q.device, non_blocking=True)", mode="eval").body
            if _dump(node.value) != _dump(expected):
                raise ValueError("Native FlashAttention cumulative-length anchor changed")
            before = copy.deepcopy(node.value)
            node.value = ast.parse(
                f"_native_execution_cumulative({side}, b, {length})", mode="eval").body
            replacements.append((node, before, copy.deepcopy(node.value)))
            cumulative_count += 1
        if cumulative_count != 4:
            raise ValueError("Expected the native FA2 and FA3 q/k cumulative metadata")
        # Revert the six declared expressions in place and compare the entire
        # native function AST; preserve every kernel call and all its kwargs.
        changed = copy.deepcopy(tree)
        for node, before, _ in replacements:
            node.value = before
        if _dump(tree) != _dump(original):
            raise AssertionError("Unexpected native attention math/source change")
        ast.fix_missing_locations(changed)
        namespace = dict(native.__globals__)
        namespace.update(_native_execution_length=self._length_tensor,
                         _native_execution_cumulative=self._cumulative_tensor)
        exec(compile(changed, "<native_execution_cache_v1:flash_metadata>", "exec"), namespace)
        transformed = namespace[native.__name__]

        def dispatch(q, k, v, q_lens=None, k_lens=None, **kwargs):
            # Preserve all non-native/explicit variable-length callers exactly.
            if q_lens is not None or k_lens is not None or q.device != k.device or q.device != v.device:
                return native(q, k, v, q_lens=q_lens, k_lens=k_lens, **kwargs)
            return transformed(q, k, v, q_lens=None, k_lens=None, **kwargs)

        self.contract.update(attention_native_source_sha256=_sha(source),
                             attention_restored_entire_AST_equal=True,
                             attention_changed_expression_count=len(replacements),
                             attention_kernel_and_all_call_kwargs_unchanged=True)
        return dispatch

    def install(self):
        if getattr(self.camera_module, "_native_execution_cache_v1_owner", None) is not None:
            raise RuntimeError("Native execution cache already installed; refuse duplicate")
        attention = self._build_attention()
        # model_2_2 imported flash_attention by value; update that known native
        # binding as well.  Native self/camera attention imports attention(),
        # whose globals already reference attention_module.flash_attention.
        base_model = importlib.import_module("wan.modules.model_2_2")
        if base_model.flash_attention is not self.attention_original:
            raise RuntimeError("Unexpected existing cross-attention wrapper; refuse overwrite")
        bindings = [(self.camera_module, "_prepare_apply_fns_all_dim", self.prepare_camera),
                    (self.attention_module, "flash_attention", attention),
                    (base_model, "flash_attention", attention)]
        for module, name, replacement in bindings:
            old = getattr(module, name)
            setattr(module, name, replacement)
            self.bindings.append((module, name, old, replacement))
        self.camera_module._native_execution_cache_v1_owner = self
        self.contract.update(camera_cache_exact_native_partial_functions=True,
                             camera_key_includes_strong_tensor_identity_pointer_version=True,
                             model_source_files_modified=False, model_weights_modified=False,
                             kernel_precision_sampler_or_noise_changed=False,
                             native_latent_RGB_parity_acceptance="NOT_RUN_BY_INSTALLER")
        return self

    def audit(self):
        return dict(version="native_execution_cache_v1", counters=dict(self.counters),
                    camera_entries=len(self.camera), attention_entries=len(self.lengths),
                    max_camera_entries=self.max_camera_entries,
                    max_attention_entries=self.max_attention_entries,
                    contract=dict(self.contract))

    def remove(self):
        for module, name, old, replacement in reversed(self.bindings):
            if getattr(module, name) is not replacement:
                raise RuntimeError("Binding changed after installation; refuse unsafe restore")
            setattr(module, name, old)
        self.bindings.clear()
        if getattr(self.camera_module, "_native_execution_cache_v1_owner", None) is self:
            delattr(self.camera_module, "_native_execution_cache_v1_owner")
        self.camera.clear()
        self.lengths.clear()


def install_native_execution_cache(pipeline=None, *, camera_module=None,
                                   attention_module=None, max_camera_entries=4,
                                   max_attention_entries=32):
    """Install once in an isolated idle-GPU worker; return audit/remove handle.

    pipeline is accepted for a clear installer interface but is never mutated.
    Import resolution must already point at the caller's frozen native repo.
    """
    camera_module = camera_module or importlib.import_module("wan.modules.camera_prope")
    attention_module = attention_module or importlib.import_module("wan.modules.attention")
    return NativeExecutionCache(camera_module, attention_module,
                                max_camera_entries, max_attention_entries).install()


# Short interface consumed by the independent bounded benchmark worker.
install_execution_cache = install_native_execution_cache
