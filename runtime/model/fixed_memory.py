# Portions adapted from Transformers Qwen3.5; modified for YANchor inference.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Licensed under Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations
from typing import Any
from torch import Tensor, nn
import torch
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer, LinearAttentionLayer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5DecoderLayer, Qwen3_5ForCausalLM, Qwen3_5GatedDeltaNet, Qwen3_5ModelOutputWithPast, Qwen3_5TextModel, apply_rotary_pos_emb
from .independent_memory import IndependentGQAMemory, INDEPENDENT_MEMORY_REVISION

class Qwen3_5SegmentAwareGatedDeltaNet(Qwen3_5GatedDeltaNet):

    def forward(self, hidden_states: Tensor, cache_params: Cache | None=None, attention_mask: Tensor | None=None, **kwargs: Any) -> Tensor:
        if attention_mask is not None:
            current_mask = attention_mask[:, -hidden_states.shape[1]:]
            hidden_states = hidden_states * current_mask[..., None].to(hidden_states.dtype)
            attention_mask = None
        return super().forward(hidden_states=hidden_states, cache_params=cache_params, attention_mask=attention_mask)

def bind_native_qwen_gdn_kernels(model: nn.Module) -> dict[str, Any]:
    import fla
    import tilelang
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.chunk_gated_delta_rule = chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
            module.causal_conv1d_fn = causal_conv1d_fn
            module.causal_conv1d_update = causal_conv1d_update
            names.append(name)
    if not names:
        raise RuntimeError('no native Qwen3.5 GDN layers were found')
    return {'backend': 'fla', 'fla_version': getattr(fla, '__version__', 'unknown'), 'tilelang_version': None if tilelang is None else getattr(tilelang, '__version__', 'unknown'), 'native_gdn_layers': names, 'causal_conv1d_bound': True}

def reference_swa(q: Tensor, k: Tensor, v: Tensor, *, window: int, key_valid: Tensor | None=None, query_start: int=0) -> Tensor:
    batch, query_len, heads, head_dim = q.shape
    key_len = k.shape[1]
    query_pos = torch.arange(query_len, device=q.device) + int(query_start)
    key_pos = torch.arange(key_len, device=q.device)
    allowed = key_pos[None, :] <= query_pos[:, None]
    allowed &= key_pos[None, :] >= query_pos[:, None] - int(window) + 1
    allowed = allowed[None, None].expand(batch, 1, -1, -1)
    if key_valid is not None:
        allowed = allowed & key_valid[:, None, None, :].to(dtype=torch.bool)
    output = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed, dropout_p=0.0, is_causal=False, enable_gqa=heads != k.shape[2])
    output = output.transpose(1, 2).contiguous().view(batch, query_len, heads, head_dim)
    if key_valid is not None:
        query_valid = key_valid[:, query_start:query_start + query_len]
        output = output * query_valid[:, :, None, None].to(dtype=output.dtype)
    return output

def flash_swa(q: Tensor, k: Tensor, v: Tensor, *, window: int, attention_mask: Tensor | None) -> Tensor:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
    except ImportError as error:
        raise RuntimeError('CUDA inference requires FlashAttention') from error
    left = max(0, int(window) - 1)
    if attention_mask is None:
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=True, window_size=(left, 0))
    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    except ImportError as error:
        raise RuntimeError('padded SWA requires flash_attn.bert_padding') from error
    mask = attention_mask[:, -q.shape[1]:].to(dtype=torch.bool)
    q_flat, indices, cu, max_length = unpad_input(q.reshape(q.shape[0], q.shape[1], -1), mask)[:4]
    q_flat = q_flat.view(-1, q.shape[2], q.shape[3])
    if q_flat.shape[0] == 0:
        return q * 0.0
    k_flat = index_first_axis(k.reshape(-1, k.shape[2] * k.shape[3]), indices).view(-1, k.shape[2], k.shape[3])
    v_flat = index_first_axis(v.reshape(-1, v.shape[2] * v.shape[3]), indices).view(-1, v.shape[2], v.shape[3])
    output = flash_attn_varlen_func(q_flat, k_flat, v_flat, cu, cu, int(max_length), int(max_length), dropout_p=0.0, causal=True, window_size=(left, 0))
    return pad_input(output.reshape(output.shape[0], -1), indices, q.shape[0], q.shape[1]).view_as(q)
SUPPORTED_FIXED_MEMORY_REVISIONS = (INDEPENDENT_MEMORY_REVISION,)
DEFAULT_FIXED_MEMORY_WINDOW = 1024

def is_fixed_memory_revision(value: Any) -> bool:
    if isinstance(value, str):
        revision = value
    elif isinstance(value, dict):
        revision = value.get('fixed_memory_revision')
    else:
        revision = getattr(value, 'fixed_memory_revision', None)
        if revision is None and hasattr(value, 'config'):
            revision = getattr(value.config, 'fixed_memory_revision', None)
    return revision in SUPPORTED_FIXED_MEMORY_REVISIONS

def _linear_cache_tensor(state: Any) -> Tensor:
    if isinstance(state, (dict, list, tuple)):
        if len(state) != 1:
            raise RuntimeError('linear-attention cache state must contain one tensor')
        return next(iter(state.values())) if isinstance(state, dict) else state[0]
    if not isinstance(state, Tensor):
        raise TypeError(f'unsupported linear-attention cache state: {type(state).__name__}')
    return state

def _batch_scatter_state_(target: Any, source: Any, target_indices: Tensor, source_indices: Tensor) -> Any:
    if target is None or source is None:
        if target is not source:
            raise ValueError('source and target cache initialization differs')
        return target
    if isinstance(target, Tensor):
        if not isinstance(source, Tensor):
            raise TypeError('source and target cache state types differ')
        if target.ndim == 0 or source.ndim == 0:
            if not torch.equal(target, source.to(target.device)):
                raise ValueError('scalar cache metadata differs')
            return target
        if target.shape[1:] != source.shape[1:] or target.dtype != source.dtype:
            raise ValueError('source and target cache tensor geometry differs')
        if target.dtype == torch.float8_e4m3fn:
            target.view(torch.uint8).index_copy_(0, target_indices.to(target.device), source.view(torch.uint8).index_select(0, source_indices.to(source.device)).to(target.device))
        else:
            target.index_copy_(0, target_indices.to(target.device), source.index_select(0, source_indices.to(source.device)).to(target.device))
        return target
    if isinstance(target, dict):
        if not isinstance(source, dict) or set(target) != set(source):
            raise ValueError('source and target cache dictionaries differ')
        for key in target:
            target[key] = _batch_scatter_state_(target[key], source[key], target_indices, source_indices)
        return target
    if isinstance(target, list):
        if not isinstance(source, list) or len(target) != len(source):
            raise ValueError('source and target cache lists differ')
        for index in range(len(target)):
            target[index] = _batch_scatter_state_(target[index], source[index], target_indices, source_indices)
        return target
    if isinstance(target, tuple):
        if not isinstance(source, tuple) or len(target) != len(source):
            raise ValueError('source and target cache tuples differ')
        return tuple((_batch_scatter_state_(target_value, source_value, target_indices, source_indices) for target_value, source_value in zip(target, source)))
    if target != source:
        raise ValueError('source and target cache metadata differs')
    return target

class FixedMemoryLinearAttentionLayer(LinearAttentionLayer):

    @staticmethod
    def _map_batch_state(state: Any, function: Any) -> Any:
        if state is None:
            return None
        if isinstance(state, dict):
            return {key: function(value) if value is not None else None for key, value in state.items()}
        if isinstance(state, list):
            return [function(value) if value is not None else None for value in state]
        if isinstance(state, tuple):
            return tuple((function(value) if value is not None else None for value in state))
        return function(state)

    def _refresh_batch_size(self) -> None:
        for state in (self.conv_states, self.recurrent_states):
            tensor = _linear_cache_tensor(state) if state is not None else None
            if tensor is not None:
                self.max_batch_size = int(tensor.shape[0])
                return

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError('cache batch repeats must be positive')
        self.conv_states = self._map_batch_state(self.conv_states, lambda tensor: tensor.repeat_interleave(repeats, dim=0))
        self.recurrent_states = self._map_batch_state(self.recurrent_states, lambda tensor: tensor.repeat_interleave(repeats, dim=0))
        self._refresh_batch_size()

    def batch_select_indices(self, indices: Tensor) -> None:
        self.conv_states = self._map_batch_state(self.conv_states, lambda tensor: tensor.index_select(0, indices.to(tensor.device)))
        self.recurrent_states = self._map_batch_state(self.recurrent_states, lambda tensor: tensor.index_select(0, indices.to(tensor.device)))
        self._refresh_batch_size()

    def batch_scatter_from(self, source: 'FixedMemoryLinearAttentionLayer', target_indices: Tensor, source_indices: Tensor | None=None) -> None:
        if type(source) is not type(self):
            raise TypeError('linear cache layer types differ')
        if source_indices is None:
            source_indices = torch.arange(target_indices.numel(), device=target_indices.device)
        if target_indices.numel() != source_indices.numel():
            raise ValueError('source and target cache row counts differ')
        self.conv_states = _batch_scatter_state_(self.conv_states, source.conv_states, target_indices, source_indices)
        self.recurrent_states = _batch_scatter_state_(self.recurrent_states, source.recurrent_states, target_indices, source_indices)
        self._refresh_batch_size()

    def reorder_cache(self, beam_idx: Tensor) -> None:
        self.batch_select_indices(beam_idx)

class FixedMemorySlidingWindowLayer(DynamicSlidingWindowLayer):

    def __init__(self, sliding_window: int) -> None:
        super().__init__(sliding_window=sliding_window)
        self.valid: Tensor | None = None
        self.all_valid_host = True
        self.cuda_graph_decode = False
        self.physical_ring_cuda_graph_update = False
        self.memory_state = None

    def update_with_valid(self, key_states: Tensor, value_states: Tensor, token_valid: Tensor, all_tokens_valid: bool) -> tuple[Tensor, Tensor, Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.valid = torch.empty(token_valid.shape[0], 0, dtype=torch.bool, device=token_valid.device)
        token_valid = token_valid.detach().to(device=key_states.device, dtype=torch.bool)
        self.all_valid_host = self.all_valid_host and bool(all_tokens_valid)
        self.cumulative_length += key_states.shape[-2]
        if self.cuda_graph_decode:
            if key_states.shape[-2] != 1:
                raise ValueError('CUDA Graph SWA cache only accepts one decode token')
            keep = max(self.sliding_window - 1, 0)
            if self.keys.shape[-2] != keep or self.valid.shape[-1] != keep:
                raise ValueError('CUDA Graph SWA cache requires a prefill at least as long as the window')
            if self.physical_ring_cuda_graph_update:
                from .inference_kernels import triton_fixed_memory_swa_physical_ring_update
                return triton_fixed_memory_swa_physical_ring_update(self, key_states, value_states, token_valid)
            full_keys = torch.cat((self.keys, key_states), dim=-2)
            full_values = torch.cat((self.values, value_states), dim=-2)
            full_valid = torch.cat((self.valid, token_valid), dim=-1)
            if keep:
                self.keys.copy_(full_keys[:, :, 1:])
                self.values.copy_(full_values[:, :, 1:])
                self.valid.copy_(full_valid[:, 1:])
            return (full_keys, full_values, full_valid)
        full_keys = torch.cat((self.keys, key_states), dim=-2)
        full_values = torch.cat((self.values, value_states), dim=-2)
        full_valid = torch.cat((self.valid, token_valid), dim=-1)
        keep = max(self.sliding_window - 1, 0)
        self.keys = (full_keys[:, :, -keep:] if keep else full_keys[:, :, :0]).clone()
        self.values = (full_values[:, :, -keep:] if keep else full_values[:, :, :0]).clone()
        self.valid = (full_valid[:, -keep:] if keep else full_valid[:, :0]).clone()
        return (full_keys, full_values, full_valid)

    def update(self, key_states: Tensor, value_states: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        token_valid = kwargs.pop('token_valid', None)
        if token_valid is None:
            token_valid = torch.ones(key_states.shape[0], key_states.shape[-2], dtype=torch.bool, device=key_states.device)
        keys, values, _ = self.update_with_valid(key_states, value_states, token_valid, all_tokens_valid=bool(kwargs.pop('all_tokens_valid', False)))
        return (keys, values)

    def offload(self) -> None:
        super().offload()
        if self.valid is not None:
            self.valid = self.valid.to('cpu', non_blocking=True)
        if self.memory_state is not None:
            self.memory_state.to_('cpu')

    def prefetch(self) -> None:
        super().prefetch()
        if self.valid is not None and self.valid.device != self.device:
            self.valid = self.valid.to(self.device, non_blocking=True)
        if self.memory_state is not None:
            self.memory_state.to_(self.device)

    def reset(self) -> None:
        if self.is_initialized:
            self.keys = self.keys[:, :, :0]
            self.values = self.values[:, :, :0]
            self.valid = self.valid[:, :0]
        self.cumulative_length = 0
        self.all_valid_host = True
        if self.memory_state is not None:
            self.memory_state.reset_()

    def reorder_cache(self, beam_idx: Tensor) -> None:
        super().reorder_cache(beam_idx)
        if self.valid is not None:
            self.valid = self.valid.index_select(0, beam_idx.to(self.valid.device))
        if self.memory_state is not None:
            self.memory_state = self.memory_state.select(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.valid is not None:
            self.valid = self.valid.repeat_interleave(repeats, dim=0)
        if self.memory_state is not None:
            self.memory_state = self.memory_state.repeat(repeats)

    def batch_select_indices(self, indices: Tensor) -> None:
        super().batch_select_indices(indices)
        if self.valid is not None:
            self.valid = self.valid.index_select(0, indices.to(self.valid.device))
        if self.memory_state is not None:
            self.memory_state = self.memory_state.select(indices)

    def batch_scatter_from(self, source: 'FixedMemorySlidingWindowLayer', target_indices: Tensor, source_indices: Tensor | None=None) -> None:
        if type(source) is not type(self):
            raise TypeError('SWA cache layer types differ')
        if source_indices is None:
            source_indices = torch.arange(target_indices.numel(), device=target_indices.device)
        if target_indices.numel() != source_indices.numel():
            raise ValueError('source and target cache row counts differ')
        if self.is_initialized != source.is_initialized:
            raise ValueError('SWA cache initialization differs')
        if not self.is_initialized:
            return
        self.keys = _batch_scatter_state_(self.keys, source.keys, target_indices, source_indices)
        self.values = _batch_scatter_state_(self.values, source.values, target_indices, source_indices)
        self.valid = _batch_scatter_state_(self.valid, source.valid, target_indices, source_indices)
        if self.memory_state is not None:
            self.memory_state.scatter_(source.memory_state, target_indices, source_indices)
        for prefix in ('_qwen35_swa_physical_ring_', '_qwen35_swa_scratch_'):
            for suffix in ('keys', 'values', 'valid'):
                name = prefix + suffix
                target = getattr(self, name, None)
                value = getattr(source, name, None)
                if target is not None or value is not None:
                    setattr(self, name, _batch_scatter_state_(target, value, target_indices, source_indices))
        for name in ('_qwen35_swa_physical_ring_index',):
            target = getattr(self, name, None)
            value = getattr(source, name, None)
            if target is not None or value is not None:
                if target is None or value is None or (not torch.equal(target, value.to(target.device))):
                    raise ValueError('SWA ring phases differ during slot refill')
        self.all_valid_host = self.all_valid_host and source.all_valid_host

class FixedMemoryHybridDynamicCache(Cache):

    def __init__(self, config: Any):
        text_config = config.get_text_config(decoder=True) if hasattr(config, 'get_text_config') else config
        swa_layers = set((int(index) for index in text_config.fixed_memory_swa_layers))
        window = int(text_config.fixed_memory_swa_window)
        layers = []
        for index, layer_type in enumerate(text_config.layer_types):
            if index in swa_layers:
                layers.append(FixedMemorySlidingWindowLayer(window))
            elif layer_type == 'linear_attention':
                layers.append(FixedMemoryLinearAttentionLayer())
            else:
                raise ValueError(f'unsupported fixed-memory cache layer {index}: {layer_type}')
        super().__init__(layers=layers)
        self.cuda_graph_decode = False

    def update_recurrent_state(self, recurrent_state: Tensor, layer_idx: int, *args: Any, **kwargs: Any) -> Any:
        layer = self.layers[layer_idx]
        if self.cuda_graph_decode and getattr(layer, 'recurrent_states', None) is not None:
            cache_state = _linear_cache_tensor(layer.recurrent_states)
            aliases_cache = cache_state.data_ptr() == recurrent_state.data_ptr() and cache_state.shape == recurrent_state.shape and (cache_state.stride() == recurrent_state.stride()) and (cache_state.dtype == recurrent_state.dtype) and (cache_state.device == recurrent_state.device)
            if not aliases_cache:
                cache_state.copy_(recurrent_state)
            return None
        return super().update_recurrent_state(recurrent_state, layer_idx, *args, **kwargs)

    def update_conv_state(self, conv_state: Tensor, layer_idx: int, *args: Any, **kwargs: Any) -> Any:
        layer = self.layers[layer_idx]
        if self.cuda_graph_decode and getattr(layer, 'conv_states', None) is not None:
            _linear_cache_tensor(layer.conv_states).copy_(conv_state)
            return conv_state
        return super().update_conv_state(conv_state, layer_idx, *args, **kwargs)

    def reset(self) -> None:
        with torch.inference_mode():
            super().reset()

    def batch_scatter_from(self, source: 'FixedMemoryHybridDynamicCache', target_indices: Tensor, source_indices: Tensor | None=None) -> None:
        if len(self.layers) != len(source.layers):
            raise ValueError('source and target cache layer counts differ')
        if source_indices is None:
            source_indices = torch.arange(target_indices.numel(), device=target_indices.device)
        for target_layer, source_layer in zip(self.layers, source.layers):
            if not hasattr(target_layer, 'batch_scatter_from'):
                raise TypeError('fixed-memory cache layer lacks row scatter support')
            target_layer.batch_scatter_from(source_layer, target_indices, source_indices)

def convert_fixed_memory_recurrent_state_storage_(cache: FixedMemoryHybridDynamicCache, dtype: torch.dtype) -> dict[str, int | str]:
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('GDN recurrent state storage must be FP32, BF16 or FP16')
    converted_layers = 0
    bytes_before = 0
    bytes_after = 0
    for layer in cache.layers:
        if not isinstance(layer, FixedMemoryLinearAttentionLayer) or layer.recurrent_states is None:
            continue
        state = _linear_cache_tensor(layer.recurrent_states)
        bytes_before += state.numel() * state.element_size()
        if state.dtype != dtype:
            layer.recurrent_states = layer._map_batch_state(layer.recurrent_states, lambda tensor: tensor.to(dtype=dtype))
            converted_layers += 1
            layer._refresh_batch_size()
        converted = _linear_cache_tensor(layer.recurrent_states)
        bytes_after += converted.numel() * converted.element_size()
    return {'storage_dtype': str(dtype).removeprefix('torch.'), 'converted_layers': converted_layers, 'bytes_before': bytes_before, 'bytes_after': bytes_after, 'bytes_saved': bytes_before - bytes_after}

def align_fixed_memory_refill_cache_(source: FixedMemoryHybridDynamicCache, target: FixedMemoryHybridDynamicCache, *, replace_all_rows: bool=False) -> None:
    if len(source.layers) != len(target.layers):
        raise ValueError('source and target cache layer counts differ')
    for source_layer, target_layer in zip(source.layers, target.layers):
        if isinstance(target_layer, FixedMemoryLinearAttentionLayer):
            if not isinstance(source_layer, FixedMemoryLinearAttentionLayer):
                raise TypeError('linear cache layer types differ')
            if source_layer.recurrent_states is not None and target_layer.recurrent_states is not None:
                target_state = _linear_cache_tensor(target_layer.recurrent_states)
                source_state = _linear_cache_tensor(source_layer.recurrent_states)
                if isinstance(source_state, Tensor) and isinstance(target_state, Tensor) and (source_state.dtype != target_state.dtype):
                    source_layer.recurrent_states = source_layer._map_batch_state(source_layer.recurrent_states, lambda tensor: tensor.to(dtype=target_state.dtype))
                    source_layer._refresh_batch_size()
        if not isinstance(target_layer, FixedMemorySlidingWindowLayer):
            continue
        if not isinstance(source_layer, FixedMemorySlidingWindowLayer):
            raise TypeError('SWA cache layer types differ')
        target_index = getattr(target_layer, '_qwen35_swa_physical_ring_index', None)
        if target_index is None:
            raise ValueError('target cache does not use the physical SWA ring')
        source_index = getattr(source_layer, '_qwen35_swa_physical_ring_index', None)
        if source_index is None:
            from .inference_kernels import initialize_fixed_memory_swa_physical_ring
            initialize_fixed_memory_swa_physical_ring(source_layer)
            source_index = source_layer._qwen35_swa_physical_ring_index
        window = int(source_layer._qwen35_swa_physical_ring_valid.shape[-1])
        source_phase = int(source_index.item())
        if replace_all_rows:
            target_index.copy_(source_index)
        target_phase = int(target_index.item())
        shift = (target_phase - source_phase) % window
        if shift:
            for name, dimension in (('_qwen35_swa_physical_ring_keys', 2), ('_qwen35_swa_physical_ring_values', 2), ('_qwen35_swa_physical_ring_valid', 1)):
                tensor = getattr(source_layer, name)
                tensor.copy_(torch.roll(tensor, shifts=shift, dims=dimension))
        source_index.copy_(target_index)

def set_fixed_memory_cuda_graph_decode(model: nn.Module, cache: FixedMemoryHybridDynamicCache | None, enabled: bool) -> None:
    if cache is not None:
        cache.cuda_graph_decode = bool(enabled)
        for layer in cache.layers:
            if isinstance(layer, FixedMemorySlidingWindowLayer):
                if not enabled and layer.physical_ring_cuda_graph_update:
                    from .inference_kernels import materialize_fixed_memory_swa_physical_ring
                    materialize_fixed_memory_swa_physical_ring(layer)
                layer.cuda_graph_decode = bool(enabled)
                physical_ring = bool(enabled and getattr(model.config, 'fixed_memory_physical_ring_swa_cuda_graph_update', False))
                layer.physical_ring_cuda_graph_update = False
                if physical_ring and layer.is_initialized:
                    from .inference_kernels import initialize_fixed_memory_swa_physical_ring
                    layer.keys = layer.keys.contiguous()
                    layer.values = layer.values.contiguous()
                    layer.valid = layer.valid.contiguous()
                    layer._qwen35_swa_shared_ring_storage = bool(getattr(model.config, 'fixed_memory_swa_shared_ring_storage', False))
                    initialize_fixed_memory_swa_physical_ring(layer)
                layer.physical_ring_cuda_graph_update = physical_ring

class Qwen3_5FixedWindowAttention(Qwen3_5Attention):

    def __init__(self, config: Any, layer_idx: int):
        super().__init__(config, layer_idx)
        self.swa_window = int(getattr(config, 'fixed_memory_swa_window', DEFAULT_FIXED_MEMORY_WINDOW))

    @classmethod
    def from_attention(cls, source: Qwen3_5Attention) -> 'Qwen3_5FixedWindowAttention':
        module = cls(source.config, source.layer_idx)
        module.to(device=source.q_proj.weight.device, dtype=source.q_proj.weight.dtype)
        incompatible = module.load_state_dict(source.state_dict(), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'unexpected SWA conversion keys: {incompatible}')
        return module

    def forward(self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor], attention_mask: Tensor | None, past_key_values: Cache | None=None, **kwargs: Any) -> tuple[Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if hidden_states.shape[-2] == 1 and getattr(self, '_qwen35_packed_swa_projection_enabled', False):
            projected = F.linear(hidden_states, self._qwen35_swa_qkv_weight, self._qwen35_swa_qkv_bias)
            q_projected, k_projected, v_projected = projected.split(self._qwen35_swa_qkv_sizes, dim=-1)
            k_projected = k_projected.contiguous()
            v_projected = v_projected.contiguous()
        else:
            q_projected = self.q_proj(hidden_states)
            k_projected = self.k_proj(hidden_states)
            v_projected = self.v_proj(hidden_states)
        query, gate = torch.chunk(q_projected.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(k_projected.view(hidden_shape)).transpose(1, 2)
        value = v_projected.view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        window = int(kwargs.get('swa_window_size', self.swa_window))
        if window <= 0 or window > self.swa_window:
            raise ValueError(f'runtime SWA window must be in [1, {self.swa_window}], got {window}')
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError('fixed-window Qwen attention expects a 2D padding mask')
        current_valid = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device) if attention_mask is None else attention_mask[:, -query.shape[1]:].bool()
        if past_key_values is None:
            if query.is_cuda:
                local = flash_swa(query, key, value, window=window, attention_mask=attention_mask)
            else:
                local = reference_swa(query, key, value, window=window, key_valid=current_valid)
        else:
            layer_cache = past_key_values.layers[self.layer_idx]
            if not isinstance(layer_cache, FixedMemorySlidingWindowLayer):
                raise TypeError('fixed-memory SWA requires FixedMemoryHybridDynamicCache')
            previous = 0 if not layer_cache.is_initialized else layer_cache.keys.shape[-2]
            all_key, all_value, all_valid = layer_cache.update_with_valid(key.transpose(1, 2), value.transpose(1, 2), current_valid, all_tokens_valid=attention_mask is None)
            if layer_cache.physical_ring_cuda_graph_update:
                from .inference_kernels import fixed_memory_physical_ring_decode_attention
                local = fixed_memory_physical_ring_decode_attention(query, all_key, all_value, all_valid, reproducible=bool(getattr(self.config, 'fixed_memory_swa_reproducible_decode', False)))
            elif query.is_cuda and (previous == 0 or layer_cache.all_valid_host):
                local = flash_swa(query, all_key.transpose(1, 2), all_value.transpose(1, 2), window=window, attention_mask=None if layer_cache.all_valid_host else current_valid)
            else:
                local = reference_swa(query, all_key.transpose(1, 2), all_value.transpose(1, 2), window=window, key_valid=all_valid, query_start=previous)
        output = local.reshape(*input_shape, -1).contiguous()
        if getattr(self, '_qwen35_fused_swa_output_gate_enabled', False) and output.is_cuda and (output.dtype == torch.bfloat16) and gate.is_contiguous():
            from .inference_kernels import triton_qwen35_sigmoid_mul
            output = triton_qwen35_sigmoid_mul(output, gate)
        else:
            output = output * torch.sigmoid(gate)
        return (self.o_proj(output), None)

class Qwen3_5FixedMemoryDecoderLayer(Qwen3_5DecoderLayer):

    def forward(self, hidden_states: Tensor | tuple[Tensor, Tensor], position_embeddings: tuple[Tensor, Tensor], attention_mask: Tensor | None=None, position_ids: Tensor | None=None, past_key_values: Cache | None=None, **kwargs: Any) -> Tensor | tuple[Tensor, Tensor]:

        def run_attention(values: Tensor) -> Tensor:
            if hasattr(self, 'linear_attn'):
                return self.linear_attn(hidden_states=values, cache_params=past_key_values, attention_mask=attention_mask, **kwargs)
            attention_output, _ = self.self_attn(hidden_states=values, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, position_embeddings=position_embeddings, **kwargs)
            return attention_output
        if isinstance(hidden_states, tuple):
            raise RuntimeError('precomputed input norm requires cross-layer fusion')
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = run_attention(hidden_states)
        if hasattr(self, 'memory'):
            memory_output = self.memory(residual, position_embeddings, window=kwargs.get('swa_window_size'), attention_mask=attention_mask, cache_layer=None if past_key_values is None else past_key_values.layers[self.self_attn.layer_idx])
            hidden_states = hidden_states + memory_output
        if getattr(self, '_qwen35_fused_residual_rmsnorm_enabled', False):
            from .inference_kernels import fused_residual_qwen35_rmsnorm
            residual, hidden_states = fused_residual_qwen35_rmsnorm(residual, hidden_states, self.post_attention_layernorm)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3_5FixedMemoryTextModel(Qwen3_5TextModel):
    cache_class = FixedMemoryHybridDynamicCache

    def forward(self, input_ids: Tensor | None=None, attention_mask: Tensor | dict[str, Tensor] | None=None, position_ids: Tensor | None=None, past_key_values: Cache | None=None, inputs_embeds: Tensor | None=None, use_cache: bool | None=None, **kwargs: Any) -> Qwen3_5ModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('specify exactly one of input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = self.cache_class(self.config)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(seen, seen + inputs_embeds.shape[1], device=inputs_embeds.device).view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids
        masks = attention_mask if isinstance(attention_mask, dict) else {'full_attention': attention_mask, 'linear_attention': attention_mask}
        output_hidden_states = bool(kwargs.pop('output_hidden_states', False))
        all_hidden_states = () if output_hidden_states else None
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
        active_layers = self.layers[:self.config.num_hidden_layers]
        for index, decoder_layer in enumerate(active_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_kwargs = {'position_embeddings': position_embeddings, 'attention_mask': masks[self.config.layer_types[index]], 'position_ids': text_position_ids, 'past_key_values': past_key_values, 'use_cache': use_cache, **kwargs}
            hidden_states = decoder_layer(hidden_states, **layer_kwargs)
        if isinstance(hidden_states, tuple):
            raise RuntimeError('unexpected precomputed final RMSNorm')
        else:
            hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values, hidden_states=all_hidden_states)

class Qwen3_5FixedMemoryForCausalLM(Qwen3_5ForCausalLM):
    cache_class = FixedMemoryHybridDynamicCache
    _no_split_modules = ['Qwen3_5FixedMemoryDecoderLayer', 'Qwen3_5FixedWindowAttention', 'IndependentGQAMemory']

    def __init__(self, config: Any):
        super().__init__(config)
        _install_fixed_memory_modules(self)

    def bind_runtime_kernels(self) -> dict[str, Any]:
        return bind_native_qwen_gdn_kernels(self)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=True, logits_to_keep=0, **kwargs):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

def _install_fixed_memory_modules(model: nn.Module) -> None:
    config = model.config
    if not is_fixed_memory_revision(config):
        raise ValueError('fixed-memory config revision is missing or unsupported')
    swa_layers = tuple((int(index) for index in config.fixed_memory_swa_layers))
    independent = config.fixed_memory_revision == INDEPENDENT_MEMORY_REVISION
    for index in swa_layers:
        source = model.model.layers[index].self_attn
        if type(source) is Qwen3_5Attention:
            model.model.layers[index].self_attn = Qwen3_5FixedWindowAttention.from_attention(source)
        elif not isinstance(source, Qwen3_5FixedWindowAttention):
            raise TypeError(f'layer {index} is not an original Qwen attention layer')
        if independent:
            layer = model.model.layers[index]
            layer.memory = IndependentGQAMemory(config).to(device=source.q_proj.weight.device, dtype=source.q_proj.weight.dtype)
    for index, layer_type in enumerate(config.layer_types):
        if layer_type != 'linear_attention':
            continue
        source = model.model.layers[index].linear_attn
        if type(source) is Qwen3_5GatedDeltaNet:
            source.__class__ = Qwen3_5SegmentAwareGatedDeltaNet
        elif not isinstance(source, Qwen3_5SegmentAwareGatedDeltaNet):
            raise TypeError(f'layer {index} is not an original Qwen GDN layer')
    for layer in model.model.layers:
        layer.__class__ = Qwen3_5FixedMemoryDecoderLayer
    model.model.__class__ = Qwen3_5FixedMemoryTextModel

def load_fixed_memory_causal_lm(pretrained_model_name_or_path: str | Any, **kwargs: Any) -> Qwen3_5FixedMemoryForCausalLM:
    config = kwargs.pop('config', None)
    if config is None:
        config = Qwen3_5TextConfig.from_pretrained(pretrained_model_name_or_path, local_files_only=bool(kwargs.get('local_files_only', False)))
    if not is_fixed_memory_revision(config):
        raise ValueError('checkpoint is not an original-Qwen fixed-memory model')
    layer_count = int(config.num_hidden_layers)
    swa_layers = tuple((int(index) for index in config.fixed_memory_swa_layers))
    if any((index < 0 or index >= layer_count for index in swa_layers)):
        raise ValueError('fixed-memory checkpoint topology is invalid')
    construction_types = ['linear_attention'] * layer_count
    for index in swa_layers:
        construction_types[index] = 'full_attention'
    config.layer_types = construction_types
    config.architectures = ['Qwen3_5FixedMemoryForCausalLM']
    return Qwen3_5FixedMemoryForCausalLM.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)

def load_model(path, **kwargs):
    model = load_fixed_memory_causal_lm(path, **kwargs)
    bind_native_qwen_gdn_kernels(model)
    return model
