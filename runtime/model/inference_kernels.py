# Portions adapted from Transformers Qwen3.5; modified for YANchor inference.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Licensed under Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
import torch
from torch import Tensor
import importlib
import os
import time
import weakref
from types import MethodType
from torch.nn import functional as F
COUNTER_SAMPLING_REVISION = 'counter_cdf_v2'
_UINT32_MASK = 4294967295
_UINT32_SCALE = float(1 << 32)

def _as_int64_tensor(value: Tensor | int, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.int64)
    return torch.tensor(value, device=device, dtype=torch.int64)

def counter_uniform(attempt_ids: Tensor, steps: Tensor | int, *, seed: int, stream: int=0, dtype: torch.dtype=torch.float32) -> Tensor:
    if attempt_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError('attempt ids must be integer tensors')
    ids = attempt_ids.to(dtype=torch.int64) & _UINT32_MASK
    step_values = _as_int64_tensor(steps, device=attempt_ids.device) & _UINT32_MASK
    state = ids * 747796405 + step_values * 289133645 + (int(seed) & _UINT32_MASK) * 277803737 + (int(stream) & _UINT32_MASK) * 144269504 & _UINT32_MASK
    shift = (state >> 28) + 4
    word = (state >> shift ^ state) * 277803737 & _UINT32_MASK
    word = (word >> 22 ^ word) & _UINT32_MASK
    return ((word.to(torch.float64) + 0.5) / _UINT32_SCALE).to(dtype=dtype).clamp_max(1.0 - torch.finfo(dtype).eps / 2)

def _sample_cdf_counter(probabilities, attempt_ids, steps, *, seed, stream):
    cumulative = probabilities.cumsum(dim=-1)
    total = cumulative[:, -1:]
    uniform = counter_uniform(attempt_ids, steps, seed=seed, stream=stream, dtype=probabilities.dtype)[:, None]
    threshold = torch.minimum(uniform * total, torch.nextafter(total, torch.zeros_like(total)))
    return torch.searchsorted(cumulative, threshold, right=False)

def top_k_top_p_distribution(scores: Tensor, *, temperature: float, top_k: int, top_p: float) -> tuple[Tensor, Tensor]:
    if scores.ndim != 2:
        raise ValueError('sampling scores must have shape [batch, vocabulary]')
    if temperature <= 0.0 or top_k <= 0 or (not 0.0 < top_p <= 1.0):
        raise ValueError('invalid top-k/top-p sampling parameters')
    if top_k > scores.shape[-1]:
        raise ValueError('top-k exceeds the vocabulary')
    if temperature == 1.0:
        values, indices = torch.topk(scores, top_k, dim=-1)
        values = values.float()
    else:
        values, indices = torch.topk(scores.float() / temperature, top_k, dim=-1)
    probabilities = torch.softmax(values, dim=-1)
    keep = probabilities.cumsum(dim=-1) - probabilities < top_p
    probabilities = probabilities * keep
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return (indices, probabilities)

def sample_probabilities_counter(probabilities: Tensor, attempt_ids: Tensor, steps: Tensor | int, *, seed: int, stream: int=0) -> Tensor:
    if probabilities.ndim != 2:
        raise ValueError('probabilities must have shape [batch, support]')
    if attempt_ids.shape != probabilities.shape[:1]:
        raise ValueError('attempt ids must contain one id per probability row')
    if torch.any(probabilities < 0) or not torch.isfinite(probabilities).all():
        raise ValueError('probabilities must be finite and non-negative')
    totals = probabilities.sum(dim=-1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError('probability rows must have positive mass')
    normalized = probabilities / totals
    return _sample_cdf_counter(normalized, attempt_ids, steps, seed=seed, stream=stream)

def sample_top_k_top_p_counter(scores: Tensor, attempt_ids: Tensor, steps: Tensor | int, *, seed: int, temperature: float, top_k: int, top_p: float, stream: int=0) -> Tensor:
    if top_k <= 0 and top_p == 1.0:
        probabilities = torch.softmax(scores.float() / temperature, dim=-1)
        ranks = _sample_cdf_counter(probabilities, attempt_ids, steps, seed=seed, stream=stream)
        return ranks
    if top_k == 1:
        indices = scores.argmax(dim=-1, keepdim=True)
        return indices
    indices, probabilities = top_k_top_p_distribution(scores, temperature=temperature, top_k=top_k, top_p=top_p)
    ranks = sample_probabilities_counter(probabilities, attempt_ids, steps, seed=seed, stream=stream)
    return indices.gather(1, ranks)

@dataclass(frozen=True)
class PrefillExpansionPlan:
    representative_indices: Tensor
    inverse_indices: Tensor
    group_ids: Tensor

    @property
    def unique_count(self) -> int:
        return int(self.representative_indices.numel())

    @property
    def expanded_count(self) -> int:
        return int(self.inverse_indices.numel())

def build_prefill_expansion(group_ids: Tensor) -> PrefillExpansionPlan:
    if group_ids.ndim != 1 or group_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('prefill group ids must be a one-dimensional integer tensor')
    mapping: dict[int, int] = {}
    representatives: list[int] = []
    inverse: list[int] = []
    for row, raw_group in enumerate(group_ids.detach().cpu().tolist()):
        group = int(raw_group)
        if group not in mapping:
            mapping[group] = len(representatives)
            representatives.append(row)
        inverse.append(mapping[group])
    return PrefillExpansionPlan(representative_indices=torch.tensor(representatives, device=group_ids.device), inverse_indices=torch.tensor(inverse, device=group_ids.device), group_ids=group_ids.clone())

def assert_group_rows_identical(values: Tensor, plan: PrefillExpansionPlan) -> None:
    representatives = values.index_select(0, plan.representative_indices.to(values.device))
    expanded = representatives.index_select(0, plan.inverse_indices.to(values.device))
    if not torch.equal(values, expanded):
        raise ValueError('prefill group contains non-identical rows')
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

    class _TritonStub:

        @staticmethod
        def jit(function):
            return function

        @staticmethod
        def next_power_of_2(value):
            raise RuntimeError('Triton is unavailable')

    class _TritonLanguageStub:
        constexpr = object
    triton = _TritonStub()
    tl = _TritonLanguageStub()
    libdevice = None

def _inplace_native_gdn_state_forward(q, k, v, g=None, gk=None, gv=None, beta=None, A_log=None, dt_bias=None, scale=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, state_v_first=False, cu_seqlens=None):
    module = importlib.import_module('fla.ops.gated_delta_rule.fused_recurrent')
    original = module._qwen35_original_native_gdn_fwd
    if not (getattr(module, '_qwen35_inplace_native_gdn_state_enabled', False) and q.shape[1] == 1 and (initial_state is not None) and output_final_state):
        return original(q, k, v, g, gk, gv, beta, A_log, dt_bias, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval, state_v_first, None)
    batch, sequence, heads, key_dim, value_dim = (*k.shape, v.shape[-1])
    adaptive_small_batch = getattr(module, '_qwen35_native_gdn_adaptive', False) and batch < 128 and ('QWEN35_NATIVE_GDN_BV' not in os.environ)
    value_heads = int(v.shape[2])
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(8, triton.next_power_of_2(value_dim)) if gv is None else triton.next_power_of_2(value_dim)
    requested_block_value = int(os.environ.get('QWEN35_NATIVE_GDN_BV', str(32 if adaptive_small_batch else getattr(module, '_qwen35_native_gdn_bv', 0))))
    if requested_block_value:
        if requested_block_value not in (8, 16, 32, 64, 128):
            raise ValueError('QWEN35_NATIVE_GDN_BV must be one of 8, 16, 32, 64, 128')
        if value_dim % requested_block_value:
            raise ValueError('QWEN35_NATIVE_GDN_BV must divide the value dimension')
        block_value = requested_block_value
    num_warps = int(os.environ.get('QWEN35_NATIVE_GDN_WARPS', str(2 if adaptive_small_batch else getattr(module, '_qwen35_native_gdn_warps', 1))))
    num_stages = int(os.environ.get('QWEN35_NATIVE_GDN_STAGES', '3'))
    if num_warps not in (1, 2, 4, 8):
        raise ValueError('QWEN35_NATIVE_GDN_WARPS must be one of 1, 2, 4, 8')
    if num_stages not in (1, 2, 3, 4):
        raise ValueError('QWEN35_NATIVE_GDN_STAGES must be one of 1, 2, 3, 4')
    value_blocks = triton.cdiv(value_dim, block_value)
    output = torch.empty_like(v)
    module.fused_recurrent_gated_delta_rule_fwd_kernel[value_blocks, batch * value_heads](q=q, k=k, v=v, g=g, gk=gk, gv=gv, beta=beta, A_log=A_log, dt_bias=dt_bias, o=output, h0=initial_state, ht=initial_state, cu_seqlens=None, scale=scale, T=sequence, H=heads, HV=value_heads, K=key_dim, V=value_dim, BK=block_key, BV=block_value, IS_BETA_HEADWISE=beta.ndim != v.ndim, USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel, ALLOW_NEG_EIGVAL=allow_neg_eigval, STATE_V_FIRST=state_v_first, num_warps=num_warps, num_stages=num_stages)
    return (output, initial_state)

def enable_inplace_native_gdn_state_inference(block_value: int=8, num_warps: int=1, *, adaptive: bool=False) -> bool:
    module = importlib.import_module('fla.ops.gated_delta_rule.fused_recurrent')
    if not hasattr(module, '_qwen35_original_native_gdn_fwd'):
        module._qwen35_original_native_gdn_fwd = module.fused_recurrent_gated_delta_rule_fwd
        module.fused_recurrent_gated_delta_rule_fwd = _inplace_native_gdn_state_forward
    module._qwen35_inplace_native_gdn_state_enabled = True
    module._qwen35_native_gdn_bv = block_value
    module._qwen35_native_gdn_warps = num_warps
    module._qwen35_native_gdn_adaptive = adaptive
    return True

@triton.jit
def _qwen35_square_to_fp32_kernel(input_ptr, output_ptr, total_elements: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, values * values, mask=mask)

def triton_square_to_fp32(values: torch.Tensor) -> torch.Tensor:
    if not _TRITON_AVAILABLE or not values.is_cuda or (not values.is_contiguous()):
        raise ValueError('fused BF16 square requires a contiguous CUDA tensor')
    output = torch.empty_like(values, dtype=torch.float32)
    total_elements = int(values.numel())
    block_size = 256
    _qwen35_square_to_fp32_kernel[triton.cdiv(total_elements, block_size),](values, output, total_elements=total_elements, block_size=block_size, num_warps=4, num_stages=1)
    return output

@triton.jit
def _qwen35_rmsnorm_epilogue_kernel(input_ptr, gamma_ptr, mean_square_ptr, output_ptr, row_width: tl.constexpr, eps: tl.constexpr, block_size: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    values = tl.load(input_ptr + row * row_width + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)
    mean_square = tl.load(mean_square_ptr + row)
    normalized = values * tl.rsqrt(mean_square + eps) * gamma
    tl.store(output_ptr + row * row_width + offsets, normalized, mask=mask)

def triton_qwen35_rmsnorm_exact_epilogue(values: torch.Tensor, gamma: torch.Tensor, mean_square: torch.Tensor, eps: float, output_dtype: torch.dtype) -> torch.Tensor:
    if not _TRITON_AVAILABLE or not values.is_cuda:
        raise RuntimeError('exact RMSNorm epilogue requires CUDA Triton')
    row_width = int(gamma.numel())
    if values.shape[-1] != row_width or not values.is_contiguous():
        raise ValueError('exact RMSNorm epilogue requires contiguous matching rows')
    rows = values.numel() // row_width
    if mean_square.numel() != rows or not mean_square.is_contiguous():
        raise ValueError('exact RMSNorm epilogue mean shape differs from its rows')
    output = torch.empty_like(values, dtype=output_dtype)
    block_size = triton.next_power_of_2(row_width)
    _qwen35_rmsnorm_epilogue_kernel[rows,](values, gamma, mean_square, output, row_width=row_width, eps=float(eps), block_size=block_size, num_warps=8 if block_size >= 2048 else 4, num_stages=1)
    return output

@triton.jit
def _qwen35_rmsnorm_torch_tree_kernel(X, U, Gamma, Out, Residual, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr, ADD: tl.constexpr, BW: tl.constexpr):
    row = tl.program_id(0)
    lane, component = (tl.arange(0, BW), tl.arange(0, 4))
    accum = tl.full((BW, 4), 0.0, tl.float32)
    for start in range(tl.cdiv(WIDTH, BW * 4)):
        col = start * BW * 4 + lane[:, None] * 4 + component[None, :]
        x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        if ADD:
            u = tl.load(U + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
            x = (x + u).to(X.dtype.element_ty).to(tl.float32)
            tl.store(Residual + row * WIDTH + col, x, col < WIDTH)
        accum = accum + x * x
    value = tl.sum(tl.where(component[None, :] == 0, accum, 0.0), 1)
    for i in tl.static_range(1, 4):
        value = value + tl.sum(tl.where(component[None, :] == i, accum, 0.0), 1)
    for shift in tl.static_range(0, 9):
        if BW >> shift > 1:
            value = value + tl.gather(value, (lane + (BW >> shift + 1)) % BW, 0)
    mean = tl.sum(tl.where(lane == 0, value, 0.0), 0) * (1.0 / WIDTH)
    inv = tl.rsqrt(mean + EPS)
    col = tl.arange(0, BLOCK)
    if ADD:
        x = tl.load(Residual + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    else:
        x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    gamma = tl.load(Gamma + col, col < WIDTH, 0)
    tl.store(Out + row * WIDTH + col, x * inv * gamma, col < WIDTH)

def triton_qwen35_rmsnorm_torch_tree(values, gamma, eps, update=None):
    values = values.contiguous()
    width = values.shape[-1]
    rows = values.numel() // width
    block_height = min(1 << rows.bit_length() - 1, 16)
    block_width = min(1 << (width // 4).bit_length() - 1, 512 // block_height)
    output = torch.empty_like(values)
    summed = torch.empty_like(values) if update is not None else values
    _qwen35_rmsnorm_torch_tree_kernel[rows,](values, values if update is None else update, gamma, output, summed, width, float(eps), triton.next_power_of_2(width), update is not None, block_width, num_warps=4, enable_fp_fusion=False)
    return (summed, output)

def _native_rmsnorm_forward(module, values: torch.Tensor) -> torch.Tensor:
    if not values.is_cuda or not getattr(module, '_qwen35_native_rms_enabled', True):
        return module._qwen35_native_rms_eager_forward(values)
    if module._qwen35_native_rms_torch_tree:
        return triton_qwen35_rmsnorm_torch_tree(values, module._qwen35_native_rms_gamma, module.eps)[1]
    if values.is_contiguous():
        mean_square = triton_square_to_fp32(values).mean(dim=-1, keepdim=True)
    else:
        work = values.float()
        mean_square = work.pow(2).mean(dim=-1, keepdim=True)
    return triton_qwen35_rmsnorm_exact_epilogue(values.contiguous(), module._qwen35_native_rms_gamma, mean_square, float(module.eps), values.dtype)

def enable_fused_residual_rmsnorm_inference(model: torch.nn.Module) -> int:
    from .fixed_memory import Qwen3_5FixedMemoryDecoderLayer
    if not _TRITON_AVAILABLE:
        raise RuntimeError('fused residual RMSNorm requires Triton')
    patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5FixedMemoryDecoderLayer):
            continue
        if not hasattr(module.post_attention_layernorm, '_qwen35_native_rms_gamma'):
            raise RuntimeError('enable native RMSNorm before fused residual RMSNorm')
        module._qwen35_fused_residual_rmsnorm_enabled = True
        patched += 1
    model.config.fixed_memory_fused_residual_rmsnorm_inference = True
    return patched

@torch.no_grad()
def refresh_native_rmsnorm_inference(model: torch.nn.Module) -> int:
    refreshed = 0
    for module in model.modules():
        if not hasattr(module, '_qwen35_native_rms_gamma'):
            continue
        module._qwen35_native_rms_gamma.copy_(1.0 + module.weight.float())
        refreshed += 1
    return refreshed

def enable_native_rmsnorm_inference(model: torch.nn.Module, *, torch_tree: bool=False) -> int:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5RMSNorm):
            continue
        module._qwen35_native_rms_torch_tree = torch_tree and module.weight.numel() in (256, 2560)
        if hasattr(module, '_qwen35_native_rms_gamma'):
            continue
        module.register_buffer('_qwen35_native_rms_gamma', torch.empty_like(module.weight, dtype=torch.float32), persistent=False)
        module._qwen35_native_rms_eager_forward = module.forward
        module._qwen35_native_rms_enabled = True
        module.forward = MethodType(_native_rmsnorm_forward, module)
        patched += 1
    refresh_native_rmsnorm_inference(model)
    return patched

@triton.jit
def _qwen35_graph_safe_conv_update_kernel(input_ptr, state_ptr, weight_ptr, bias_ptr, output_ptr, input_sb: tl.constexpr, input_sc: tl.constexpr, state_sb: tl.constexpr, state_sc: tl.constexpr, state_sk: tl.constexpr, weight_sc: tl.constexpr, weight_sk: tl.constexpr, output_sb: tl.constexpr, output_sc: tl.constexpr, CHANNELS: tl.constexpr, KERNEL_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr, APPLY_SILU: tl.constexpr, BLOCK_C: tl.constexpr):
    batch = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = channels < CHANNELS
    current = tl.load(input_ptr + batch * input_sb + channels * input_sc, mask=mask, other=0.0)
    result = tl.zeros((BLOCK_C,), tl.float32)
    for offset in tl.static_range(0, KERNEL_SIZE - 1):
        shifted = tl.load(state_ptr + batch * state_sb + channels * state_sc + (offset + 1) * state_sk, mask=mask, other=0.0)
        result += shifted.to(tl.float32) * tl.load(weight_ptr + channels * weight_sc + offset * weight_sk, mask=mask, other=0.0).to(tl.float32)
        tl.store(state_ptr + batch * state_sb + channels * state_sc + offset * state_sk, shifted, mask=mask)
    result += current.to(tl.float32) * tl.load(weight_ptr + channels * weight_sc + (KERNEL_SIZE - 1) * weight_sk, mask=mask, other=0.0).to(tl.float32)
    tl.store(state_ptr + batch * state_sb + channels * state_sc + (KERNEL_SIZE - 1) * state_sk, current, mask=mask)
    if HAS_BIAS:
        result += tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    if APPLY_SILU:
        result *= tl.sigmoid(result)
    tl.store(output_ptr + batch * output_sb + channels * output_sc, result, mask=mask)

def triton_graph_safe_causal_conv1d_update(hidden_states: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None=None, activation: str | None=None) -> torch.Tensor:
    if not _TRITON_AVAILABLE or not hidden_states.is_cuda:
        raise RuntimeError('graph-safe causal convolution requires CUDA Triton')
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != 1:
        raise ValueError('graph-safe causal convolution accepts one decode token')
    if conv_state.ndim != 3 or weight.ndim != 2:
        raise ValueError('invalid causal-convolution state or weight rank')
    batch, channels, _ = hidden_states.shape
    if conv_state.shape[:2] != (batch, channels) or weight.shape[0] != channels:
        raise ValueError('causal-convolution tensors have incompatible shapes')
    kernel_size = int(conv_state.shape[-1])
    if weight.shape[1] != kernel_size:
        raise ValueError('causal-convolution state and weight widths differ')
    if activation not in (None, 'silu', 'swish'):
        raise ValueError(f'unsupported causal-convolution activation: {activation}')
    output = torch.empty_like(hidden_states)
    block_c = 256
    _qwen35_graph_safe_conv_update_kernel[batch, triton.cdiv(channels, block_c)](hidden_states, conv_state, weight, bias if bias is not None else hidden_states, output, input_sb=hidden_states.stride(0), input_sc=hidden_states.stride(1), state_sb=conv_state.stride(0), state_sc=conv_state.stride(1), state_sk=conv_state.stride(2), weight_sc=weight.stride(0), weight_sk=weight.stride(1), output_sb=output.stride(0), output_sc=output.stride(1), CHANNELS=channels, KERNEL_SIZE=kernel_size, HAS_BIAS=bias is not None, APPLY_SILU=activation in ('silu', 'swish'), BLOCK_C=block_c, num_warps=4, num_stages=1)
    return output

def enable_triton_graph_safe_conv_update(model: torch.nn.Module) -> int:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5GatedDeltaNet):
            continue
        module.causal_conv1d_update = triton_graph_safe_causal_conv1d_update
        patched += 1
    return patched

def snapshot_cuda_graph_cache(cache, *, include_ring_backing: bool=False) -> dict[str, object]:
    entries = []
    for layer_index, layer in enumerate(cache.layers):
        for name, value in vars(layer).items():
            if not include_ring_backing and name.startswith(('_qwen35_swa_scratch_', '_qwen35_swa_physical_ring_')):
                continue
            if isinstance(value, torch.Tensor):
                entries.append((f'{layer_index}.{name}', value, value.clone()))
            elif isinstance(value, dict):
                entries.extend(((f'{layer_index}.{name}.{key}', item, item.clone()) for key, item in value.items() if isinstance(item, torch.Tensor)))
            elif name == 'memory_state' and value is not None:
                entries.extend(((f'{layer_index}.memory_state.{key}', item, item.clone()) for key, item in vars(value).items()))
    return {'entries': entries, 'include_ring_backing': bool(include_ring_backing), 'seen_tokens': getattr(cache, '_seen_tokens', None), 'layer_metadata': [{'cumulative_length': getattr(layer, 'cumulative_length', None), 'all_valid_host': getattr(layer, 'all_valid_host', None)} for layer in cache.layers]}

def restore_cuda_graph_cache(cache, snapshot: dict[str, object]) -> None:
    current = {}
    for layer_index, layer in enumerate(cache.layers):
        for name, value in vars(layer).items():
            if not snapshot.get('include_ring_backing', False) and name.startswith(('_qwen35_swa_scratch_', '_qwen35_swa_physical_ring_')):
                continue
            if isinstance(value, torch.Tensor):
                current[f'{layer_index}.{name}'] = value
            elif isinstance(value, dict):
                current.update({f'{layer_index}.{name}.{key}': item for key, item in value.items() if isinstance(item, torch.Tensor)})
            elif name == 'memory_state' and value is not None:
                current.update({f'{layer_index}.memory_state.{key}': item for key, item in vars(value).items()})
    for name, tensor, saved in snapshot['entries']:
        if current.get(name) is not tensor:
            raise RuntimeError(f'CUDA Graph capture replaced cache tensor {name}')
        tensor.copy_(saved)
    if snapshot['seen_tokens'] is not None:
        cache._seen_tokens = int(snapshot['seen_tokens'])
    for layer, metadata in zip(cache.layers, snapshot['layer_metadata']):
        for name, value in metadata.items():
            if hasattr(layer, name):
                setattr(layer, name, value)
        if not snapshot.get('include_ring_backing', False):
            if hasattr(layer, '_qwen35_swa_physical_ring_index'):
                reset_fixed_memory_swa_physical_ring(layer)

def resolve_fixed_memory_recurrent_state_storage(storage: str, batch_size: int) -> str:
    if batch_size <= 0:
        raise ValueError('recurrent-state storage requires a positive batch')
    if storage == 'fp16_non_anchor_auto_b64':
        return 'fp16_non_anchor' if batch_size >= 64 else 'fp32'
    if storage not in {'fp32', 'bf16_non_anchor', 'fp16_non_anchor'}:
        raise ValueError(f'unsupported GDN recurrent state storage: {storage}')
    return storage

class FixedMemoryDecodeGraph:

    @torch.no_grad()
    def __init__(self, model: torch.nn.Module, cache, token: torch.Tensor, position: int | torch.Tensor, scores_dtype: torch.dtype=torch.float32, consume_capture_steps: bool=False) -> None:
        from .fixed_memory import convert_fixed_memory_recurrent_state_storage_, set_fixed_memory_cuda_graph_decode
        if token.ndim != 2 or token.shape[1] != 1 or (not token.is_cuda):
            raise ValueError('fixed-memory CUDA Graph requires CUDA [batch, 1] tokens')
        if scores_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('decode scores must use FP32 or BF16')
        self.model = model
        self.cache = cache
        state_storage = resolve_fixed_memory_recurrent_state_storage(str(getattr(model.config, 'gdn_recurrent_state_storage', 'fp32')), int(token.shape[0]))
        if state_storage == 'bf16_non_anchor':
            state_dtype = torch.bfloat16
        elif state_storage == 'fp16_non_anchor':
            state_dtype = torch.float16
        elif state_storage == 'fp32':
            state_dtype = torch.float32
        else:
            raise ValueError(f'unsupported GDN recurrent state storage: {state_storage}')
        self.recurrent_state_storage_receipt = convert_fixed_memory_recurrent_state_storage_(cache, state_dtype)
        self.static_ids = token.clone()
        if isinstance(position, torch.Tensor):
            position_values = position.to(device=token.device, dtype=torch.long)
            if position_values.ndim == 1:
                position_values = position_values[:, None]
            if position_values.shape not in (token.shape, (4, *token.shape)):
                raise ValueError('decode positions require one value per token row')
            self.static_position_ids = position_values.clone()
        else:
            self.static_position_ids = torch.full((token.shape[0], 1), int(position), dtype=torch.long, device=token.device)
        initial_position_ids = self.static_position_ids.clone()
        self.replays = 0
        self.graph = None
        self.scores = None
        self._pending_scores = None
        self.consume_capture_steps = bool(consume_capture_steps)
        set_fixed_memory_cuda_graph_decode(model, cache, True)
        if self.consume_capture_steps:
            host_metadata = self._snapshot_host_metadata()
            warm_output = model(input_ids=self.static_ids, position_ids=self.static_position_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            self.scores = warm_output.logits[:, -1].to(dtype=scores_dtype)
            self._pending_scores = self.scores
            self.static_position_ids.add_(1)
            self._restore_host_metadata(host_metadata)
            self._record_completed_decode()
            torch.cuda.synchronize(token.device)
            return
        warm_snapshot = snapshot_cuda_graph_cache(cache)
        model(input_ids=self.static_ids, position_ids=self.static_position_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize(token.device)
        restore_cuda_graph_cache(cache, warm_snapshot)
        del warm_snapshot
        capture_snapshot = snapshot_cuda_graph_cache(cache)
        self.static_position_ids.copy_(initial_position_ids)
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(token.device)
        with torch.cuda.graph(self.graph):
            output = model(input_ids=self.static_ids, position_ids=self.static_position_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            self.scores = output.logits[:, -1].to(dtype=scores_dtype)
            self.static_position_ids.add_(1)
        torch.cuda.synchronize(token.device)
        restore_cuda_graph_cache(cache, capture_snapshot)
        self.static_position_ids.copy_(initial_position_ids)

    def _snapshot_host_metadata(self) -> tuple[object, list[tuple[object, object]]]:
        return (getattr(self.cache, '_seen_tokens', None), [(getattr(layer, 'cumulative_length', None), getattr(layer, 'all_valid_host', None)) for layer in self.cache.layers])

    def _restore_host_metadata(self, snapshot: tuple[object, list[tuple[object, object]]]) -> None:
        seen_tokens, layer_metadata = snapshot
        if seen_tokens is not None:
            self.cache._seen_tokens = int(seen_tokens)
        for layer, (cumulative_length, all_valid_host) in zip(self.cache.layers, layer_metadata):
            if cumulative_length is not None and hasattr(layer, 'cumulative_length'):
                layer.cumulative_length = cumulative_length
            if all_valid_host is not None and hasattr(layer, 'all_valid_host'):
                layer.all_valid_host = all_valid_host

    def _record_completed_decode(self) -> None:
        self.replays += 1

    def _copy_position_ids(self, position_ids: torch.Tensor | None) -> None:
        if position_ids is None:
            return
        position_values = position_ids.to(device=self.static_position_ids.device, dtype=torch.long)
        if position_values.ndim == 1:
            position_values = position_values[:, None]
        if position_values.shape != self.static_position_ids.shape:
            raise ValueError('decode positions require one value per captured row')
        self.static_position_ids.copy_(position_values)

    @torch.no_grad()
    def replay(self, token: torch.Tensor, position_ids: torch.Tensor | None=None) -> torch.Tensor:
        if self._pending_scores is not None:
            if not torch.equal(token, self.static_ids):
                raise RuntimeError('capture-consuming warm token changed before use')
            scores = self._pending_scores
            self._pending_scores = None
            return scores
        self.static_ids.copy_(token)
        self._copy_position_ids(position_ids)
        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            host_metadata = self._snapshot_host_metadata()
            torch.cuda.synchronize(token.device)
            with torch.cuda.graph(self.graph):
                output = self.model(input_ids=self.static_ids, position_ids=self.static_position_ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
                self.scores = output.logits[:, -1].to(dtype=self.scores.dtype)
                self.static_position_ids.add_(1)
            self._restore_host_metadata(host_metadata)
            self.graph.replay()
            torch.cuda.synchronize(token.device)
            self._record_completed_decode()
            return self.scores
        self.graph.replay()
        self._record_completed_decode()
        return self.scores

def initialize_fixed_memory_swa_physical_ring(layer_cache) -> None:
    tensors = (layer_cache.keys, layer_cache.values, layer_cache.valid)
    if any((tensor is None or not tensor.is_cuda for tensor in tensors)):
        raise ValueError('physical SWA ring requires CUDA cache tensors')
    keep = int(layer_cache.keys.shape[-2])
    window = keep + 1
    if getattr(layer_cache, '_qwen35_swa_shared_ring_storage', False):
        for field, dimension in (('keys', 2), ('values', 2), ('valid', 1)):
            source = getattr(layer_cache, field)
            shape = list(source.shape)
            shape[dimension] = window
            name = '_qwen35_swa_physical_ring_' + field
            target = getattr(layer_cache, name, None)
            if target is not None and target.shape == tuple(shape) and (source.data_ptr() == target.data_ptr()):
                target.narrow(dimension, keep, 1).zero_()
            else:
                target = source.new_zeros(shape)
                target.narrow(dimension, 0, keep).copy_(source)
            setattr(layer_cache, name, target)
            setattr(layer_cache, field, target.narrow(dimension, 0, keep))
        index = getattr(layer_cache, '_qwen35_swa_physical_ring_index', None)
        if index is None:
            index = torch.tensor(keep, dtype=torch.int32, device=layer_cache.keys.device)
            layer_cache._qwen35_swa_physical_ring_index = index
        else:
            index.fill_(keep)
        return
    shapes = (('_qwen35_swa_physical_ring_keys', (*layer_cache.keys.shape[:-2], window, layer_cache.keys.shape[-1]), layer_cache.keys), ('_qwen35_swa_physical_ring_values', (*layer_cache.values.shape[:-2], window, layer_cache.values.shape[-1]), layer_cache.values), ('_qwen35_swa_physical_ring_valid', (layer_cache.valid.shape[0], window), layer_cache.valid))
    for name, shape, source in shapes:
        target = getattr(layer_cache, name, None)
        if target is None or target.shape != shape or target.device != source.device:
            target = source.new_zeros(shape)
            setattr(layer_cache, name, target)
        else:
            target.zero_()
    layer_cache._qwen35_swa_physical_ring_keys[:, :, :keep].copy_(layer_cache.keys)
    layer_cache._qwen35_swa_physical_ring_values[:, :, :keep].copy_(layer_cache.values)
    layer_cache._qwen35_swa_physical_ring_valid[:, :keep].copy_(layer_cache.valid)
    index = getattr(layer_cache, '_qwen35_swa_physical_ring_index', None)
    if index is None or index.device != layer_cache.keys.device:
        index = torch.tensor(keep, dtype=torch.int32, device=layer_cache.keys.device)
        layer_cache._qwen35_swa_physical_ring_index = index
    else:
        index.fill_(keep)

def prepare_fixed_memory_swa_cache_for_physical_ring_(cache, window: int) -> int:
    keep = int(window) - 1
    if keep <= 0:
        raise ValueError('physical SWA ring window must exceed one token')
    padded_rows = 0
    for layer_cache in cache.layers:
        if not all((hasattr(layer_cache, name) for name in ('keys', 'values', 'valid', 'sliding_window'))):
            continue
        if layer_cache.keys is None or layer_cache.valid is None:
            raise ValueError('physical SWA ring requires initialized cache rows')
        if int(layer_cache.sliding_window) != int(window):
            raise ValueError('SWA cache window differs from the model contract')
        current = int(layer_cache.keys.shape[-2])
        if current > keep or int(layer_cache.valid.shape[-1]) != current:
            raise ValueError('SWA cache shape is incompatible with the fixed ring')
        if current == keep:
            continue
        padding = keep - current
        keys = layer_cache.keys.new_zeros((*layer_cache.keys.shape[:-2], keep, layer_cache.keys.shape[-1]))
        values = layer_cache.values.new_zeros((*layer_cache.values.shape[:-2], keep, layer_cache.values.shape[-1]))
        valid = layer_cache.valid.new_zeros((layer_cache.valid.shape[0], keep))
        if current:
            keys[:, :, padding:].copy_(layer_cache.keys)
            values[:, :, padding:].copy_(layer_cache.values)
            valid[:, padding:].copy_(layer_cache.valid)
        layer_cache.keys = keys
        layer_cache.values = values
        layer_cache.valid = valid
        layer_cache.all_valid_host = False
        padded_rows += padding
    return padded_rows

def reset_fixed_memory_swa_physical_ring(layer_cache) -> None:
    initialize_fixed_memory_swa_physical_ring(layer_cache)

def enable_physical_ring_fixed_memory_swa_update(model: torch.nn.Module) -> int:
    from .fixed_memory import Qwen3_5FixedWindowAttention
    if not _TRITON_AVAILABLE:
        raise RuntimeError('physical SWA ring requires Triton')
    model.config.fixed_memory_physical_ring_swa_cuda_graph_update = True
    return sum((isinstance(module, Qwen3_5FixedWindowAttention) for module in model.modules()))

@triton.jit
def _qwen35_silu_mul_kernel(gate_up_ptr, output_ptr, row_width: tl.constexpr, total_elements: tl.constexpr, block_size: tl.constexpr, use_int64_offsets: tl.constexpr):
    if use_int64_offsets:
        offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    else:
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    row = offsets // row_width
    column = offsets - row * row_width
    gate_up_row = row * (2 * row_width)
    gate = tl.load(gate_up_ptr + gate_up_row + column, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + gate_up_row + row_width + column, mask=mask, other=0.0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(output_ptr + offsets, activated * up, mask=mask)

def triton_qwen35_silu_mul(gate_up: torch.Tensor, row_width: int) -> torch.Tensor:
    if not _TRITON_AVAILABLE or not gate_up.is_cuda or (not gate_up.is_contiguous()):
        raise ValueError('fused SiLU-mul requires a contiguous CUDA tensor')
    if gate_up.shape[-1] != 2 * int(row_width):
        raise ValueError('fused SiLU-mul input width differs from gate plus up')
    output = gate_up.new_empty((*gate_up.shape[:-1], int(row_width)))
    total_elements = int(output.numel())
    block_size = 256
    _qwen35_silu_mul_kernel[triton.cdiv(total_elements, block_size),](gate_up, output, row_width=int(row_width), total_elements=total_elements, block_size=block_size, use_int64_offsets=gate_up.numel() > torch.iinfo(torch.int32).max, num_warps=4, num_stages=1)
    return output

@triton.jit
def _qwen35_fused_gdn_gates_kernel(b_ptr, a_ptr, a_log_ptr, dt_bias_ptr, beta_ptr, decay_ptr, b_row_stride: tl.constexpr, a_row_stride: tl.constexpr, heads: tl.constexpr, total_elements: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    row = offsets // heads
    head = offsets - row * heads
    b = tl.load(b_ptr + row * b_row_stride + head, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + row * a_row_stride + head, mask=mask, other=0.0).to(tl.float32)
    a_log = tl.load(a_log_ptr + head, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head, mask=mask, other=0.0).to(tl.float32)
    decay_input = a + dt_bias
    softplus = tl.where(decay_input > 20.0, decay_input, libdevice.log1p(libdevice.exp(decay_input)))
    tl.store(beta_ptr + offsets, 1.0 / (1.0 + libdevice.exp(-b)), mask=mask)
    tl.store(decay_ptr + offsets, -libdevice.exp(a_log) * softplus, mask=mask)

def triton_qwen35_fused_gdn_gates(b: torch.Tensor, a: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE or not b.is_cuda or b.dtype != torch.bfloat16 or (a.dtype != b.dtype) or (b.shape != a.shape) or (b.ndim != 3) or (b.shape[1] != 1) or (b.stride(-1) != 1) or (a.stride(-1) != 1) or (a_log.numel() != b.shape[-1]) or (dt_bias.numel() != b.shape[-1]):
        raise ValueError('fused GDN gates require one-token CUDA BF16 head logits')
    beta = torch.empty(b.shape, device=b.device, dtype=b.dtype)
    decay = torch.empty(b.shape, device=b.device, dtype=torch.float32)
    total_elements = int(b.numel())
    block_size = 256
    _qwen35_fused_gdn_gates_kernel[triton.cdiv(total_elements, block_size),](b, a, a_log, dt_bias, beta, decay, b_row_stride=int(b.stride(-2)), a_row_stride=int(a.stride(-2)), heads=int(b.shape[-1]), total_elements=total_elements, block_size=block_size, num_warps=4, num_stages=1)
    return (beta, decay)

def _fused_gdn_gate_native_decode_forward(module, hidden_states: torch.Tensor, cache_params=None, attention_mask: torch.Tensor | None=None, **kwargs):
    if not getattr(module, '_qwen35_fused_gdn_gates_enabled', False) or not hidden_states.is_cuda or hidden_states.shape[-2] != 1 or (cache_params is None) or (not cache_params.has_previous_state(module.layer_idx)):
        return module._qwen35_fused_gdn_gates_eager_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    layer_cache = cache_params.layers[module.layer_idx]
    conv_state = layer_cache.conv_states
    recurrent_state = layer_cache.recurrent_states
    mixed_qkv = module.in_proj_qkv(hidden_states).transpose(1, 2)
    z = module.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, module.head_v_dim)
    b = module.in_proj_b(hidden_states)
    a = module.in_proj_a(hidden_states)
    beta, g = triton_qwen35_fused_gdn_gates(b, a, module.A_log, module.dt_bias)
    mixed_qkv = module.causal_conv1d_update(mixed_qkv, conv_state, module.conv1d.weight.squeeze(1), module.conv1d.bias, module.activation).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, (module.key_dim, module.key_dim, module.value_dim), dim=-1)
    query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)
    if module.num_v_heads // module.num_k_heads > 1 and (not getattr(module, '_qwen35_gdn_grouped_query_enabled', False)):
        repeats = module.num_v_heads // module.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    core_attn_out, last_recurrent_state = module.recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=recurrent_state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    cache_params.update_recurrent_state(last_recurrent_state, module.layer_idx)
    core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
    z = z.reshape(-1, module.head_v_dim)
    core_attn_out = module.norm(core_attn_out, z)
    return module.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))

def enable_fused_gdn_gate_inference(model: torch.nn.Module, *, grouped_query: bool=False) -> int:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    if not _TRITON_AVAILABLE:
        raise RuntimeError('fused GDN gates require Triton')
    changed = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5GatedDeltaNet):
            continue
        module._qwen35_fused_gdn_gates_enabled = True
        module._qwen35_gdn_grouped_query_enabled = grouped_query
        if not hasattr(module, '_qwen35_fused_gdn_gates_eager_forward'):
            module._qwen35_fused_gdn_gates_eager_forward = module.forward
            module.forward = MethodType(_fused_gdn_gate_native_decode_forward, module)
        changed += 1
    model.config.fixed_memory_fused_gdn_gate_inference = True
    return changed

def enable_fused_swa_output_gate_inference(model: torch.nn.Module) -> int:
    from .fixed_memory import Qwen3_5FixedWindowAttention
    if not _TRITON_AVAILABLE:
        raise RuntimeError('fused SWA output gate requires Triton')
    changed = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5FixedWindowAttention):
            continue
        module._qwen35_fused_swa_output_gate_enabled = True
        changed += 1
    model.config.fixed_memory_fused_swa_output_gate_inference = True
    return changed

@triton.jit
def _int8_group_mlp_gemv_kernel(X, W, S, Y, N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    chunk = triton.cdiv(K, 256 * SPLIT) * 256
    acc = tl.zeros((BN,), tl.float32)
    for start in range(0, chunk, 256):
        cols = split * chunk + start + tl.arange(0, 256)
        x = tl.load(X + cols, cols < K, 0).to(tl.float32)
        w = tl.load(W + rows[:, None] * K + cols[None, :], (rows[:, None] < N) & (cols[None, :] < K), 0).to(tl.float32)
        group = (split * chunk + start) // 256
        scale = tl.load(S + rows * triton.cdiv(K, 256) + group, (rows < N) & (group < triton.cdiv(K, 256)), 0)
        acc += tl.sum(w * x[None, :], 1) * scale
    tl.store(Y + split * N + rows, acc, rows < N)

@triton.jit
def _int8_group_mlp_reduce_kernel(P, Y, N: tl.constexpr, SPLIT: tl.constexpr):
    rows = tl.program_id(0) * 256 + tl.arange(0, 256)
    splits = tl.arange(0, SPLIT)
    values = tl.load(P + splits[:, None] * N + rows[None, :], rows[None, :] < N, 0)
    tl.store(Y + rows, tl.sum(values, 0), rows < N)

def _int8_group_mlp_linear(values, weight, scales):
    n, k = weight.shape
    split = 4 if n < 8192 and k >= 4096 else 1
    bn = 16 if n >= 8192 else 8
    output = torch.empty((*values.shape[:-1], n), device=values.device, dtype=values.dtype)
    partial = torch.empty((split, n), device=values.device, dtype=torch.float32) if split > 1 else output
    _int8_group_mlp_gemv_kernel[triton.cdiv(n, bn), split](values, weight, scales, partial, n, k, split, bn, num_warps=4, num_stages=1)
    if split > 1:
        _int8_group_mlp_reduce_kernel[triton.cdiv(n, 256),](partial, output, n, split, num_warps=4)
    return output

@torch.no_grad()
def _refresh_int8_group_mlp(module):
    for name, weight in (('gate_up', module._qwen35_gate_up_weight), ('down', module.down_proj.weight)):
        grouped = weight.float().reshape(weight.shape[0], -1, 256)
        scales = grouped.abs().amax(-1).clamp_min(1e-12) / 127
        quantized = torch.round(grouped / scales[:, :, None]).clamp(-127, 127).to(torch.int8)
        for suffix, tensor in (('weight', quantized.reshape_as(weight)), ('scales', scales)):
            key = f'_qwen35_int8_{name}_{suffix}'
            if hasattr(module, key):
                getattr(module, key).copy_(tensor)
            else:
                module.register_buffer(key, tensor.contiguous(), persistent=False)

def _int8_mlp_decode_context_forward(model, *args, **kwargs):
    ids = kwargs.get('input_ids', args[0] if args else None)
    model._qwen35_int8_mlp_decode_context[0] = ids is not None and ids.shape[-1] == 1 and (kwargs.get('past_key_values') is not None)
    try:
        return model._qwen35_int8_mlp_previous_forward(*args, **kwargs)
    finally:
        model._qwen35_int8_mlp_decode_context[0] = False

@torch.no_grad()
def enable_int8_mlp_b1_inference(model: torch.nn.Module) -> int:
    if not hasattr(model, '_qwen35_int8_mlp_decode_context'):
        model._qwen35_int8_mlp_decode_context = [False]
        model._qwen35_int8_mlp_previous_forward = model.forward
        model.forward = MethodType(_int8_mlp_decode_context_forward, model)
    changed = 0
    for module in model.modules():
        if hasattr(module, '_qwen35_gate_up_weight'):
            _refresh_int8_group_mlp(module)
            module._qwen35_int8_mlp_decode_context = model._qwen35_int8_mlp_decode_context
            module._qwen35_int8_mlp_b1_enabled = True
            changed += 1
    model.config.fixed_memory_int8_mlp_b1 = True
    return changed

def _packed_mlp_forward(module, values: torch.Tensor) -> torch.Tensor:
    if not getattr(module, '_qwen35_packed_mlp_enabled', True):
        return module._qwen35_eager_forward(values)
    if getattr(module, '_qwen35_int8_mlp_b1_enabled', False) and module._qwen35_int8_mlp_decode_context[0] and (values.numel() == values.shape[-1]):
        gate_up = _int8_group_mlp_linear(values, module._qwen35_int8_gate_up_weight, module._qwen35_int8_gate_up_scales)
        hidden = triton_qwen35_silu_mul(gate_up, module.intermediate_size)
        return _int8_group_mlp_linear(hidden, module._qwen35_int8_down_weight, module._qwen35_int8_down_scales)
    gate_up = F.linear(values, module._qwen35_gate_up_weight)
    if getattr(module, '_qwen35_fused_mlp_activation_enabled', False):
        return module.down_proj(triton_qwen35_silu_mul(gate_up, module.intermediate_size))
    gate, up = gate_up.split(module.intermediate_size, dim=-1)
    return module.down_proj(module.act_fn(gate) * up)

@torch.no_grad()
def refresh_packed_mlp_inference(model: torch.nn.Module) -> int:
    refreshed = 0
    for module in model.modules():
        if not hasattr(module, '_qwen35_gate_up_weight'):
            continue
        width = int(module.intermediate_size)
        module._qwen35_gate_up_weight[:width].copy_(module.gate_proj.weight)
        module._qwen35_gate_up_weight[width:].copy_(module.up_proj.weight)
        if getattr(module, '_qwen35_int8_mlp_b1_enabled', False):
            _refresh_int8_group_mlp(module)
        refreshed += 1
    return refreshed

def enable_packed_mlp_inference(model: torch.nn.Module) -> int:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
    patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5MLP):
            continue
        if hasattr(module, '_qwen35_gate_up_weight'):
            continue
        packed = torch.empty((2 * module.intermediate_size, module.hidden_size), device=module.gate_proj.weight.device, dtype=module.gate_proj.weight.dtype)
        module.register_buffer('_qwen35_gate_up_weight', packed, persistent=False)
        module._qwen35_eager_forward = module.forward
        module._qwen35_packed_mlp_enabled = True
        module.forward = MethodType(_packed_mlp_forward, module)
        patched += 1
    refresh_packed_mlp_inference(model)
    return patched

@torch.no_grad()
def enable_independent_memory_packed_projections(model: torch.nn.Module) -> int:
    from .independent_memory import IndependentGQAMemory
    count = 0
    for module in model.modules():
        if isinstance(module, IndependentGQAMemory):
            for name, first, second in (('_inference_writer_gate_up', module.write_gate_proj, module.write_up_proj), ('_inference_kv', module.k_proj, module.v_proj)):
                weights = torch.cat((first.weight, second.weight), 0)
                if hasattr(module, name):
                    getattr(module, name).copy_(weights)
                else:
                    module.register_buffer(name, weights, persistent=False)
            count += 1
    return count

def enable_fused_mlp_activation_inference(model: torch.nn.Module) -> int:
    if not _TRITON_AVAILABLE:
        raise RuntimeError('fused MLP activation requires Triton')
    hidden_act = str(getattr(model.config, 'hidden_act', ''))
    if hidden_act not in {'silu', 'swish'}:
        raise ValueError(f'fused MLP activation does not support {hidden_act!r}')
    changed = 0
    for module in model.modules():
        if not hasattr(module, '_qwen35_gate_up_weight'):
            continue
        module._qwen35_fused_mlp_activation_enabled = True
        changed += 1
    return changed

def _packed_all_native_projection_forward(module, values: torch.Tensor) -> torch.Tensor:
    owner = module.__dict__['_qwen35_all_native_owner_ref']()
    if owner is None or not getattr(owner, '_qwen35_packed_all_native_enabled', True):
        return module._qwen35_all_native_eager_forward(values)
    kind = module.__dict__['_qwen35_all_native_projection_kind']
    cached = getattr(owner, '_qwen35_all_native_projection_cache', None)
    if kind == 'qkv' or cached is None or cached[0] != id(values):
        projected = F.linear(values, owner._qwen35_all_native_weight)
        owner._qwen35_all_native_projection_cache = (id(values), projected)
    else:
        projected = cached[1]
    qkv_end, z_end, b_end = owner._qwen35_all_native_splits
    if kind == 'qkv':
        return projected[..., :qkv_end]
    if kind == 'z':
        return projected[..., qkv_end:z_end]
    if kind == 'b':
        return projected[..., z_end:b_end]
    owner._qwen35_all_native_projection_cache = None
    return projected[..., b_end:]

@torch.no_grad()
def refresh_packed_all_native_gdn_inference(model: torch.nn.Module) -> int:
    refreshed = 0
    for owner in model.modules():
        if not hasattr(owner, '_qwen35_all_native_weight'):
            continue
        start = 0
        for projection in (owner.in_proj_qkv, owner.in_proj_z, owner.in_proj_b, owner.in_proj_a):
            stop = start + int(projection.out_features)
            owner._qwen35_all_native_weight[start:stop].copy_(projection.weight)
            start = stop
        refreshed += 1
    return refreshed

def enable_packed_all_native_gdn_inference(model: torch.nn.Module) -> int:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    patched = 0
    for owner in model.modules():
        if not isinstance(owner, Qwen3_5GatedDeltaNet):
            continue
        if hasattr(owner, '_qwen35_all_native_weight'):
            continue
        projections = (('qkv', owner.in_proj_qkv), ('z', owner.in_proj_z), ('b', owner.in_proj_b), ('a', owner.in_proj_a))
        sizes = tuple((int(projection.out_features) for _, projection in projections))
        owner.register_buffer('_qwen35_all_native_weight', torch.empty((sum(sizes), owner.hidden_size), device=owner.in_proj_qkv.weight.device, dtype=owner.in_proj_qkv.weight.dtype), persistent=False)
        owner._qwen35_all_native_splits = (sizes[0], sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2])
        owner._qwen35_packed_all_native_enabled = True
        owner._qwen35_all_native_projection_cache = None
        owner_ref = weakref.ref(owner)
        for kind, projection in projections:
            projection._qwen35_all_native_eager_forward = getattr(projection, '_qwen35_native_eager_forward', projection.forward)
            projection.__dict__['_qwen35_all_native_owner_ref'] = owner_ref
            projection.__dict__['_qwen35_all_native_projection_kind'] = kind
            projection.forward = MethodType(_packed_all_native_projection_forward, projection)
        patched += 1
    refresh_packed_all_native_gdn_inference(model)
    for owner in model.modules():
        if hasattr(owner, '_qwen35_all_native_weight') and hasattr(owner, '_qwen35_native_zba_weight'):
            del owner._qwen35_native_zba_weight
            owner._qwen35_native_projection_cache = None
    model.config.fixed_memory_packed_all_native_gdn_inference = True
    return patched

@torch.no_grad()
def refresh_packed_swa_projection_inference(model: torch.nn.Module) -> int:
    refreshed = 0
    for module in model.modules():
        if not hasattr(module, '_qwen35_swa_qkv_weight'):
            continue
        start = 0
        for projection in (module.q_proj, module.k_proj, module.v_proj):
            end = start + projection.out_features
            module._qwen35_swa_qkv_weight[start:end].copy_(projection.weight)
            if module._qwen35_swa_qkv_bias is not None:
                if projection.bias is None:
                    module._qwen35_swa_qkv_bias[start:end].zero_()
                else:
                    module._qwen35_swa_qkv_bias[start:end].copy_(projection.bias)
            start = end
        refreshed += 1
    return refreshed

def enable_packed_swa_projection_inference(model: torch.nn.Module) -> int:
    from .fixed_memory import Qwen3_5FixedWindowAttention
    patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5FixedWindowAttention):
            continue
        if hasattr(module, '_qwen35_swa_qkv_weight'):
            continue
        projections = (module.q_proj, module.k_proj, module.v_proj)
        sizes = tuple((int(projection.out_features) for projection in projections))
        module.register_buffer('_qwen35_swa_qkv_weight', torch.empty((sum(sizes), module.q_proj.in_features), device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype), persistent=False)
        if any((projection.bias is not None for projection in projections)):
            module.register_buffer('_qwen35_swa_qkv_bias', torch.empty(sum(sizes), device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype), persistent=False)
        else:
            module.register_buffer('_qwen35_swa_qkv_bias', None, persistent=False)
        module._qwen35_swa_qkv_sizes = sizes
        module._qwen35_packed_swa_projection_enabled = True
        patched += 1
    refresh_packed_swa_projection_inference(model)
    model.config.fixed_memory_packed_swa_projection_inference = True
    return patched

@triton.jit
def _qwen35_bf16_b1_lm_head_kernel(hidden_ptr, weight_ptr, output_ptr, output_elements: tl.constexpr, hidden_elements: tl.constexpr, block_output: tl.constexpr, block_hidden: tl.constexpr):
    rows = tl.program_id(0) * block_output + tl.arange(0, block_output)
    row_mask = rows < output_elements
    accumulator = tl.zeros((block_output,), tl.float32)
    for start in range(0, hidden_elements, block_hidden):
        columns = start + tl.arange(0, block_hidden)
        column_mask = columns < hidden_elements
        hidden = tl.load(hidden_ptr + columns, mask=column_mask, other=0.0)
        offsets = rows[:, None] * hidden_elements + columns[None, :]
        weight = tl.load(weight_ptr + offsets, mask=row_mask[:, None] & column_mask[None, :], other=0.0)
        accumulator += tl.sum(weight.to(tl.float32) * hidden[None, :], axis=1)
    tl.store(output_ptr + rows, accumulator, mask=row_mask)

def _bf16_b1_lm_head_forward(module: torch.nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    token_count = hidden_states.numel() // hidden_states.shape[-1]
    if hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16 and (token_count == 1) and (hidden_states.stride(-1) == 1):
        output = torch.empty((*hidden_states.shape[:-1], module.out_features), device=hidden_states.device, dtype=hidden_states.dtype)
        _qwen35_bf16_b1_lm_head_kernel[triton.cdiv(module.out_features, 8),](hidden_states, module.weight, output, output_elements=module.out_features, hidden_elements=module.in_features, block_output=8, block_hidden=256, num_warps=4, num_stages=1)
        return output
    return module._qwen35_bf16_b1_eager_forward(hidden_states)

def enable_bf16_b1_lm_head_inference(model: torch.nn.Module) -> int:
    lm_head = getattr(model, 'lm_head', None)
    if not isinstance(lm_head, torch.nn.Linear):
        raise TypeError('BF16 B1 lm_head inference requires a Linear lm_head')
    if not _TRITON_AVAILABLE or not lm_head.weight.is_cuda or lm_head.weight.dtype != torch.bfloat16 or (not lm_head.weight.is_contiguous()) or (lm_head.bias is not None):
        raise RuntimeError('BF16 B1 lm_head inference requires Triton and a contiguous bias-free CUDA BF16 lm_head')
    if hasattr(lm_head, '_qwen35_bf16_b1_eager_forward'):
        return 1
    lm_head._qwen35_bf16_b1_eager_forward = lm_head.forward
    lm_head.forward = MethodType(_bf16_b1_lm_head_forward, lm_head)
    model.config.fixed_memory_bf16_b1_lm_head = True
    return 1

def install_inference(model, *, fp16_state=False, int8_mlp_b1=False) -> dict[str, int | str | bool]:
    gdn_bv, gdn_warps = (128, 4)
    recurrent_storage = 'fp16_non_anchor' if fp16_state else 'fp32'
    stack = {'fixed_memory_operator_profile': 'max_throughput', 'operator_revision': 'independent-gqa-grouped-fused-decode-v6', 'memory_backend': 'triton_direct_indexed_gqa', 'memory_decode_backend': 'triton_committed_gqa_masked_commit', 'memory_cache_dtype': 'bfloat16', 'packed_memory_layers': enable_independent_memory_packed_projections(model), 'packed_mlp_layers': enable_packed_mlp_inference(model), 'fused_mlp_activation_layers': enable_fused_mlp_activation_inference(model), 'packed_native_gdn_layers': enable_packed_all_native_gdn_inference(model), 'packed_all_native_gdn': True, 'graph_safe_conv_layers': enable_triton_graph_safe_conv_update(model), 'packed_swa_projection_layers': enable_packed_swa_projection_inference(model), 'native_rmsnorm_layers': enable_native_rmsnorm_inference(model, torch_tree=True), 'rmsnorm_reduction': 'torch_fp32_tree_fused', 'fused_residual_rmsnorm_layers': enable_fused_residual_rmsnorm_inference(model), 'fused_swa_output_gate_layers': enable_fused_swa_output_gate_inference(model), 'inplace_native_gdn_state': enable_inplace_native_gdn_state_inference(gdn_bv, gdn_warps, adaptive=True), 'gdn_block_value': gdn_bv, 'gdn_num_warps': gdn_warps, 'gdn_small_batch_block_value': 32, 'fused_gdn_gate_layers': enable_fused_gdn_gate_inference(model, grouped_query=True), 'gdn_grouped_query_decode': True, 'physical_ring_swa_layers': enable_physical_ring_fixed_memory_swa_update(model), 'swa_backend': 'physical', 'gdn_recurrent_state_storage': recurrent_storage, 'fused_decode_state': True, 'bf16_b1_lm_head': enable_bf16_b1_lm_head_inference(model)}
    if int8_mlp_b1:
        stack['int8_mlp_b1_layers'] = enable_int8_mlp_b1_inference(model)
        stack['int8_mlp_group_size'] = 256
        stack['int8_mlp_down_split_k'] = 4
        stack['int8_mlp_up_tile'] = 16
        stack['operator_revision'] += '-int8-mlp-b1-up16'
    model.config.fixed_memory_operator_profile = 'max_throughput'
    model.config.fixed_memory_operator_revision = stack['operator_revision']
    model.config.gdn_recurrent_state_storage = recurrent_storage
    model.config.fixed_memory_fused_decode_state = True
    return stack

@triton.jit
def _qwen35_dense_presence_penalty_kernel(scores_ptr, seen_ptr, total_elements: tl.constexpr, penalty: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    scores = tl.load(scores_ptr + offsets, mask=mask)
    seen = tl.load(seen_ptr + offsets, mask=mask, other=0).to(tl.int1)
    scores = scores.to(tl.float32) - tl.where(seen, penalty, 0.0)
    tl.store(scores_ptr + offsets, scores, mask=mask)

def _apply_dense_presence_penalty_(scores: torch.Tensor, seen: torch.Tensor, penalty: float) -> None:
    if penalty == 0.0:
        return
    if scores.shape != seen.shape or seen.dtype != torch.bool:
        raise ValueError('dense presence state must be a boolean scores-shaped tensor')
    if scores.is_cuda and _TRITON_AVAILABLE and scores.is_contiguous() and seen.is_contiguous():
        total_elements = scores.numel()
        block_size = 256
        _qwen35_dense_presence_penalty_kernel[triton.cdiv(total_elements, block_size),](scores, seen, total_elements=total_elements, penalty=float(penalty), block_size=block_size, num_warps=4, num_stages=1)
        return
    scores.sub_(seen.to(dtype=scores.dtype), alpha=float(penalty))

def _sample_top_k_top_p_counter(scores: torch.Tensor, attempt_ids: torch.Tensor, step: torch.Tensor | int, *, seed: int, temperature: float, top_k: int, top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    return sample_top_k_top_p_counter(scores, attempt_ids, step, seed=seed, temperature=temperature, top_k=top_k, top_p=top_p)

@triton.jit
def _qwen35_top20_counter_sample_kernel(values_ptr, indices_ptr, attempt_ids_ptr, steps_ptr, output_token_ptr, seed_term, scalar_step: tl.constexpr, step_is_tensor: tl.constexpr, top_p: tl.constexpr, support: tl.constexpr, block_size: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    mask = columns < support
    offsets = row * support + columns
    values = tl.load(values_ptr + offsets, mask=mask, other=-float('inf')).to(tl.float32)
    maximum = tl.max(values, axis=0)
    weights = tl.exp(values - maximum)
    total = tl.sum(weights, axis=0)
    probabilities = weights / total
    prefix_before = tl.cumsum(probabilities, axis=0) - probabilities
    truncated = tl.where(mask & (prefix_before < top_p), probabilities, 0.0)
    retained = tl.sum(truncated, axis=0)
    cumulative = tl.cumsum(truncated, axis=0)
    attempt_id = tl.load(attempt_ids_ptr + row).to(tl.uint32)
    step = tl.load(steps_ptr + row).to(tl.uint32) if step_is_tensor else tl.full((1,), scalar_step, tl.uint32)
    state = (attempt_id * 747796405 + step * 289133645 + seed_term).to(tl.uint32)
    shift = (state >> 28) + 4
    word = ((state >> shift ^ state) * 277803737).to(tl.uint32)
    word = (word >> 22 ^ word).to(tl.uint32)
    uniform = ((word.to(tl.float64) + 0.5) / 4294967296.0).to(tl.float32)
    threshold = uniform * retained
    rank = tl.sum((mask & (cumulative < threshold)).to(tl.int32), axis=0)
    rank = tl.minimum(rank, support - 1)
    selected_token = tl.sum(tl.where(columns == rank, tl.load(indices_ptr + offsets, mask=mask, other=0), 0), axis=0)
    tl.store(output_token_ptr + row, selected_token)

def _sample_top_k_top_p_counter_triton(scores: torch.Tensor, attempt_ids: torch.Tensor, step: torch.Tensor | int, *, seed: int, temperature: float, top_k: int, top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE or not scores.is_cuda or top_k != 20 or (not 0.0 < top_p <= 1.0) or (temperature <= 0.0):
        raise ValueError('fused counter sampler requires CUDA top20 scores')
    values, indices = torch.topk(scores, top_k, dim=-1)
    if temperature != 1.0:
        values = values / temperature
    if attempt_ids.shape != scores.shape[:1]:
        raise ValueError('fused counter sampler requires one attempt id per row')
    if isinstance(step, torch.Tensor):
        if step.shape != scores.shape[:1] or step.dtype not in (torch.int32, torch.int64):
            raise ValueError('fused counter sampler requires one integer step per row')
        steps = step.contiguous()
        scalar_step = 0
        step_is_tensor = True
    else:
        steps = attempt_ids
        scalar_step = int(step) & 4294967295
        step_is_tensor = False
    tokens = torch.empty((scores.shape[0], 1), device=scores.device, dtype=torch.long)
    _qwen35_top20_counter_sample_kernel[scores.shape[0],](values, indices, attempt_ids, steps, tokens, seed_term=(int(seed) & 4294967295) * 277803737 & 4294967295, scalar_step=scalar_step, step_is_tensor=step_is_tensor, top_p=float(top_p), support=20, block_size=32, num_warps=1, num_stages=1)
    return tokens

@triton.jit
def _qwen35_decode_state_update_kernel(sampled_token_ptr, storage_ptr, dense_seen_ptr, generated_lengths_ptr, finished_by_eos_ptr, finished_ptr, slot_rows_ptr, sink_rows_ptr, slot_caps_ptr, row_steps_ptr, row_positions_ptr, decode_ids_ptr, decode_useful_events_ptr, storage_stride, seen_stride, maximum_new_tokens, slots: tl.constexpr, block_size: tl.constexpr, pad_token: tl.constexpr, eos_token_0: tl.constexpr, eos_token_1: tl.constexpr, eos_count: tl.constexpr):
    row = tl.arange(0, block_size)
    mask = row < slots
    sampled_token = tl.load(sampled_token_ptr + row, mask, 0).to(tl.int64)
    finished = tl.load(finished_ptr + row, mask, 1).to(tl.int1)
    slot_row = tl.load(slot_rows_ptr + row, mask, -1).to(tl.int64)
    active = mask & ~finished & (slot_row >= 0)
    sink_row = tl.load(sink_rows_ptr + row, mask, 0).to(tl.int64)
    step = tl.load(row_steps_ptr + row, mask, 0).to(tl.int64)
    write_row = tl.where(active, slot_row, sink_row)
    write_column = tl.where(active, step, maximum_new_tokens)
    stored_token = tl.where(active, sampled_token, pad_token)
    storage_offset = write_row * storage_stride + write_column
    tl.store(storage_ptr + storage_offset, stored_token, mask)
    tl.store(dense_seen_ptr + row * seen_stride + stored_token, 1, mask)
    next_step = step + active.to(tl.int64)
    old_generated_length = tl.load(generated_lengths_ptr + write_row, mask, 0)
    tl.store(generated_lengths_ptr + write_row, tl.where(active, next_step, old_generated_length), mask)
    if eos_count == 0:
        is_eos = stored_token != stored_token
    elif eos_count == 1:
        is_eos = stored_token == eos_token_0
    else:
        is_eos = (stored_token == eos_token_0) | (stored_token == eos_token_1)
    old_finished_by_eos = tl.load(finished_by_eos_ptr + write_row, mask, 0).to(tl.int1)
    tl.store(finished_by_eos_ptr + write_row, old_finished_by_eos | active & is_eos, mask)
    cap = tl.load(slot_caps_ptr + row, mask, 0).to(tl.int64)
    next_finished = finished | active & (is_eos | (next_step >= cap))
    tl.store(finished_ptr + row, next_finished, mask)
    tl.store(row_steps_ptr + row, next_step, mask)
    position = tl.load(row_positions_ptr + row, mask, 0).to(tl.int64)
    tl.store(row_positions_ptr + row, position + active.to(tl.int64), mask)
    tl.store(decode_ids_ptr + row, tl.where(next_finished, pad_token, stored_token), mask)
    useful_events = tl.load(decode_useful_events_ptr).to(tl.int64)
    tl.store(decode_useful_events_ptr, useful_events + tl.sum(active.to(tl.int64), 0))

def _update_fixed_decode_state_(sampled_token: torch.Tensor, storage: torch.Tensor, dense_seen: torch.Tensor, generated_lengths: torch.Tensor, finished_by_eos: torch.Tensor, finished: torch.Tensor, slot_rows: torch.Tensor, sink_rows: torch.Tensor, slot_caps: torch.Tensor, row_steps: torch.Tensor, row_positions: torch.Tensor, decode_ids: torch.Tensor, decode_useful_events: torch.Tensor, *, maximum_new_tokens: int, pad_token_id: int, eos_token_ids: tuple[int, ...]) -> None:
    eos_count = min(len(eos_token_ids), 2)
    _qwen35_decode_state_update_kernel[1,](sampled_token, storage, dense_seen, generated_lengths, finished_by_eos, finished, slot_rows, sink_rows, slot_caps, row_steps, row_positions, decode_ids, decode_useful_events, storage_stride=int(storage.shape[1]), seen_stride=int(dense_seen.shape[1]), maximum_new_tokens=int(maximum_new_tokens), slots=int(sampled_token.shape[0]), block_size=triton.next_power_of_2(sampled_token.shape[0]), pad_token=int(pad_token_id), eos_token_0=int(eos_token_ids[0]) if eos_count else 0, eos_token_1=int(eos_token_ids[1]) if eos_count > 1 else 0, eos_count=eos_count, num_warps=1 if sampled_token.shape[0] <= 32 else 4, num_stages=1)

def _prefill_logits_and_cache(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, scores_dtype: torch.dtype, fixed_memory: bool, prefill_chunk_size: int | None=None, prefill_token_chunk_size: int | None=None, prefill_group_ids: torch.Tensor | None=None, resolved_recurrent_state_storage: str | None=None, prefill_context=None, logical_row_indices: torch.Tensor | None=None) -> tuple[object, torch.Tensor, int]:

    def prepare_recurrent_storage(cache):
        if not fixed_memory:
            return
        storage = resolved_recurrent_state_storage
        if storage is None:
            storage = resolve_fixed_memory_recurrent_state_storage(str(getattr(getattr(model, 'config', None), 'gdn_recurrent_state_storage', 'fp32')), expanded_batch)
        if storage == 'fp32':
            return
        if storage not in {'bf16_non_anchor', 'fp16_non_anchor'}:
            raise ValueError(f'unsupported GDN recurrent state storage: {storage}')
        from .fixed_memory import convert_fixed_memory_recurrent_state_storage_
        convert_fixed_memory_recurrent_state_storage_(cache, torch.bfloat16 if storage == 'bf16_non_anchor' else torch.float16)
    expanded_batch = int(input_ids.shape[0])
    if prefill_group_ids is not None:
        if prefill_group_ids.ndim != 1 or int(prefill_group_ids.shape[0]) != expanded_batch or prefill_group_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('prefill group ids require one integer id per input row')
        prefill_group_ids = prefill_group_ids.to(input_ids.device, dtype=torch.int64)
        plan = build_prefill_expansion(prefill_group_ids)
        assert_group_rows_identical(input_ids, plan)
        assert_group_rows_identical(attention_mask, plan)
        representative_indices = plan.representative_indices.to(input_ids.device)
        prefill_ids = input_ids.index_select(0, representative_indices)
        prefill_mask = attention_mask.index_select(0, representative_indices)
        if logical_row_indices is not None:
            logical_row_indices = logical_row_indices.index_select(0, representative_indices)
    else:
        plan = None
        prefill_ids = input_ids
        prefill_mask = attention_mask
    unique_batch = int(prefill_ids.shape[0])
    chunk_size = min(unique_batch, int(prefill_chunk_size or unique_batch))
    if chunk_size < unique_batch and (not fixed_memory):
        raise ValueError('chunked prefill currently requires fixed-memory inference')
    compact = prefill_context is None and bool(getattr(model.config, 'independent_memory', None))
    token_chunk = int(prefill_token_chunk_size) if compact else 0
    if token_chunk and prefill_ids.shape[1] > token_chunk:
        chunk_size = 1
    cache = scores = None
    virtual_padding = 0
    for start in range(0, unique_batch, chunk_size):
        stop = min(start + chunk_size, unique_batch)
        ids, mask = (prefill_ids[start:stop], prefill_mask[start:stop])
        if prefill_context is None:
            positions = {}
            if compact:
                width = int(mask.sum(-1).max().item())
                ids, mask = (ids[:, -width:], mask[:, -width:])
                positions['position_ids'] = (mask.long().cumsum(-1) - 1).clamp_min(0)
            if token_chunk and ids.shape[1] > token_chunk:
                token_cache = None
                for begin in range(0, ids.shape[1], token_chunk):
                    end = min(begin + token_chunk, ids.shape[1])
                    prefill = model(input_ids=ids[:, begin:end], attention_mask=None, position_ids=positions['position_ids'][:, begin:end], past_key_values=token_cache, use_cache=True, logits_to_keep=1)
                    token_cache = prefill.past_key_values
                    if end < ids.shape[1]:
                        del prefill
                del token_cache
            else:
                prefill = model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1, **positions)
        else:
            prefill = prefill_context(ids, mask, logical_row_indices[start:stop])
        chunk_cache = prefill.past_key_values
        prepare_recurrent_storage(chunk_cache)
        if fixed_memory:
            virtual_padding += prepare_fixed_memory_swa_cache_for_physical_ring_(chunk_cache, int(model.config.fixed_memory_swa_window))
        chunk_scores = prefill.logits[:, -1, :].to(dtype=scores_dtype)
        del prefill
        if chunk_size == unique_batch:
            cache, scores = (chunk_cache, chunk_scores)
            break
        target_indices = torch.arange(start, stop, device=input_ids.device, dtype=torch.long)
        source_indices = torch.arange(stop - start, device=input_ids.device, dtype=torch.long)
        if cache is None:
            initial_indices = torch.zeros(unique_batch, device=input_ids.device, dtype=torch.long)
            initial_indices.index_copy_(0, target_indices, source_indices)
            chunk_cache.batch_select_indices(initial_indices)
            cache = chunk_cache
            scores = chunk_scores.index_select(0, initial_indices)
        else:
            cache.batch_scatter_from(chunk_cache, target_indices, source_indices)
            scores.index_copy_(0, target_indices, chunk_scores)
        del chunk_cache, chunk_scores
    cache._prefill_virtual_swa_padding = virtual_padding
    del prefill_ids, prefill_mask
    if plan is not None and unique_batch != expanded_batch:
        inverse_indices = plan.inverse_indices.to(input_ids.device)
        cache.batch_select_indices(inverse_indices)
        scores = scores.index_select(0, inverse_indices)
    return (cache, scores, unique_batch)

@dataclass
class DynamicSlotRefillOutput:
    generated_lengths: torch.Tensor
    metrics: dict[str, object]

def _left_pad_refill_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor, *, width: int, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError('refill inputs require aligned two-dimensional tensors')
    padding = int(width) - int(input_ids.shape[1])
    if padding < 0:
        raise ValueError('refill width is shorter than the supplied prompt')
    if padding == 0:
        return (input_ids, attention_mask)
    padded_ids = input_ids.new_full((input_ids.shape[0], width), int(pad_token_id))
    padded_mask = attention_mask.new_zeros((attention_mask.shape[0], width))
    padded_ids[:, padding:].copy_(input_ids)
    padded_mask[:, padding:].copy_(attention_mask)
    return (padded_ids, padded_mask)

def _initial_attempt_order(difficulty: torch.Tensor, attempt_ids: list[int], prefill_group_ids: list[int] | None=None) -> list[int]:
    rows = range(len(attempt_ids))
    if prefill_group_ids is None:
        return sorted(rows, key=lambda row: (-float(difficulty[row]), int(attempt_ids[row])))
    groups: dict[int, list[int]] = {}
    for row, group_id in enumerate(prefill_group_ids):
        groups.setdefault(int(group_id), []).append(row)
    ordered_groups = sorted(groups.values(), key=lambda group: (-max((float(difficulty[row]) for row in group)), min((int(attempt_ids[row]) for row in group))))
    return [row for group in ordered_groups for row in sorted(group, key=lambda value: (-float(difficulty[value]), int(attempt_ids[value])))]

def _group_aligned_prefix_length(ordered_rows: list[int], prefill_group_ids: list[int], limit: int) -> int:
    if limit <= 0 or not ordered_rows:
        return 0
    accepted = 0
    while accepted < len(ordered_rows):
        group_id = int(prefill_group_ids[ordered_rows[accepted]])
        group_stop = accepted + 1
        while group_stop < len(ordered_rows) and int(prefill_group_ids[ordered_rows[group_stop]]) == group_id:
            group_stop += 1
        if group_stop > limit:
            break
        accepted = group_stop
    if accepted == 0:
        first_group = int(prefill_group_ids[ordered_rows[0]])
        raise ValueError(f'prefill group {first_group} is larger than the ready-bank limit')
    return accepted

def _should_compact_dynamic_decode_tail(live_slots: int, physical_slots: int, *, queue_exhausted: bool, live_fraction: float) -> bool:
    return queue_exhausted and 0 < live_slots < physical_slots and (live_slots <= max(1, int(physical_slots * live_fraction)))

@torch.no_grad()
def cuda_graph_generate_top20_dynamic_refill(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, slot_batch_size: int, max_new_tokens: int, temperature: float, top_k: int=20, top_p: float=0.95, presence_penalty: float=1.5, eos_token_ids: tuple[int, ...]=(248046, 248044), pad_token_id: int=248044, stop_check_interval_tokens: int=128, counter_seed: int=20260820, attempt_ids: torch.Tensor | None=None, prefill_group_ids: torch.Tensor | None=None, max_new_tokens_per_attempt: torch.Tensor | None=None, refill_batch_size: int | None=None, refill_prefetch_size: int | None=None, prefill_chunk_size: int | None=None, prefill_token_chunk_size: int | None=None, score_dtype: str='float32', sampling_vocab_size: int | None=None, counter_sampling_backend: str='torch', adaptive_tail_compaction: bool=False, tail_compaction_live_fraction: float=0.5, mechanical_repetition_early_stop: bool=True, prefill_context=None, stream=None) -> DynamicSlotRefillOutput:
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError('dynamic refill requires aligned two-dimensional inputs')
    if not input_ids.is_cuda or attention_mask.device != input_ids.device:
        raise ValueError('dynamic refill requires CUDA inputs on one device')
    total_attempts = int(input_ids.shape[0])
    if total_attempts <= 0:
        raise ValueError('dynamic refill requires at least one attempt')
    if slot_batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError('slot batch and generation cap must be positive')
    if stop_check_interval_tokens <= 0:
        raise ValueError('stop-check interval must be positive')
    if score_dtype not in {'float32', 'bfloat16'}:
        raise ValueError('score dtype must be float32 or bfloat16')
    if sampling_vocab_size is not None and (not 0 < int(sampling_vocab_size) <= int(model.config.vocab_size)):
        raise ValueError('sampling vocabulary is incompatible with the model head')
    if counter_sampling_backend not in {'torch', 'triton_top20'}:
        raise ValueError('counter sampling backend must be torch or triton_top20')
    if getattr(model.config, 'fixed_memory_revision', None) is None:
        raise ValueError('dynamic refill requires a fixed-memory model')
    if not getattr(model.config, 'fixed_memory_physical_ring_swa_cuda_graph_update', False):
        raise ValueError('dynamic refill requires the physical SWA ring')
    if mechanical_repetition_early_stop:
        from scoring.protocols import MECHANICAL_REPETITION_EARLY_STOP_MAX_PATTERN_TOKENS, MECHANICAL_REPETITION_EARLY_STOP_MIN_PATTERN_TOKENS, MECHANICAL_REPETITION_EARLY_STOP_MIN_REPETITIONS, MECHANICAL_REPETITION_EARLY_STOP_REVISION, mechanical_repetition_early_stop_patterns
        mechanical_repetition_minimum_tokens = MECHANICAL_REPETITION_EARLY_STOP_MIN_PATTERN_TOKENS * MECHANICAL_REPETITION_EARLY_STOP_MIN_REPETITIONS
        mechanical_repetition_history_tokens = MECHANICAL_REPETITION_EARLY_STOP_MAX_PATTERN_TOKENS * MECHANICAL_REPETITION_EARLY_STOP_MIN_REPETITIONS
        mechanical_repetition_relative_positions = torch.arange(-mechanical_repetition_history_tokens, 0, device=input_ids.device, dtype=torch.long)[None, :]
        mechanical_repetition_revision = MECHANICAL_REPETITION_EARLY_STOP_REVISION
    else:
        mechanical_repetition_revision = None
        mechanical_repetition_minimum_tokens = 0
        mechanical_repetition_history_tokens = 0
        mechanical_repetition_relative_positions = None
    device = input_ids.device
    initial_slots = min(int(slot_batch_size), total_attempts)
    slots = initial_slots
    decode_recurrent_state_storage = resolve_fixed_memory_recurrent_state_storage(str(getattr(model.config, 'gdn_recurrent_state_storage', 'fp32')), slots)
    refill_rows = min(slots, int(refill_batch_size) if refill_batch_size is not None else min(slots, 32))
    if refill_rows <= 0:
        raise ValueError('refill batch size must be positive')
    prefetch_rows = min(total_attempts - slots if total_attempts > slots else refill_rows, int(refill_prefetch_size) if refill_prefetch_size is not None else max(refill_rows, min(slots, 64)))
    prefetch_rows = max(prefetch_rows, refill_rows)
    scores_dtype = torch.float32 if score_dtype == 'float32' else torch.bfloat16
    if attempt_ids is None:
        attempt_ids = torch.arange(total_attempts, device=device, dtype=torch.long)
    else:
        attempt_ids = attempt_ids.to(device=device, dtype=torch.long)
    if attempt_ids.shape != (total_attempts,):
        raise ValueError('attempt ids require one value per logical trajectory')
    if int(torch.unique(attempt_ids).numel()) != total_attempts:
        raise ValueError('attempt ids must be unique')
    prefill_group_ids_cpu = None
    if prefill_group_ids is not None:
        prefill_group_ids = prefill_group_ids.to(device=device, dtype=torch.long)
        if prefill_group_ids.shape != (total_attempts,):
            raise ValueError('prefill group ids require one value per attempt')
        prefill_plan = build_prefill_expansion(prefill_group_ids)
        assert_group_rows_identical(input_ids, prefill_plan)
        assert_group_rows_identical(attention_mask, prefill_plan)
        prefill_group_ids_cpu = prefill_group_ids.detach().cpu().tolist()
    if max_new_tokens_per_attempt is None:
        attempt_caps = torch.full((total_attempts,), max_new_tokens, device=device, dtype=torch.long)
    else:
        attempt_caps = max_new_tokens_per_attempt.to(device=device, dtype=torch.long)
    if attempt_caps.shape != (total_attempts,) or bool(((attempt_caps <= 0) | (attempt_caps > max_new_tokens)).any().item()):
        raise ValueError('per-attempt caps must lie within the generation cap')
    difficulty_cpu = attempt_caps.detach().cpu().to(torch.float64)
    attempt_ids_cpu = attempt_ids.detach().cpu().tolist()
    counter_sampler = _sample_top_k_top_p_counter_triton if counter_sampling_backend == 'triton_top20' else _sample_top_k_top_p_counter
    order = _initial_attempt_order(difficulty_cpu, attempt_ids_cpu, prefill_group_ids_cpu)
    if prefill_group_ids_cpu is not None and slots < total_attempts:
        aligned_slots = _group_aligned_prefix_length(order, prefill_group_ids_cpu, slots)
        if aligned_slots != slots:
            raise ValueError('slot batch size splits a repeated-prompt prefill group')
    initial_rows = order[:slots]
    pending_rows = order[slots:]
    pending_cursor = 0
    free_rows: list[int] = []
    window = int(model.config.fixed_memory_swa_window)
    memory_chunk = min(window, model.config.independent_memory['max_commit_chunk'])
    if window <= 1 or memory_chunk <= 0:
        raise ValueError('sliding window and memory chunk must be positive')
    base_width = int(input_ids.shape[1])
    from .fixed_memory import align_fixed_memory_refill_cache_, set_fixed_memory_cuda_graph_decode
    initial_index = torch.tensor(initial_rows, device=device, dtype=torch.long)
    initial_ids, initial_mask = _left_pad_refill_inputs(input_ids.index_select(0, initial_index), attention_mask.index_select(0, initial_index), width=base_width, pad_token_id=pad_token_id)
    initial_prefill_group_ids = prefill_group_ids.index_select(0, initial_index) if prefill_group_ids is not None else None
    set_fixed_memory_cuda_graph_decode(model, None, False)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    prefill_started = started
    cache, scores, initial_prefill_unique_rows = _prefill_logits_and_cache(model, initial_ids, initial_mask, scores_dtype=scores_dtype, fixed_memory=True, prefill_chunk_size=prefill_chunk_size, prefill_token_chunk_size=prefill_token_chunk_size, prefill_group_ids=initial_prefill_group_ids, resolved_recurrent_state_storage=decode_recurrent_state_storage, prefill_context=prefill_context, logical_row_indices=initial_index)
    if sampling_vocab_size is not None:
        scores = scores[:, :int(sampling_vocab_size)]
    initial_virtual_swa_padding = cache._prefill_virtual_swa_padding
    initial_attempt_ids = attempt_ids.index_select(0, initial_index)
    first_token = counter_sampler(scores, initial_attempt_ids, 0, seed=counter_seed, temperature=temperature, top_k=top_k, top_p=top_p)
    torch.cuda.synchronize(device)
    initial_prefill_seconds = time.perf_counter() - prefill_started
    storage = torch.full((total_attempts + slots, max_new_tokens + 1), int(pad_token_id), device=device, dtype=torch.int32)
    finished_by_eos = torch.zeros(total_attempts + slots, device=device, dtype=torch.bool)
    finished_by_mechanical_repetition = torch.zeros(total_attempts + slots, device=device, dtype=torch.bool)
    generated_lengths = torch.zeros(total_attempts + slots, device=device, dtype=torch.long)
    slot_rows = initial_index.clone()
    slot_attempt_ids = initial_attempt_ids.clone()
    slot_caps = attempt_caps.index_select(0, initial_index)
    row_steps = torch.ones(slots, device=device, dtype=torch.long)
    row_positions = torch.full((slots,), base_width, device=device, dtype=torch.long) if prefill_context is None else prefill_context.decode_positions(initial_index)
    logical_positions = prefill_context is None and bool(getattr(model.config, 'independent_memory', None))
    microbatch_refill = bool(stream is not None and logical_positions and prefill_token_chunk_size)
    if logical_positions:
        row_positions = initial_mask.long().sum(-1)
    storage[slot_rows, 0] = first_token[:, 0].to(dtype=storage.dtype)
    generated_lengths[slot_rows] = 1
    dense_seen = torch.zeros_like(scores, dtype=torch.bool)
    dense_seen.scatter_(1, first_token, True)
    finished = row_steps >= slot_caps
    for token_id in eos_token_ids:
        finished |= first_token[:, 0] == int(token_id)
        finished_by_eos[slot_rows] |= first_token[:, 0] == int(token_id)
    decode_ids = torch.where(finished[:, None], first_token.new_full((), pad_token_id), first_token)
    fixed_graph = None
    sink_rows = total_attempts + torch.arange(slots, device=device)
    decode_useful_events = torch.zeros((), device=device, dtype=torch.long)
    replay_count = 0
    scheduled_decode_events = 0
    refill_count = 0
    refill_waves = 0
    refill_prefill_waves = 0
    refill_virtual_swa_padding = 0
    prefill_expanded_rows = slots
    prefill_unique_rows = initial_prefill_unique_rows
    refill_prefill_seconds = 0.0
    maximum_pending = len(pending_rows)
    ready_cache = None
    ready_rows: list[int] = []
    ready_first_token = None
    ready_attempt_ids = None
    ready_caps = None
    ready_offset = 0
    ready_width = base_width
    ready_positions = None
    tail_compaction_shapes = [slots]
    tail_compaction_count = 0
    mechanical_repetition_checks = 0
    mechanical_repetition_pattern_counts: dict[int, int] = {}

    def poll_and_refill() -> int:
        nonlocal pending_cursor
        nonlocal pending_rows
        nonlocal ready_width
        nonlocal ready_positions
        nonlocal refill_count
        nonlocal refill_waves
        nonlocal refill_prefill_waves
        nonlocal refill_virtual_swa_padding
        nonlocal prefill_expanded_rows
        nonlocal prefill_unique_rows
        nonlocal refill_prefill_seconds
        nonlocal ready_cache
        nonlocal ready_rows
        nonlocal ready_first_token
        nonlocal ready_attempt_ids
        nonlocal ready_caps
        nonlocal ready_offset
        nonlocal mechanical_repetition_checks
        if mechanical_repetition_early_stop:
            candidate_slots_tensor = torch.nonzero(~finished & (slot_rows >= 0) & (row_steps >= mechanical_repetition_minimum_tokens), as_tuple=False).flatten()
            if int(candidate_slots_tensor.numel()):
                candidate_rows_tensor = slot_rows.index_select(0, candidate_slots_tensor)
                candidate_lengths_tensor = row_steps.index_select(0, candidate_slots_tensor)
                positions = candidate_lengths_tensor[:, None] + mechanical_repetition_relative_positions
                valid = positions >= 0
                tails = storage[candidate_rows_tensor[:, None], positions.clamp_min(0)]
                tails_cpu = tails.detach().cpu().numpy()
                valid_widths = valid.sum(dim=1).detach().cpu().numpy()
                candidate_slots = candidate_slots_tensor.detach().cpu().tolist()
                candidate_rows = candidate_rows_tensor.detach().cpu().tolist()
                detected_slots = []
                patterns = mechanical_repetition_early_stop_patterns(tails_cpu, valid_widths)
                mechanical_repetition_checks += len(patterns)
                for offset, pattern_tokens in enumerate(patterns):
                    if not pattern_tokens:
                        continue
                    detected_slots.append(int(candidate_slots[offset]))
                    logical_row = int(candidate_rows[offset])
                    finished_by_mechanical_repetition[logical_row] = True
                    pattern_tokens = int(pattern_tokens)
                    mechanical_repetition_pattern_counts[pattern_tokens] = mechanical_repetition_pattern_counts.get(pattern_tokens, 0) + 1
                if detected_slots:
                    finished[torch.tensor(detected_slots, device=device, dtype=torch.long)] = True
        target_slots = torch.nonzero(finished, as_tuple=False).flatten().detach().cpu().tolist()
        if target_slots:
            target_tensor = torch.tensor(target_slots, device=device, dtype=torch.long)
            if stream is not None:
                completed_rows = slot_rows.index_select(0, target_tensor).cpu().tolist()
                completed_rows = [row for row in completed_rows if row >= 0]
                for row in completed_rows:
                    length = int(generated_lengths[row].item())
                    reason = 'mechanical_repetition' if finished_by_mechanical_repetition[row].item() else 'eos' if finished_by_eos[row].item() else 'prompt_length_cap' if attempt_caps[row].item() < max_new_tokens else 'length'
                    stream.emit(row, storage[row, :length].cpu(), reason)
                free_rows.extend(completed_rows)
            slot_rows.index_fill_(0, target_tensor, -1)
            slot_attempt_ids.index_fill_(0, target_tensor, 0)
            slot_caps.index_fill_(0, target_tensor, 0)
        if stream is not None:
            replacements = stream.acquire(free_rows, block=len(target_slots) == slots and ready_cache is None and (pending_cursor >= len(pending_rows)))
            if replacements:
                new_rows, new_attempts, new_groups, new_caps = zip(*replacements)
                new_index = torch.tensor(new_rows, device=device, dtype=torch.long)
                attempt_ids[new_index] = torch.tensor(new_attempts, device=device)
                attempt_caps[new_index] = torch.tensor(new_caps, device=device)
                prefill_group_ids[new_index] = torch.tensor(new_groups, device=device)
                for row, group in zip(new_rows, new_groups):
                    prefill_group_ids_cpu[row] = group
                finished_by_eos[new_index] = False
                finished_by_mechanical_repetition[new_index] = False
                generated_lengths[new_index] = 0
                pending_rows = pending_rows[pending_cursor:] + list(new_rows)
                pending_cursor = 0
        available = target_slots
        while available and (ready_cache is not None or pending_cursor < len(pending_rows)):
            if ready_cache is None:
                ready_count = min(prefetch_rows, len(pending_rows) - pending_cursor)
                if microbatch_refill:
                    ready_count = min(ready_count, len(available), refill_rows)
                if prefill_group_ids_cpu is not None:
                    ready_count = _group_aligned_prefix_length(pending_rows[pending_cursor:], prefill_group_ids_cpu, ready_count)
                ready_rows = pending_rows[pending_cursor:pending_cursor + ready_count]
                if microbatch_refill:
                    ready_rows = stream.prefill_microbatch_rows(ready_rows, prefill_token_chunk_size)
                    ready_count = len(ready_rows)
                pending_cursor += ready_count
                ready_index = torch.tensor(ready_rows, device=device, dtype=torch.long)
                if stream is None:
                    ready_ids, ready_mask = _left_pad_refill_inputs(input_ids.index_select(0, ready_index), attention_mask.index_select(0, ready_index), width=base_width, pad_token_id=pad_token_id)
                else:
                    ready_ids, ready_mask = stream.inputs(ready_rows, device)
                ready_width = int(ready_ids.shape[1])
                if logical_positions:
                    ready_positions = ready_mask.long().sum(-1)
                ready_prefill_group_ids = prefill_group_ids.index_select(0, ready_index) if prefill_group_ids is not None else None
                torch.cuda.synchronize(device)
                refill_started = time.perf_counter()
                set_fixed_memory_cuda_graph_decode(model, None, False)
                ready_cache, ready_scores, ready_unique_rows = _prefill_logits_and_cache(model, ready_ids, ready_mask, scores_dtype=scores_dtype, fixed_memory=True, prefill_chunk_size=prefill_chunk_size if stream is None else stream.prefill_chunk_size(ready_width), prefill_token_chunk_size=prefill_token_chunk_size, prefill_group_ids=ready_prefill_group_ids, resolved_recurrent_state_storage=decode_recurrent_state_storage, prefill_context=prefill_context, logical_row_indices=ready_index)
                if sampling_vocab_size is not None:
                    ready_scores = ready_scores[:, :int(sampling_vocab_size)]
                prefill_expanded_rows += ready_count
                prefill_unique_rows += ready_unique_rows
                refill_virtual_swa_padding += ready_cache._prefill_virtual_swa_padding
                ready_attempt_ids = attempt_ids.index_select(0, ready_index)
                ready_caps = attempt_caps.index_select(0, ready_index)
                ready_first_token = counter_sampler(ready_scores, ready_attempt_ids, 0, seed=counter_seed, temperature=temperature, top_k=top_k, top_p=top_p)
                set_fixed_memory_cuda_graph_decode(model, ready_cache, True)
                torch.cuda.synchronize(device)
                refill_prefill_seconds += time.perf_counter() - refill_started
                refill_prefill_waves += 1
                ready_offset = 0
                del ready_scores, ready_ids, ready_mask
            ready_remaining = len(ready_rows) - ready_offset
            count = min(len(available), refill_rows, ready_remaining)
            targets = available[:count]
            available = available[count:]
            source_rows = ready_rows[ready_offset:ready_offset + count]
            source_index = torch.tensor(source_rows, device=device, dtype=torch.long)
            target_index = torch.tensor(targets, device=device, dtype=torch.long)
            align_fixed_memory_refill_cache_(ready_cache, cache, replace_all_rows=count == slots)
            ready_source_index = torch.arange(ready_offset, ready_offset + count, device=device, dtype=torch.long)
            cache.batch_scatter_from(ready_cache, target_index, ready_source_index)
            source_attempt_ids = ready_attempt_ids.index_select(0, ready_source_index)
            source_first_token = ready_first_token.index_select(0, ready_source_index)
            source_caps = ready_caps.index_select(0, ready_source_index)
            source_finished = source_caps <= 1
            for token_id in eos_token_ids:
                source_finished |= source_first_token[:, 0] == int(token_id)
            slot_rows.index_copy_(0, target_index, source_index)
            slot_attempt_ids.index_copy_(0, target_index, source_attempt_ids)
            slot_caps.index_copy_(0, target_index, source_caps)
            row_steps.index_fill_(0, target_index, 1)
            if logical_positions:
                row_positions.index_copy_(0, target_index, ready_positions.index_select(0, ready_source_index))
            elif prefill_context is None:
                row_positions.index_fill_(0, target_index, ready_width)
            else:
                row_positions.index_copy_(1, target_index, prefill_context.decode_positions(source_index))
            finished.index_copy_(0, target_index, source_finished)
            decode_ids.index_copy_(0, target_index, torch.where(source_finished[:, None], source_first_token.new_full((), pad_token_id), source_first_token))
            dense_seen.index_fill_(0, target_index, False)
            dense_seen[target_index, source_first_token[:, 0]] = True
            storage[source_index, 0] = source_first_token[:, 0].to(dtype=storage.dtype)
            for token_id in eos_token_ids:
                finished_by_eos[source_index] |= source_first_token[:, 0] == int(token_id)
            generated_lengths[source_index] = 1
            refill_count += count
            refill_waves += 1
            ready_offset += count
            if stream is not None:
                for row, done in zip(source_rows, source_finished.cpu().tolist()):
                    if done:
                        reason = 'eos' if finished_by_eos[row].item() else 'prompt_length_cap'
                        stream.emit(row, storage[row, :1].cpu(), reason)
                        free_rows.append(row)
            available.extend((target for target, done in zip(targets, source_finished.detach().cpu().tolist()) if done))
            if ready_offset == len(ready_rows):
                del ready_cache
                ready_cache = None
                ready_rows = []
                ready_first_token = None
                ready_attempt_ids = None
                ready_caps = None
                ready_offset = 0
        if available:
            empty_index = torch.tensor(available, device=device, dtype=torch.long)
            slot_rows.index_fill_(0, empty_index, -1)
            slot_attempt_ids.index_fill_(0, empty_index, 0)
            slot_caps.index_fill_(0, empty_index, 0)
            finished.index_fill_(0, empty_index, True)
            decode_ids.index_fill_(0, empty_index, int(pad_token_id))
            dense_seen.index_fill_(0, empty_index, False)
        live = int((slot_rows >= 0).sum().item())
        if stream is not None:
            stream.progress(live, slots, scheduled_decode_events, int(decode_useful_events.item()), refill_count, elapsed=time.perf_counter() - started, prefill=initial_prefill_seconds + refill_prefill_seconds)
        return live
    try:
        live_slots = poll_and_refill()
        while not live_slots and stream is not None and (not stream.exhausted):
            live_slots = poll_and_refill()
        if live_slots:
            fixed_graph = FixedMemoryDecodeGraph(model, cache, decode_ids, position=row_positions, scores_dtype=scores_dtype, consume_capture_steps=True)
        decode_cohort_allocated_bytes = int(torch.cuda.memory_allocated(device))
        decode_cohort_reserved_bytes = int(torch.cuda.memory_reserved(device))
        decode_cohort_device_total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
        while live_slots:
            graph_scores = fixed_graph.replay(decode_ids, position_ids=row_positions)
            scheduled_decode_events += slots
            if sampling_vocab_size is not None:
                graph_scores = graph_scores[:, :int(sampling_vocab_size)]
            _apply_dense_presence_penalty_(graph_scores, dense_seen, presence_penalty)
            next_token = counter_sampler(graph_scores, slot_attempt_ids, row_steps, seed=counter_seed, temperature=temperature, top_k=top_k, top_p=top_p)
            if _TRITON_AVAILABLE and prefill_context is None and (len(eos_token_ids) <= 2) and (slots == 1 or getattr(model.config, 'fixed_memory_fused_decode_state', False)):
                _update_fixed_decode_state_(next_token, storage, dense_seen, generated_lengths, finished_by_eos, finished, slot_rows, sink_rows, slot_caps, row_steps, row_positions, decode_ids, decode_useful_events, maximum_new_tokens=max_new_tokens, pad_token_id=pad_token_id, eos_token_ids=eos_token_ids)
            else:
                active = ~finished & (slot_rows >= 0)
                next_token = torch.where(active[:, None], next_token, next_token.new_full((), pad_token_id))
                write_rows = torch.where(active, slot_rows, sink_rows)
                write_columns = torch.where(active, row_steps, row_steps.new_full((), max_new_tokens))
                storage[write_rows, write_columns] = next_token[:, 0].to(dtype=storage.dtype)
                dense_seen.scatter_(1, next_token, True)
                next_steps = row_steps + active.to(dtype=row_steps.dtype)
                generated_lengths[write_rows] = torch.where(active, next_steps, generated_lengths[write_rows])
                is_eos = torch.zeros_like(finished)
                for token_id in eos_token_ids:
                    is_eos |= next_token[:, 0] == int(token_id)
                finished_by_eos[write_rows] |= active & is_eos
                finished |= active & (is_eos | (next_steps >= slot_caps))
                row_steps.copy_(next_steps)
                position_increment = active if prefill_context is None else active[None, :, None]
                row_positions.add_(position_increment.to(dtype=row_positions.dtype))
                decode_ids.copy_(torch.where(finished[:, None], next_token.new_full((), pad_token_id), next_token))
                decode_useful_events.add_(active.sum())
            replay_count += 1
            if replay_count % stop_check_interval_tokens == 0:
                live_slots = poll_and_refill()
                while not live_slots and stream is not None and (not stream.exhausted):
                    live_slots = poll_and_refill()
                if slots > 1 and _should_compact_dynamic_decode_tail(live_slots, slots, queue_exhausted=adaptive_tail_compaction and ready_cache is None and (pending_cursor >= len(pending_rows)) and (stream is None or stream.exhausted), live_fraction=tail_compaction_live_fraction):
                    keep = torch.nonzero(~finished & (slot_rows >= 0), as_tuple=False).flatten()
                    if int(keep.numel()) != live_slots:
                        raise RuntimeError('dynamic tail compaction live-slot accounting diverged')
                    torch.cuda.synchronize(device)
                    set_fixed_memory_cuda_graph_decode(model, cache, False)
                    del (graph_scores, next_token)
                    fixed_graph = None
                    cache.batch_select_indices(keep)
                    slot_rows = slot_rows.index_select(0, keep)
                    slot_attempt_ids = slot_attempt_ids.index_select(0, keep)
                    slot_caps = slot_caps.index_select(0, keep)
                    row_steps = row_steps.index_select(0, keep)
                    row_positions = row_positions.index_select(0 if prefill_context is None else 1, keep)
                    finished = finished.index_select(0, keep)
                    decode_ids = decode_ids.index_select(0, keep)
                    dense_seen = dense_seen.index_select(0, keep)
                    slots = live_slots
                    sink_rows = total_attempts + torch.arange(slots, device=device)
                    tail_compaction_count += 1
                    tail_compaction_shapes.append(slots)
                    fixed_graph = FixedMemoryDecodeGraph(model, cache, decode_ids, position=row_positions, scores_dtype=scores_dtype, consume_capture_steps=True)
    except Exception:
        set_fixed_memory_cuda_graph_decode(model, cache, False)
        raise
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    set_fixed_memory_cuda_graph_decode(model, cache, False)
    logical_lengths = generated_lengths[:total_attempts]
    if stream is None and bool((logical_lengths <= 0).any().item()):
        raise RuntimeError('dynamic refill lost one or more logical attempts')
    if bool((logical_lengths > attempt_caps).any().item()):
        raise RuntimeError('dynamic refill exceeded one or more attempt caps')
    maximum_length = int(logical_lengths.max().item())
    response_ids = storage[:total_attempts, :maximum_length]
    total_useful_tokens = int(logical_lengths.sum().item())
    decode_useful_tokens = int(decode_useful_events.item())
    metrics: dict[str, object] = {'scheduler_revision': 'fixed_batch_length_mix_slot_refill_group_prefill_tailcompact_v8', 'sampling_revision': COUNTER_SAMPLING_REVISION, 'counter_sampling_backend': counter_sampling_backend, 'decode_logits_dtype': str(scores_dtype).removeprefix('torch.'), 'cuda_graph_capture_mode': 'consume_warm_and_capture_steps', 'sampling_vocab_size': int(sampling_vocab_size or model.config.vocab_size), 'logical_attempts': total_attempts, 'response_id_storage_dtype': str(storage.dtype).removeprefix('torch.'), 'physical_slots': initial_slots, 'final_physical_slots': slots, 'adaptive_tail_compaction': bool(adaptive_tail_compaction), 'tail_compaction_live_fraction': float(tail_compaction_live_fraction), 'tail_compaction_count': tail_compaction_count, 'tail_compaction_shapes': tail_compaction_shapes, 'mechanical_repetition_early_stop': bool(mechanical_repetition_early_stop), 'mechanical_repetition_early_stop_revision': mechanical_repetition_revision, 'mechanical_repetition_checks': mechanical_repetition_checks, 'mechanical_repetition_stops': int(finished_by_mechanical_repetition[:total_attempts].sum().item()), 'mechanical_repetition_pattern_counts': dict(sorted(mechanical_repetition_pattern_counts.items())), 'initial_prompt_width': int(input_ids.shape[1]), 'internal_base_prompt_width': base_width, 'maximum_generated_tokens': maximum_length, 'total_useful_generated_tokens': total_useful_tokens, 'decode_useful_token_events': decode_useful_tokens, 'decode_scheduled_token_events': scheduled_decode_events, 'decode_useful_to_scheduled_ratio': decode_useful_tokens / scheduled_decode_events if scheduled_decode_events else 1.0, 'decode_replays': replay_count, 'refilled_attempts': refill_count, 'refill_waves': refill_waves, 'refill_batch_size': refill_rows, 'refill_prefetch_size': prefetch_rows, 'prefill_token_chunk_size': int(prefill_token_chunk_size), 'refill_cache_writeback': 'per_microbatch' if microbatch_refill else 'ready_bank', 'refill_prefill_waves': refill_prefill_waves, 'maximum_pending_attempts': maximum_pending, 'initial_virtual_swa_padding_rows_total': initial_virtual_swa_padding, 'refill_virtual_swa_padding_rows_total': refill_virtual_swa_padding, 'prefill_once_per_group': prefill_group_ids is not None, 'prefill_microbatch_unique_prompts': min(initial_prefill_unique_rows, 1 if logical_positions and prefill_token_chunk_size and (input_ids.shape[1] > prefill_token_chunk_size) else int(prefill_chunk_size or initial_prefill_unique_rows)), 'prefill_expanded_rows': prefill_expanded_rows, 'prefill_unique_rows': prefill_unique_rows, 'prefill_reuse_ratio': prefill_expanded_rows / prefill_unique_rows if prefill_unique_rows else 1.0, 'initial_prefill_seconds': initial_prefill_seconds, 'refill_prefill_seconds': refill_prefill_seconds, 'decode_cohort_initial_allocated_bytes': decode_cohort_allocated_bytes, 'decode_cohort_initial_reserved_bytes': decode_cohort_reserved_bytes, 'decode_cohort_initial_allocated_headroom_bytes': max(0, decode_cohort_device_total_bytes - decode_cohort_allocated_bytes), 'decode_cohort_initial_reserved_headroom_bytes': max(0, decode_cohort_device_total_bytes - decode_cohort_reserved_bytes), 'generation_wall_seconds': wall_seconds, 'useful_generated_tokens_per_second': total_useful_tokens / wall_seconds if wall_seconds else 0.0, 'decode_scheduled_tokens_per_second': scheduled_decode_events / wall_seconds if wall_seconds else 0.0}
    if stream is not None:
        metrics.update(scheduler_revision='global_persistent_slot_refill_v1', logical_attempts=stream.completed_attempts, total_useful_generated_tokens=stream.completed_tokens, maximum_generated_tokens=stream.maximum_generated_tokens, mechanical_repetition_stops=stream.finish_counts.get('mechanical_repetition', 0), useful_generated_tokens_per_second=stream.completed_tokens / wall_seconds)
        stream.finish(metrics)
    return DynamicSlotRefillOutput(generated_lengths=logical_lengths, metrics=metrics)

@triton.jit
def _qwen35_residual_add_square_kernel(residual_ptr, update_ptr, summed_ptr, square_ptr, total_elements: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    summed = (residual + update).to(tl.bfloat16)
    tl.store(summed_ptr + offsets, summed, mask=mask)
    summed_fp32 = summed.to(tl.float32)
    tl.store(square_ptr + offsets, summed_fp32 * summed_fp32, mask=mask)

def triton_residual_add_square_to_fp32(residual: torch.Tensor, update: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE or not residual.is_cuda or residual.dtype != torch.bfloat16 or (update.dtype != residual.dtype) or (update.shape != residual.shape) or (not residual.is_contiguous()) or (not update.is_contiguous()):
        raise ValueError('fused residual RMSNorm requires contiguous CUDA BF16 tensors')
    summed = torch.empty_like(residual)
    square = torch.empty_like(residual, dtype=torch.float32)
    total_elements = int(residual.numel())
    block_size = 256
    _qwen35_residual_add_square_kernel[triton.cdiv(total_elements, block_size),](residual, update, summed, square, total_elements=total_elements, block_size=block_size, num_warps=4, num_stages=1)
    return (summed, square)

def fused_residual_qwen35_rmsnorm(residual: torch.Tensor, update: torch.Tensor, norm: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.dtype != torch.bfloat16 or not residual.is_cuda or residual.ndim != 3 or (residual.shape[-2] != 1) or (not residual.is_contiguous()) or (not update.is_contiguous()) or (not hasattr(norm, '_qwen35_native_rms_gamma')):
        summed = residual + update
        return (summed, norm(summed))
    if norm._qwen35_native_rms_torch_tree:
        return triton_qwen35_rmsnorm_torch_tree(residual, norm._qwen35_native_rms_gamma, norm.eps, update)
    summed, square = triton_residual_add_square_to_fp32(residual, update)
    mean_square = square.mean(dim=-1, keepdim=True)
    normalized = triton_qwen35_rmsnorm_exact_epilogue(summed, norm._qwen35_native_rms_gamma, mean_square, float(norm.eps), summed.dtype)
    return (summed, normalized)

def materialize_fixed_memory_swa_physical_ring(layer_cache) -> None:
    index = int(layer_cache._qwen35_swa_physical_ring_index.item())
    ring_key = layer_cache._qwen35_swa_physical_ring_keys
    ring_value = layer_cache._qwen35_swa_physical_ring_values
    ring_valid = layer_cache._qwen35_swa_physical_ring_valid
    keep = int(layer_cache.keys.shape[-2])
    window = keep + 1
    start = (index + 1) % window
    first = min(keep, window - start)
    if getattr(layer_cache, '_qwen35_swa_shared_ring_storage', False):
        for field, dimension in (('keys', 2), ('values', 2), ('valid', 1)):
            name = '_qwen35_swa_physical_ring_' + field
            ring = getattr(layer_cache, name)
            pieces = [ring.narrow(dimension, start, first)]
            if first < keep:
                pieces.append(ring.narrow(dimension, 0, keep - first))
            setattr(layer_cache, field, torch.cat(pieces, dim=dimension))
            delattr(layer_cache, name)
        delattr(layer_cache, '_qwen35_swa_physical_ring_index')
        return
    layer_cache.keys[:, :, :first].copy_(ring_key[:, :, start:start + first])
    layer_cache.values[:, :, :first].copy_(ring_value[:, :, start:start + first])
    layer_cache.valid[:, :first].copy_(ring_valid[:, start:start + first])
    if first < keep:
        remainder = keep - first
        layer_cache.keys[:, :, first:].copy_(ring_key[:, :, :remainder])
        layer_cache.values[:, :, first:].copy_(ring_value[:, :, :remainder])
        layer_cache.valid[:, first:].copy_(ring_valid[:, :remainder])

@triton.jit
def _qwen35_fixed_swa_chrono_ring_write_kernel(ring_key_ptr, ring_value_ptr, ring_valid_ptr, ring_index_ptr, new_key_ptr, new_value_ptr, new_valid_ptr, new_key_sb: tl.constexpr, new_key_sh: tl.constexpr, new_key_sd: tl.constexpr, new_value_sb: tl.constexpr, new_value_sh: tl.constexpr, new_value_sd: tl.constexpr, ring_key_sb: tl.constexpr, ring_key_sh: tl.constexpr, ring_key_ss: tl.constexpr, ring_value_sb: tl.constexpr, ring_value_sh: tl.constexpr, ring_value_ss: tl.constexpr, ring_valid_sb: tl.constexpr, heads: tl.constexpr, head_dim: tl.constexpr, batch: tl.constexpr, total_kv_elements: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    ring_index = tl.load(ring_index_ptr)
    key_mask = offsets < total_kv_elements
    key_row = offsets // head_dim
    key_dimension = offsets - key_row * head_dim
    key_batch = key_row // heads
    key_head = key_row - key_batch * heads
    key_new_offset = key_batch * new_key_sb + key_head * new_key_sh + key_dimension * new_key_sd
    key_ring_offset = key_batch * ring_key_sb + key_head * ring_key_sh + ring_index * ring_key_ss + key_dimension
    key = tl.load(new_key_ptr + key_new_offset, mask=key_mask, other=0.0)
    tl.store(ring_key_ptr + key_ring_offset, key, mask=key_mask)
    value_offsets = offsets - total_kv_elements
    value_mask = (offsets >= total_kv_elements) & (value_offsets < total_kv_elements)
    value_row = value_offsets // head_dim
    value_dimension = value_offsets - value_row * head_dim
    value_batch = value_row // heads
    value_head = value_row - value_batch * heads
    value_new_offset = value_batch * new_value_sb + value_head * new_value_sh + value_dimension * new_value_sd
    value_ring_offset = value_batch * ring_value_sb + value_head * ring_value_sh + ring_index * ring_value_ss + value_dimension
    value = tl.load(new_value_ptr + value_new_offset, mask=value_mask, other=0.0)
    tl.store(ring_value_ptr + value_ring_offset, value, mask=value_mask)
    valid_offsets = offsets - 2 * total_kv_elements
    valid_mask = (offsets >= 2 * total_kv_elements) & (valid_offsets < batch)
    valid = tl.load(new_valid_ptr + valid_offsets, mask=valid_mask, other=0)
    tl.store(ring_valid_ptr + valid_offsets * ring_valid_sb + ring_index, valid, mask=valid_mask)

@triton.jit
def _qwen35_fixed_swa_chrono_ring_advance_kernel(index_ptr, keep: tl.constexpr):
    index = tl.load(index_ptr)
    tl.store(index_ptr, tl.where(index + 1 == keep, 0, index + 1))

def triton_fixed_memory_swa_physical_ring_update(layer_cache, key: torch.Tensor, value: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE or any((not tensor.is_cuda for tensor in (key, value, valid))):
        raise RuntimeError('physical SWA ring requires CUDA Triton')
    if key.shape[-2] != 1 or value.shape[-2] != 1 or valid.shape[-1] != 1:
        raise ValueError('physical SWA ring accepts one token')
    if key.shape != value.shape or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError('physical SWA ring requires matching contiguous KV heads')
    if not hasattr(layer_cache, '_qwen35_swa_physical_ring_index'):
        initialize_fixed_memory_swa_physical_ring(layer_cache)
    ring_key = layer_cache._qwen35_swa_physical_ring_keys
    ring_value = layer_cache._qwen35_swa_physical_ring_values
    ring_valid = layer_cache._qwen35_swa_physical_ring_valid
    heads = int(key.shape[1])
    head_dim = int(key.shape[-1])
    total_kv_elements = int(key.shape[0] * heads * head_dim)
    block_size = 256
    _qwen35_fixed_swa_chrono_ring_write_kernel[triton.cdiv(2 * total_kv_elements + int(key.shape[0]), block_size),](ring_key, ring_value, ring_valid, layer_cache._qwen35_swa_physical_ring_index, key, value, valid, new_key_sb=int(key.stride(0)), new_key_sh=int(key.stride(1)), new_key_sd=int(key.stride(3)), new_value_sb=int(value.stride(0)), new_value_sh=int(value.stride(1)), new_value_sd=int(value.stride(3)), ring_key_sb=int(ring_key.stride(0)), ring_key_sh=int(ring_key.stride(1)), ring_key_ss=int(ring_key.stride(2)), ring_value_sb=int(ring_value.stride(0)), ring_value_sh=int(ring_value.stride(1)), ring_value_ss=int(ring_value.stride(2)), ring_valid_sb=int(ring_valid.stride(0)), heads=heads, head_dim=head_dim, batch=int(key.shape[0]), total_kv_elements=total_kv_elements, block_size=block_size, num_warps=4, num_stages=1)
    _qwen35_fixed_swa_chrono_ring_advance_kernel[1,](layer_cache._qwen35_swa_physical_ring_index, keep=int(ring_key.shape[-2]), num_warps=1, num_stages=1)
    return (ring_key, ring_value, ring_valid)

def fixed_memory_physical_ring_decode_attention(query: torch.Tensor, ring_key: torch.Tensor, ring_value: torch.Tensor, valid: torch.Tensor, *, reproducible: bool=False) -> torch.Tensor:
    if query.ndim != 4 or query.shape[1] != 1:
        raise ValueError('physical-ring attention accepts one decode query')
    if ring_key.ndim != 4 or ring_value.ndim != 4:
        raise ValueError('physical-ring KV tensors must have shape [B,H,W,D]')
    if ring_key.shape[:3] != ring_value.shape[:3]:
        raise ValueError('physical-ring key/value geometry differs')
    if valid.shape != (ring_key.shape[0], ring_key.shape[2]):
        raise ValueError('physical-ring validity geometry differs')
    query_heads = int(query.shape[2])
    key_heads = int(ring_key.shape[1])
    if query_heads % key_heads:
        raise ValueError('query heads must be a multiple of physical-ring KV heads')
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH) if reproducible else nullcontext():
        output = F.scaled_dot_product_attention(query.transpose(1, 2), ring_key, ring_value, attn_mask=valid[:, None, None, :], dropout_p=0.0, is_causal=False, enable_gqa=query_heads != key_heads)
    return output.transpose(1, 2)

@triton.jit
def _qwen35_sigmoid_mul_kernel(values_ptr, gate_ptr, output_ptr, total_elements: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    activated_gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    tl.store(output_ptr + offsets, values * activated_gate, mask=mask)

def triton_qwen35_sigmoid_mul(values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if not _TRITON_AVAILABLE or not values.is_cuda or values.dtype != torch.bfloat16 or (gate.dtype != values.dtype) or (gate.shape != values.shape) or (not values.is_contiguous()) or (not gate.is_contiguous()):
        raise ValueError('fused sigmoid-mul requires contiguous CUDA BF16 tensors')
    output = torch.empty_like(values)
    total_elements = int(values.numel())
    block_size = 256
    _qwen35_sigmoid_mul_kernel[triton.cdiv(total_elements, block_size),](values, gate, output, total_elements=total_elements, block_size=block_size, num_warps=4, num_stages=1)
    return output
