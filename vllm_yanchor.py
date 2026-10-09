# Portions adapted from vLLM Qwen3.5; licensed under Apache-2.0.
# Copyright contributors to the vLLM project.
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import torch
from vllm import ModelRegistry
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5DecoderLayer
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from modeling_yanchor import IndependentGQAMemory, IndependentMemoryState, Qwen3_5TextRotaryEmbedding


def register():
    ModelRegistry.register_model('YANchorForCausalLM', 'vllm_yanchor:YANchorForCausalLM')


def memory_shapes(config):
    groups, dim = config.num_key_value_heads, config.head_dim
    capacity, chunk = config.independent_memory['capacity'], config.independent_memory['max_commit_chunk']
    return ((2 * groups * (capacity + chunk) * dim,), (groups * (2 * capacity + chunk) + 4,))


class YANchorMemory(IndependentGQAMemory, MambaBase):
    def __init__(self, config, vllm_config, prefix):
        IndependentGQAMemory.__init__(self, config)
        self.prefix = prefix
        self.state_shapes = memory_shapes(config)
        self.state_dtype = (vllm_config.model_config.dtype, torch.float32)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.backend = 'cuda' if torch.version.hip is None else 'torch'
        if self.backend == 'cuda':
            sys.path.insert(0, str(Path(__file__).parent / 'runtime'))
        vllm_config.compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self):
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_shape(self):
        return self.state_shapes

    def get_state_dtype(self):
        return self.state_dtype

    def state(self):
        values, meta = self.kv_cache
        batch, groups, capacity, chunk, dim = values.shape[0], self.groups, self.capacity, self.max_chunk, self.dim
        tensors = []
        offset = 0
        for width in (capacity, capacity, chunk, chunk):
            size = groups * width * dim
            tensors.append(values[:, offset:offset+size].view(batch, groups, width, dim))
            offset += size
        width = groups * capacity
        scores = meta[:, :width].view(batch, groups, capacity)
        positions = meta[:, width:2*width].view(torch.int32).view(batch, groups, capacity)
        pending_scores = meta[:, 2*width:-4].view(batch, groups, chunk)
        count = meta[:, -4:-2].view(torch.int64).squeeze(-1)
        total = meta[:, -2:].view(torch.int64).squeeze(-1)
        return IndependentMemoryState(tensors[0], tensors[1], scores, positions, tensors[2], tensors[3], pending_scores, count, total)

    def forward(self, hidden, positions):
        metadata = get_forward_context().attn_metadata
        angles = positions.unsqueeze(0).expand(3, -1) if positions.ndim == 1 else positions
        embeddings = self.rotary_emb(hidden.unsqueeze(0), angles[:, None])
        if metadata is None:
            return super().forward(hidden.unsqueeze(0), embeddings).squeeze(0)
        metadata = metadata[self.prefix]
        indices = metadata.non_spec_state_indices_tensor.long()
        pool = self.state()
        output = torch.empty_like(hidden)
        n = metadata.num_decodes
        if n:
            state = pool if self.backend == 'cuda' else pool.select(indices[:n])
            local_angles = tuple(x[:, :n].transpose(0, 1) for x in embeddings)
            output[:n] = super().forward(hidden[:n, None], local_angles,
                cache_layer=SimpleNamespace(memory_state=state),
                state_indices=indices[:n] if self.backend == 'cuda' else None).squeeze(1)
            if self.backend != 'cuda':
                pool.scatter_(state, indices[:n], torch.arange(n, device=hidden.device))
        starts = metadata.non_spec_query_start_loc.tolist() if metadata.num_prefills else ()
        for i in range(n, len(starts)-1):
            start, stop = starts[i:i+2]
            if stop == start:
                continue
            state = pool.select(indices[i:i+1])
            if positions[..., start].reshape(-1)[0].item() == 0:
                state.reset_()
            holder = SimpleNamespace(memory_state=state)
            output[start:stop] = super().forward(hidden[None, start:stop],
                tuple(x[:, start:stop] for x in embeddings), cache_layer=holder).squeeze(0)
            pool.scatter_(holder.memory_state, indices[i:i+1], torch.zeros(1, device=hidden.device, dtype=torch.long))
        return output


class YANchorDecoderLayer(Qwen3_5DecoderLayer):
    def forward(self, hidden_states, residual, positions=None, **kwargs):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states=hidden_states, positions=positions) + self.memory(residual, positions)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class YANchorForCausalLM(Qwen3_5ForCausalLM):
    supports_lora = False
    supports_eagle3 = False

    def __init__(self, *, vllm_config, prefix=''):
        if vllm_config.speculative_config or vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError('YANchor currently requires speculative decoding and prefix caching disabled.')
        config = vllm_config.model_config.hf_text_config
        self._native_decode = [False]
        config.model_type = 'qwen3_5_text'
        vllm_config.cache_config.sliding_window = config.fixed_memory_swa_window
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for index in config.fixed_memory_swa_layers:
            if self.model.start_layer <= index < self.model.end_layer:
                layer = self.model.layers[index]
                layer.__class__ = YANchorDecoderLayer
                layer.memory = YANchorMemory(config, vllm_config, f'{prefix + "." if prefix else ""}model.layers.{index}.memory')
                self._memory_prefix = layer.memory.prefix

    def forward(self, *args, **kwargs):
        metadata = get_forward_context().attn_metadata
        self._native_decode[0] = metadata is not None and metadata[self._memory_prefix].num_prefills == 0
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._native_decode[0] = False

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return memory_shapes(vllm_config.model_config.hf_text_config)

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        loaded = set()
        def ordinary_weights():
            for name, value in weights:
                if '.memory.' in name:
                    if name in parameters:
                        default_weight_loader(parameters[name], value)
                        loaded.add(name)
                else:
                    yield name, value
        loaded.update(super().load_weights(ordinary_weights()))
        if torch.version.hip is None:
            from model.inference_kernels import enable_independent_memory_packed_projections, enable_native_rmsnorm_inference, enable_framework_rmsnorm_inference
            from vllm.model_executor.layers.layernorm import GemmaRMSNorm
            enable_independent_memory_packed_projections(self)
            enable_native_rmsnorm_inference(self, torch_tree=True)
            enable_framework_rmsnorm_inference(self, GemmaRMSNorm)
            if os.environ.get('YANCHOR_NATIVE_FAST') == '1':
                from model.inference_kernels import enable_native_int8_mlp_b1, enable_native_packed_gdn_b1
                enable_native_int8_mlp_b1(self, self._native_decode)
                enable_native_packed_gdn_b1(self, self._native_decode)
        return loaded
