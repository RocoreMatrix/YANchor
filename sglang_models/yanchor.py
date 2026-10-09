# Portions adapted from SGLang Qwen3.5; licensed under Apache-2.0.
# Copyright 2025 Qwen Team and SGLang Team.
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import torch
from torch import nn
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5AttentionDecoderLayer
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.model_executor.forward_context import get_req_to_token_pool
from modeling_yanchor import IndependentGQAMemory, IndependentMemoryState, Qwen3_5TextRotaryEmbedding


class MemorySlots:
    def __init__(self, memory, size, device, dtype, layer_id):
        self.state = IndependentMemoryState.empty(size + 1, memory.groups, memory.capacity,
            memory.max_chunk, memory.dim, device, dtype)
        self.layer_id = layer_id

    def reset_slots(self, indices):
        self.state.reset_(indices.long())

    def copy_slots(self, src_index, dst_index):
        self.state.scatter_(self.state, dst_index.long(), src_index.long())

    def get_cpu_slots(self, indices):
        return {f.name: getattr(self.state, f.name)[indices.long()].cpu() for f in fields(self.state)}

    def load_cpu_slots(self, data, indices):
        for name, value in data.items():
            target = getattr(self.state, name)
            target[indices.long()] = value.to(target.device)

    def iter_transfer_state_entries(self):
        for f in fields(self.state):
            yield f'yanchor_{f.name}', getattr(self.state, f.name), None, self.layer_id


class YANchorMemory(IndependentGQAMemory):
    def __init__(self, config, layer_id):
        super().__init__(config)
        self.layer_id = layer_id
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.backend = 'cuda' if torch.version.hip is None else 'torch'
        self.slots = None
        if self.backend == 'cuda':
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))

    def forward(self, hidden, positions, forward_batch):
        requests = get_req_to_token_pool()
        pool = requests.mamba_pool
        if self.slots is None:
            self.slots = MemorySlots(self, pool.size, hidden.device, hidden.dtype, self.layer_id)
            pool.register_slot_state(self.slots)
        indices = requests.translate_mamba_indices(requests.get_mamba_indices(forward_batch.req_pool_indices)).long()
        angles = positions.unsqueeze(0).expand(3, -1) if positions.ndim == 1 else positions
        embeddings = self.rotary_emb(hidden.unsqueeze(0), angles[:, None])
        if forward_batch.forward_mode.is_decode():
            state = self.slots.state if self.backend == 'cuda' else self.slots.state.select(indices)
            holder = SimpleNamespace(memory_state=state)
            output = super().forward(hidden[:, None], tuple(x.transpose(0, 1) for x in embeddings),
                cache_layer=holder, state_indices=indices if self.backend == 'cuda' else None).squeeze(1)
            if self.backend != 'cuda':
                self.slots.state.scatter_(holder.memory_state, indices, torch.arange(len(indices), device=hidden.device))
            return output
        output = torch.empty_like(hidden)
        start = 0
        for row, length in enumerate(forward_batch.extend_seq_lens_cpu):
            if not length:
                continue
            state = self.slots.state.select(indices[row:row+1])
            if forward_batch.extend_prefix_lens_cpu[row] == 0:
                state.reset_()
            holder = SimpleNamespace(memory_state=state)
            output[start:start+length] = super().forward(hidden[None, start:start+length],
                tuple(x[:, start:start+length] for x in embeddings), cache_layer=holder).squeeze(0)
            self.slots.state.scatter_(holder.memory_state, indices[row:row+1], torch.zeros(1, device=hidden.device, dtype=torch.long))
            start += length
        return output


class YANchorDecoderLayer(Qwen3_5AttentionDecoderLayer):
    def forward(self, positions, hidden_states, residual, forward_batch,
                captured_last_layer_outputs=None, **kwargs):
        hidden_states, residual = self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
            hidden_states, residual, forward_batch, captured_last_layer_outputs=captured_last_layer_outputs)
        if not forward_batch.forward_mode.is_idle() and hidden_states.shape[0]:
            hidden_states = self.self_attention(positions, hidden_states, forward_batch) + self.memory(residual, positions, forward_batch)
        hidden_states, residual = self.layer_communicator.prepare_mlp(hidden_states, residual, forward_batch)
        hidden_states = self.mlp(hidden_states)
        return self.layer_communicator.postprocess_layer(hidden_states, residual, forward_batch)


class YANchorForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=''):
        super().__init__()
        self.config = config
        self._native_decode = [False]
        config.model_type = 'qwen3_5_text'
        self.model = Qwen3_5ForCausalLM(config, quant_config, prefix='model')
        self.lm_head = self.model.embed_tokens if config.tie_word_embeddings else ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config, prefix='lm_head')
        self.logits_processor = LogitsProcessor(config)
        for index in config.fixed_memory_swa_layers:
            layer = self.model.layers[index]
            layer.__class__ = YANchorDecoderLayer
            layer.attn.sliding_window_size = config.fixed_memory_swa_window - 1
            layer.memory = YANchorMemory(config, index)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        self._native_decode[0] = forward_batch.forward_mode.is_decode()
        try:
            hidden = self.model(input_ids, positions, forward_batch, input_embeds=input_embeds)
            return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)
        finally:
            self._native_decode[0] = False

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        loaded = set()
        def ordinary_weights():
            for name, value in weights:
                if '.memory.' in name:
                    default_weight_loader(parameters[name], value)
                    loaded.add(name)
                else:
                    yield name, value
        loaded.update(Qwen3_5ForCausalLM.load_weights(self, ordinary_weights()))
        if torch.version.hip is None:
            from model.inference_kernels import enable_independent_memory_packed_projections, enable_native_rmsnorm_inference
            enable_independent_memory_packed_projections(self)
            enable_native_rmsnorm_inference(self, torch_tree=True)
            if os.environ.get('YANCHOR_NATIVE_FAST') == '1':
                from model.inference_kernels import enable_native_int8_mlp_b1
                enable_native_int8_mlp_b1(self, self._native_decode)
        return loaded


EntryClass = YANchorForCausalLM
