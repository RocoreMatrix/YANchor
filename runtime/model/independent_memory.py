from __future__ import annotations
import torch
import triton as tr
import triton.language as tl
from dataclasses import dataclass, fields
import math
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, rotate_half

@tr.jit(do_not_specialize=['NB', 'S'])
def _indexed_forward(Q, K, V, Ref, Out, NB, C: tl.constexpr, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, S, KB: tl.constexpr, CAP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    group = tl.program_id(1)
    g, block, b = (group % G, group // G % NB, group // (G * NB))
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    row = ((b * NB + block).to(tl.int64) * C + m % C) * H + g * (H // G) + m // C
    q = tl.load(Q + row[:, None] * D + d[None, :], (m[:, None] < C * (H // G)) & (d[None, :] < D), 0)
    maximum = tl.zeros((BM,), tl.float32)
    denominator = tl.full((BM,), 1.0, tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    key_batch = b if KB != 1 else 0
    for start in range(0, CAP, BN):
        n = start + tl.arange(0, BN)
        ref = tl.load(Ref + group * CAP + n, n < CAP, -1)
        offset = ((key_batch * S + ref) * G + g) * D
        k = tl.load(K + offset[None, :] + d[:, None], (ref[None, :] >= 0) & (d[:, None] < D), 0)
        score = tl.dot(q, k, input_precision='ieee') * D ** (-0.5)
        score = tl.where(ref[None, :] >= 0, score, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        scale = tl.exp(maximum - next_max)
        p = tl.exp(score - next_max[:, None])
        v = tl.load(V + offset[:, None] + d[None, :], (ref[:, None] >= 0) & (d[None, :] < D), 0)
        p_hi = p.to(v.dtype)
        acc = acc * scale[:, None] + tl.dot(p_hi, v, input_precision='ieee')
        if V.dtype.element_ty != tl.float32:
            p_lo = (p - p_hi.to(tl.float32)).to(v.dtype)
            acc += tl.dot(p_lo, v, input_precision='ieee')
        denominator = denominator * scale + tl.sum(p, 1)
        maximum = next_max
    tl.store(Out + row[:, None] * D + d[None, :], acc / denominator[:, None], (m[:, None] < C * (H // G)) & (d[None, :] < D))

def indexed_gqa(query, key, value, references):
    query, key, value, references = (x.contiguous() for x in (query, key, value, references))
    b, blocks, chunk, heads, dim = query.shape
    groups, capacity = references.shape[-2:]
    shape = (blocks, chunk, heads, groups, dim, key.shape[1], key.shape[0], capacity)
    output = torch.empty_like(query)
    _indexed_forward[tr.cdiv(chunk * (heads // groups), 64), b * blocks * groups](query, key, value, references, output, *shape, 64, 64, max(16, tr.next_power_of_2(dim)), num_warps=4, num_stages=1)
    return output

@tr.jit
def _read_parts(Q, K, V, Pos, Acc, Max, Den, G: tl.constexpr, R: tl.constexpr, D: tl.constexpr, KSIZE: tl.constexpr, SPLITS: tl.constexpr, SPAN: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr, BN: tl.constexpr):
    bg, split = (tl.program_id(0), tl.program_id(1))
    r, d = (tl.arange(0, BR), tl.arange(0, BD))
    q = tl.load(Q + bg * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    maximum = tl.full((BR,), -float('inf'), tl.float32)
    denominator = tl.zeros((BR,), tl.float32)
    acc = tl.zeros((BR, BD), tl.float32)
    for start in range(split * SPAN, (split + 1) * SPAN, BN):
        n = start + tl.arange(0, BN)
        pos = tl.load(Pos + bg * KSIZE + n, n < KSIZE, -1)
        k = tl.load(K + bg * KSIZE * D + n[None, :] * D + d[:, None], (n[None, :] < KSIZE) & (d[:, None] < D), 0)
        logits = tl.dot(q, k, input_precision='ieee') * D ** (-0.5)
        logits = tl.where(pos[None, :] >= 0, logits, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        safe_max = tl.where(next_max == -float('inf'), 0.0, next_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.exp(logits - safe_max[:, None])
        v = tl.load(V + bg * KSIZE * D + n[:, None] * D + d[None, :], (n[:, None] < KSIZE) & (d[None, :] < D), 0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision='ieee')
        denominator = denominator * alpha + tl.sum(p, 1)
        maximum = next_max
    base = (bg * SPLITS + split) * R + r
    tl.store(Acc + base[:, None] * D + d[None, :], acc, (r[:, None] < R) & (d[None, :] < D))
    tl.store(Max + base, maximum, r < R)
    tl.store(Den + base, denominator, r < R)

@tr.jit
def _read_reduce(Acc, Max, Den, Out, R: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr, BD: tl.constexpr):
    bg, r = (tl.program_id(0), tl.program_id(1))
    s, d = (tl.arange(0, SPLITS), tl.arange(0, BD))
    base = (bg * SPLITS + s) * R + r
    m = tl.load(Max + base)
    maximum = tl.maximum(0.0, tl.max(m, 0))
    scale = tl.exp(m - maximum)
    den = tl.exp(-maximum) + tl.sum(tl.load(Den + base) * scale, 0)
    acc = tl.load(Acc + base[:, None] * D + d[None, :], d[None, :] < D, 0)
    out = tl.sum(acc * scale[:, None], 0) / den
    tl.store(Out + (bg * R + r) * D + d, out, d < D)

@tr.jit
def _push_select(Key, Value, Score, Active, PK, PV, PS, Scores, Pos, Count, Total, Selected, NextScores, NextPos, G: tl.constexpr, D: tl.constexpr, KSIZE: tl.constexpr, C: tl.constexpr, PC: tl.constexpr, BD: tl.constexpr, N: tl.constexpr):
    bg = tl.program_id(0)
    b = bg // G
    active = tl.load(Active + b)
    if active:
        count = tl.load(Count + b)
        d = tl.arange(0, BD)
        tl.store(PK + bg * PC * D + count * D + d, tl.load(Key + bg * D + d, d < D, 0), d < D)
        tl.store(PV + bg * PC * D + count * D + d, tl.load(Value + bg * D + d, d < D, 0), d < D)
        score = tl.load(Score + bg)
        tl.store(PS + bg * PC + count, score)
        if count + 1 == C:
            n = tl.arange(0, N)
            p = tl.load(Pos + bg * KSIZE + n, n < KSIZE, -1).to(tl.int32)
            s = tl.load(Scores + bg * KSIZE + n, n < KSIZE, -float('inf'))
            pending = (n >= KSIZE) & (n < KSIZE + C)
            ps = tl.load(PS + bg * PC + n - KSIZE, pending, -float('inf'))
            ps = tl.where(n - KSIZE == count, score, ps)
            p = tl.where(pending, tl.load(Total + b) - count + n - KSIZE, p).to(tl.int32)
            s = tl.where(pending, ps, s)
            position_key = tl.where(p >= 0, p, 2147483647).to(tl.uint64)
            ordered = tl.sort(position_key << 32 | n.to(tl.uint64), descending=False)
            refs = (ordered & 4294967295).to(tl.int32)
            sorted_scores = tl.gather(s, refs, 0)
            canonical = tl.where(sorted_scores == 0.0, 0.0, sorted_scores)
            bits = canonical.to(tl.int32, bitcast=True)
            score_key = tl.where(bits < 0, ~bits, bits ^ -2147483648).to(tl.uint32).to(tl.uint64)
            ordered = tl.sort(score_key << 32 | (N - 1 - n).to(tl.uint64), descending=True)
            rank = (N - 1 - (ordered & 4294967295)).to(tl.int32)
            chosen = tl.gather(refs, rank, 0)
            chosen_pos = tl.gather(p, chosen, 0)
            tl.store(Selected + bg * KSIZE + n, tl.where(chosen_pos >= 0, chosen, 0), n < KSIZE)
            tl.store(NextScores + bg * KSIZE + n, tl.gather(s, chosen, 0), n < KSIZE)
            tl.store(NextPos + bg * KSIZE + n, chosen_pos, n < KSIZE)

@tr.jit
def _gather_commit(K, V, PK, PV, Count, Active, Selected, SK, SV, G: tl.constexpr, D: tl.constexpr, KSIZE: tl.constexpr, C: tl.constexpr, PC: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    bg = tl.program_id(0)
    if tl.load(Active + bg // G) & (tl.load(Count + bg // G) + 1 == C):
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        d = tl.arange(0, BD)
        ref = tl.load(Selected + bg * KSIZE + n, n < KSIZE, 0)
        old_mask = (n[:, None] < KSIZE) & (ref[:, None] < KSIZE) & (d[None, :] < D)
        new_mask = (n[:, None] < KSIZE) & (ref[:, None] >= KSIZE) & (d[None, :] < D)
        old = bg * KSIZE * D + ref[:, None] * D + d[None, :]
        new = bg * PC * D + (ref[:, None] - KSIZE) * D + d[None, :]
        dest = bg * KSIZE * D + n[:, None] * D + d[None, :]
        mask = (n[:, None] < KSIZE) & (d[None, :] < D)
        tl.store(SK + dest, tl.load(K + old, old_mask, 0) + tl.load(PK + new, new_mask, 0), mask)
        tl.store(SV + dest, tl.load(V + old, old_mask, 0) + tl.load(PV + new, new_mask, 0), mask)

@tr.jit
def _write_commit(K, V, Scores, Pos, Count, Active, SK, SV, NS, NP, G: tl.constexpr, D: tl.constexpr, KSIZE: tl.constexpr, C: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    bg = tl.program_id(0)
    if tl.load(Active + bg // G) & (tl.load(Count + bg // G) + 1 == C):
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        d = tl.arange(0, BD)
        offset = bg * KSIZE * D + n[:, None] * D + d[None, :]
        mask = (n[:, None] < KSIZE) & (d[None, :] < D)
        tl.store(K + offset, tl.load(SK + offset, mask, 0), mask)
        tl.store(V + offset, tl.load(SV + offset, mask, 0), mask)
        meta = bg * KSIZE + n
        tl.store(Scores + meta, tl.load(NS + meta, n < KSIZE, 0), n < KSIZE)
        tl.store(Pos + meta, tl.load(NP + meta, n < KSIZE, -1), n < KSIZE)

@tr.jit
def _advance(Count, Total, Active, B: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.arange(0, BLOCK)
    active = tl.load(Active + b, b < B, 0).to(tl.int64)
    count = tl.load(Count + b, b < B, 0) + active
    total = tl.load(Total + b, b < B, 0) + active
    tl.store(Count + b, tl.where(count == C, 0, count), b < B)
    tl.store(Total + b, total, b < B)

def committed_gqa(query, state):
    b, _, h, d = query.shape
    g, k = state.keys.shape[1:3]
    r, splits = (h // g, 4)
    span = tr.cdiv(k, splits * 128) * 128
    acc = torch.empty((b * g, splits, r, d), device=query.device, dtype=torch.float32)
    maximum = torch.empty((b * g, splits, r), device=query.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    out = torch.empty_like(query)
    _read_parts[b * g, splits](query, state.keys, state.values, state.positions, acc, maximum, denominator, g, r, d, k, splits, span, max(32, tr.next_power_of_2(d)), max(16, tr.next_power_of_2(r)), 64, num_stages=1)
    _read_reduce[b * g, r](acc, maximum, denominator, out, r, d, splits, tr.next_power_of_2(d))
    return out

def commit_decode(key, value, scores, active, chunk, state):
    b, g, k, d = state.keys.shape
    pc = state.pending_keys.shape[2]
    selected = torch.empty_like(state.positions)
    ns, np = (torch.empty_like(state.scores), torch.empty_like(state.positions))
    sk, sv = (torch.empty_like(state.keys), torch.empty_like(state.values))
    _push_select[b * g,](key, value, scores, active, state.pending_keys, state.pending_values, state.pending_scores, state.scores, state.positions, state.count, state.total, selected, ns, np, g, d, k, chunk, pc, tr.next_power_of_2(d), tr.next_power_of_2(k + chunk))
    _gather_commit[b * g, tr.cdiv(k, 32)](state.keys, state.values, state.pending_keys, state.pending_values, state.count, active, selected, sk, sv, g, d, k, chunk, pc, tr.next_power_of_2(d), 32)
    _write_commit[b * g, tr.cdiv(k, 32)](state.keys, state.values, state.scores, state.positions, state.count, active, sk, sv, ns, np, g, d, k, chunk, tr.next_power_of_2(d), 32)
    _advance[1,](state.count, state.total, active, b, chunk, tr.next_power_of_2(b))
INDEPENDENT_MEMORY_REVISION = 'qwen35-text-original-gdn-swa-independent-gqa-memory-v1'

def _memory_rope(values, cos, sin):
    width = cos.shape[-1]
    rotated = values[..., :width]
    return torch.cat((rotated * cos + rotate_half(rotated) * sin, values[..., width:]), -1)

def select_topk(scores, positions, references, capacity):
    positions = positions.long().masked_fill(references < 0, -1)
    order = positions.masked_fill(references < 0, torch.iinfo(torch.long).max).argsort(dim=-1, stable=True)
    scores, positions, references = [x.gather(-1, order) for x in (scores, positions, references)]
    order = scores.argsort(dim=-1, descending=True, stable=True)[..., :capacity]
    result = [x.gather(-1, order) for x in (scores, positions, references)]
    missing = capacity - order.shape[-1]
    if missing:
        result = [F.pad(x, (0, missing), value=value) for x, value in zip(result, (-math.inf, -1, -1))]
    return tuple(result)

def merge_topk(left, right, capacity):
    return select_topk(*(torch.cat((a, b), -1) for a, b in zip(left, right)), capacity)

def exclusive_prefix_topk(scores, positions, references, capacity):
    count = scores.shape[1]
    size = 1 << (count - 1).bit_length()
    leaf = select_topk(scores.detach(), positions, references, capacity)
    levels = [tuple((F.pad(x, (0, 0, 0, 0, 0, size - count), value=v) for x, v in zip(leaf, (-math.inf, -1, -1))))]
    while levels[-1][0].shape[1] > 1:
        previous = levels[-1]
        levels.append(merge_topk(tuple((x[:, 0::2] for x in previous)), tuple((x[:, 1::2] for x in previous)), capacity))
    prefix = tuple((torch.full_like(x, v) for x, v in zip(levels[-1], (-math.inf, -1, -1))))
    for children in reversed(levels[:-1]):
        right = merge_topk(prefix, tuple((x[:, 0::2] for x in children)), capacity)
        prefix = tuple((torch.stack((a, b), 2).flatten(1, 2) for a, b in zip(prefix, right)))
    final = merge_topk(tuple((x[:, count - 1:count] for x in prefix)), tuple((x[:, count - 1:count] for x in leaf)), capacity)
    return tuple((torch.cat((x[:, :count], y), 1) for x, y in zip(prefix, final)))

@dataclass
class IndependentMemoryState:
    keys: torch.Tensor
    values: torch.Tensor
    scores: torch.Tensor
    positions: torch.Tensor
    pending_keys: torch.Tensor
    pending_values: torch.Tensor
    pending_scores: torch.Tensor
    count: torch.Tensor
    total: torch.Tensor

    @classmethod
    def empty(cls, batch, groups, capacity, chunk, dim, device, dtype):
        shape = (batch, groups, capacity, dim)
        pending = (batch, groups, chunk, dim)
        return cls(torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype), torch.full(shape[:-1], -math.inf, device=device, dtype=torch.float32), torch.full(shape[:-1], -1, device=device, dtype=torch.int32), torch.zeros(pending, device=device, dtype=dtype), torch.zeros(pending, device=device, dtype=dtype), torch.full(pending[:-1], -math.inf, device=device, dtype=torch.float32), torch.zeros(batch, device=device, dtype=torch.long), torch.zeros(batch, device=device, dtype=torch.long))

    def select(self, indices):
        return type(self)(*(getattr(self, f.name).index_select(0, indices.to(self.keys.device)) for f in fields(self)))

    def repeat(self, repeats):
        return type(self)(*(getattr(self, f.name).repeat_interleave(repeats, 0) for f in fields(self)))

    def scatter_(self, source, target_indices, source_indices):
        for f in fields(self):
            target, value = (getattr(self, f.name), getattr(source, f.name))
            target.index_copy_(0, target_indices.to(target.device), value.index_select(0, source_indices.to(value.device)).to(target.device))

    def reset_(self, rows=None):
        for f in fields(self):
            value = getattr(self, f.name)
            fill = -math.inf if 'scores' in f.name else -1 if f.name == 'positions' else 0
            if rows is None:
                value.fill_(fill)
            else:
                value.index_fill_(0, rows.to(value.device), fill)

    def to_(self, device):
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name).to(device, non_blocking=True))

class IndependentGQAMemory(nn.Module):

    def __init__(self, config):
        super().__init__()
        settings = config.independent_memory
        self.hidden_size, self.heads = (config.hidden_size, config.num_attention_heads)
        self.groups, self.dim = (config.num_key_value_heads, config.head_dim)
        self.capacity, self.max_chunk = (settings['capacity'], settings['max_commit_chunk'])
        self.window = config.fixed_memory_swa_window
        hidden, dim, rank, intermediate = (self.hidden_size, self.dim, settings['admission_rank'], settings['writer_intermediate_size'])
        self.input_norm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.q_norm = Qwen3_5RMSNorm(dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(dim, eps=config.rms_norm_eps)
        for name, n_in, n_out in (('q_proj', hidden, self.heads * dim), ('k_proj', hidden, self.groups * dim), ('v_proj', hidden, self.groups * dim), ('o_proj', self.heads * dim, hidden), ('write_gate_proj', hidden, intermediate), ('write_up_proj', hidden, intermediate), ('write_down_proj', intermediate, hidden), ('admission_down', hidden, rank), ('admission_up', rank, self.groups)):
            setattr(self, name, nn.Linear(n_in, n_out, bias=False))
        self.read_scale = nn.Parameter(torch.zeros(self.heads))
        self.read_gate_weight = nn.Parameter(torch.zeros(self.heads, dim))
        self.read_gate_bias = nn.Parameter(torch.zeros(self.heads))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
        nn.init.zeros_(self.write_down_proj.weight)
        nn.init.zeros_(self.admission_up.weight)

    def _write_kv(self, u, position_embeddings):
        packed = hasattr(self, '_inference_writer_gate_up')

        def write(x):
            if packed:
                from .inference_kernels import triton_qwen35_silu_mul
                activation = F.linear(x, self._inference_writer_gate_up)
                if x.is_cuda and x.dtype == torch.bfloat16:
                    activation = triton_qwen35_silu_mul(activation, self.write_up_proj.out_features)
                else:
                    gate, up = activation.chunk(2, -1)
                    activation = F.silu(gate) * up
                return x + self.write_down_proj(activation)
            return x + self.write_down_proj(F.silu(self.write_gate_proj(x)) * self.write_up_proj(x))
        z = torch.cat([write(part) for part in u.split(4096)], 0)
        key, value = F.linear(z, self._inference_kv).chunk(2, -1) if packed else (self.k_proj(z), self.v_proj(z))
        key = self.k_norm(key.unflatten(-1, (self.groups, self.dim)))
        value = value.unflatten(-1, (self.groups, self.dim)).contiguous()
        cos, sin = (x[:, None] for x in position_embeddings)
        return (_memory_rope(key, cos, sin), value)

    def _query_scores(self, u, position_embeddings):
        query = self.q_norm(self.q_proj(u).unflatten(-1, (self.heads, self.dim)))
        gate = self.read_scale.tanh() * 2 * torch.sigmoid((query * self.read_gate_weight).sum(-1) / math.sqrt(self.dim) + self.read_gate_bias)
        cos, sin = (x[:, :, None] for x in position_embeddings)
        query = _memory_rope(query, cos, sin)
        features = F.silu(self.admission_down(u.detach()))
        with torch.autocast(device_type=u.device.type, enabled=False):
            scores = F.linear(features.float(), self.admission_up.weight.float())
        return (query, scores, gate)

    def _sequence(self, u, query, scores, position_embeddings, valid, chunk, state=None):
        batch, length, heads, dim = query.shape
        b = torch.arange(batch, device=query.device)[:, None]
        if valid is None:
            order = None
            q, s = (query, scores)
            counts = torch.full((batch,), length, device=query.device, dtype=torch.long)
        else:
            order = valid.long().argsort(dim=1, descending=True, stable=True)
            q, s = (query[b, order], scores[b, order])
            counts = valid.sum(1)
        old_count = torch.zeros_like(counts) if state is None else state.count
        old_total = torch.zeros_like(counts) if state is None else state.total
        prefix_length = 0 if state is None else chunk
        extent = math.ceil((length + prefix_length) / chunk) * chunk
        at = torch.arange(extent, device=query.device)[None].expand(batch, -1)
        valid_extent = at < old_count[:, None] + counts[:, None]
        if state is None:
            q = F.pad(q, (0, 0, 0, 0, 0, extent - length))
            s = F.pad(s, (0, 0, 0, extent - length), value=-math.inf)
            shift = 0
        else:
            select = torch.where(at < old_count[:, None], at, chunk + at - old_count[:, None])
            select = select.clamp_max(chunk + length - 1)
            q = torch.cat((q.new_zeros(batch, chunk, heads, dim), q), 1)[b, select]
            s = torch.cat((state.pending_scores.transpose(1, 2)[:, :chunk], s), 1)[b, select]
            shift = self.capacity
        s = s.masked_fill(~valid_extent[..., None], -math.inf)
        positions = (old_total - old_count)[:, None] + at
        refs = (at + shift).masked_fill(~valid_extent, -1)
        blocks = extent // chunk
        metadata = [x[:, :, None].expand(-1, -1, self.groups) if x.ndim == 2 else x for x in (s, positions, refs)]
        metadata = [x.reshape(batch, blocks, chunk, self.groups).transpose(2, 3) for x in metadata]
        prefix = exclusive_prefix_topk(*metadata, self.capacity)
        if state is not None:
            initial = (state.scores, state.positions, torch.arange(self.capacity, device=q.device)[None, None].expand(batch, self.groups, -1).masked_fill(state.positions < 0, -1))
            prefix = merge_topk(prefix, tuple((x[:, None].expand(-1, blocks + 1, -1, -1) for x in initial)), self.capacity)
        boundary = torch.arange(1, blocks + 1, device=q.device)[None, :, None, None] * chunk
        entrant = prefix[2][:, 1:] - shift
        used_boundary = boundary < (old_count + counts)[:, None, None, None]
        if state is not None:
            used_boundary = boundary <= (old_count + counts)[:, None, None, None]
        take = (entrant >= boundary - chunk) & (entrant < boundary) & used_boundary
        needed = torch.zeros((batch, extent + 1), device=q.device, dtype=torch.bool)
        needed.scatter_(1, entrant.masked_fill(~take, extent).flatten(1), True)
        if state is not None:
            full = (old_count + counts) // chunk
            needed[:, :extent] |= valid_extent & (at >= full[:, None] * chunk)
        needed = needed[:, :extent] & valid_extent & (at >= old_count[:, None])
        selected_batch, selected_position = needed.nonzero(as_tuple=True)
        original = selected_position - old_count[selected_batch]
        if order is not None:
            original = order[selected_batch, original]
        angles = tuple((x.expand(batch, -1, -1)[selected_batch, original] for x in position_embeddings))
        k, v = self._write_kv(u[selected_batch, original], angles)
        references = torch.full((batch, shift + extent), -1, device=q.device, dtype=torch.long)
        source_k, source_v = (k[None], v[None])
        offset = 0
        if state is not None:
            offset = batch * (self.capacity + chunk)
            old_k = torch.cat((state.keys, state.pending_keys[:, :, :chunk]), 2).transpose(1, 2)
            old_v = torch.cat((state.values, state.pending_values[:, :, :chunk]), 2).transpose(1, 2)
            source_k = torch.cat((old_k.reshape(1, -1, self.groups, dim), source_k), 1)
            source_v = torch.cat((old_v.reshape(1, -1, self.groups, dim), source_v), 1)
            base = b * (self.capacity + chunk)
            references[:, :shift] = base + torch.arange(shift, device=q.device)
            references[:, shift:] = torch.where(at < old_count[:, None], base + self.capacity + at, -1)
        references[selected_batch, shift + selected_position] = offset + torch.arange(len(k), device=q.device)
        source_k, source_v = (F.pad(x, (0, 0, 0, 0, 0, 1)) for x in (source_k, source_v))
        read_refs = references[b[:, :, None, None], prefix[2][:, :-1].clamp_min(0)]
        read_refs = read_refs.masked_fill(prefix[2][:, :-1] < 0, -1)
        read = indexed_gqa(q.reshape(batch, blocks, chunk, heads, dim), source_k, source_v, read_refs).flatten(1, 2)
        if state is None:
            output = read[:, :length]
        else:
            output = read[b, (old_count[:, None] + torch.arange(length, device=q.device)[None]).clamp_max(extent - 1)]
        if order is not None:
            output = output[b, order.argsort(1)] * valid[..., None, None]
        if state is not None:
            score, position, reference = [x[torch.arange(batch, device=q.device), full] for x in prefix]
            g = torch.arange(self.groups, device=q.device)[None, :, None]
            batch_index = torch.arange(batch, device=q.device)[:, None, None]
            compact = references[batch_index, reference.clamp_min(0)]
            state.keys.copy_(source_k[0, compact.clamp_min(0), g].masked_fill((reference < 0)[..., None], 0))
            state.values.copy_(source_v[0, compact.clamp_min(0), g].masked_fill((reference < 0)[..., None], 0))
            state.scores.copy_(score)
            state.positions.copy_(position)
            pending = full[:, None] * chunk + torch.arange(self.max_chunk, device=q.device)[None]
            remaining = (old_count + counts) % chunk
            mask = torch.arange(self.max_chunk, device=q.device)[None] < remaining[:, None]
            pending = pending.clamp_max(extent - 1)
            compact = references[b, pending + shift].clamp_min(0)
            state.pending_keys.copy_(source_k[0, compact].masked_fill(~mask[..., None, None], 0).transpose(1, 2))
            state.pending_values.copy_(source_v[0, compact].masked_fill(~mask[..., None, None], 0).transpose(1, 2))
            state.pending_scores.copy_(s[b, pending].masked_fill(~mask[..., None], -math.inf).transpose(1, 2))
            state.count.copy_(remaining)
            state.total.add_(counts)
        return output

    def _decode(self, query, key, value, scores, valid, chunk, state):
        read = committed_gqa(query, state)
        commit_decode(key, value, scores, valid[:, 0].contiguous(), chunk, state)
        return read * valid[..., None, None]

    def forward(self, hidden, position_embeddings, *, window=None, attention_mask=None, cache_layer=None):
        chunk = min(self.max_chunk, self.window if window is None else window)
        valid = None if attention_mask is None else attention_mask[:, -hidden.shape[1]:].bool()
        u = self.input_norm(hidden)
        query, scores, gate = self._query_scores(u, position_embeddings)
        if cache_layer is not None:
            if cache_layer.memory_state is None:
                cache_layer.memory_state = IndependentMemoryState.empty(hidden.shape[0], self.groups, self.capacity, self.max_chunk, self.dim, hidden.device, hidden.dtype)
            with torch.no_grad():
                if hidden.shape[1] == 1:
                    angles = tuple((x.expand(hidden.shape[0], -1, -1).reshape(-1, x.shape[-1]) for x in position_embeddings))
                    key, value = self._write_kv(u.reshape(-1, self.hidden_size), angles)
                    key, value = (x.reshape(hidden.shape[0], 1, self.groups, self.dim) for x in (key, value))
                    if valid is None:
                        valid = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
                    output = self._decode(query, key, value, scores, valid, chunk, cache_layer.memory_state)
                else:
                    output = self._sequence(u, query, scores, position_embeddings, valid, chunk, cache_layer.memory_state)
        else:
            output = self._sequence(u, query, scores, position_embeddings, valid, chunk)
        output = self.o_proj((output * gate[..., None]).flatten(-2))
        return output
