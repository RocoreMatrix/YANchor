from __future__ import annotations
import torch
import triton as tr
import triton.language as tl

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
def _read_parts(Q, K, V, Pos, Acc, Max, Den, Indices, KS: tl.constexpr, MS: tl.constexpr, G: tl.constexpr, R: tl.constexpr, D: tl.constexpr, KSIZE: tl.constexpr, SPLITS: tl.constexpr, SPAN: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr, BN: tl.constexpr):
    bg, split = (tl.program_id(0), tl.program_id(1))
    b = bg // G if Indices is None else tl.load(Indices + bg // G)
    kv_base = b * KS + bg % G * KSIZE * D
    meta_base = b * MS + bg % G * KSIZE
    r, d = (tl.arange(0, BR), tl.arange(0, BD))
    q = tl.load(Q + bg * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    maximum = tl.full((BR,), -float('inf'), tl.float32)
    denominator = tl.zeros((BR,), tl.float32)
    acc = tl.zeros((BR, BD), tl.float32)
    for start in range(split * SPAN, (split + 1) * SPAN, BN):
        n = start + tl.arange(0, BN)
        pos = tl.load(Pos + meta_base + n, (b >= 0) & (n < KSIZE), -1)
        k = tl.load(K + kv_base + n[None, :] * D + d[:, None], (b >= 0) & (n[None, :] < KSIZE) & (d[:, None] < D), 0)
        logits = tl.dot(q, k, input_precision='ieee') * D ** (-0.5)
        logits = tl.where(pos[None, :] >= 0, logits, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        safe_max = tl.where(next_max == -float('inf'), 0.0, next_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.exp(logits - safe_max[:, None])
        v = tl.load(V + kv_base + n[:, None] * D + d[None, :], (b >= 0) & (n[:, None] < KSIZE) & (d[None, :] < D), 0)
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
def _push_select(Key, Value, Score, Active, PK, PV, PS, Scores, Pos, Count, Total, Selected, NextScores, NextPos, Indices, PKS: tl.constexpr, MS: tl.constexpr, PMS: tl.constexpr, CS: tl.constexpr, G: tl.constexpr, D: tl.constexpr, KSIZE: tl.constexpr, C: tl.constexpr, PC: tl.constexpr, BD: tl.constexpr, N: tl.constexpr):
    bg = tl.program_id(0)
    row = bg // G
    b = row if Indices is None else tl.load(Indices + row)
    active = tl.load(Active + row) & (b >= 0)
    if active:
        count = tl.load(Count + b * CS)
        pending_base = b * PKS + bg % G * PC * D
        pending_meta = b * PMS + bg % G * PC
        committed_meta = b * MS + bg % G * KSIZE
        d = tl.arange(0, BD)
        tl.store(PK + pending_base + count * D + d, tl.load(Key + bg * D + d, d < D, 0), d < D)
        tl.store(PV + pending_base + count * D + d, tl.load(Value + bg * D + d, d < D, 0), d < D)
        score = tl.load(Score + bg)
        tl.store(PS + pending_meta + count, score)
        if count + 1 == C:
            n = tl.arange(0, N)
            p = tl.load(Pos + committed_meta + n, n < KSIZE, -1).to(tl.int32)
            s = tl.load(Scores + committed_meta + n, n < KSIZE, -float('inf'))
            pending = (n >= KSIZE) & (n < KSIZE + C)
            ps = tl.load(PS + pending_meta + n - KSIZE, pending, -float('inf'))
            ps = tl.where(n - KSIZE == count, score, ps)
            p = tl.where(pending, tl.load(Total + b * CS) - count + n - KSIZE, p).to(tl.int32)
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
def _gather_commit(K, V, PK, PV, Count, Active, Selected, SK, SV, Indices, KS: tl.constexpr, PKS: tl.constexpr, CS: tl.constexpr, G: tl.constexpr, D: tl.constexpr, KSIZE: tl.constexpr, C: tl.constexpr, PC: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    bg = tl.program_id(0)
    b = bg // G if Indices is None else tl.load(Indices + bg // G)
    count = tl.load(Count + b * CS, b >= 0, -2)
    if tl.load(Active + bg // G) & (b >= 0) & (count + 1 == C):
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        d = tl.arange(0, BD)
        ref = tl.load(Selected + bg * KSIZE + n, n < KSIZE, 0)
        old_mask = (n[:, None] < KSIZE) & (ref[:, None] < KSIZE) & (d[None, :] < D)
        new_mask = (n[:, None] < KSIZE) & (ref[:, None] >= KSIZE) & (d[None, :] < D)
        old = b * KS + bg % G * KSIZE * D + ref[:, None] * D + d[None, :]
        new = b * PKS + bg % G * PC * D + (ref[:, None] - KSIZE) * D + d[None, :]
        dest = bg * KSIZE * D + n[:, None] * D + d[None, :]
        mask = (n[:, None] < KSIZE) & (d[None, :] < D)
        tl.store(SK + dest, tl.load(K + old, old_mask, 0) + tl.load(PK + new, new_mask, 0), mask)
        tl.store(SV + dest, tl.load(V + old, old_mask, 0) + tl.load(PV + new, new_mask, 0), mask)

@tr.jit
def _write_commit(K, V, Scores, Pos, Count, Active, SK, SV, NS, NP, Indices, KS: tl.constexpr, MS: tl.constexpr, CS: tl.constexpr, G: tl.constexpr, D: tl.constexpr, KSIZE: tl.constexpr, C: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    bg = tl.program_id(0)
    b = bg // G if Indices is None else tl.load(Indices + bg // G)
    count = tl.load(Count + b * CS, b >= 0, -2)
    if tl.load(Active + bg // G) & (b >= 0) & (count + 1 == C):
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        d = tl.arange(0, BD)
        offset = bg * KSIZE * D + n[:, None] * D + d[None, :]
        target = b * KS + bg % G * KSIZE * D + n[:, None] * D + d[None, :]
        mask = (n[:, None] < KSIZE) & (d[None, :] < D)
        tl.store(K + target, tl.load(SK + offset, mask, 0), mask)
        tl.store(V + target, tl.load(SV + offset, mask, 0), mask)
        meta = bg * KSIZE + n
        target_meta = b * MS + bg % G * KSIZE + n
        tl.store(Scores + target_meta, tl.load(NS + meta, n < KSIZE, 0), n < KSIZE)
        tl.store(Pos + target_meta, tl.load(NP + meta, n < KSIZE, -1), n < KSIZE)

@tr.jit
def _advance(Count, Total, Active, Indices, CS: tl.constexpr, B: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.arange(0, BLOCK)
    row = b if Indices is None else tl.load(Indices + b, b < B, -1)
    mask = (b < B) & (row >= 0)
    active = tl.load(Active + b, b < B, 0).to(tl.int64)
    count = tl.load(Count + row * CS, mask, 0) + active
    total = tl.load(Total + row * CS, mask, 0) + active
    tl.store(Count + row * CS, tl.where(count == C, 0, count), mask)
    tl.store(Total + row * CS, total, mask)

def committed_gqa(query, state, indices=None):
    b, _, h, d = query.shape
    g, k = state.keys.shape[1:3]
    r, splits = (h // g, 4)
    span = tr.cdiv(k, splits * 128) * 128
    acc = torch.empty((b * g, splits, r, d), device=query.device, dtype=torch.float32)
    maximum = torch.empty((b * g, splits, r), device=query.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    out = torch.empty_like(query)
    _read_parts[b * g, splits](query, state.keys, state.values, state.positions, acc, maximum, denominator, indices, state.keys.stride(0), state.positions.stride(0), g, r, d, k, splits, span, max(32, tr.next_power_of_2(d)), max(16, tr.next_power_of_2(r)), 64, num_stages=1)
    _read_reduce[b * g, r](acc, maximum, denominator, out, r, d, splits, tr.next_power_of_2(d))
    return out

def commit_decode(key, value, scores, active, chunk, state, indices=None):
    b = key.shape[0]
    g, k, d = state.keys.shape[1:]
    pc = state.pending_keys.shape[2]
    selected = state.positions.new_empty((b, g, k))
    ns, np = (state.scores.new_empty((b, g, k)), torch.empty_like(selected))
    sk, sv = (state.keys.new_empty((b, g, k, d)), state.values.new_empty((b, g, k, d)))
    ks, pks, ms, pms, cs = (state.keys.stride(0), state.pending_keys.stride(0), state.scores.stride(0), state.pending_scores.stride(0), state.count.stride(0))
    _push_select[b * g,](key, value, scores, active, state.pending_keys, state.pending_values, state.pending_scores, state.scores, state.positions, state.count, state.total, selected, ns, np, indices, pks, ms, pms, cs, g, d, k, chunk, pc, tr.next_power_of_2(d), tr.next_power_of_2(k + chunk))
    _gather_commit[b * g, tr.cdiv(k, 32)](state.keys, state.values, state.pending_keys, state.pending_values, state.count, active, selected, sk, sv, indices, ks, pks, cs, g, d, k, chunk, pc, tr.next_power_of_2(d), 32)
    _write_commit[b * g, tr.cdiv(k, 32)](state.keys, state.values, state.scores, state.positions, state.count, active, sk, sv, ns, np, indices, ks, ms, cs, g, d, k, chunk, tr.next_power_of_2(d), 32)
    _advance[1,](state.count, state.total, active, indices, cs, b, chunk, tr.next_power_of_2(b))
from modeling_yanchor import IndependentGQAMemory, IndependentMemoryState, INDEPENDENT_MEMORY_REVISION
