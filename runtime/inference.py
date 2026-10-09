import json
from pathlib import Path
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def prompt_text(prompt, messages=None):
    messages = messages or [{'role': 'user', 'content': prompt}]
    return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + '<|im_start|>assistant\n<think>\n'


class Engine:
    def __init__(self, model=ROOT, batch=1, profile='fast', backend='auto', device=None, dtype=None):
        import importlib.util
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.batch = batch
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
        if backend == 'auto':
            accelerated = self.device.type == 'cuda' and torch.version.hip is None and torch.cuda.get_device_capability(self.device)[0] == 9
            accelerated = accelerated and all(importlib.util.find_spec(name) is not None for name in ['fla', 'flash_attn', 'causal_conv1d', 'tilelang'])
            backend = 'cuda' if accelerated else 'torch'
        self.backend = backend
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        dtype = dtype or (torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 248044
        start = time.perf_counter()
        if backend == 'cuda':
            import os
            os.environ.setdefault('FLA_TILELANG', '1')
            os.environ.setdefault('FLA_CONFIG_DIR', str(ROOT/'runtime/fla_h100'))
            os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
            from model.fixed_memory import load_model
            from model.inference_kernels import install_inference
            torch.set_num_threads(1)
            self.model = load_model(model, backend='cuda', torch_dtype=dtype,
                local_files_only=True, attn_implementation='flash_attention_2').to(self.device).eval().requires_grad_(False)
            self.model.config.fixed_memory_swa_reproducible_decode = profile == 'reference'
            self.model.config.fixed_memory_swa_shared_ring_storage = batch > 1
            self.operators = install_inference(self.model, fp16_state=batch > 1 and profile == 'fast',
                int8_mlp_b1=batch == 1 and profile == 'fast')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True,
                local_files_only=True, dtype=dtype, backend='torch', attn_implementation='eager').to(self.device).eval().requires_grad_(False)
            self.operators = dict(backend='torch', dtype=str(dtype), recurrent_state='float32')
        self.operators['gdn_norm_classes'] = sorted({type(m.norm).__module__ + '.' + type(m.norm).__name__
            for m in self.model.modules() if hasattr(m, 'recurrent_gated_delta_rule')})
        self.load_seconds = time.perf_counter() - start

    def encode(self, rows):
        if not rows:
            return []
        texts = [r.get('prompt_text') or self.tokenizer.apply_chat_template(
            r.get('messages') or [{'role': 'user', 'content': r.get('prompt', '')}],
            tokenize=False, add_generation_prompt=True) for r in rows]
        encoded = self.tokenizer(texts, add_special_tokens=False, padding=False)['input_ids']
        return [dict(row, input_ids=ids) for row, ids in zip(rows, encoded)]

    def run_stream(self, stream, first, *, cap=131072, temperature=0.6, top_k=20,
                   top_p=0.95, presence_penalty=0.0, seed=20260923, fixed=False,
                   mechanical=True, stop_interval=128, sampling_backend='triton_top20',
                   prefill_context=None,eos_ids=(248046,248044)):
        if self.backend == 'torch':
            return self.run_torch_stream(stream, first, cap=cap, temperature=temperature, top_k=top_k,
                top_p=top_p, presence_penalty=presence_penalty, seed=seed, fixed=fixed,
                mechanical=mechanical, eos_ids=eos_ids, stop_interval=stop_interval)
        if temperature <= 0:
            temperature, top_k, top_p = 1.0, 1, 1.0
        if top_k != 20:
            sampling_backend = 'torch'
        from model.inference_kernels import cuda_graph_generate_top20_dynamic_refill
        torch = self.torch
        stream.initialize(first)
        ids, mask = stream.inputs(list(range(len(first))), 'cuda')
        with torch.inference_mode():
            output = cuda_graph_generate_top20_dynamic_refill(self.model, ids, mask,
                slot_batch_size=len(first), max_new_tokens=cap, temperature=temperature,
                top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                eos_token_ids=() if fixed else eos_ids, pad_token_id=self.tokenizer.pad_token_id,
                stop_check_interval_tokens=stop_interval, counter_seed=seed,
                attempt_ids=torch.tensor([r.get('sampling_id',r['attempt_id']) for r in first], device='cuda'),
                max_new_tokens_per_attempt=torch.tensor([min(cap, r.get('max_tokens', cap)) for r in first], device='cuda'),
                prefill_group_ids=torch.tensor([r.get('prefill_group_id',r['attempt_id']) for r in first], device='cuda'),
                refill_batch_size=min(16, self.batch), refill_prefetch_size=min(16, self.batch),
                prefill_chunk_size=stream.prefill_chunk_size(ids.shape[1]), prefill_token_chunk_size=16384,
                sampling_vocab_size=len(self.tokenizer), counter_sampling_backend=sampling_backend,
                adaptive_tail_compaction=not fixed, tail_compaction_live_fraction=0.75,
                mechanical_repetition_early_stop=mechanical and not fixed, stream=stream,prefill_context=prefill_context)
        metrics = dict(output.metrics)
        metrics['decode_seconds'] = metrics['generation_wall_seconds'] - metrics['initial_prefill_seconds'] - metrics['refill_prefill_seconds']
        metrics['decode_tokens_per_second'] = metrics['decode_useful_token_events'] / metrics['decode_seconds']
        metrics.update(model_load_seconds=self.load_seconds, operators=self.operators,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        return metrics

    def generate(self, prompts, on_tokens=None, **settings):
        settings.setdefault('stop_interval', 16 if self.batch == 1 else 128)
        rows = [dict(p, attempt_id=i, sample_index=p.get('sample_index', 0)) if isinstance(p, dict)
                else dict(prompt=p, attempt_id=i, sample_index=0) for i, p in enumerate(prompts)]
        for row in rows:
            row['max_tokens'] = min(settings.get('cap',131072),row.get('max_tokens',settings.get('cap',131072)))
        queue = deque(rows)
        result = []
        def fetch(count):
            return self.encode([queue.popleft() for _ in range(min(count, len(queue)))])
        stream = Stream(self, fetch, result.extend, on_tokens=on_tokens)
        first = fetch(self.batch)
        metrics = self.run_stream(stream, first, **settings)
        return sorted(result, key=lambda r:r['attempt_id']), metrics

    def run_torch_stream(self, stream, first, *, cap, temperature, top_k, top_p, presence_penalty, seed, fixed, mechanical, eos_ids, stop_interval):
        from model.inference_kernels import sample_top_k_top_p_counter, prepare_fixed_memory_swa_cache_for_physical_ring_
        from scoring.protocols import mechanical_repetition_early_stop_patterns
        import numpy as np
        torch = self.torch
        stream.initialize(first)
        started = time.perf_counter()
        refill_seconds = 0.0
        decode_tokens = refilled = 0
        size = len(first)
        active = set(range(size))
        generated = {r: [] for r in active}
        caps = {r: min(cap, stream.rows[r].get('max_tokens', cap)) for r in active}
        attempts = torch.zeros(size, dtype=torch.long, device=self.device)
        positions = torch.zeros(size, 1, dtype=torch.long, device=self.device)

        def synchronize():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            elif self.device.type == 'mps':
                torch.mps.synchronize()

        def prefill(rows):
            ids, mask = stream.inputs(rows, self.device)
            pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
            output = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
            prepare_fixed_memory_swa_cache_for_physical_ring_(output.past_key_values, self.model.config.fixed_memory_swa_window)
            return output, mask.sum(-1, keepdim=True)

        with torch.inference_mode():
            output, initial_positions = prefill(list(range(size)))
            cache, scores = output.past_key_values, output.logits[:, -1, :len(self.tokenizer)].float()
            positions.copy_(initial_positions)
            attempts.copy_(torch.tensor([stream.rows[r].get('sampling_id', stream.rows[r]['attempt_id']) for r in range(size)], device=self.device))
            synchronize()
            initial_seconds = time.perf_counter() - started
            ttft = None
            while active or not stream.exhausted:
                if active:
                    live = sorted(active)
                    adjusted = scores.clone() if presence_penalty else scores
                    if presence_penalty:
                        for r in live:
                            if generated[r]:
                                adjusted[r, list(set(generated[r]))] -= presence_penalty
                    steps = torch.tensor([len(generated.get(r, ())) for r in range(size)], device=self.device)
                    tokens = adjusted.argmax(-1, keepdim=True) if temperature <= 0 else sample_top_k_top_p_counter(
                        adjusted, attempts, steps, seed=seed, temperature=temperature, top_k=top_k, top_p=top_p)
                    values = tokens[:, 0].tolist()
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    completed = {}
                    for r in live:
                        decode_tokens += bool(generated[r])
                        generated[r].append(values[r])
                        stream.push_tokens(r, [values[r]])
                        if not fixed and values[r] in eos_ids:
                            completed[r] = 'eos'
                        elif len(generated[r]) >= caps[r]:
                            completed[r] = 'length'
                    candidates = [r for r in live if r not in completed and len(generated[r]) >= 1024 and len(generated[r]) % stop_interval == 0]
                    if mechanical and not fixed and candidates:
                        lengths = np.asarray([min(2048, len(generated[r])) for r in candidates])
                        tails = np.zeros((len(candidates), int(lengths.max())), dtype=np.int64)
                        for i, r in enumerate(candidates):
                            tails[i, :lengths[i]] = generated[r][-lengths[i]:]
                        for r, period in zip(candidates, mechanical_repetition_early_stop_patterns(tails, lengths)):
                            if period:
                                completed[r] = 'mechanical_repetition'
                    for r, reason in completed.items():
                        stream.emit(r, torch.tensor(generated.pop(r)), reason)
                        active.remove(r)
                    if active:
                        mask = torch.zeros(size, 1, dtype=torch.long, device=self.device)
                        mask[list(active)] = 1
                        output = self.model(input_ids=tokens, attention_mask=mask, position_ids=positions,
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
                        scores = output.logits[:, -1, :len(self.tokenizer)].float()
                        positions += mask
                free = [r for r in range(size) if r not in active]
                replacements = stream.acquire(free, block=not active) if free else []
                if replacements:
                    synchronize()
                    begin = time.perf_counter()
                    rows = [r for r, *_ in replacements]
                    ready, next_positions = prefill(rows)
                    target = torch.tensor(rows, device=self.device)
                    cache.batch_scatter_from(ready.past_key_values, target)
                    scores.index_copy_(0, target, ready.logits[:, -1, :len(self.tokenizer)].float())
                    positions.index_copy_(0, target, next_positions)
                    for r, attempt, _, limit in replacements:
                        active.add(r)
                        generated[r] = []
                        caps[r] = min(cap, limit)
                        attempts[r] = attempt
                    del ready
                    synchronize()
                    refill_seconds += time.perf_counter() - begin
                    refilled += len(rows)
                stream.progress(len(active), size, len(active), decode_tokens, refilled,
                    elapsed=time.perf_counter()-started, prefill=initial_seconds+refill_seconds)
            synchronize()
        wall = time.perf_counter() - started
        decode_seconds = wall - initial_seconds - refill_seconds
        metrics = dict(backend='torch', model_load_seconds=self.load_seconds, operators=self.operators,
            initial_prefill_seconds=initial_seconds, refill_prefill_seconds=refill_seconds,
            time_to_first_token_seconds=ttft, decode_seconds=decode_seconds,
            decode_tokens_per_second=decode_tokens / max(decode_seconds, 1e-9),
            decode_useful_token_events=decode_tokens, generation_wall_seconds=wall,
            total_useful_generated_tokens=stream.completed_tokens, refilled=refilled)
        if self.device.type == 'cuda':
            metrics.update(peak_allocated_gib=torch.cuda.max_memory_allocated(self.device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(self.device)/2**30)
        stream.finish(metrics)
        return metrics


class Stream:
    def __init__(self, engine, fetch, write, progress=None, on_tokens=None):
        self.engine, self.fetch_rows, self.write_rows, self.report = engine, fetch, write, progress
        self.rows, self.prepared = {}, deque()
        self.exhausted = self.source_drained = False
        self.pool = ThreadPoolExecutor(1)
        self.io_pool = ThreadPoolExecutor(1)
        self.commits = []
        self.fetch = None
        self.completed_attempts = self.completed_tokens = self.maximum_generated_tokens = 0
        self.finish_counts = Counter()
        self.fragment = []
        self.started = self.last_progress = time.perf_counter()
        self.last_useful = 0
        self.last_decode_seconds = 0
        self.on_tokens = on_tokens
        self.sent_lengths = {}

    def push_tokens(self, row, tokens):
        if self.on_tokens is not None and tokens:
            self.on_tokens(self.rows[row]['attempt_id'], tokens)
        self.sent_lengths[row] = self.sent_lengths.get(row, 0) + len(tokens)

    def initialize(self, first):
        self.rows = dict(enumerate(first))
        self.fetch = self.pool.submit(self.fetch_rows, self.engine.batch)

    def acquire(self, free_rows, *, block=False):
        if self.fetch is not None and (self.fetch.done() or block and not self.prepared):
            values = self.fetch.result()
            self.fetch = None
            self.prepared.extend(values)
            self.source_drained = not values
        replacements = []
        while free_rows and self.prepared:
            slot, row = free_rows.pop(0), self.prepared.popleft()
            self.rows[slot] = row
            self.sent_lengths[slot] = 0
            replacements.append((slot, row.get('sampling_id',row['attempt_id']), row.get('prefill_group_id',row['attempt_id']), row.get('max_tokens', 131072)))
        if not self.source_drained and self.fetch is None and len(self.prepared) <= self.engine.batch // 2:
            self.fetch = self.pool.submit(self.fetch_rows, self.engine.batch)
        self.exhausted = self.source_drained and not self.prepared
        return replacements

    def inputs(self, rows, device):
        torch = self.engine.torch
        sequences = [self.rows[r]['input_ids'] for r in rows]
        width = max(map(len, sequences))
        ids = torch.full((len(rows), width), self.engine.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, -len(sequence):] = torch.tensor(sequence)
            mask[i, -len(sequence):] = 1
        return ids.to(device), mask.to(device)

    def prefill_chunk_size(self, width):
        return max(1, min(4 if width >= 8192 else 16, 32768 // width))

    def prefill_microbatch_rows(self, rows, token_chunk_size):
        selected, width = [], 0
        for row in rows:
            width = max(width, len(self.rows[row]['input_ids']))
            limit = 1 if width > token_chunk_size else self.prefill_chunk_size(width)
            if len(selected) >= limit:
                break
            selected.append(row)
        return selected

    def emit(self, row, ids, reason):
        entry = self.rows.pop(row)
        ids = ids.tolist()
        content = ids[:-1] if ids and ids[-1] in (248046, 248044) else ids
        response = self.engine.tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        output = {k:v for k,v in entry.items() if k not in {'input_ids','vision_inputs'}}
        output.update(checkpoint_id='YANchor-4B', protocol_key='main', response=response,
            response_ids=ids, generated_tokens=len(ids), prompt_tokens=len(entry['input_ids']),
            finish_reason=reason, finished_by_eos=reason=='eos', cap_hit=reason in {'length','prompt_length_cap'})
        self.completed_attempts += 1
        self.completed_tokens += len(ids)
        self.maximum_generated_tokens = max(self.maximum_generated_tokens, len(ids))
        self.finish_counts[reason] += 1
        self.fragment.append(output)
        if len(self.fragment) >= 32:
            self.flush()

    def flush(self):
        if self.fragment:
            self.commits.append(self.io_pool.submit(self.write_rows, self.fragment))
            self.fragment = []

    def progress(self, live, physical, scheduled, useful, refilled, *, elapsed=None, prefill=None):
        now = time.perf_counter()
        if self.report is None or now - self.last_progress < 10:
            return
        elapsed = elapsed if elapsed is not None else now - self.started
        seconds = elapsed - (prefill or 0)
        self.report(dict(completed=self.completed_attempts, output_tokens=self.completed_tokens,
            useful_decode_tokens=useful, decode_seconds=seconds,
            decode_tokens_per_second=useful / max(seconds, 1e-9),
            recent_decode_tokens_per_second=(useful-self.last_useful)/max(seconds-self.last_decode_seconds,1e-9),
            prefill_seconds=prefill, live_slots=live, physical_slots=physical,
            refilled=refilled, timestamp=time.time(), status='RUNNING'))
        self.last_progress = now
        self.last_useful,self.last_decode_seconds=useful,seconds

    def finish(self, metrics):
        self.flush()
        for task in self.commits:
            task.result()
        self.io_pool.shutdown(wait=True)
        self.pool.shutdown(wait=True)
        metrics['stream_committed_groups'] = self.completed_attempts
