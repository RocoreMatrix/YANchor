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
    def __init__(self, model=ROOT, batch=1, profile='fast'):
        import torch
        from transformers import AutoTokenizer
        from model.fixed_memory import load_model
        from model.inference_kernels import install_inference
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        self.torch = torch
        self.batch = batch
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 248044
        start = time.perf_counter()
        self.model = load_model(model, torch_dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
        self.model.config.fixed_memory_swa_reproducible_decode = profile == 'reference'
        self.model.config.fixed_memory_swa_shared_ring_storage = batch > 1
        self.operators = install_inference(self.model, fp16_state=batch > 1 and profile == 'fast',
            int8_mlp_b1=batch == 1 and profile == 'fast')
        self.operators['gdn_norm_classes'] = sorted({type(m.norm).__module__ + '.' + type(m.norm).__name__
            for m in self.model.modules() if hasattr(m, 'recurrent_gated_delta_rule')})
        self.load_seconds = time.perf_counter() - start

    def encode(self, rows):
        if not rows:
            return []
        texts = [r.get('prompt_text') or prompt_text(r.get('prompt', ''), r.get('messages')) for r in rows]
        encoded = self.tokenizer(texts, add_special_tokens=False, padding=False)['input_ids']
        return [dict(row, input_ids=ids) for row, ids in zip(rows, encoded)]

    def run_stream(self, stream, first, *, cap=131072, temperature=0.6, top_k=20,
                   top_p=0.95, presence_penalty=0.0, seed=20260923, fixed=False,
                   mechanical=True, stop_interval=128, sampling_backend='triton_top20',
                   prefill_context=None,eos_ids=(248046,248044)):
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

    def generate(self, prompts, **settings):
        settings.setdefault('stop_interval', 16 if self.batch == 1 else 128)
        rows = [dict(p, attempt_id=i, sample_index=p.get('sample_index', 0)) if isinstance(p, dict)
                else dict(prompt=p, attempt_id=i, sample_index=0) for i, p in enumerate(prompts)]
        for row in rows:
            row['max_tokens'] = min(settings.get('cap',131072),row.get('max_tokens',settings.get('cap',131072)))
        queue = deque(rows)
        result = []
        def fetch(count):
            return self.encode([queue.popleft() for _ in range(min(count, len(queue)))])
        stream = Stream(self, fetch, result.extend)
        first = fetch(self.batch)
        metrics = self.run_stream(stream, first, **settings)
        return sorted(result, key=lambda r:r['attempt_id']), metrics


class Stream:
    def __init__(self, engine, fetch, write, progress=None):
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
