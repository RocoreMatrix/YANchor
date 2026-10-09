from __future__ import annotations
from collections import Counter, defaultdict
import contextlib
import functools
import importlib.util
import io
import json
import math
from multiprocessing.managers import BaseManager, BaseProxy
from pathlib import Path
import re
import shlex
import statistics
import sys
import textwrap
import threading
from typing import Any, Iterable, Mapping
import numpy as np
from scoring.protocols import PYTHON_FENCE_CONTRACT, code_contract_status, mechanical_repetitive_tail_metrics, think_protocol_status
from scoring.verification import CPU_VERIFIER_REVISION, code_blocks, compare_answers, deterministic_verifier_contract, typed_declared_answer
from evaluation.serve_verifier import ExternalVerifier, VerifierEngine, prepare_semantic_record, record_visible_response, missing_answer_outcome
OFFICIAL_RUNNERS = Path(__file__).resolve().parents[1] / 'third_party' / 'official_evaluators'
MACRO_AVERAGE_SUBSET_FIELDS = {'BBH': 'task', 'BBH128': 'task', 'BBEH': 'task', 'CMMLU': 'subject', 'CMMLU128': 'subject'}
ACCEPTED_CODE_LANGUAGES = {'', 'py', 'python', 'python3'}
PYTHON_OPEN = re.compile('```[ \\t]*(?:python|py|python3)[ \\t]*\\r?\\n', re.I)
TRAILING_IM_END = re.compile('(?:<\\|im_end\\|>)?\\s*$')
MATH_SOURCES = {'AIME2024', 'AIME2025', 'AIME2026', 'HMMT2025', 'MATH500', 'HMMT2026'}
CHOICE_SOURCES = {'GPQA68', 'MMLU256', 'HellaSwag128', 'WinoGrande128', 'ARC-C128', 'ARC-E128', 'CEval128', 'CMMLU128', 'MMLUPro', 'MMLURedux', 'SuperGPQA'}
CODE_SOURCES = {'HumanEval', 'MBPP128', 'LiveCodeBenchV6'}
_BBEH_LOAD_LOCK = threading.Lock()

def evaluation_finish_reason(row: Mapping[str, Any]) -> str:
    return str(row.get('finish_reason') or ('eos' if row.get('finished_by_eos') else 'max_tokens')).strip().lower()

def references(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[Any]:
    alternatives = metadata.get('reference_answers')
    if isinstance(alternatives, list) and alternatives:
        return alternatives
    return [row['reference_answer']]

def adapt_score_row(row):
    adapted = {**row, 'source': row['score_source']}
    if row['score_source'] == 'HellaSwag128':
        adapted['reference_answer'] = str(ord(row['reference_answer']) - 65)
    elif row['score_source'] == 'WinoGrande128':
        adapted['reference_answer'] = str(ord(row['reference_answer']) - 64)
    return adapted

def choice_references(source: str, values: Iterable[Any]) -> list[str]:
    if source == 'HellaSwag128':
        return [chr(ord('A') + int(value)) for value in values]
    if source == 'WinoGrande128':
        return [chr(ord('A') + int(value) - 1) for value in values]
    return [str(value).strip().upper() for value in values]

def math_contract(row: Mapping[str, Any], values: list[Any]) -> dict[str, Any]:
    extra = {'verifier_mode': 'math_equivalence', 'problem': str(row['prompt'])}
    if len(values) > 1:
        extra['reference_answers'] = values
    routed = deterministic_verifier_contract({'task_type': 'math', 'ground_truth': values[0], 'prompt': [{'role': 'user', 'content': str(row['prompt'])}], 'extra_info': extra})
    if routed['eligible']:
        return routed
    return {'eligible': True, 'reason': routed['reason'], 'verifier_mode': 'math_equivalence', 'answer_type': 'symbolic', 'resolution': 'math_verify_then_symbolic'}

def standard_verifier_record(row: Mapping[str, Any], response: str, *, accept_unmarked_final: bool=False) -> tuple[dict[str, Any], dict[str, Any]]:
    source = str(row['source'])
    metadata = json.loads(str(row.get('metadata_json') or '{}'))
    values = references(row, metadata)
    finish_reason = evaluation_finish_reason(row)
    record: dict[str, Any] = {'sample_id': f"{row['checkpoint_id']}:{row['source_sample_id']}:{row['sample_index']}", 'problem': str(row['prompt']), 'response': response, 'finish_reason': finish_reason, 'reference_answer': values[0]}
    if len(values) > 1:
        record['reference_answers'] = values
    routing: dict[str, Any]
    if source in MATH_SOURCES:
        routing = math_contract(row, values)
        record.update(task_type='math', verifier_mode=routing['verifier_mode'], answer_type=routing.get('answer_type') or 'symbolic')
    elif source == 'GSM8K128':
        routing = {'verifier_mode': 'numeric_equivalence', 'resolution': 'frozen_numeric'}
        record.update(task_type='math', verifier_mode='numeric_equivalence', answer_type='numeric', allow_hash_answer=True)
        solution = str(metadata.get('answer', '')).split('####', 1)[0].replace('\\n', '\n').strip()
        unit = re.search('\\d\\s*(seconds?|minutes?|hours?)(?:/(?:day|week|month|year)s?)?[.\\s]*$', solution, re.I)
        if unit:
            record['reference_unit'] = unit.group(1).lower()
    elif source in CHOICE_SOURCES:
        normalized = choice_references(source, values)
        record['reference_answer'] = normalized[0]
        if len(normalized) > 1:
            record['reference_answers'] = normalized
        routing = {'verifier_mode': 'exact_choice', 'resolution': 'frozen_choice_with_source_index_normalization'}
        record.update(task_type='choice', verifier_mode='exact_choice', answer_type='choice')
    elif source == 'BBH128':
        routing = {'verifier_mode': 'exact_text', 'resolution': 'frozen_exact_text'}
        record.update(task_type='general', source_dataset='bbh', verifier_mode='exact_text', answer_type='exact_text')
    elif source in {'IFEval128', 'IFBench128'}:
        mode = 'ifeval' if source == 'IFEval128' else 'ifbench'
        routing = {'verifier_mode': mode, 'resolution': 'official_instruction'}
        record.update(task_type=mode, verifier_mode=mode, verifier_payload=metadata)
    else:
        raise ValueError(f'unsupported fixed evaluation source: {source}')
    if source == 'BBH128':
        record['accept_unmarked_final'] = accept_unmarked_final or '</think>' in response
        if metadata.get('task') == 'word_sorting':
            record['ordered_words'] = True
            routing['resolution'] = 'bbh_ordered_words'
    return (record, routing)

def code_payload(source: str, metadata: Mapping[str, Any], code: str) -> tuple[str, str]:
    code = textwrap.dedent(code).strip()
    if source == 'HumanEval':
        entry = str(metadata['entry_point'])
        if not re.search(f'\\bdef\\s+{re.escape(entry)}\\s*\\(', code):
            code = str(metadata['prompt']).rstrip() + '\n' + textwrap.indent(code, '    ')
        else:
            context = re.split(f'(?m)^def\\s+{re.escape(entry)}\\s*\\(', str(metadata['prompt']), maxsplit=1)[0].strip()
            if context:
                code = context + '\n\n' + code
        return (code, str(metadata['test']).rstrip() + f'\n\ncheck({entry})\n')
    tests = [*metadata.get('test_list', []), *metadata.get('challenge_test_list', [])]
    harness = '\n'.join((value for value in [str(metadata.get('test_setup_code') or ''), *map(str, tests)] if value.strip()))
    return (code, harness)

@functools.lru_cache(maxsize=4)
def load_bbeh_evaluator(path: str):
    with _BBEH_LOAD_LOCK:
        spec = importlib.util.spec_from_file_location('yanchor_frozen_bbeh_evaluate', path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load BBEH evaluator: {path}')
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
        return module.evaluate_correctness

def score_bbeh(row: Mapping[str, Any], visible: str, *, extract_final: bool=True) -> dict[str, Any]:
    metadata = json.loads(str(row.get('metadata_json') or '{}'))
    declared = typed_declared_answer(visible, answer_type='exact_text') if extract_final else None
    ambiguous = bool(declared and declared['ambiguous'])
    candidate = declared['candidate_answer'] if declared else visible.strip().splitlines()[-1] if extract_final and visible.strip() else visible
    correct = bool(not ambiguous and load_bbeh_evaluator(str(OFFICIAL_RUNNERS / 'BBEH/evaluate.py'))(candidate, str(row['reference_answer'])))
    return {'verdict': 'PASS' if correct else 'FAIL', 'method': 'bbeh_official_compare_declared_answer' if declared else 'bbeh_official_evaluate_correctness', 'candidate': declared, 'details': {'task': str(metadata['task']), 'revision': str(metadata['official_evaluator_revision'])}}

def execute_code(engine: VerifierEngine, row: Mapping[str, Any], metadata: Mapping[str, Any], code: str) -> dict[str, Any]:
    source = str(row['source'])
    if source == 'LiveCodeBenchV6':
        candidate = textwrap.dedent(code).strip()
        return engine.verify({'sample_id': f"{row['checkpoint_id']}:{row['source_sample_id']}:{row['sample_index']}", 'task_type': 'code', 'verifier_mode': 'code', 'problem': str(row['prompt']), 'response': f'</think>\n```python\n{candidate}\n```', 'finish_reason': 'eos', 'verifier_payload': {'language': 'python', 'lcb_payload_path': str(metadata['lcb_payload_path']), 'lcb_runner_path': str(OFFICIAL_RUNNERS / 'LiveCodeBench'), 'time_limit_s': 6}})
    candidate, harness = code_payload(source, metadata, code)
    return engine.verify({'sample_id': f"{row['checkpoint_id']}:{row['source_sample_id']}:{row['sample_index']}", 'task_type': 'code', 'verifier_mode': 'code', 'problem': str(row['prompt']), 'response': f'</think>\n```python\n{candidate}\n```', 'finish_reason': 'eos', 'verifier_payload': {'python_harness': harness, 'language': 'python', 'time_limit_s': 10}})

def score_code(engine: VerifierEngine, row: Mapping[str, Any], visible: str, *, accept_unfenced: bool=False) -> dict[str, Any]:
    metadata = json.loads(str(row.get('metadata_json') or '{}'))
    blocks = [value for value in code_blocks(visible) if str(value['language']).lower() in ACCEPTED_CODE_LANGUAGES]
    strict_result = None
    if blocks:
        block = blocks[-1]
        strict_result = execute_code(engine, row, metadata, visible[int(block['start']):int(block['end'])])
    elif accept_unfenced and '```' not in visible:
        strict_result = execute_code(engine, row, metadata, visible)
    strict_pass = bool(strict_result and strict_result['verdict'] == 'PASS')
    relaxed_result = strict_result
    trimmed = TRAILING_IM_END.sub('', visible).rstrip()
    opens = list(PYTHON_OPEN.finditer(trimmed))
    repair_eligible = bool(row['finished_by_eos'] and (not row['cap_hit']) and (not blocks) and (len(opens) == 1) and (trimmed.count('```') == 1) and (opens[0].start() == 0))
    if repair_eligible:
        relaxed_result = execute_code(engine, row, metadata, trimmed[opens[0].end():])
    return {'semantic_correct': bool(relaxed_result and relaxed_result['verdict'] == 'PASS'), 'strict_result': strict_result, 'relaxed_correct': bool(relaxed_result and relaxed_result['verdict'] == 'PASS'), 'relaxed_result': relaxed_result, 'repair_eligible': repair_eligible}

def instruction_counts(result: Mapping[str, Any]) -> tuple[int, int]:
    external = (result.get('outcome') or {}).get('external') or {}
    details = external.get('details') or {}
    values = details.get('follow_instruction_list') or []
    return (sum((bool(value) for value in values)), len(values))

def strict_gsm_correct(visible: str, reference: Any) -> bool:
    candidate = last_balanced_boxed_answer(visible)
    return bool(candidate is not None and compare_answers(candidate, [reference], answer_type='numeric_equivalence')['verdict'] == 'PASS')

def score_one(engine: VerifierEngine, row: Mapping[str, Any], think_open_id: int | None, think_close_id: int | None, *, prompt_think_closed: bool=False, allow_empty_reasoning: bool=False, accept_unmarked_final: bool=False) -> dict[str, Any]:
    source_name = str(row['source'])
    metadata = json.loads(str(row.get('metadata_json') or '{}'))
    score_source = str(row.get('score_source') or metadata.get('score_source') or source_name)
    accept_unmarked_final = accept_unmarked_final or bool(metadata.get('accept_unmarked_final'))
    if source_name != score_source:
        row = adapt_score_row({**row, 'score_source': score_source})
    response = str(row['response'])
    response_ids = row['response_ids']
    finish_reason = evaluation_finish_reason(row)
    think = think_protocol_status(response, finish_reason, response_ids, think_open_id=think_open_id, think_close_id=think_close_id, prompt_think_closed=prompt_think_closed, allow_empty_reasoning=allow_empty_reasoning)
    mechanical = mechanical_repetitive_tail_metrics(response_ids, finish_reason)
    source = str(row['source'])
    instruction_pass = instruction_total = 0
    relaxed_code = strict_code = None
    code_status: Mapping[str, Any] | None = None
    routing: Mapping[str, Any] = {}
    if source in CODE_SOURCES or source == 'BBEH':
        record = dict(response=response, problem=str(row['prompt']), task_type='code' if source in CODE_SOURCES else 'general', verifier_mode='code' if source in CODE_SOURCES else 'exact_text', finish_reason=finish_reason)
    else:
        record, routing = standard_verifier_record(row, response, accept_unmarked_final=accept_unmarked_final)
    record['prompt_think_closed'] = prompt_think_closed
    record = prepare_semantic_record(record, think_status=think, mechanical=mechanical['degenerate'])
    visible = record_visible_response(record)
    recovered_cot = record['answer_selection']['source'] == 'cot'
    if source in CODE_SOURCES:
        code_status = code_contract_status(response, response_ids, finish_reason, PYTHON_FENCE_CONTRACT, prompt_think_closed=prompt_think_closed)
        if visible.strip():
            code = score_code(engine, row, visible, accept_unfenced=accept_unmarked_final)
            semantic = bool(code['semantic_correct'])
            relaxed_code = bool(code['relaxed_correct'])
            strict_code = bool(semantic and think['verdict'] == 'PASS' and code_status['contract_pass'] and (not mechanical['degenerate']) and (not recovered_cot))
            verifier_result = code['relaxed_result'] or {'verdict': 'UNCERTAIN', 'method': 'no_complete_visible_python_fence'}
        else:
            semantic = relaxed_code = strict_code = False
            verifier_result = missing_answer_outcome(record)
        contract = strict_code
        routing = {'verifier_mode': 'code', 'resolution': 'livecodebench_official' if source == 'LiveCodeBenchV6' else 'python_harness'}
    elif source == 'BBEH':
        verifier_result = score_bbeh(row, visible, extract_final=accept_unmarked_final or recovered_cot or prompt_think_closed or ('</think>' in response)) if visible.strip() else missing_answer_outcome(record)
        semantic = verifier_result['verdict'] == 'PASS'
        contract = bool(semantic and think['verdict'] == 'PASS' and (not mechanical['degenerate']) and (not recovered_cot))
        routing = {'verifier_mode': 'bbeh_official', 'resolution': 'google_deepmind_evaluate_correctness'}
    else:
        verifier_result = engine.verify(record)
        semantic = verifier_result['verdict'] == 'PASS'
        contract = bool(semantic and think['verdict'] == 'PASS' and (not mechanical['degenerate']) and (not recovered_cot))
        if source in {'IFEval128', 'IFBench128'}:
            instruction_pass, instruction_total = instruction_counts(verifier_result)
    verifier_result.setdefault('outcome', {})['answer_selection'] = record['answer_selection']
    if recovered_cot:
        verifier_result['outcome']['format_kind'] = 'missing_final'
    original_label_correct = None
    primary_view_eligible = None
    if source == 'MMLURedux':
        primary_view_eligible = bool(metadata['primary_corrected_choice_eligible'])
        original_record = dict(record)
        original_record['reference_answer'] = str(metadata['original_answer_letter'])
        original_record.pop('reference_answers', None)
        original_label_correct = bool(visible.strip() and engine.verify(original_record)['verdict'] == 'PASS')
    subgroup = ''
    if source in MACRO_AVERAGE_SUBSET_FIELDS:
        subgroup = str(metadata[MACRO_AVERAGE_SUBSET_FIELDS[source]])
    elif source == 'MMLUPro':
        subgroup = str(metadata['category'])
    elif source == 'MMLURedux':
        subgroup = str(metadata['annotation_error_type'])
    elif source == 'SuperGPQA':
        subgroup = str(metadata['discipline'])
    elif source == 'LiveCodeBenchV6':
        subgroup = str(metadata['difficulty'])
    elif source == 'HMMT2026':
        subgroup = ','.join(map(str, metadata['problem_type']))
    gsm = strict_gsm_correct(visible, row['reference_answer']) if source == 'GSM8K128' else None
    return {'checkpoint_id': str(row['checkpoint_id']), 'protocol_key': str(row['protocol_key']), 'source': source_name, 'report_source': source_name, 'benchmark_subgroup': subgroup, 'source_sample_id': str(row['source_sample_id']), 'problem_family_id': str(row['problem_family_id']), 'sample_index': int(row['sample_index']), 'semantic_correct': bool(semantic), 'original_label_correct': original_label_correct, 'primary_view_eligible': primary_view_eligible, 'contract_correct': bool(contract), 'strict_code_correct': strict_code, 'relaxed_code_correct': relaxed_code, 'strict_gsm_correct': gsm, 'instruction_pass': instruction_pass, 'instruction_total': instruction_total, 'verdict': str(verifier_result['verdict']), 'method': str(verifier_result.get('method') or ''), 'effective_verifier_mode': str(routing.get('verifier_mode') or ''), 'routing_resolution': str(routing.get('resolution') or ''), 'routing_reason': str(routing.get('reason') or ''), 'think_verdict': str(think['verdict']), 'think_reason': str(think['reason']), 'think_open_count': int(think['opening_count']), 'think_close_count': int(think['closing_count']), 'answer_visible': bool(visible) and (not recovered_cot), 'finished_by_eos': bool(row['finished_by_eos']), 'cap_hit': bool(row['cap_hit']), 'mechanical_repetition': bool(mechanical['degenerate']), 'mechanical_onset_index': mechanical['failure_onset_index'], 'code_contract_reason': '' if code_status is None else str(code_status['contract_reason']), 'generated_tokens': int(row['generated_tokens']), 'scoring_json': json.dumps(verifier_result, ensure_ascii=False, sort_keys=True)}

def build_engine(config: Mapping[str, Any], *, external_factory=ExternalVerifier) -> VerifierEngine:
    project = Path(__file__).resolve().parents[1]
    verifier = config['execution_contract']['verifier_service']
    assets = Path(verifier['external_verifier_assets'])
    python = shlex.quote(sys.executable)
    math_command = f"{python} {shlex.quote(str(project.parent / 'run.py'))}"
    code_command = f"{python} {shlex.quote(str(project.parent / 'run.py'))}"
    instruction_command = ' '.join(map(shlex.quote, [sys.executable, str(project.parent / 'run.py'), '_verify', 'instruction', '--ifbench-repo', str(assets / 'ifbench_official'), '--ifeval-repo', str(assets / 'ifeval_official'), '--nltk-data', str(assets / 'nltk_data')]))
    math_command += ' _verify math'
    code_command += ' _verify code'
    timeout = float(verifier['external_timeout_seconds'])
    return VerifierEngine(math_verifier=external_factory(math_command, timeout=timeout, worker_count=int(verifier['math_workers'])), code_verifier=external_factory(code_command, timeout=None, worker_count=int(verifier['code_workers'])), instruction_verifier=external_factory(instruction_command, timeout=timeout, worker_count=int(verifier['instruction_workers'])), service_instance_id='fixed-eval-frozen-protocol')

class _ExternalVerifierProxy(BaseProxy):
    _exposed_ = ('run', 'close', '__getattribute__')

    @property
    def enabled(self):
        return self._callmethod('__getattribute__', ('enabled',))

    @property
    def worker_count(self):
        return self._callmethod('__getattribute__', ('worker_count',))

    def run(self, request):
        return self._callmethod('run', (request,))

    def close(self):
        return self._callmethod('close')

class _ScoringVerifierManager(BaseManager):
    pass

@contextlib.contextmanager
def shared_scoring_engine(config, context):
    with _ScoringVerifierManager(ctx=context) as manager:
        engine = build_engine(config, external_factory=manager.ExternalVerifier)
        try:
            yield engine
        finally:
            engine.close()

def percentile(values: list[int], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))

def pass_estimate(correct: int, samples: int, k: int) -> float:
    if correct <= 0:
        return 0.0
    if samples - correct < k:
        return 1.0
    return 1.0 - math.comb(samples - correct, k) / math.comb(samples, k)

def summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    source = str(records[0]['report_source'])
    cases: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        cases[str(record['source_sample_id'])].append(record)
    tokens = [int(record['generated_tokens']) for record in records]
    attempts = len(records)
    semantic = sum((bool(record['semantic_correct']) for record in records))
    contract = sum((bool(record['contract_correct']) for record in records))
    sample_counts = sorted({len(value) for value in cases.values()})
    relaxed = [bool(record['relaxed_code_correct']) for record in records if record['relaxed_code_correct'] is not None]
    strict_code = [bool(record['strict_code_correct']) for record in records if record['strict_code_correct'] is not None]
    strict_gsm = [bool(record['strict_gsm_correct']) for record in records if record['strict_gsm_correct'] is not None]
    instruction_total = sum((int(record['instruction_total']) for record in records))
    subgroups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get('benchmark_subgroup') or ''):
            subgroups[str(record['benchmark_subgroup'])].append(record)
    subgroup_semantic = {name: statistics.fmean((bool(record['semantic_correct']) for record in values)) for name, values in sorted(subgroups.items())}
    primary_records = [record for record in records if record.get('primary_view_eligible') is True]
    original_label = [bool(record['original_label_correct']) for record in records if record.get('original_label_correct') is not None]
    clean_redux = [bool(record['semantic_correct']) for record in records if record.get('benchmark_subgroup') == 'ok']
    observed = {metric: statistics.fmean((any((bool(row[metric]) for row in value)) for value in cases.values())) for metric in ('semantic_correct', 'contract_correct')}
    pass_ks = [k for k in sorted({1, 2, 4, 8, *sample_counts}) if all((k <= len(value) for value in cases.values()))]
    execution_counts = Counter()
    for row in records:
        outcome = json.loads(row.get('scoring_json') or '{}').get('outcome') or {}
        details = (outcome.get('external') or {}).get('details') or {}
        if isinstance(details, dict):
            for name in ('global_timeout', 'missing_result', 'missing_metadata', 'scorer_error'):
                if details.get(name) is True:
                    execution_counts[name] += 1
    result = {'attempts': attempts, 'upstream_execution_counts': dict(execution_counts), 'cases': len(cases), 'samples_per_case': sample_counts, 'semantic_mean_pass_at_1': semantic / attempts, 'contract_mean_pass_at_1': contract / attempts, 'semantic_observed_pass_at_k': observed['semantic_correct'], 'contract_observed_pass_at_k': observed['contract_correct'], 'unbiased_pass_at_k': {metric: {str(k): statistics.fmean((pass_estimate(sum((bool(row[metric]) for row in value)), len(value), k) for value in cases.values())) for k in pass_ks} for metric in ('semantic_correct', 'contract_correct')}, 'strict_code_mean_pass_at_1': statistics.fmean(strict_code) if strict_code else None, 'relaxed_code_mean_pass_at_1': statistics.fmean(relaxed) if relaxed else None, 'strict_gsm_accuracy': statistics.fmean(strict_gsm) if strict_gsm else None, 'instruction_level_accuracy': sum((int(record['instruction_pass']) for record in records)) / instruction_total if instruction_total else None, 'all_correct_cases': sum((all((bool(row['semantic_correct']) for row in value)) for value in cases.values())), 'mixed_cases': sum((any((bool(row['semantic_correct']) for row in value)) and (not all((bool(row['semantic_correct']) for row in value))) for value in cases.values())), 'all_wrong_cases': sum((not any((bool(row['semantic_correct']) for row in value)) for value in cases.values())), 'thinking_closed_exactly_once_rate': statistics.fmean((int(record['think_close_count'] == 1) for record in records)), 'thinking_protocol_pass_rate': statistics.fmean((record['think_verdict'] == 'PASS' for record in records)), 'thinking_verdicts': dict(Counter((record['think_verdict'] for record in records))), 'thinking_reasons': dict(Counter((record['think_reason'] for record in records))), 'eos_rate': statistics.fmean((bool(record['finished_by_eos']) for record in records)), 'cap_hit_rate': statistics.fmean((bool(record['cap_hit']) for record in records)), 'answer_visible_rate': statistics.fmean((bool(record['answer_visible']) for record in records)), 'mechanical_repetition_rate': statistics.fmean((bool(record['mechanical_repetition']) for record in records)), 'mean_generated_tokens': statistics.fmean(tokens), 'p50_generated_tokens': percentile(tokens, 50), 'p90_generated_tokens': percentile(tokens, 90), 'p95_generated_tokens': percentile(tokens, 95), 'p99_generated_tokens': percentile(tokens, 99), 'maximum_generated_tokens': max(tokens), 'semantic_correct_per_million_tokens': semantic * 1000000 / sum(tokens) if sum(tokens) else None, 'verdicts': dict(Counter((str(record['verdict']) for record in records))), 'methods': dict(Counter((str(record['method']) for record in records))), 'effective_verifier_modes': dict(Counter((str(record['effective_verifier_mode']) for record in records))), 'routing_resolutions': dict(Counter((str(record['routing_resolution']) for record in records)))}
    result['subgroup_semantic_mean_pass_at_1'] = subgroup_semantic
    result['subgroup_semantic_macro_pass_at_1'] = statistics.fmean(subgroup_semantic.values()) if subgroup_semantic else None
    if source in MACRO_AVERAGE_SUBSET_FIELDS:
        result['official_primary_mean_pass_at_1'] = result['subgroup_semantic_macro_pass_at_1']
    elif source == 'MMLURedux':
        result['official_primary_mean_pass_at_1'] = statistics.fmean((bool(record['semantic_correct']) for record in primary_records)) if primary_records else None
        result['original_label_accuracy'] = statistics.fmean(original_label) if original_label else None
        result['clean_accuracy'] = statistics.fmean(clean_redux) if clean_redux else None
        result['corrected_choice_accuracy'] = result['official_primary_mean_pass_at_1']
        result['corrected_choice_attempts'] = len(primary_records)
    else:
        result['official_primary_mean_pass_at_1'] = result['semantic_mean_pass_at_1']
    return result

def last_balanced_boxed_answer(response_text: str) -> str | None:
    text = str(response_text)
    values: list[str] = []
    for match in re.finditer('\\\\boxed\\s*\\{', text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            character = text[index]
            if character == '{':
                depth += 1
            elif character == '}':
                depth -= 1
                if depth == 0:
                    value = text[start:index].strip()
                    if value:
                        values.append(value)
                    break
    return values[-1] if values else None

_ScoringVerifierManager.register('ExternalVerifier', ExternalVerifier, _ExternalVerifierProxy)
