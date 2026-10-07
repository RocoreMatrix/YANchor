from __future__ import annotations
import json
import os
import queue
import re
import select
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoring.protocols import CAP_FINISH_REASONS, NATURAL_FINISH_REASONS
from scoring.verification import CPU_VERIFIER_REVISION, DIRECT_REFERENCE_VERIFIER_MODES, INSTRUCTION_VERIFIER_MODES, MATH_EXTERNAL_VERIFIER_MODES, RESULT_SCHEMA, canonical_json, code_blocks, compare_answers, external_request, normalize_answer_text, reference_alternatives, sha256_json, sha256_text, typed_declared_answer, validate_external_result, validate_request
DETERMINISTIC_ANSWER_MODES = frozenset({*DIRECT_REFERENCE_VERIFIER_MODES, 'math_equivalence'})
VERIFIER_EXECUTION_REVISION = 'isolated_jsonl_retire_failed_connection_v5'

def legacy_bbh_choice_contract(record: Mapping[str, Any]) -> bool:
    if str(record.get('source_dataset') or '').strip().casefold() != 'bbh' or str(record.get('verifier_mode') or '').strip().lower() != 'exact_text':
        return False
    references = record.get('reference_answers')
    if not isinstance(references, list):
        reference = record.get('reference_answer')
        references = [reference] if reference is not None else []
    labels = []
    for reference in references:
        matched = re.fullmatch('\\s*\\(([A-Z])\\)\\s*', str(reference), re.I)
        if matched is None:
            return False
        labels.append(matched.group(1).upper())
    problem_labels = set(re.findall('(?<![A-Za-z])\\(([A-Z])\\)(?![A-Za-z])', str(record.get('problem') or ''), re.I))
    return bool(labels) and set(labels).issubset({label.upper() for label in problem_labels})

def effective_answer_type(record: Mapping[str, Any]) -> str:
    verifier_mode = str(record.get('verifier_mode') or '').strip().lower()
    source_dataset = str(record.get('source_dataset') or '').strip().casefold()
    if legacy_bbh_choice_contract(record):
        return 'choice'
    if verifier_mode in {'exact_choice', 'choice'}:
        return 'choice'
    if verifier_mode in {'exact_text', 'text'}:
        return 'exact_text'
    if verifier_mode == 'choice_set':
        return 'choice_set'
    if verifier_mode == 'numeric_equivalence':
        return 'numeric'
    if verifier_mode in {'numeric', 'integer', 'fraction', 'decimal', 'set', 'symbolic', 'expression', 'drop_exact'}:
        return verifier_mode
    configured = str(record.get('answer_type') or verifier_mode or 'auto').lower()
    if configured == 'exact_choice':
        return 'choice'
    if configured == 'numeric_equivalence':
        return 'numeric'
    return configured
NON_FINITE_NUMERIC_ANSWERS = frozenset({'inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity', 'nan', '+nan', '-nan', '∞', '+∞', '-∞', '\\infty', '+\\infty', '-\\infty'})

class BackendError(RuntimeError):
    pass

class VerifierTimeout(BackendError):
    pass

def visible_response(response: Any) -> str:
    text = str(response or '')
    close = text.rfind('</think>')
    if close >= 0:
        text = text[close + len('</think>'):]
    return text.strip()

def record_visible_response(record: Mapping[str, Any]) -> str:
    if 'answer_selection' in record:
        selected = record['answer_selection']
        return str(record['response'])[selected['start']:selected['end']].strip()
    return visible_response(record.get('response'))

def missing_answer_outcome(record):
    finish = record.get('finish_reason', '')
    known_failure = finish in CAP_FINISH_REASONS | NATURAL_FINISH_REASONS or finish == 'mechanical_repetition'
    verdict = 'FAIL' if known_failure else 'UNCERTAIN'
    method = 'budget_exhausted_without_answer' if finish in CAP_FINISH_REASONS else 'mechanical_stop_without_answer' if finish == 'mechanical_repetition' else 'no_completed_answer'
    return dict(verdict=verdict, cpu_verdict=verdict, method=method, candidate=None)

def prepare_semantic_record(record, *, think_status=None, mechanical=False):
    from scoring.protocols import think_protocol_status
    text = str(record['response'])
    closed = record.get('prompt_think_closed', False)
    if think_status is None:
        think_status = think_protocol_status(text, record.get('finish_reason', ''), record.get('response_ids'), think_open_id=record.get('think_open_id'), think_close_id=record.get('think_close_id'), prompt_think_closed=closed, allow_empty_reasoning=record.get('allow_empty_reasoning', False))
    start = 0 if closed else text.find('</think>') + len('</think>') if think_status['literal_closing_count'] else len(text)
    end = len(text)
    ending = re.search('\\s*<\\|im_end\\|>\\s*$', text)
    if ending:
        end = ending.start()
    visible = text[start:end].strip()
    absent = not visible or bool(re.fullmatch('(?:see (?:the )?(?:answer |solution )?above|as above|见上文|如上|答案如上)[.。!！]?', visible, re.I))
    selection = dict(source='answer', start=start, end=end, reason='submitted_answer')
    declared = None
    mode = effective_answer_type(record)
    if absent:
        selection['reason'] = 'no_submitted_answer'
        if record.get('finish_reason') == 'eos' and (not mechanical) and (not closed):
            cot_end = text.find('</think>') if think_status['literal_closing_count'] else end
            cot = text[:cot_end]
            if record['task_type'] == 'code':
                blocks = code_blocks(cot)
                candidate = blocks[-1] if blocks else None
                if candidate and (not cot[candidate['end']:].strip('` \n\r\t')):
                    selection.update(source='cot', start=candidate['start'], end=candidate['end'], reason='terminal_code_without_final')
            elif str(record.get('verifier_mode')) not in INSTRUCTION_VERIFIER_MODES:
                declared = typed_declared_answer(cot, answer_type=mode, allow_hash_answer=record.get('allow_hash_answer', False), problem=record['problem'])
                if declared:
                    before = cot[:declared.get('start', 0)].splitlines()[-1:]
                    if before and re.search('\\b(?:if|maybe|perhaps|suppose|assuming|for example)\\b[^.!?]*$|(?:如果|假设|也许|例如)[^。！？]*$', before[0], re.I):
                        declared = None
                if declared and mode in {'choice', 'exact_choice'}:
                    from scoring.verification import choice_answer_value, numeric_value
                    if choice_answer_value(declared['candidate_answer'], record['problem']) is None and numeric_value(declared['candidate_answer']) is None:
                        declared = None
                if declared and 'end' in declared and (not declared['ambiguous']) and (not cot[declared['end']:].strip(' $*`_\\[]().。;；\n\r\t')):
                    selection.update(source='cot', start=declared.get('start', 0), end=cot_end, reason='terminal_conclusion_without_final')
                else:
                    declared = None
    result = dict(record, answer_selection=selection, think_status=think_status)
    for key in ('response_ids', 'think_open_id', 'think_close_id', 'allow_empty_reasoning'):
        result.pop(key, None)
    if declared is not None and selection['source'] == 'cot':
        result['declared_answer'] = declared
    return result

class JSONLVerifierWorker:

    def __init__(self, argv: Sequence[str], *, timeout: float | None, worker_id: int) -> None:
        self.argv = [*argv, '--jsonl-worker']
        self.timeout = timeout
        self.worker_id = worker_id
        self.process: subprocess.Popen[str] | None = None
        self.stdout_buffer = b''
        self._start()

    def _start(self) -> None:
        self.process = subprocess.Popen(self.argv, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1, start_new_session=True)

    def request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        process = self.process
        if process is None or process.poll() is not None:
            raise BackendError('persistent external verifier is not running')
        assert process is not None and process.stdin is not None and (process.stdout is not None)
        try:
            process.stdin.write(canonical_json(request) + '\n')
            process.stdin.flush()
            deadline = time.monotonic() + self.timeout if self.timeout is not None else None
            while True:
                newline = self.stdout_buffer.find(b'\n')
                if newline >= 0:
                    raw_line = self.stdout_buffer[:newline]
                    self.stdout_buffer = self.stdout_buffer[newline + 1:]
                    line = raw_line.decode('utf-8', errors='replace')
                else:
                    remaining = deadline - time.monotonic() if deadline is not None else None
                    if remaining is not None and remaining <= 0.0:
                        raise VerifierTimeout('persistent external verifier timed out')
                    ready, _, _ = select.select([process.stdout], [], [], remaining)
                    if not ready:
                        raise VerifierTimeout('persistent external verifier timed out')
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        raise BackendError('persistent external verifier exited without a result')
                    self.stdout_buffer += chunk
                    if len(self.stdout_buffer) > 1000000:
                        raise BackendError('persistent external verifier emitted an oversized line')
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BackendError('persistent external verifier returned invalid JSON') from exc
                result = validate_external_result(payload)
                result['worker_id'] = self.worker_id
                return result
        except BrokenPipeError as exc:
            raise BackendError(f'persistent external verifier protocol failed: {exc}') from exc

    def close(self) -> None:
        process = self.process
        self.process = None
        self.stdout_buffer = b''
        if process is None:
            return
        try:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                if process.poll() is None:
                    process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                if process.poll() is None:
                    process.kill()
            process.wait()
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

class ExternalVerifier:

    def __init__(self, command: str | None, *, timeout: float | None, worker_count: int=0) -> None:
        self.argv = shlex.split(command) if command else []
        self.timeout = timeout
        self.worker_count = int(worker_count)
        if command and (not self.argv):
            raise ValueError('external verifier command is empty')
        if self.worker_count < 0:
            raise ValueError('external verifier worker count cannot be negative')
        self.workers = [JSONLVerifierWorker(self.argv, timeout=timeout, worker_id=index) for index in range(self.worker_count)] if self.argv else []
        self.available: queue.Queue[JSONLVerifierWorker] = queue.Queue()
        for worker in self.workers:
            self.available.put(worker)

    @property
    def enabled(self) -> bool:
        return bool(self.argv)

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not self.argv:
            raise BackendError('required external verifier is disabled')
        started = time.perf_counter()
        if self.workers:
            worker = self.available.get()
            try:
                if worker.process is None:
                    worker._start()
                result = worker.request(request)
            except BaseException:
                worker.close()
                raise
            finally:
                self.available.put(worker)
            result.update({'command': self.argv, 'persistent_worker': True, 'request_sha256': sha256_json(request), 'elapsed_s': time.perf_counter() - started})
            return result
        try:
            completed = subprocess.run(self.argv, input=canonical_json(request), text=True, capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired:
            raise VerifierTimeout('external verifier timed out')
        if completed.returncode != 0:
            stderr = completed.stderr.strip()[-1000:]
            raise BackendError(f'external verifier exited {completed.returncode}: {stderr}')
        try:
            raw = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BackendError('external verifier did not return JSON') from exc
        result = validate_external_result(raw)
        result.update({'command': self.argv, 'request_sha256': sha256_json(request), 'elapsed_s': time.perf_counter() - started})
        return result

    def close(self) -> None:
        for worker in self.workers:
            worker.close()

class VerifierEngine:

    def __init__(self, *, math_verifier: ExternalVerifier, code_verifier: ExternalVerifier, instruction_verifier: ExternalVerifier | None=None, service_instance_id: str='') -> None:
        self.math_verifier = math_verifier
        self.code_verifier = code_verifier
        self.instruction_verifier = instruction_verifier
        self.config = {'project_release': str(Path(__file__).resolve().parents[1]), 'service_execution_revision': VERIFIER_EXECUTION_REVISION, 'service_instance_id': service_instance_id, 'math_external_enabled': math_verifier.enabled, 'math_external_workers': math_verifier.worker_count, 'code_external_enabled': code_verifier.enabled, 'code_external_workers': code_verifier.worker_count, 'instruction_external_enabled': bool(instruction_verifier and instruction_verifier.enabled), 'instruction_external_workers': 0 if instruction_verifier is None else instruction_verifier.worker_count, 'rule_equivalence_revision': CPU_VERIFIER_REVISION}
        self.config_sha256 = sha256_json(self.config)

    def close(self) -> None:
        seen: set[int] = set()
        for verifier in (self.math_verifier, self.code_verifier, self.instruction_verifier):
            if verifier is not None and id(verifier) not in seen:
                seen.add(id(verifier))
                verifier.close()

    def _verify_code(self, record: Mapping[str, Any]) -> dict[str, Any]:
        visible = record_visible_response(record)
        blocks = code_blocks(visible)
        unterminated_block = False
        if not blocks and visible.count('```') == 1:
            match = re.fullmatch('\\s*```([^\\n`]*)\\n([\\s\\S]+)', visible)
            if match and match.group(1).strip().lower() in {'', 'py', 'python'}:
                code = match.group(2).strip()
                if code:
                    blocks = [{'index': 0, 'language': match.group(1).strip().lower() or 'python', 'code': code}]
                    unterminated_block = True
        selection: dict[str, Any]
        if len(blocks) == 1:
            selection = {'code_block_index': 0, 'language': blocks[0]['language'], 'confidence': 'high', 'method': 'single_unterminated_python_fence' if unterminated_block else 'single_fenced_block'}
        elif len(blocks) > 1:
            selection = {'code_block_index': len(blocks) - 1, 'language': blocks[-1]['language'], 'confidence': 'deterministic', 'method': 'last_fenced_block'}
        elif record.get('code_is_full_response') is True:
            blocks = [{'index': 0, 'language': str(record.get('language', '')).lower(), 'code': visible}]
            selection = {'code_block_index': 0, 'language': blocks[0]['language'], 'confidence': 'high', 'method': 'explicit_full_response'}
        else:
            return {'verdict': 'UNCERTAIN', 'method': 'no_unambiguous_code_submission', 'candidate': None, 'external': None, 'cpu_verdict': 'UNCERTAIN'}
        index = int(selection['code_block_index'])
        if index < 0:
            return {'verdict': 'UNCERTAIN', 'method': 'no_submitted_code_selected', 'candidate': {'selection': selection}, 'external': None, 'cpu_verdict': 'UNCERTAIN'}
        code = str(blocks[index]['code'])
        language = str(selection.get('language') or blocks[index].get('language') or '')
        candidate = {'selection': selection, 'language': language, 'characters': len(code), 'sha256': sha256_text(code)}
        request = external_request(record, candidate=code, language=language)
        external = self.code_verifier.run(request)
        return {'verdict': external['verdict'], 'method': 'executable_code_verifier', 'candidate': candidate, 'external': external, 'cpu_verdict': external['verdict']}

    def _verify_answer(self, record: Mapping[str, Any]) -> dict[str, Any]:
        answer_type = effective_answer_type(record)
        visible = record_visible_response(record)
        verifier_mode = str(record.get('verifier_mode') or '').strip().lower()
        allow_hash_answer = bool(record.get('allow_hash_answer') is True)
        declared = record.get('declared_answer') or typed_declared_answer(visible, answer_type=answer_type, allow_hash_answer=allow_hash_answer, problem=record['problem'])
        if declared is None and record.get('accept_unmarked_final') and visible.strip():
            candidate = visible.strip().splitlines()[-1]
            declared = {'candidate_answer': candidate, 'ambiguous': False, 'method': 'unmarked_final_line'}
        if declared is None:
            return missing_answer_outcome(record)
        extraction = {**declared, 'answer_type': answer_type}
        extraction['source'] = record.get('answer_selection', {}).get('source', 'answer')
        if extraction.get('ambiguous') is True:
            return {'verdict': 'FAIL', 'method': 'conflicting_submitted_answers', 'candidate': extraction, 'deterministic': None, 'external': None, 'judgments': [], 'cpu_verdict': 'FAIL'}
        candidate = extraction['candidate_answer']
        if record.get('ordered_words'):
            candidate = ' '.join(candidate.split()).casefold()
            verdict = 'PASS' if any((candidate == ' '.join(str(v).split()).casefold() for v in reference_alternatives(record))) else 'FAIL'
            return dict(verdict=verdict, method='bbh_ordered_words', candidate=extraction, cpu_verdict=verdict)
        normalized_candidate = normalize_answer_text(candidate)
        normalized_references = {normalize_answer_text(value) for value in reference_alternatives(record)}
        if normalized_candidate in NON_FINITE_NUMERIC_ANSWERS and normalized_candidate not in normalized_references:
            return {'verdict': 'FAIL', 'method': 'non_finite_numeric_candidate', 'candidate': extraction, 'deterministic': {'verdict': 'FAIL', 'method': 'non_finite_numeric_candidate', 'candidate': normalized_candidate, 'references': sorted(normalized_references)}, 'external': None, 'judgments': [], 'cpu_verdict': 'FAIL'}
        deterministic = compare_answers(candidate, reference_alternatives(record), answer_type=answer_type, problem=record['problem'], numeric_scale=str(record.get('numeric_scale') or ''), reference_unit=str(record.get('reference_unit') or ''))
        if deterministic['verdict'] in {'PASS', 'FAIL'}:
            return {'verdict': deterministic['verdict'], 'method': 'deterministic_answer_verifier', 'candidate': extraction, 'deterministic': deterministic, 'external': None, 'judgments': [], 'cpu_verdict': deterministic['verdict']}
        external = None
        if record['task_type'] in {'math', 'stem'} or str(record.get('verifier_mode') or '').lower() in MATH_EXTERNAL_VERIFIER_MODES:
            if not self.math_verifier.enabled:
                raise BackendError('required math verifier is disabled')
            request = external_request(record, candidate=candidate)
            try:
                external = self.math_verifier.run(request)
            except VerifierTimeout as exc:
                external = {'verdict': 'UNCERTAIN', 'details': {'method': 'math_verifier_timeout', 'error': str(exc), 'request_sha256': sha256_json(request)}}
            if external['verdict'] in {'PASS', 'FAIL'}:
                return {'verdict': external['verdict'], 'method': 'external_math_verifier', 'candidate': extraction, 'deterministic': deterministic, 'external': external, 'judgments': [], 'cpu_verdict': external['verdict']}
        return {'verdict': 'UNCERTAIN', 'method': 'deterministic_verifiers_inconclusive', 'candidate': extraction, 'deterministic': deterministic, 'external': external, 'judgments': [], 'cpu_verdict': 'UNCERTAIN'}

    def _verify_instruction(self, record: Mapping[str, Any]) -> dict[str, Any]:
        visible = record_visible_response(record)
        if self.instruction_verifier is None or not self.instruction_verifier.enabled:
            raise BackendError('instruction verifier is disabled')
        external = self.instruction_verifier.run(external_request(record, candidate=visible))
        return {'verdict': external['verdict'], 'method': 'external_instruction_verifier', 'candidate': {'source': 'visible_response', 'characters': len(visible), 'sha256': sha256_text(visible)}, 'external': external, 'cpu_verdict': external['verdict']}

    def verify(self, raw_record: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        record = validate_request(raw_record)
        instruction_mode = str(record.get('verifier_mode', '')).lower()
        if not record_visible_response(record):
            outcome = missing_answer_outcome(record)
        elif instruction_mode in INSTRUCTION_VERIFIER_MODES:
            outcome = self._verify_instruction(record)
        elif instruction_mode == 'bbeh_official':
            from evaluation.fixed_protocol_score import score_bbeh
            outcome = score_bbeh({'reference_answer': record['reference_answer'], 'metadata_json': json.dumps(record['verifier_payload'])}, record_visible_response(record))
        elif record['task_type'] == 'code':
            outcome = self._verify_code(record)
        elif record['task_type'] == 'general' and (not (str(record.get('verifier_mode', '')).lower() in DETERMINISTIC_ANSWER_MODES or str(record.get('answer_type', '')).lower() in DETERMINISTIC_ANSWER_MODES)):
            visible = record_visible_response(record)
            outcome = {'verdict': 'UNCERTAIN', 'method': 'unsupported_general_semantic_mode', 'candidate': {'source': 'visible_response', 'characters': len(visible), 'sha256': sha256_text(visible)}, 'judgments': [], 'cpu_verdict': 'UNCERTAIN'}
        else:
            outcome = self._verify_answer(record)
        if 'answer_selection' in record:
            candidate = outcome.get('candidate') or {}
            outcome['answer_selection'] = dict(record['answer_selection'], candidate_answer=candidate.get('candidate_answer'))
            outcome['think_status'] = record['think_status']
        semantic_verdict = str(outcome['verdict']).upper()
        cpu_verdict = str(outcome.get('cpu_verdict') or semantic_verdict).upper()
        gpu_verdict = str(outcome.get('gpu_verdict') or 'NOT_RUN').upper()
        result = {'schema': RESULT_SCHEMA, 'sample_id': record['sample_id'], 'task_type': record['task_type'], 'verifier_mode': record['verifier_mode'], 'verdict': semantic_verdict, 'cpu_verdict': cpu_verdict, 'gpu_verdict': gpu_verdict, 'verdict_provenance': 'gpu_fallback' if gpu_verdict in {'PASS', 'FAIL'} else 'cpu', 'method': outcome.get('method'), 'outcome': outcome, 'input_sha256': sha256_json(record), 'verifier_config_sha256': self.config_sha256, 'elapsed_s': time.perf_counter() - started}
        return result
