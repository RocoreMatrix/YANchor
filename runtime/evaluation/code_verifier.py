from __future__ import annotations
import functools
import importlib
import json
import multiprocessing
import os
import re
import select
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from scoring.verification import executable_code_payload_contract
_PYTHON_HARNESS_RUNNER = '\nimport builtins\nimport json\nimport os\nimport signal\nimport sys\nimport threading\nimport traceback\n\ndef run():\n    trusted_exec = exec\n    trusted_compile = compile\n    trusted_json_loads = json.loads\n    trusted_readline = sys.stdin.readline\n    trusted_write = os.write\n    trusted_traceback = traceback.print_exc\n    builtins_before = dict(builtins.__dict__)\n    signal_handlers_before = {\n        number: signal.getsignal(number)\n        for number in signal.valid_signals()\n        if isinstance(number, int)\n    }\n    namespace = {"__name__": "candidate", "__file__": sys.argv[1]}\n    try:\n        with open(sys.argv[1], encoding="utf-8") as handle:\n            candidate_code = trusted_compile(handle.read(), "candidate.py", "exec")\n        trusted_exec(candidate_code, namespace)\n    except BaseException:\n        trusted_traceback()\n        raise SystemExit(1)\n\n    builtins_changed = (\n        builtins.__dict__.keys() != builtins_before.keys()\n        or any(\n            builtins.__dict__[name] is not value\n            for name, value in builtins_before.items()\n        )\n    )\n    signal_changed = any(\n        signal.getsignal(number) is not handler\n        for number, handler in signal_handlers_before.items()\n    )\n    if (\n        builtins_changed\n        or signal_changed\n        or sys.gettrace() is not None\n        or sys.getprofile() is not None\n        or len(threading.enumerate()) != 1\n    ):\n        raise SystemExit(91)\n\n    trusted_write(1, ("__Q35_CANDIDATE_READY__" + sys.argv[2] + "\\n").encode())\n    payload = trusted_json_loads(trusted_readline())\n    harness_code = trusted_compile(payload["harness"], "hidden_tests.py", "exec")\n    try:\n        trusted_exec(harness_code, namespace)\n    except BaseException:\n        trusted_traceback()\n        raise SystemExit(1)\n    trusted_write(\n        1,\n        ("__Q35_HARNESS_COMPLETE__" + payload["completion_token"] + "\\n").encode(),\n    )\n\nrun()\n'

def normalized_output(value: str) -> str:
    return '\n'.join((line.rstrip() for line in value.strip().splitlines()))

def output_matches(actual: str, expected: str, checker: str) -> bool:
    if checker == 'tokens':
        return actual.split() == expected.split()
    return normalized_output(actual) == normalized_output(expected)

def run_process(argv: list[str], *, cwd: Path, stdin: str, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, input=stdin, text=True, capture_output=True, timeout=timeout, check=False)

def run_python_harness(candidate: str, harness: str, *, cwd: Path, timeout: float) -> tuple[subprocess.CompletedProcess[str], bool]:
    code_path = cwd / 'candidate.py'
    code_path.write_text(candidate, encoding='utf-8')
    ready_token = secrets.token_hex(32)
    completion_token = secrets.token_hex(32)
    argv = [sys.executable, '-I', '-S', '-c', _PYTHON_HARNESS_RUNNER, str(code_path), ready_token]
    process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert process.stdout is not None
    deadline = time.monotonic() + timeout
    stdout_prefix = bytearray()
    stderr_prefix = bytearray()
    pending = bytearray()
    streams = [process.stdout, process.stderr]
    ready_marker = f'__Q35_CANDIDATE_READY__{ready_token}'.encode()
    ready = False
    try:
        while not ready and process.stdout in streams:
            while b'\n' in pending:
                line, _, tail = pending.partition(b'\n')
                pending = bytearray(tail)
                if line.rstrip(b'\r') == ready_marker:
                    ready = True
                    break
            if ready:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            readable, _, _ = select.select(streams, [], [], remaining)
            if not readable:
                raise subprocess.TimeoutExpired(argv, timeout)
            for stream in readable:
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    streams.remove(stream)
                elif stream is process.stdout:
                    stdout_prefix.extend(chunk)
                    pending.extend(chunk)
                else:
                    stderr_prefix.extend(chunk)
            if len(stdout_prefix) > 1000000:
                process.kill()
                break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(argv, timeout)
        stdout, stderr = process.communicate(input=(json.dumps({'harness': harness, 'completion_token': completion_token}) + '\n').encode() if ready else None, timeout=remaining)
    except BaseException:
        process.kill()
        process.communicate()
        raise
    completed = subprocess.CompletedProcess(argv, process.returncode, (stdout_prefix + stdout).decode('utf-8', errors='replace'), (stderr_prefix + stderr).decode('utf-8', errors='replace'))
    marker = f'__Q35_HARNESS_COMPLETE__{completion_token}'
    return (completed, ready and marker in completed.stdout.splitlines())

def select_runtime(language: str, code_path: Path, root: Path) -> tuple[list[str], dict[str, Any]]:
    if language in {'python', 'py', 'python3', ''}:
        return ([sys.executable, '-I', str(code_path)], {'language': 'python'})
    if language in {'cpp', 'c++', 'cxx', 'cc'}:
        compiler = shutil.which('g++')
        if compiler is None:
            raise FileNotFoundError('g++ is unavailable')
        executable = root / 'candidate'
        compiled = subprocess.run([compiler, '-O2', '-std=c++17', str(code_path), '-o', str(executable)], cwd=root, text=True, capture_output=True, timeout=30, check=False)
        if compiled.returncode != 0:
            return ([], {'language': 'cpp', 'compile_failed': True})
        return ([str(executable)], {'language': 'cpp', 'compile_failed': False})
    raise ValueError(f'unsupported language: {language}')

@functools.lru_cache(maxsize=1)
def livecodebench_runtime():
    runner = Path(__file__).resolve().parents[1] / 'third_party/official_evaluators/LiveCodeBench'
    sys.path.insert(0, str(runner))
    benchmark = importlib.import_module('lcb_runner.benchmarks.code_generation')
    evaluation = importlib.import_module('lcb_runner.evaluation.compute_code_generation_metrics')
    return (benchmark.CodeGenerationProblem, functools.partial(check_livecodebench_correctness, evaluation._temp_run))

def check_livecodebench_correctness(run_test, sample, generation, *, timeout, debug=False):
    cases = len(json.loads(sample['input_output'])['inputs'])
    budget = (timeout + 1) * cases + 5
    with multiprocessing.Manager() as manager:
        results, metadata = (manager.list(), manager.list())
        process = multiprocessing.Process(target=run_test, args=(sample, generation, debug, results, metadata, timeout))
        process.start()
        try:
            process.join(timeout=budget)
            global_timeout = process.is_alive()
            if global_timeout:
                process.kill()
            process.join()
            facts = {'exitcode': process.exitcode, 'global_timeout': global_timeout, 'global_timeout_seconds': budget, 'missing_result': not bool(results), 'missing_metadata': not bool(metadata), 'scorer_error': bool(process.exitcode and (not global_timeout))}
            return (results[0] if results else None, metadata[0] if metadata else None, facts)
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            process.close()

def run_livecodebench(candidate: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    problem_class, check_correctness = livecodebench_runtime()
    raw = json.loads(Path(str(payload['lcb_payload_path'])).read_text(encoding='utf-8'))
    problem = problem_class(**raw)
    started = time.perf_counter()
    results, metadata, facts = check_correctness(problem.get_evaluation_sample(), candidate, timeout=timeout, debug=False)
    passed = [bool(value == True) for value in results or []]
    first_failed = next((index for index, value in enumerate(passed) if not value), None)
    if facts['global_timeout']:
        verdict = 'FAIL'
    elif facts['missing_result'] or facts['missing_metadata'] or facts['scorer_error']:
        verdict = 'UNCERTAIN'
    else:
        verdict = 'PASS' if passed and all(passed) else 'FAIL'
    return {'verdict': verdict, 'details': {**facts, 'language': 'python', 'livecodebench': True, 'cases': len(passed), 'first_failed_case': first_failed, 'elapsed_seconds': time.perf_counter() - started, 'official_metadata_type': type(metadata).__name__}}

def verify(request: Mapping[str, Any]) -> dict[str, Any]:
    code = request.get('candidate')
    if not isinstance(code, str) or not code.strip():
        return {'verdict': 'FAIL', 'details': 'empty code submission'}
    payload = request.get('tests') if 'tests' in request else request.get('verifier_payload')
    contract = executable_code_payload_contract(payload, language=request.get('language'))
    if not contract['eligible']:
        return {'verdict': 'UNCERTAIN', 'details': {'method': 'code_payload_contract_rejected', 'reason': contract['reason']}}
    tests = {'cases': payload} if isinstance(payload, list) else payload
    assert isinstance(tests, Mapping)
    timeout = min(max(int(float(tests.get('time_limit_s', 2))), 1), 30)
    checker = str(tests.get('checker', 'exact')).strip().lower()
    language = str(request.get('language') or tests.get('language') or 'python').strip().lower()
    if contract['mode'] == 'livecodebench':
        return run_livecodebench(code, tests, timeout)
    with tempfile.TemporaryDirectory(prefix='yanchor-code-verifier-') as temporary:
        root = Path(temporary)
        suffix = '.cpp' if language in {'cpp', 'c++', 'cxx', 'cc'} else '.py'
        code_path = root / f'candidate{suffix}'
        source = code
        if contract['mode'] == 'python_harness':
            harness = tests['python_harness']
            assert isinstance(harness, str)
            try:
                completed, harness_completed = run_python_harness(code, harness, cwd=root, timeout=timeout)
            except subprocess.TimeoutExpired:
                return {'verdict': 'FAIL', 'details': {'language': 'python', 'harness': True, 'timeout': True}}
            return {'verdict': 'PASS' if completed.returncode == 0 and harness_completed else 'UNCERTAIN' if 'ModuleNotFoundError:' in completed.stderr else 'FAIL', 'details': {'language': 'python', 'harness': True, 'harness_completed': harness_completed, 'returncode': completed.returncode, 'missing_modules': re.findall("ModuleNotFoundError: No module named '([^']+)'", completed.stderr)}}
        code_path.write_text(source, encoding='utf-8')
        try:
            argv, runtime = select_runtime(language, code_path, root)
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired) as exc:
            return {'verdict': 'UNCERTAIN', 'details': f'runtime unavailable: {type(exc).__name__}: {exc}'}
        if runtime.get('compile_failed'):
            return {'verdict': 'FAIL', 'details': runtime}
        cases = tests.get('cases')
        assert isinstance(cases, list) and cases
        case_results = []
        for index, case in enumerate(cases):
            assert isinstance(case, Mapping)
            stdin = str(case.get('stdin', ''))
            expected = case.get('stdout', case.get('expected_stdout'))
            assert isinstance(expected, str)
            try:
                completed = run_process(argv, cwd=root, stdin=stdin, timeout=timeout)
            except subprocess.TimeoutExpired:
                return {'verdict': 'FAIL', 'details': {**runtime, 'case': index, 'timeout': True}}
            passed = completed.returncode == 0 and output_matches(completed.stdout, expected, checker)
            case_results.append({'index': index, 'passed': passed, 'returncode': completed.returncode})
            if not passed:
                return {'verdict': 'FAIL', 'details': {**runtime, 'cases': case_results}}
        return {'verdict': 'PASS', 'details': {**runtime, 'cases': case_results}}

def main() -> int:
    result_stream = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    if '--jsonl-worker' in sys.argv[1:]:
        for line in sys.stdin:
            try:
                result = verify(json.loads(line))
            except Exception as exc:
                result = {'verdict': 'UNCERTAIN', 'details': {'method': 'code_worker_fail_closed', 'error': f'{type(exc).__name__}: {str(exc)[:1000]}'}}
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=result_stream, flush=True)
        return 0
    print(json.dumps(verify(json.load(sys.stdin)), ensure_ascii=False, sort_keys=True), file=result_stream, flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
