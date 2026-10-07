from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
import string
import sys
import types
from pathlib import Path
from typing import Any, Mapping
from scoring.verification import EXTERNAL_REQUEST_SCHEMA
_EVALUATION_LIB_CACHE: dict[tuple[str, str, str], Any] = {}

class VerificationError(ValueError):
    pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ifbench-repo', type=Path, required=True)
    parser.add_argument('--ifeval-repo', type=Path, required=True)
    parser.add_argument('--nltk-data', type=Path, required=True)
    parser.add_argument('--jsonl-worker', action='store_true')
    return parser.parse_args()

def _json_object(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise VerificationError(f'{name} is not valid JSON') from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise VerificationError(f'{name} must be a JSON object')

def _request_from_stdin() -> dict[str, Any]:
    try:
        value = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise VerificationError('stdin is not valid JSON') from exc
    if not isinstance(value, Mapping):
        raise VerificationError('request must be a JSON object')
    if value.get('schema') != EXTERNAL_REQUEST_SCHEMA:
        raise VerificationError('unsupported external verifier request schema')
    return dict(value)

def _evaluation_lib_path(repo: Path) -> Path:
    path = repo.expanduser().resolve()
    if path.is_file():
        if path.name != 'evaluation_lib.py':
            raise VerificationError(f'expected evaluation_lib.py, got {path}')
        return path
    candidates = (path / 'evaluation_lib.py', path / 'instruction_following_eval' / 'evaluation_lib.py', path / 'ifbench' / 'evaluation_lib.py', path / 'ifeval' / 'evaluation_lib.py')
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise VerificationError(f'evaluation_lib.py not found under {path}')

def _load_evaluation_lib(mode: str, repo: Path) -> Any:
    module_path = _evaluation_lib_path(repo)
    cache_key = (mode, str(module_path.resolve()))
    cached = _EVALUATION_LIB_CACHE.get(cache_key)
    if cached is not None:
        return cached
    search_paths = [module_path.parent, module_path.parent.parent]
    for path in reversed(search_paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    package_name = f'_yanchor_{mode}_official'
    package = types.ModuleType(package_name)
    package.__path__ = [str(module_path.parent)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    module_name = f'{package_name}.evaluation_lib'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise VerificationError(f'cannot import {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in ('InputExample', 'test_instruction_following_strict'):
        if not hasattr(module, name):
            raise VerificationError(f'official evaluation_lib is missing {name}')
    _EVALUATION_LIB_CACHE[cache_key] = module
    return module

def _normalize_kwarg(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize_kwarg(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_kwarg(item) for key, item in value.items() if item is not None}
    return value

def _literal_nonalpha_letter_frequency(candidate: str, kwargs: Mapping[str, Any]) -> bool | None:
    letter = kwargs.get('letter')
    frequency = kwargs.get('let_frequency')
    relation = kwargs.get('let_relation')
    if not isinstance(letter, str) or len(letter) != 1 or letter in string.ascii_letters or (not isinstance(frequency, int)) or isinstance(frequency, bool) or (frequency < 0):
        return None
    count = candidate.casefold().count(letter.casefold())
    if relation == 'at least':
        return count >= frequency
    if relation == 'less than':
        return count < frequency
    return None

def verify(request: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mode = str(request.get('verifier_mode', '')).strip().lower()
    if mode not in {'ifbench', 'ifeval'}:
        raise VerificationError(f'unsupported verifier_mode: {mode}')
    problem = request.get('problem')
    candidate = request.get('candidate')
    if not isinstance(problem, str) or not problem.strip():
        raise VerificationError('problem must contain the original prompt')
    if not isinstance(candidate, str):
        raise VerificationError('candidate must be the visible response text')
    payload = _json_object(request.get('verifier_payload'), name='verifier_payload')
    instruction_ids = payload.get('instruction_id_list')
    kwargs = payload.get('kwargs')
    if not isinstance(instruction_ids, list) or not instruction_ids:
        raise VerificationError('instruction_id_list must be a non-empty list')
    if not all((isinstance(value, str) and value for value in instruction_ids)):
        raise VerificationError('instruction_id_list contains an invalid id')
    if not isinstance(kwargs, list) or len(kwargs) != len(instruction_ids):
        raise VerificationError('kwargs must align one-to-one with instruction_id_list')
    normalized_kwargs = []
    for item in kwargs:
        if not isinstance(item, Mapping):
            raise VerificationError('each kwargs item must be an object')
        normalized_kwargs.append(_normalize_kwarg(item))
    repo = args.ifbench_repo if mode == 'ifbench' else args.ifeval_repo
    nltk_data = args.nltk_data
    if nltk_data is None:
        candidate_nltk = _evaluation_lib_path(repo).parent / '.nltk_data'
        if candidate_nltk.is_dir():
            nltk_data = candidate_nltk
    if nltk_data is not None:
        os.environ['NLTK_DATA'] = str(nltk_data.expanduser().resolve())
    evaluation_lib = _load_evaluation_lib(mode, repo)
    item = evaluation_lib.InputExample(key=payload.get('key', request.get('sample_id', 0)), instruction_id_list=list(instruction_ids), prompt=problem, kwargs=normalized_kwargs)
    outcome = evaluation_lib.test_instruction_following_strict(copy.deepcopy(item), {problem: candidate})
    followed = [bool(value) for value in outcome.follow_instruction_list]
    if len(followed) != len(instruction_ids):
        raise VerificationError('official scorer returned the wrong instruction count')
    literal_overrides = []
    for index, (instruction_id, instruction_kwargs) in enumerate(zip(instruction_ids, normalized_kwargs, strict=True)):
        if instruction_id != 'keywords:letter_frequency':
            continue
        literal_result = _literal_nonalpha_letter_frequency(candidate, instruction_kwargs)
        if literal_result is None:
            continue
        followed[index] = literal_result
        literal_overrides.append(index)
    passed = all(followed)
    return {'verdict': 'PASS' if passed else 'FAIL', 'details': {'method': f'{mode}_official_strict_with_literal_letter_frequency' if literal_overrides else f'{mode}_official_strict_prompt_level', 'instruction_id_list': list(instruction_ids), 'follow_instruction_list': followed, 'follow_all_instructions': passed, 'literal_letter_frequency_indices': literal_overrides}}

def main() -> int:
    result_stream = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    args = parse_args()

    def evaluate(request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return verify(request, args)
        except Exception as exc:
            return {'verdict': 'UNCERTAIN', 'details': {'method': 'instruction_verifier_fail_closed', 'error': f'{type(exc).__name__}: {str(exc)[:1000]}'}}
    if args.jsonl_worker:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, Mapping):
                    raise VerificationError('request must be a JSON object')
                result = evaluate(request)
            except Exception as exc:
                result = evaluate({'verifier_mode': 'invalid', 'error': str(exc)})
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=result_stream, flush=True)
        return 0
    result = evaluate(_request_from_stdin())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=result_stream, flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
