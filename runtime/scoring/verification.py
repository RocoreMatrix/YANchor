from __future__ import annotations
import ast
import hashlib
import json
import math
import string
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence
import re
import unicodedata
from fractions import Fraction
from typing import Any, Sequence
RESULT_SCHEMA = 'yanchor-verdict-v1'
EXTERNAL_REQUEST_SCHEMA = 'yanchor-verifier-request-v1'
TASK_TYPES = frozenset({'math', 'choice', 'code', 'stem', 'general', 'retrieval', 'ifeval', 'ifbench'})
VERDICTS = frozenset({'PASS', 'FAIL', 'UNCERTAIN'})
CPU_VERIFIER_REVISION = 'cpu_first_typed_tristate_final_answer_equivalence_v41'
CODE_VERIFIER_MODES = frozenset({'python_harness', 'code'})
INSTRUCTION_VERIFIER_MODES = frozenset({'ifbench', 'ifeval'})
MATH_EXTERNAL_VERIFIER_MODES = frozenset({'math_equivalence', 'symbolic', 'expression', 'numeric_equivalence', 'set'})
DIRECT_REFERENCE_VERIFIER_MODES = frozenset({'numeric_equivalence', 'numeric', 'integer', 'fraction', 'decimal', 'exact_choice', 'choice', 'choice_set', 'exact_text', 'exact', 'text', 'set', 'symbolic', 'expression', 'drop_exact', 'drop_official', 'bbeh_official', 'python_literal_output'})

def _unwrap_complete_latex_reference(value: str) -> str:
    text = value.strip().strip('$').strip()
    for _ in range(4):
        previous = text
        for command in ('boxed', 'fbox', 'text', 'mbox', 'mathrm'):
            prefix = f'\\{command}{{'
            if not text.startswith(prefix) or not text.endswith('}'):
                continue
            depth = 0
            closing = -1
            for index, character in enumerate(text[len(prefix) - 1:], len(prefix) - 1):
                if character == '{':
                    depth += 1
                elif character == '}':
                    depth -= 1
                    if depth == 0:
                        closing = index
                        break
            if closing == len(text) - 1:
                text = text[len(prefix):-1].strip()
                break
        if text == previous:
            break
    return text

def _choice_reference_labels(reference: str) -> tuple[str, ...] | None:
    text = _unwrap_complete_latex_reference(reference)
    if len(text) > 2 and (text[0], text[-1]) in {('(', ')'), ('[', ']'), ('{', '}')}:
        text = text[1:-1].strip()
    if re.fullmatch('[A-Z]{2,26}', text):
        return tuple(text)
    text = text.replace('\\,', ' ')
    text = re.sub('\\band\\b|[;，、和及]', ',', text, flags=re.I)
    fields = [field for field in re.split('\\s*,\\s*', text) if field.strip()]
    labels = [choice_value(field) for field in fields]
    if not labels or any((label is None for label in labels)):
        return None
    return tuple((str(label) for label in labels))

def choice_set_value(value: Any) -> frozenset[str] | None:
    labels = _choice_reference_labels(str(value))
    if not labels or len(labels) < 2 or len(set(labels)) != len(labels):
        return None
    return frozenset(labels)

def _choice_problem_labels(problem: str) -> set[str]:
    labels: set[str] = set()
    patterns = ('(?<![A-Za-z])\\(([A-Z])\\)(?![A-Za-z])', '(?m)^\\s*(?:[-*•]|\\\\bullet)?\\s*\\$?\\s*([A-Z])[.)]\\s+', '\\\\textbf\\s*\\{\\s*\\(?([A-Z])\\)?\\s*\\}')
    for pattern in patterns:
        labels.update(re.findall(pattern, problem))
    instruction = re.search('(?is)(?:enter|write|give|list|answer).{0,120}(?:letter|choice|option)', problem)
    if instruction:
        window = problem[instruction.start():instruction.end() + 160]
        labels.update(re.findall('(?<![A-Za-z])([A-Z])(?![A-Za-z])', window))
    return labels
_CHOICE_OPTION_MARKER = re.compile('\\\\text\\s*\\{\\s*\\((?P<latex>[A-Z])\\)\\s*\\}|(?<![A-Za-z])\\((?P<paren>[A-Z])\\)(?![A-Za-z])|^[ \\t]*(?P<line>[A-Z])[.):：]\\s+', re.MULTILINE)

def choice_option_values(problem: str) -> dict[str, str] | None:
    question_starts = list(re.finditer('(?mi)^[ \\t]*(?:Q|Question|问题|题目)\\s*[:：]', problem))
    if question_starts:
        problem = problem[question_starts[-1].end():]
    headings = list(re.finditer('(?mi)^[ \\t]*(?:Options(?:\\s+are)?|Choices|选项)\\s*[:：]', problem))
    if headings:
        problem = problem[headings[-1].end():]
    tail = re.search("(?mi)^[ \\t]*(?:(?:give|enter|write|put|provide|return)\\b[^\\n]*(?:answer|letter|option|final|boxed)|reason through[^\\n]*finish|最后一行[^\\n]*答案|(?:请)?(?:在|将)[^\\n]*(?:答案|选项)|A:\\s*Let's think)", problem)
    if tail:
        problem = problem[:tail.start()]
    matches = list(_CHOICE_OPTION_MARKER.finditer(problem))
    line_matches = [match for match in matches if not problem[problem.rfind('\n', 0, match.start()) + 1:match.start()].strip()]
    if len(line_matches) >= 2 and any(('A' in match.groupdict().values() for match in line_matches)):
        matches = line_matches
    markers = []
    for match in matches:
        label = next((value for value in match.groupdict().values() if value))
        markers.append((match.start(), match.end(), label.upper()))
    labels = [label for _, _, label in markers]
    if len(markers) < 2 or 'A' not in labels or len(labels) != len(set(labels)):
        return None
    options = {}
    for index, (_, end, label) in enumerate(markers):
        next_start = markers[index + 1][0] if index + 1 < len(markers) else len(problem)
        body = problem[end:next_start].strip().lstrip(':：').strip()
        body = re.sub('^(?:\\$+|\\\\(?:quad|qquad|enspace)|\\\\[ ,;!])+', '', body)
        body = re.sub('(?:\\$+|\\\\(?:quad|qquad|enspace)|\\\\[ ,;!])+\\s*$', '', body)
        normalized = _choice_content_text(body)
        if not normalized:
            return None
        options[label] = normalized
    return options

def compare_choice_option_content(candidate: Any, references: Sequence[Any], problem: str) -> dict[str, Any] | None:
    candidate_label = choice_value(candidate)
    options = choice_option_values(problem)
    if candidate_label is None or options is None or candidate_label not in options or any((choice_value(reference) is not None for reference in references)):
        return None
    reference_labels = {label for label, option in options.items() for reference in references if option == normalize_answer_text(reference)}
    if len(reference_labels) != 1:
        return None
    return {'verdict': 'PASS' if candidate_label in reference_labels else 'FAIL', 'method': 'choice_label_to_signed_option_content', 'candidate': candidate_label, 'references': sorted(reference_labels)}

def _row_reference_values(row: Mapping[str, Any], extra: Mapping[str, Any]) -> list[Any]:
    values = extra.get('reference_answers')
    if values is not None:
        if not isinstance(values, list):
            return []
        return [value for value in values if str(value).strip()]
    value = row.get('ground_truth')
    return [value] if value is not None and str(value).strip() else []

def executable_code_payload_contract(payload: Any, *, language: Any=None) -> dict[str, Any]:
    if isinstance(payload, list):
        cases = payload
        config: Mapping[str, Any] = {}
    elif isinstance(payload, Mapping):
        config = payload
        cases = payload.get('cases')
    else:
        return {'eligible': False, 'reason': 'code_verifier_payload_missing'}
    checker = str(config.get('checker', 'exact')).strip().lower()
    if checker not in {'exact', 'tokens'}:
        return {'eligible': False, 'reason': 'code_verifier_checker_invalid'}
    raw_timeout = config.get('time_limit_s', 2.0)
    try:
        timeout = math.nan if isinstance(raw_timeout, bool) else float(raw_timeout)
    except (TypeError, ValueError):
        timeout = math.nan
    if not math.isfinite(timeout) or timeout <= 0.0:
        return {'eligible': False, 'reason': 'code_verifier_timeout_invalid'}
    runtime = str(language or config.get('language') or 'python').strip().lower()
    python_runtime = runtime in {'', 'python', 'py', 'python3'}
    if not python_runtime and runtime not in {'cpp', 'c++', 'cxx', 'cc'}:
        return {'eligible': False, 'reason': 'code_verifier_language_invalid'}
    if config.get('lcb_payload_path') and config.get('lcb_runner_path'):
        if not python_runtime:
            return {'eligible': False, 'reason': 'livecodebench_python_required'}
        return {'eligible': True, 'reason': 'livecodebench_official_payload', 'mode': 'livecodebench'}
    harness = config.get('python_harness')
    if harness is not None:
        if not python_runtime or not isinstance(harness, str) or (not harness.strip()):
            return {'eligible': False, 'reason': 'code_verifier_python_harness_invalid'}
        return {'eligible': True, 'reason': 'code_verifier_python_harness', 'mode': 'python_harness'}
    if not isinstance(cases, list) or not cases:
        return {'eligible': False, 'reason': 'code_verifier_cases_missing'}
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get('stdout', case.get('expected_stdout')), str) or ('stdin' in case and (not isinstance(case['stdin'], str))):
            return {'eligible': False, 'reason': 'code_verifier_case_invalid'}
    return {'eligible': True, 'reason': 'code_verifier_cases', 'mode': 'cases'}

def bound_tool_verdict_contract(value: Any) -> dict[str, Any]:
    if value is None:
        return {'eligible': True, 'reason': 'absent'}
    if not isinstance(value, Mapping):
        return {'eligible': False, 'reason': 'code_tool_verdict_invalid'}
    hashes = (value.get('candidate_sha256'), value.get('tests_sha256'))
    if any((not isinstance(item, str) or len(item) != 64 or any((character not in '0123456789abcdef' for character in item)) for item in hashes)) or str(value.get('verdict') or '').upper() not in VERDICTS:
        return {'eligible': False, 'reason': 'code_tool_verdict_invalid'}
    return {'eligible': True, 'reason': 'bound_tool_verdict'}

def deterministic_verifier_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    extra = row['extra_info']
    mode = 'math_equivalence'
    answer_type = ''
    references = _row_reference_values(row, extra)
    if not references or any((len(str(reference)) > 4096 for reference in references)):
        return {'eligible': False, 'reason': 'math_reference_empty_or_too_long', 'verifier_mode': mode, 'answer_type': answer_type, 'resolution': 'rejected'}
    problem = str(extra.get('problem') or '')
    if not problem:
        prompt = row.get('prompt') or []
        if isinstance(prompt, list) and prompt and isinstance(prompt[-1], Mapping):
            problem = str(prompt[-1].get('content') or '')
    choice_labels = [_choice_reference_labels(str(reference)) for reference in references]
    problem_labels = _choice_problem_labels(problem)
    if all(choice_labels) and len(problem_labels) >= 2 and all((set(labels or ()).issubset(problem_labels) for labels in choice_labels)) and (len({len(labels or ()) for labels in choice_labels}) == 1):
        single = len(choice_labels[0] or ()) == 1
        return {'eligible': True, 'reason': 'math_reference_routed_exact_choice' if single else 'math_reference_routed_choice_set', 'verifier_mode': 'exact_choice' if single else 'choice_set', 'answer_type': 'choice' if single else 'choice_set', 'resolution': 'choice_metadata_rebound'}
    unwrapped_references = [_unwrap_complete_latex_reference(str(reference)) for reference in references]
    fixed_text = all((2 < len(unwrapped) <= 128 and re.fullmatch("[A-Za-z\\u3400-\\u9fff][A-Za-z\\u3400-\\u9fff '\\u2019.,，。]*", unwrapped) and (not re.search('\\b(?:if|otherwise|when|where|such\\s+that)\\b|(?:如果|否则|当|其中|满足|时)', unwrapped, re.I)) for unwrapped in unwrapped_references))
    if fixed_text:
        return {'eligible': True, 'reason': 'math_reference_routed_exact_text', 'verifier_mode': 'exact_text', 'answer_type': 'exact_text', 'resolution': 'fixed_text_metadata_rebound'}
    exact_text_fallback = all((re.search('\\\\begin\\{(?:cases|array)\\}|\\\\(?:text|mbox|mathrm)\\{|\\b(?:if|otherwise|when|where|such\\s+that)\\b|\\d\\s*\\\\?[!:：]\\\\?!?\\s*\\d', str(reference), re.I) for reference in references))
    if exact_text_fallback:
        return {'eligible': True, 'reason': 'math_reference_routed_exact_text_fallback', 'verifier_mode': 'exact_text', 'answer_type': 'exact_text', 'resolution': 'unparsed_reference_exact_text'}
    if all((symbolic_reference_is_supported(reference, problem=problem) for reference in references)):
        return {'eligible': True, 'reason': 'math_equivalence_symbolic', 'verifier_mode': 'math_equivalence', 'answer_type': answer_type or 'symbolic', 'resolution': 'symbolic_reference'}
    return {'eligible': False, 'reason': 'math_reference_not_supported_by_symbolic_verifier', 'verifier_mode': mode, 'answer_type': answer_type, 'resolution': 'rejected'}

class ContractError(ValueError):
    pass

def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()

def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def validate_request(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ContractError('request must be a JSON object')
    sample_id = str(record.get('sample_id', '')).strip()
    task_type = str(record.get('task_type', '')).strip().lower()
    problem = record.get('problem')
    response = record.get('response')
    if not sample_id:
        raise ContractError('sample_id is required')
    if task_type not in TASK_TYPES:
        raise ContractError(f'unsupported task_type: {task_type}')
    if not isinstance(problem, str) or not problem.strip():
        raise ContractError('problem must be non-empty text')
    if not isinstance(response, str):
        raise ContractError('response must be text')
    verifier_mode = str(record.get('verifier_mode') or ('code' if task_type == 'code' else 'exact' if task_type == 'choice' else 'semantic' if task_type == 'general' else 'auto')).lower()
    source_dataset = str(record.get('source_dataset') or '').strip().casefold()
    effective_drop = verifier_mode == 'drop_official' or (verifier_mode in {'exact', 'exact_text', 'text'} and (source_dataset == 'drop' or source_dataset.startswith('drop_')))
    if (task_type == 'code') != (verifier_mode in CODE_VERIFIER_MODES):
        raise ContractError('code_task_verifier_mode_mismatch')
    if task_type != 'code' and (not (any((key in record for key in ('reference_answer', 'reference_answers', 'rubric'))) or (effective_drop and bool(drop_answer_sets(record))))):
        raise ContractError('non-code verification requires reference_answer, reference_answers, or rubric')
    if task_type == 'code' and (not any((key in record for key in ('tests', 'verifier_payload', 'tool_verdict')))):
        raise ContractError('code verification requires tests, verifier_payload, or a bound tool_verdict')
    if task_type == 'code':
        executable = executable_code_payload_contract(record.get('tests', record.get('verifier_payload')), language=record.get('language'))
        if not executable['eligible']:
            raise ContractError(str(executable['reason']))
        bound = bound_tool_verdict_contract(record.get('tool_verdict'))
        if not bound['eligible']:
            raise ContractError(str(bound['reason']))
    value = dict(record)
    value['sample_id'] = sample_id
    value['task_type'] = task_type
    value['problem'] = problem.strip()
    value['response'] = response
    value['verifier_mode'] = verifier_mode
    return value

def _unwrap_latex_command(value: str, command: str) -> str:
    prefix = f'\\{command}{{'
    text = value.strip()
    if not text.startswith(prefix) or not text.endswith('}'):
        return text
    depth = 0
    for index, character in enumerate(text[len(prefix) - 1:], len(prefix) - 1):
        if character == '{':
            depth += 1
        elif character == '}':
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return text
    return text[len(prefix):-1].strip() if depth == 0 else text

def normalize_answer_text(value: Any) -> str:
    text = unicodedata.normalize('NFKC', str(value)).strip()
    text = re.sub('^\\s*(?:answer|答案|final\\s+answer)\\s*[:：]\\s*', '', text, flags=re.I)
    for _ in range(8):
        previous = text
        text = text.strip().rstrip('。.;；').strip()
        if len(text) > 4 and text.startswith('$$') and text.endswith('$$'):
            text = text[2:-2].strip()
        elif len(text) > 2 and text.startswith('$') and text.endswith('$'):
            text = text[1:-1].strip()
        elif len(text) > 4 and text.startswith('\\(') and text.endswith('\\)'):
            text = text[2:-2].strip()
        elif len(text) > 4 and text.startswith('\\[') and text.endswith('\\]'):
            text = text[2:-2].strip()
        for command in ('boxed', 'fbox', 'text', 'mbox', 'mathrm'):
            text = _unwrap_latex_command(text, command)
        if text == previous:
            break
    text = text.replace('\\left', '').replace('\\right', '')
    text = text.replace('\\{', '{').replace('\\}', '}')
    text = text.replace('−', '-').replace('–', '-')
    text = re.sub('^([+-]?\\d+)\\s+(\\d+\\s*/\\s*\\d+)(\\\\?%)?$', '\\1又\\2\\3', text)
    text = re.sub('\\s+', '', text)
    text = _normalize_symbolic_number_format(text)
    text = text.rstrip('。.;；')
    return text.casefold()

def _normalize_symbolic_number_format(text: str) -> str:
    text = re.sub('^\\\\\\\\?\\$(?=[+-]?(?:\\d|\\.\\d))', '', text)
    compact = re.sub('\\s+', '', text)
    if re.fullmatch('[+-]?\\d{1,3}(?:(?:,|\\{,\\}|,\\\\[!,; ])\\d{3})+(?:\\.\\d+)?(?:\\\\?%)?', compact):
        text = re.sub('\\{,\\}|,\\\\[!,; ]|,', '', compact)
    return text

def symbolic_math_text(value: Any) -> str:
    value = re.sub('\\\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle|[Bb]igg?[lrm]?)(?![A-Za-z])', '', str(value))
    value = re.sub('\\\\sqrt\\s*([0-9A-Za-z])(?![A-Za-z])', '\\\\sqrt{\\1}', value)
    value = re.sub('\\\\log_\\s*(?:\\{([^{}]+)\\}|([0-9A-Za-z]))\\s*(?:\\{([^{}]+)\\}|\\(([^()]*)\\)|([0-9A-Za-z]+))', lambda m: '\\log(' + next((v for v in m.groups()[2:] if v is not None)) + ')/\\log(' + (m.group(1) or m.group(2)) + ')', value)
    text = normalize_answer_text(value)
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    text = re.sub('\\\\frac\\s*\\{([^{}]+)\\}\\s*([0-9A-Za-z])', '\\\\frac{\\1}{\\2}', text)
    text = re.sub('\\\\frac\\s*([0-9A-Za-z])\\s*\\{([^{}]+)\\}', '\\\\frac{\\1}{\\2}', text)
    text = re.sub('\\\\frac\\s*([0-9A-Za-z])\\s*([0-9A-Za-z])', '\\\\frac{\\1}{\\2}', text)
    text = re.sub('\\\\sqrt\\s*([0-9A-Za-z])', '\\\\sqrt{\\1}', text)
    text = re.sub('(?<=[0-9A-Za-z})])(?:\\^\\{?\\\\circ\\}?|°)', '', text)
    text = re.sub('(?<=[0-9})])\\\\(?:text|mbox|mathrm)\\{[A-Za-z\\\\.\\s]{1,32}\\}\\s*$', '', text)
    text = re.sub('([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))\\\\?%', '(\\1/100)', text)
    text = re.sub('\\\\(?:quad|qquad|enspace|thinspace)', '', text)
    text = re.sub('\\\\[ ,;!]', '', text)
    text = text.replace('\\cdot', '*').replace('\\times', '*')
    text = text.replace('\\pi', 'pi').replace('\\infty', 'oo')
    text = text.replace('\\leq', '<=').replace('\\le', '<=')
    text = text.replace('\\geq', '>=').replace('\\ge', '>=')
    text = text.replace('\\cup', '|')
    text = text.replace('\\lfloor', 'floor(').replace('\\rfloor', ')')
    for name in ('sin', 'cos', 'tan', 'csc', 'arccos', 'arctan', 'log', 'ln', 'exp', 'alpha', 'beta', 'gamma', 'delta', 'theta', 'lambda', 'mu', 'phi', 'psi', 'omega', 'rho'):
        text = text.replace(f'\\{name}', name)
    text = re.sub('\\\\(?:operatorname|mathrm|text|mbox)\\{([^{}]*)\\}', '\\1', text)
    text = re.sub('_\\{([^{}]+)\\}', '_\\1', text)
    text = re.sub('\\^\\{([^{}]+)\\}', '^(\\1)', text)
    text = re.sub('log_(\\d+|[a-z])\\(([^()]*)\\)', '(log(\\2)/log(\\1))', text)
    text = re.sub('log_(\\d+|[a-z])\\{([^{}]*)\\}', '(log(\\2)/log(\\1))', text)
    fraction = re.compile('\\\\frac\\{([^{}]+)\\}\\{([^{}]+)\\}')
    square_root = re.compile('\\\\sqrt\\{([^{}]+)\\}')
    indexed_root = re.compile('\\\\sqrt\\[([^\\[\\]]+)\\]\\{([^{}]+)\\}')
    for _ in range(32):
        updated = indexed_root.sub('root(\\2,\\1)', text)
        updated = square_root.sub('sqrt(\\1)', updated)
        updated = fraction.sub('((\\1)/(\\2))', updated)
        if updated == text:
            break
        text = updated
    factorial = re.compile('(?P<base>(?:\\d+|[a-zA-Z_]\\w*|\\([^()]+\\)))!')
    for _ in range(8):
        updated = factorial.sub('factorial(\\g<base>)', text)
        if updated == text:
            break
        text = updated
    text = text.replace('{', '(').replace('}', ')')
    text = re.sub('(?<=[0-9a-zA-Z_)])(?=(?:sqrt|factorial|pi|oo)(?:\\b|\\())', '*', text)
    text = re.sub('(?<=\\))(?=[0-9a-zA-Z])', '*', text)
    text = re.sub('\\b(sin|cos|tan|csc|arccos|arctan|log|ln)(?=\\d)', '\\1*', text)
    return text

def split_symbolic_top_level(text: str) -> list[str] | None:
    fields: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for index, character in enumerate(text):
        if character in '([{':
            stack.append(character)
        elif character in ')]}':
            if not stack or stack[-1] != pairs[character]:
                return None
            stack.pop()
        elif character in ',;' and (not stack):
            fields.append(text[start:index].strip())
            start = index + 1
    if stack:
        return None
    fields.append(text[start:].strip())
    return fields if len(fields) > 1 and all(fields) else None

def symbolic_solution_context(problem: str) -> bool:
    return bool(re.search('\\b(?:solve|solutions?|roots?|all\\s+(?:(?:real|complex|possible|integer)\\s+)*values)\\b|\\b(?:what|which|find)\\s+(?:(?:real|complex|possible|integer)\\s+)*values\\s+of\\b|解集|所有.{0,12}(?:解|根|值)|求.{0,12}(?:解|根)', problem, re.I)) and (not re.search('\\b(?:uncertainty|measurement|tolerance)\\b|margin of error|误差|测量', problem, re.I))

def symbolic_structured_values(value: Any, *, problem: str='', interval: bool=False) -> tuple[str, list[str]] | None:
    text = _normalize_symbolic_number_format(_unwrap_complete_latex_reference(str(value)))
    text = text.replace('\\left', '').replace('\\right', '')
    chain = re.fullmatch('\\s*(.+?)\\s*(<=|>=|\\\\leq?|\\\\geq?|[<>≤≥])\\s*(?:[A-Za-z]|\\\\[A-Za-z]+)\\s*(<=|>=|\\\\leq?|\\\\geq?|[<>≤≥])\\s*(.+?)\\s*', text)
    if chain and numeric_value(chain[1]) is not None and (numeric_value(chain[4]) is not None):
        lower, left, right, upper = chain.groups()
        left, right = [re.sub('\\\\leq?', '≤', re.sub('\\\\geq?', '≥', operator)) for operator in (left, right)]
        if left in {'<', '<=', '≤'} and right in {'<', '<=', '≤'}:
            return ('interval:' + ('(' if left == '<' else '[') + (')' if right == '<' else ']'), [lower, upper])
        if left in {'>', '>=', '≥'} and right in {'>', '>=', '≥'}:
            return ('interval:' + ('(' if right == '>' else '[') + (')' if left == '>' else ']'), [upper, lower])
    membership = re.fullmatch('\\s*(?:[A-Za-z](?:_\\{?[A-Za-z0-9]+\\}?)?|[A-Za-z]\\s*/\\s*[A-Za-z]|\\\\frac\\{[A-Za-z]\\}\\{[A-Za-z]\\})\\s*(?:\\\\in|∈)\\s*(.+)', text)
    if membership:
        text = membership.group(1)
        interval = text.startswith(('(', '['))
    text = re.sub('\\\\{2,}\\s*', '', text)
    text = re.sub('\\\\(?:text|mathrm|mbox)\\{\\s*(?:or|或)\\s*\\}', ',', text, flags=re.I)
    text = re.sub('(?<![A-Za-z])(?:or|或)(?![A-Za-z])', ',', text, flags=re.I)
    text = re.sub('\\\\(?:quad|qquad|enspace|thinspace)', '', text)
    text = re.sub('\\\\[ ,;!]', '', text)
    text = re.sub('\\s*[,;](?:\\s*[,;])+\\s*', ',', text)
    text = text.strip()
    if text in {'\\emptyset', '\\varnothing', '∅', '{}', '\\{\\}'}:
        return ('set', [])
    solution_context = symbolic_solution_context(problem)
    if solution_context and re.fullmatch('none|(?:there (?:is|are) )?no (?:real |complex )?(?:solutions?|roots?)|无解|没有解|解集为空', text, re.I):
        return ('set', [])
    if len(text) >= 3 and text[0] in '([' and (text[-1] in ')]'):
        bounds = text[0] + text[-1]
        if interval or bounds in {'(]', '[)'} or re.search('\\binterval(?:s| notation)?\\b|区间', problem, re.I):
            fields = split_symbolic_top_level(text[1:-1])
            if fields and len(fields) == 2:
                return ('interval:' + bounds, fields)
    for left, right, kind in (('(', ')', 'sequence'), ('[', ']', 'sequence'), ('\\{', '\\}', 'set'), ('{', '}', 'set')):
        if text.startswith(left) and text.endswith(right):
            body = text[len(left):-len(right)].strip()
            fields = split_symbolic_top_level(body)
            if fields is None and kind == 'set' and body and (',' not in body) and (';' not in body):
                fields = [body]
            if fields:
                if kind == 'set':
                    expanded = []
                    for field in fields:
                        token = '\\pm' if field.count('\\pm') == 1 else '±'
                        if field.count(token) == 1 and '\\mp' not in field and ('∓' not in field):
                            expanded.extend((field.replace(token, '+'), field.replace(token, '-')))
                        else:
                            expanded.append(field)
                    fields = expanded
                return (kind, fields)
    token = '\\pm' if text.count('\\pm') == 1 else '±'
    if solution_context and text.count(token) == 1 and ('\\mp' not in text) and ('∓' not in text):
        return ('set', [text.replace(token, '+'), text.replace(token, '-')])
    fields = split_symbolic_top_level(text)
    return ('set', fields) if fields else None

def symbolic_matrix_values(value: Any) -> list[list[str]] | None:
    text = _unwrap_complete_latex_reference(str(value))
    text = text.replace('\\left', '').replace('\\right', '').strip()
    matched = re.fullmatch('\\\\begin\\{(?P<environment>[pbvBV]?matrix)\\}(?P<body>[\\s\\S]*)\\\\end\\{(?P=environment)\\}', text)
    if matched is None:
        return None
    rows = re.split('\\\\\\\\(?:\\[[^\\]]*\\])?', matched.group('body'))
    matrix = [[cell.strip() for cell in row.split('&')] for row in rows]
    if not matrix or len(matrix) > 64 or (not matrix[0]) or (len(matrix[0]) > 64) or any((len(row) != len(matrix[0]) for row in matrix)) or any((not cell for row in matrix for cell in row)):
        return None
    return matrix

def _symbolic_scalar_reference_is_supported(value: Any) -> bool:
    if numeric_value(value) is not None:
        return True
    text = symbolic_math_text(value)
    if not text or len(text) > 4096:
        return False
    if not re.fullmatch('[\\w\\s+\\-*/^().,=<>\\[\\]_:|]+', text, flags=re.UNICODE):
        return False
    if text.count('=') > 1:
        return False
    balanced_text = '' if '|' in text else text
    if balanced_text and len(text) >= 3 and (text[0] in '([') and (text[-1] in ')]') and (',' in text):
        balanced_text = text[1:-1]
    depth = 0
    for character in balanced_text:
        if character == '(':
            depth += 1
        elif character == ')':
            depth -= 1
            if depth < 0:
                return False
    if depth:
        return False
    names = set(re.findall('[^\\W\\d]\\w*', text, flags=re.UNICODE))
    functions = {'sqrt', 'sin', 'cos', 'tan', 'csc', 'arccos', 'arctan', 'log', 'ln', 'exp', 'abs', 'factorial', 'floor', 'root', 'pi', 'e', 'i', 'oo'}
    named_symbols = {'alpha', 'beta', 'gamma', 'delta', 'theta', 'lambda', 'mu', 'phi', 'psi', 'omega', 'rho'}
    return not any((name not in functions and len(name) > 2 and (name not in named_symbols) and (not ('_' in name and re.fullmatch('(?:[A-Za-z](?:_[A-Za-z0-9]{1,2})?)+', name))) for name in names))

def symbolic_reference_is_supported(value: Any, *, problem: str='') -> bool:
    matrix = symbolic_matrix_values(value)
    if matrix is not None:
        return all((_symbolic_scalar_reference_is_supported(cell) for row in matrix for cell in row))
    structured = symbolic_structured_values(value, problem=problem)
    if structured is None:
        return _symbolic_scalar_reference_is_supported(value)
    return all((symbolic_structured_values(field) is None and _symbolic_scalar_reference_is_supported(field) for field in structured[1]))

def _drop_number(value: str) -> str | None:
    text = unicodedata.normalize('NFKC', value).replace(',', '').strip()
    text = text.replace('−', '-').replace('–', '-')
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), 'f')
    if '.' in normalized:
        normalized = normalized.rstrip('0').rstrip('.')
    return '0' if normalized in {'-0', ''} else normalized

def _drop_normalize_token(value: str) -> str:
    token = unicodedata.normalize('NFKC', value).lower().strip()
    token = token.replace('\\%', '%')
    if token.endswith('%') and _drop_number(token[:-1]) is not None:
        token = token[:-1]
    number = _drop_number(token)
    if number is not None:
        return number
    token = ''.join((character for character in token if character not in string.punctuation))
    number = _drop_number(token)
    if number is not None:
        return number
    return '' if token in {'a', 'an', 'the'} else token

def drop_normalize_answer(value: Any) -> str:
    text = unicodedata.normalize('NFKC', str(value)).lower().strip()
    for _ in range(4):
        normalized = re.sub('\\\\(?:text|mathrm|operatorname|mbox)\\{([^{}]*)\\}', '\\1', text)
        if normalized == text:
            break
        text = normalized
    raw_tokens = re.split('\\s+|(?<=[A-Za-z])-(?=[A-Za-z])', text)
    tokens = [_drop_normalize_token(token) for token in raw_tokens]
    return ' '.join((token for token in tokens if token))

def _drop_spans(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        try:
            return drop_answer_json_to_strings(value)[0]
        except ContractError:
            return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if text.startswith(('[', '(')) and text.endswith((']', ')')):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    fields = [field.strip() for field in re.split('\\s*(?:;|\\|)\\s*', text) if field.strip()]
    return fields or ([text] if text else [])

def drop_answer_json_to_strings(value: Mapping[str, Any]) -> tuple[list[str], str]:
    number = value.get('number')
    if number is not None and str(number).strip():
        return ([str(number).strip()], 'number')
    spans = value.get('spans')
    if isinstance(spans, (list, tuple)) and any((str(item).strip() for item in spans)):
        normalized = [str(item).strip() for item in spans if str(item).strip()]
        return (normalized, 'span' if len(normalized) == 1 else 'spans')
    date = value.get('date')
    if isinstance(date, Mapping):
        fields = [str(date.get(key) or '').strip() for key in ('day', 'month', 'year')]
        joined = ' '.join((field for field in fields if field))
        if joined:
            return ([joined], 'date')
    raise ContractError('DROP answer lacks number, spans, or date')

def _parse_drop_object(value: Any) -> Any:
    parsed = value
    if isinstance(parsed, str) and parsed.strip().startswith('{'):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(parsed)
            except (SyntaxError, ValueError):
                parsed = value
    return parsed

def _drop_answer_set(value: Any) -> list[str] | None:
    parsed = _parse_drop_object(value)
    if isinstance(parsed, Mapping):
        nested = parsed.get('answer_sets')
        if isinstance(nested, list) and len(nested) == 1:
            return _drop_answer_set(nested[0])
        try:
            return drop_answer_json_to_strings(parsed)[0]
        except ContractError:
            return None
    return None

def _drop_answer_sets_from_object(value: Any) -> list[list[str]]:
    parsed = _parse_drop_object(value)
    if not isinstance(parsed, Mapping):
        return []
    nested = parsed.get('answer_sets')
    if isinstance(nested, list):
        answer_sets = [item for item in (_drop_answer_set(value) for value in nested) if item]
        if answer_sets:
            return answer_sets
    answers = []
    primary = parsed.get('answer') if isinstance(parsed.get('answer'), Mapping) else parsed
    primary_set = _drop_answer_set(primary)
    if primary_set:
        answers.append(primary_set)
    validated = parsed.get('validated_answers')
    if isinstance(validated, list):
        answers.extend((item for item in (_drop_answer_set(value) for value in validated) if item))
    return answers

def drop_answer_sets(record: Mapping[str, Any]) -> list[list[str]]:
    explicit = record.get('drop_answer_sets')
    if isinstance(explicit, list) and explicit:
        answer_sets = []
        for value in explicit:
            spans = _drop_answer_set(value)
            if spans is None and isinstance(value, (list, tuple)):
                spans = _drop_spans(value)
            if spans:
                answer_sets.append(spans)
        if answer_sets:
            return answer_sets
    for value in (record.get('verifier_payload'), record.get('reference_answer')):
        answer_sets = _drop_answer_sets_from_object(value)
        if answer_sets:
            return answer_sets
    references = record.get('reference_answers')
    if isinstance(references, list) and references:
        parsed_sets = []
        for value in references:
            parsed_sets.extend(_drop_answer_sets_from_object(value))
            parsed = _drop_answer_set(value)
            if parsed:
                parsed_sets.append(parsed)
        if parsed_sets:
            return parsed_sets
        return [spans for value in references if (spans := _drop_spans(value))]
    reference = record.get('reference_answer')
    parsed = _drop_answer_set(reference)
    if parsed:
        return [parsed]
    if reference is not None and str(reference).strip():
        return [_drop_spans(reference)]
    return []

def _latex_fraction(value: str) -> str:
    match = re.fullmatch('[+-]?\\\\(?:[dt]?frac)\\{([^{}]+)\\}\\{([^{}]+)\\}', value)
    if not match:
        return value
    sign = '-' if value.startswith('-') else ''
    numerator = match.group(1)
    denominator = match.group(2)
    return f'{sign}{numerator}/{denominator}'

def numeric_value(value: Any) -> Fraction | None:
    raw = _unwrap_complete_latex_reference(str(value))
    text = normalize_answer_text(raw)
    percent = text.endswith(('%', '\\%'))
    text = re.sub('(?:\\\\%|%)$', '', text)
    mixed = re.fullmatch('([+-]?)(\\d+)(?:\\\\[dt]?frac\\{(\\d+)\\}\\{(\\d+)\\}|又(\\d+)/(\\d+))', text)
    if mixed:
        sign, whole, numerator, denominator, other_numerator, other_denominator = mixed.groups()
        denominator = int(denominator or other_denominator)
        if denominator == 0:
            return None
        number = Fraction(int(whole)) + Fraction(int(numerator or other_numerator), denominator)
        number = -number if sign == '-' else number
        return number / 100 if percent else number
    text = _latex_fraction(text)

    def bounded_fraction(raw: str) -> Fraction | None:
        if '_' in raw:
            return None
        decimal = Decimal(raw)
        if not decimal.is_finite():
            return None
        digits = decimal.as_tuple().digits
        exponent = decimal.as_tuple().exponent
        if abs(exponent) > 10000 or len(digits) + max(exponent, 0) > 10000:
            return None
        return Fraction(decimal)
    try:
        if '/' in text and text.count('/') == 1:
            numerator, denominator = text.split('/', 1)
            numerator_value = bounded_fraction(numerator)
            denominator_value = bounded_fraction(denominator)
            if numerator_value is None or denominator_value in {None, 0}:
                return None
            number = numerator_value / denominator_value
        else:
            number = bounded_fraction(text)
            if number is None:
                return None
        return number / 100 if percent else number
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None

def numeric_display_value(value: Any) -> Fraction | None:
    number = numeric_value(value)
    if number is None:
        return None
    normalized = normalize_answer_text(value)
    return number * 100 if normalized.endswith(('%', '\\%')) else number

def choice_value(value: Any) -> str | None:
    text = normalize_answer_text(value).upper()
    match = re.fullmatch('(?:OPTION)?[\\(\\[]?([A-Z])[\\)\\].:]?', text)
    return match.group(1) if match else None

def choice_answer_value(value: Any, problem: str='') -> str | None:
    text = re.sub('\\s*[✅✔✓]\\ufe0f?\\s*$', '', str(value).strip())
    text = _choice_display_text(text)
    text = re.sub('^(?:(?:the\\s+)?(?:(?:correct|final)\\s+)?(?:answer|option)\\s*(?:is|[:：])\\s*|(?:最终)?答案(?:字母)?\\s*(?:为|是|[:：])\\s*|选项\\s*)', '', text, flags=re.I)
    text = _choice_display_text(text)
    standalone = re.match('^\\(([A-Z])\\)\\s*[.。](?:\\s|$)', text, re.I)
    if standalone:
        return standalone.group(1).upper()
    label = choice_value(text)
    if label is not None:
        return label
    options = choice_option_values(problem)
    matches = [label for label, body in (options or {}).items() if _choice_content_text(text) == body]
    if matches:
        return matches[0] if len(matches) == 1 else None
    labels = choice_set_value(text)
    available = set(options) if options else _choice_problem_labels(problem)
    if labels is not None and labels.issubset(available):
        return ','.join(sorted(labels))
    labelled = re.fullmatch('\\(?([A-Z])(?:\\)|[.、:：])\\s*(.+)', text, re.I)
    if labelled:
        label, body = labelled.groups()
        label = label.upper()
        if _choice_content_text(body) == (options or {}).get(label):
            return label
        return None
    annotated = re.fullmatch('([A-Z])\\s*[（(]([^A-Za-z]*?)[）)]', text)
    if annotated:
        return annotated.group(1)
    return None

def _choice_display_text(value: str) -> str:
    text = _unwrap_complete_latex_reference(_strip_answer_formatting(value))
    for command in ('text', 'mbox', 'mathrm'):
        for start, end, body in reversed(_latex_command_value_matches(text, command)):
            text = text[:start] + body + text[end:]
    return _strip_answer_formatting(text)

def _choice_content_text(value: str) -> str:
    text = re.sub('\\\\(?:quad|qquad|enspace|thinspace)|\\\\[ ,;!]', '', _choice_display_text(value))
    text = normalize_answer_text(text)
    text = text.replace('\\times', '*').replace('\\cdot', '*')
    text = re.sub('\\^(?:\\{\\\\circ\\}|\\\\circ)', '°', text)
    for name, letter in (('alpha', 'α'), ('beta', 'β'), ('gamma', 'γ'), ('delta', 'δ'), ('theta', 'θ'), ('lambda', 'λ'), ('mu', 'μ'), ('pi', 'π'), ('sigma', 'σ'), ('phi', 'φ'), ('omega', 'ω')):
        text = re.sub('\\\\' + name + '(?![a-z])', letter, text)
    if re.fullmatch('\\([a-z -]+\\)', text):
        text = text[1:-1]
    text = re.sub('\\^\\{([^{}]+)\\}', '^\\1', text)
    return re.sub('_(?:\\{(\\d+)\\}|(\\d+))', '\\1\\2', text)

def set_value(value: Any, *, allow_sequence_brackets: bool=False) -> tuple[str, ...] | None:
    text = normalize_answer_text(value)
    if text.startswith('\\{') and text.endswith('\\}'):
        text = '{' + text[2:-2] + '}'
    accepted = {('{', '}')}
    if allow_sequence_brackets:
        accepted.add(('[', ']'))
    if len(text) < 2 or (text[0], text[-1]) not in accepted:
        return None
    fields = [field for field in re.split('[,，]', text[1:-1]) if field]
    if not fields:
        return None
    normalized = []
    for field in fields:
        number = numeric_value(field)
        normalized.append(str(number) if number is not None else normalize_answer_text(field))
    return tuple(sorted(normalized))

def reference_alternatives(record: Mapping[str, Any]) -> list[Any]:
    values = record.get('reference_answers')
    if values is not None:
        if not isinstance(values, list) or not values:
            raise ContractError('reference_answers must be a non-empty list')
        return values
    if 'reference_answer' in record:
        return [record['reference_answer']]
    return []

def code_blocks(response: str) -> list[dict[str, Any]]:
    blocks = []
    pattern = re.compile('```([^\\n`]*)\\n([\\s\\S]*?)```', re.MULTILINE)
    for index, match in enumerate(pattern.finditer(response)):
        blocks.append({'index': index, 'language': match.group(1).strip().lower(), 'code': match.group(2).strip(), 'start': match.start(2), 'end': match.end(2)})
    return blocks

def _latex_brace_is_escaped(value: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and value[cursor] == '\\':
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1

def _latex_command_value_matches(value: str, command: str) -> list[tuple[int, int, str]]:
    prefix = f'\\{command}{{'
    values: list[tuple[int, int, str]] = []
    cursor = 0
    while True:
        marker = '(?<![A-Za-z\\\\])\\\\?' + re.escape(command)
        if command == 'boxed':
            marker = '(?:' + marker + '|\\x08oxed)'
        match = re.search(marker + '\\{', value[cursor:]) if command in {'boxed', 'fbox'} else None
        start = cursor + match.start() if match else value.find(prefix, cursor)
        if start < 0:
            return values
        body_start = cursor + match.end() if match else start + len(prefix)
        depth = 1
        for index in range(body_start, len(value)):
            character = value[index]
            if character == '{' and (not _latex_brace_is_escaped(value, index)):
                depth += 1
            elif character == '}' and (not _latex_brace_is_escaped(value, index)):
                depth -= 1
                if depth == 0:
                    values.append((start, index + 1, value[body_start:index].strip()))
                    cursor = index + 1
                    break
        else:
            return values
_NEXT_ANSWER_DECLARATION = '(?=\\s*(?:the\\s+)?(?:final\\s+(?:answer|result)|answer|最终答案|最终结果|答案字母|答案|结果)\\s*(?:\\*\\*|__|`)?\\s*(?:is\\b|equals?\\b|=|[:：]|为|是)|</?think>|<\\|im_end\\|>|\\n|$)'
_STRONG_FINAL_ANSWER_PATTERN = re.compile('(?:final\\s+(?:answer|result)|最终答案|最终结果)\\s*(?:\\*\\*|__|`)?\\s*(?:is\\s*|equals?\\s*|=\\s*|[:：]\\s*|为\\s*|是\\s*)(?:\\*\\*|__|`)?\\s*([^\\n]*?)' + _NEXT_ANSWER_DECLARATION, re.I)
_TAGGED_ANSWER_PATTERN = re.compile('<(?:answer|final_answer)>\\s*([\\s\\S]*?)\\s*</(?:answer|final_answer)>', re.I)
_DECLARED_ANSWER_PATTERN = re.compile('(?:final\\s+(?:answer|result)|answer|最终答案|最终结果|答案字母|答案|结果)\\s*(?:\\*\\*|__|`)?\\s*(?:is\\s*|equals?\\s*|=\\s*|[:：]\\s*|为\\s*|是\\s*)(?:\\*\\*|__|`)?\\s*([^\\n]*?)' + _NEXT_ANSWER_DECLARATION, re.I)

def _non_overlapping_generic_answer_matches(response: str, strong_matches: list[re.Match[str]]) -> list[re.Match[str]]:
    generic = []
    for match in _DECLARED_ANSWER_PATTERN.finditer(response):
        if any((match.start() >= strong.start() and match.end() <= strong.end() for strong in strong_matches)):
            continue
        generic.append(match)
    return generic

def _last_balanced_latex_command_value(value: str, command: str) -> str | None:
    prefix = f'\\{command}{{'
    starts = [match.start() for match in re.finditer(re.escape(prefix), value)]
    for start in reversed(starts):
        body_start = start + len(prefix)
        depth = 1
        for index in range(body_start, len(value)):
            character = value[index]
            if character == '{' and (not _latex_brace_is_escaped(value, index)):
                depth += 1
            elif character == '}' and (not _latex_brace_is_escaped(value, index)):
                depth -= 1
                if depth == 0:
                    candidate = value[body_start:index].strip()
                    if candidate:
                        return candidate
                    break
    return None

def _answer_declarations(response: str, allow_hash_answer: bool=False) -> list[dict[str, Any]]:
    declarations = []
    for command in ('boxed', 'fbox'):
        declarations.extend((dict(start=start, end=end, candidate_answer=body, method=f'last_latex_{command}', strong=False) for start, end, body in _latex_command_value_matches(response, command) if body))
    strong_matches = sorted([*_STRONG_FINAL_ANSWER_PATTERN.finditer(response), *_TAGGED_ANSWER_PATTERN.finditer(response)], key=lambda match: match.start())
    generic_matches = _non_overlapping_generic_answer_matches(response, strong_matches)
    headings = list(re.finditer('(?:final\\s+line|答案字母)\\s*[:：]?\\s*\\n?\\s*([^\\n]+)', response, re.I))
    for match in [*strong_matches, *generic_matches, *headings]:
        if not match.group(1).strip() or any((d['start'] <= match.start() < d['end'] for d in declarations)):
            continue
        declarations.append(dict(start=match.start(), end=match.end(), candidate_answer=match.group(1).strip(), method='last_final_answer_marker', strong=match in strong_matches or match in headings))
    if allow_hash_answer:
        declarations.extend((dict(start=match.start(), end=match.end(), candidate_answer=match.group(1).strip(), method='last_hash_answer', strong=numeric_value(match.group(1)) is not None or symbolic_reference_is_supported(match.group(1))) for match in re.finditer('(?:^|\\n)\\s*####\\s*([^\\n]+)', response) if match.group(1).strip() and (not re.match('(?:[a-z][.)]\\s|(?:step|case|part|explanation|proof|analysis)\\b)', match.group(1).strip(), re.I))))
    for declaration in declarations:
        nested = [d for d in declarations if declaration['start'] < d['start'] and d['end'] <= declaration['end']]
        if nested and len(nested) == 1:
            declaration['candidate_answer'] = nested[0]['candidate_answer']
    return sorted(declarations, key=lambda d: d['start'])

def extract_declared_answer(response: str, *, allow_hash_answer: bool=False) -> dict[str, Any] | None:
    declarations = _answer_declarations(response, allow_hash_answer)
    if declarations:
        return {**declarations[-1], 'confidence': 'high'}
    for command in ('boxed', 'fbox'):
        recovered = _last_balanced_latex_command_value(response, command)
        if recovered:
            return {'candidate_answer': recovered, 'confidence': 'high', 'method': f'last_latex_{command}_after_malformed_prefix'}
    return None

def candidate_type(value: Any) -> str:
    text = str(value).strip()
    if choice_value(text) is not None:
        return 'choice'
    if text.endswith(('%', '\\%')) and numeric_value(text) is not None:
        return 'percent'
    if '/' in text or 'frac' in text:
        return 'fraction' if numeric_value(text) is not None else 'text'
    if numeric_value(text) is not None:
        return 'number'
    if len(_drop_spans(text)) > 1:
        return 'spans'
    return 'text'

def _strip_answer_formatting(value: str) -> str:
    text = value.strip().lstrip(':：').strip().rstrip('。.;；').rstrip()
    text = re.sub('^(?:\\*\\*|__|`)+\\s*', '', text)
    return re.sub('\\s*(?:\\*\\*|__|`)+$', '', text).strip().rstrip('。.;；').rstrip()

def external_request(record: Mapping[str, Any], *, candidate: Any, language: str | None=None) -> dict[str, Any]:
    return {'schema': EXTERNAL_REQUEST_SCHEMA, 'sample_id': record['sample_id'], 'task_type': record['task_type'], 'verifier_mode': record['verifier_mode'], 'problem': record['problem'], 'candidate': candidate, 'language': language, 'reference_answer': record.get('reference_answer'), 'reference_answers': record.get('reference_answers'), 'tests': record['tests'] if record.get('tests') is not None else record.get('verifier_payload'), 'verifier_payload': record.get('verifier_payload')}

def validate_external_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError('external verifier output must be a JSON object')
    verdict = str(value.get('verdict', '')).upper()
    if verdict not in VERDICTS:
        raise ContractError('external verifier returned an invalid verdict')
    return {'verdict': verdict, 'details': value.get('details'), 'raw': dict(value)}

def compare_answers(candidate: Any, references: Sequence[Any], *, answer_type: str='auto', problem: str='', numeric_scale: str='', reference_unit: str='') -> dict[str, Any]:
    mode = answer_type.lower()
    if mode == 'exact_choice':
        mode = 'choice'
    if mode == 'numeric_equivalence':
        mode = 'numeric'
    if reference_unit and mode in {'numeric', 'integer', 'decimal', 'fraction'}:
        candidate = problem_numeric_answer(str(candidate), 'in ' + reference_unit)
    references = [value for value in references if str(value).strip()]
    if candidate is None or not str(candidate).strip() or (not references):
        return {'verdict': 'UNCERTAIN', 'method': 'missing_candidate_or_reference'}
    candidate_text = normalize_answer_text(candidate)
    normalized_references = [normalize_answer_text(value) for value in references]
    if mode in {'exact', 'exact_text', 'text'} and all((re.fullmatch('CODE-[0-9A-F]{12}', str(v).strip()) for v in references)):
        code = _unwrap_complete_latex_reference(_strip_answer_formatting(str(candidate)))
        return {'verdict': 'PASS' if code in references else 'FAIL', 'method': 'complete_exact_access_code', 'candidate': code}
    if candidate_text in normalized_references:
        return {'verdict': 'PASS', 'method': 'normalized_exact', 'candidate': candidate_text, 'references': normalized_references}
    option_content = compare_choice_option_content(candidate, references, problem)
    if option_content is not None:
        return option_content
    if mode == 'drop_exact':
        left = drop_normalize_answer(candidate)
        comparable = [drop_normalize_answer(value) for value in references]
        return {'verdict': 'PASS' if left in comparable else 'FAIL', 'method': 'drop_official_exact', 'candidate': left, 'references': comparable}
    if mode in {'choice', 'auto'}:
        selection = choice_answer_value(candidate, problem)
        left = choice_value(selection or candidate)
        right = [choice_value(value) for value in references]
        comparable = [value for value in right if value is not None]
        if left is not None and comparable:
            return {'verdict': 'PASS' if left in comparable else 'FAIL', 'method': 'choice', 'candidate': left, 'references': comparable}
        multiple = choice_set_value(selection) if selection else None
        if mode == 'choice' and multiple is not None and comparable:
            return {'verdict': 'FAIL', 'method': 'multiple_answers_in_single_choice', 'candidate': sorted(multiple), 'references': comparable}
        if mode == 'choice' and comparable:
            options = choice_option_values(problem) or {}
            number = numeric_value(candidate)
            numbers = {label: numeric_value(body) for label, body in options.items()}
            if number is not None and numbers and all((value is not None for value in numbers.values())):
                matches = [label for label, value in numbers.items() if value == number]
                return {'verdict': 'PASS' if len(matches) == 1 and matches[0] in comparable else 'FAIL' if len(matches) <= 1 else 'UNCERTAIN', 'method': 'numeric_content_against_all_options', 'candidate': str(number), 'matching_options': matches}
            if options and symbolic_reference_is_supported(candidate) and re.search('[0-9\\\\+*/^]', str(candidate)):
                from evaluation.math_verifier import symbolic_equivalence
                comparisons = {label: symbolic_equivalence(candidate, body) for label, body in options.items()}
                matches = [label for label, result in comparisons.items() if result['verdict'] == 'PASS']
                uncertain = any((result['verdict'] == 'UNCERTAIN' for result in comparisons.values()))
                verdict = 'UNCERTAIN' if uncertain or len(matches) > 1 else 'PASS' if matches and matches[0] in comparable else 'FAIL'
                return dict(verdict=verdict, method='symbolic_content_against_all_options', matching_options=matches)
            if re.search('none of (?:the |these )?(?:options|choices)|no (?:listed )?(?:option|choice) matches|选项均不|没有正确选项', str(candidate), re.I):
                return {'verdict': 'FAIL', 'method': 'no_listed_option_submitted'}
            if re.fullmatch('[\\sA-Z()\\[\\],、，/或和及]+', str(candidate)) and len(set(re.findall('[A-Z]', str(candidate)))) > 1:
                return {'verdict': 'FAIL', 'method': 'multiple_answers_in_single_choice'}
    if mode == 'choice_set':
        left = choice_set_value(candidate)
        right = [choice_set_value(value) for value in references]
        comparable = [value for value in right if value is not None]
        if left is not None and comparable:
            return {'verdict': 'PASS' if left in comparable else 'FAIL', 'method': 'unordered_choice_set', 'candidate': sorted(left), 'references': [sorted(value) for value in comparable]}
        return {'verdict': 'UNCERTAIN', 'method': 'choice_set_unparseable', 'candidate': candidate_text, 'references': normalized_references}
    if mode in {'integer', 'numeric', 'decimal', 'fraction', 'auto', 'math_equivalence', 'symbolic', 'expression'}:
        left_interval = symbolic_structured_values(candidate, problem=problem)
        if left_interval and left_interval[0].startswith('interval:'):
            right_intervals = [symbolic_structured_values(value, problem=problem, interval=True) for value in references]
            if all((value and value[0].startswith('interval:') and all((numeric_value(x) is not None for x in value[1])) for value in [left_interval, *right_intervals])):
                signature = lambda value: (value[0], tuple((numeric_value(x) for x in value[1])))
                return {'verdict': 'PASS' if any((signature(left_interval) == signature(value) for value in right_intervals)) else 'FAIL', 'method': 'numeric_interval_equivalence', 'candidate': str(candidate), 'references': list(references)}
        if re.search('volume|面积|体积', problem, re.I) and re.search('\\\\(?:leq?|geq?)\\b|[<>]', str(candidate)):
            return {'verdict': 'FAIL', 'method': 'region_submitted_for_requested_quantity'}
        value_parser = numeric_display_value if numeric_scale.strip().lower() == 'display_percent' else numeric_value
        left = value_parser(candidate)
        right = [value_parser(value) for value in references]
        comparable = [value for value in right if value is not None]
        if left is not None and comparable and (mode not in {'math_equivalence', 'symbolic', 'expression'} or left in comparable or (all((value is not None for value in right)) and (numeric_scale or not any(('%' in str(value) for value in [candidate, *references]))))):
            return {'verdict': 'PASS' if left in comparable else 'FAIL', 'method': 'numeric_display_percent' if value_parser is numeric_display_value else 'numeric_fraction', 'candidate': str(left), 'references': [str(value) for value in comparable]}
    if mode in {'set', 'auto'}:
        allow_sequence_brackets = mode == 'set'
        left = set_value(candidate, allow_sequence_brackets=allow_sequence_brackets)
        right = [set_value(value, allow_sequence_brackets=allow_sequence_brackets) for value in references]
        comparable = [value for value in right if value is not None]
        if left is not None and comparable:
            return {'verdict': 'PASS' if left in comparable else 'FAIL', 'method': 'finite_set', 'candidate': list(left), 'references': [list(value) for value in comparable]}
    if mode in {'exact', 'exact_text', 'text'}:
        return {'verdict': 'FAIL', 'method': 'normalized_exact', 'candidate': candidate_text, 'references': normalized_references}
    return {'verdict': 'UNCERTAIN', 'method': 'symbolic_or_semantic_equivalence_required', 'candidate': candidate_text, 'references': normalized_references}

def typed_declared_answer(response: str, *, answer_type: str, allow_hash_answer: bool=False, problem: str='', reference_unit: str='') -> dict[str, Any] | None:
    mode = answer_type.strip().lower()
    terminal = terminal_choice_answer(response, problem) if mode in {'choice', 'exact_choice'} else None
    declarations = [terminal] if terminal else _answer_declarations(response, allow_hash_answer)
    declared = dict(declarations[-1], confidence='high') if declarations else extract_declared_answer(response, allow_hash_answer=allow_hash_answer)
    if declared is None and mode in {'choice', 'exact_choice'}:
        lines = [(m.start(), m.end(), m.group().strip()) for m in re.finditer('[^\\n]+', response) if m.group().strip()]
        boundaries = lines[:1] + lines[-1:] if len(lines) > 1 else lines
        literals = []
        for start, end, line in boundaries:
            label = choice_answer_value(line, problem)
            if label:
                literals.append(dict(start=start, end=end, candidate_answer=label, method='boundary_choice_literal', strong=True))
        if literals:
            declarations = literals
            declared = {**literals[-1], 'confidence': 'high'}
        elif numeric_value(response) is not None:
            declared = dict(candidate_answer=response, confidence='high', method='complete_numeric_option_content', start=0, end=len(response))
    if declared is None and mode in {'exact', 'exact_text', 'text'}:
        literal = re.fullmatch('\\s*(?:\\{(CODE-[0-9A-F]{12})\\}|(CODE-[0-9A-F]{12}))\\s*', response)
        if literal:
            declared = dict(candidate_answer=literal.group(1) or literal.group(2), confidence='high', method='complete_access_code_literal', start=0, end=len(response))
    if declared is None and mode in {'auto', 'numeric', 'integer', 'fraction', 'decimal', 'symbolic', 'expression', 'math_equivalence'}:
        if numeric_value(response) is not None or symbolic_reference_is_supported(response):
            declared = dict(candidate_answer=response, confidence='high', method='complete_numeric_literal', start=0, end=len(response))
    if declared is None:
        return None
    candidate = _strip_answer_formatting(str(declared['candidate_answer']))
    candidate = re.sub('^(?:(?:the\\s+)?(?:final\\s+)?answer\\s*(?:is\\s*|[:：]\\s*)|(?:最终)?答案(?:字母)?\\s*(?:是|为|[:：])\\s*)', '', candidate, flags=re.I)
    candidate = _strip_answer_formatting(candidate)
    if not candidate or len(candidate) > 4096:
        return None
    raw_candidate = candidate
    if mode in {'choice', 'exact_choice'}:
        candidate = choice_answer_value(candidate, problem) or candidate
    if mode in {'auto', 'numeric', 'integer', 'fraction', 'decimal'}:
        candidate = problem_numeric_answer(candidate, 'in ' + reference_unit if reference_unit else problem, response)
    normalizer = drop_normalize_answer if mode in {'drop_exact', 'drop_official'} else normalize_answer_text
    normalized = normalizer(candidate)
    strong = [d for d in declarations if d['strong']]
    scope = strong[0]['start'] if strong else declared.get('start', 0)
    if mode in {'exact', 'exact_text', 'text'} and declared.get('end') is not None:
        if not response[declared['end']:].strip(' $*_`().。;；\n\r\t'):
            paragraph = response.rfind('\n\n', 0, declared['start']) + 2
            scope = max(scope, paragraph)
    if mode not in {'choice', 'exact_choice'} and declared.get('method', '').startswith('last_latex_'):
        if not response[declared.get('end', len(response)):].strip(' $*_`\\[]().。;；\n\r\t'):
            scope = declared.get('start', scope)
            boxes = [d for d in declarations if 'latex_' in d['method'] and d['start'] < scope]
            for prior in reversed(boxes):
                gap = response[prior['end']:scope].strip(' $*_`\\[]().。;；\n\r\t')
                if gap.casefold() not in {'', ',', 'and', 'or', '或', '和', '、'}:
                    break
                scope = prior['start']
    if mode in {'choice', 'exact_choice'} and (not strong) and declarations and (not any(('latex_' in d['method'] for d in declarations))):
        scope = declarations[0]['start']
    if mode in {'choice', 'exact_choice'} and (not strong):
        boxes = [d for d in declarations if 'latex_' in d['method']]
        if len(boxes) > 1:
            scope = boxes[0]['start']
    corrections = list(re.finditer('\\b(?:correction|after correction|correcting)\\b|更正|修正为|改为', response[:declared.get('start', 0)], re.I))
    if corrections:
        scope = max(scope, corrections[-1].start())
    competing = set()
    for item in declarations:
        if item['start'] < scope:
            continue
        value = _strip_answer_formatting(item['candidate_answer'])
        value = re.sub('^(?:the\\s+)?(?:final\\s+)?answer\\s*(?:is\\s*|[:：]\\s*)', '', value, flags=re.I)
        if not value.strip('$*_` :：') or all((unicodedata.category(c)[0] in {'P', 'Z'} for c in value)):
            continue
        if mode in {'choice', 'exact_choice'}:
            label = choice_answer_value(value, problem)
            if label is None and item != declarations[-1]:
                if not re.match('^\\(?[A-Z](?:\\)|[.、:：]|$)', value):
                    continue
            value = label or value
        if mode in {'auto', 'numeric', 'integer', 'fraction', 'decimal'}:
            value = problem_numeric_answer(value, 'in ' + reference_unit if reference_unit else problem, response)
        competing.add(normalizer(value))
    competing.discard('')
    return {**declared, 'raw': raw_candidate, 'visible_answer': candidate, 'candidate_answer': candidate, 'candidate_type': 'text' if mode in {'exact', 'exact_text', 'text'} else candidate_type(candidate), 'normalized': normalized, 'extraction_method': declared['method'], 'candidate_sha256': sha256_text(candidate), 'visible_suffix_sha256': sha256_text(response), 'ambiguous': len(competing) > 1, 'competing_candidate_count': len(competing) or 1}

def problem_numeric_answer(candidate: str, problem: str, response: str='') -> str:
    original = candidate
    candidate = re.sub('\\\\(?:text|mbox|mathrm)\\{([^{}]*)\\}', ' \\1 ', candidate)
    candidate = re.sub('\\\\[ ,;!]', ' ', candidate).strip()
    requested = re.search('how many\\s+(seconds?|minutes?|hours?)\\b', problem, re.I)
    if requested is None:
        requested = re.search('(?:in|total(?: number of)?)\\s+(seconds?|minutes?|hours?)\\b', problem, re.I)
    if requested:
        units = {'second': 1, 'minute': 60, 'hour': 3600}
        pieces = list(re.finditer('([0-9]+(?:\\.[0-9]+)?)\\s*(seconds?|minutes?|hours?)\\b', candidate, re.I))
        rest = re.sub('([0-9]+(?:\\.[0-9]+)?)\\s*(seconds?|minutes?|hours?)\\b', '', candidate, flags=re.I)
        if pieces and (not re.sub('\\band\\b|[ ,.;]', '', rest, flags=re.I)):
            total = sum((Fraction(m.group(1)) * units[m.group(2).lower().rstrip('s')] for m in pieces))
            return str(total / units[requested.group(1).lower().rstrip('s')])
    total = re.search('(?:total(?: of)?|altogether|in total|共|总计)\\s*[:：=]?\\s*([0-9]+(?:\\.[0-9]+)?)\\b', candidate, re.I)
    if total and re.search('total|altogether|how many|总|共|多少', problem, re.I):
        return total.group(1)
    parts = re.fullmatch('(\\d+)\\s+([A-Za-z]+)\\s+and\\s+(\\d+)\\s+([A-Za-z]+)', candidate)
    asked = re.search('how many\\s+([A-Za-z]+)\\s+and\\s+([A-Za-z]+)', problem, re.I)
    if parts and asked and ({parts[2].lower(), parts[4].lower()} == {asked[1].lower(), asked[2].lower()}):
        a, b = (int(parts[1]), int(parts[3]))
        return str(a + b)
    labelled = re.fullmatch('([+-]?[0-9]+(?:\\.[0-9]+)?)\\s+([A-Za-z]+)[.]?', candidate)
    if labelled and re.search('how many\\s+' + re.escape(labelled.group(2).rstrip('s')) + 's?\\b', problem, re.I):
        return labelled.group(1)
    return original

def evaluation_answer_text(response) -> str:
    text = str(response or '')
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[-1]
    elif text.lstrip().startswith('<think>'):
        return ''
    return re.sub('(?:<\\|im_end\\|>)?\\s*$', '', text).strip()

def terminal_choice_answer(response: str, problem: str='') -> dict[str, Any] | None:
    end = len(response.rstrip())
    if not end:
        return None
    start = response.rfind('\n', 0, end) + 1
    value = response[start:end].strip()
    if re.fullmatch('(?:give (?:only )?the answer on the final line|on the final line|最后一行给出答案字母)[.。]?', value, re.I):
        end = len(response[:start].rstrip())
        start = response.rfind('\n', 0, end) + 1
        value = response[start:end].strip()
    tagged = re.search('<(answer|final_answer)>\\s*([\\s\\S]*?)\\s*</\\1>\\s*$', response, re.I)
    if tagged:
        start, value = (tagged.start(), tagged.group(2))
    value = _choice_display_text(value)
    prefix = '^(?:(?:therefore|thus|hence)[,，]?\\s*)?(?:(?:the\\s+)?(?:(?:correct|final|best)\\s+)?(?:answer(?:\\s+letter)?(?:\\s+on\\s+the\\s+final\\s+line)?|final\\s+line\\s+answer\\s+letter|choice|option|statement|letter|conclusion)\\s*(?:(?:is|equals?)\\s*[:：]?|[:：])\\s*|(?:故选|因此选|所以选|应选|选|(?:最终)?答案(?:字母)?)\\s*(?:是|为|[:：])?\\s*)'
    for _ in range(2):
        value = _choice_display_text(re.sub(prefix, '', value, flags=re.I))
    label = choice_answer_value(value, problem)
    if label is None:
        endorsements = list(re.finditer('(?m)^\\s*(?:[-*]\\s*)?(?:\\*\\*)?([A-Z])\\s*[:：]?\\s*(?:\\*\\*)?(?:正确|is correct)\\b', response))
        labels = {m[1] for m in endorsements} & _choice_problem_labels(problem)
        if len(labels) != 1:
            return None
        label = next(iter(labels))
        start = next((m.start() for m in reversed(endorsements) if m[1] == label))
        return dict(start=start, end=end, candidate_answer=label, method='explicit_option_endorsement', strong=True)
    return dict(start=start, end=end, candidate_answer=label, method='terminal_submitted_choice', strong=True)
