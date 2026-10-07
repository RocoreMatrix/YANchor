from __future__ import annotations
import json
import os
import signal
import sys
from scoring.verification import symbolic_matrix_values, symbolic_structured_values
import re
from typing import Any
from scoring.verification import normalize_answer_text as normalize_answer, compare_answers, numeric_value, symbolic_solution_context, symbolic_math_text as latex_to_text, symbolic_matrix_values as matrix_values, symbolic_reference_is_supported, symbolic_structured_values as structured_values

def parse_expression(sp: Any, parser: Any, transformations: Any, value: Any) -> Any:
    text = latex_to_text(value)
    if not symbolic_reference_is_supported(value):
        raise ValueError('expression is outside the declared symbolic subset')
    if text.count('=') == 1:
        left, right = text.split('=', 1)
        text = f'({left})-({right})'
    elif '=' in text:
        raise ValueError('expression has multiple equality signs')
    names = set(re.findall('[^\\W\\d]\\w*', text, flags=re.UNICODE))
    functions = {'sqrt': sp.sqrt, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'csc': sp.csc, 'arccos': sp.acos, 'arctan': sp.atan, 'log': sp.log, 'ln': sp.log, 'exp': sp.exp, 'abs': sp.Abs, 'factorial': sp.factorial, 'floor': sp.floor, 'root': sp.root, 'pi': sp.pi, 'e': sp.E, 'i': sp.I, 'oo': sp.oo}
    local_dict = {name: functions.get(name, sp.Symbol(name)) for name in sorted(names)}
    global_dict = {'__builtins__': {}, 'Integer': sp.Integer, 'Float': sp.Float, 'Rational': sp.Rational, 'Symbol': sp.Symbol}
    expression = parser.parse_expr(text, local_dict=local_dict, global_dict=global_dict, transformations=transformations, evaluate=True)
    if int(sp.count_ops(expression)) > 200:
        raise ValueError('expression is too complex')
    return expression

def simple_assignment_rhs(value: Any) -> str | None:
    text = normalize_answer(value)
    if text.count('=') != 1:
        return None
    left, right = text.split('=', 1)
    left = re.sub('_\\{?[^{}=]+\\}?$', '', left)
    if not re.fullmatch('(?:[a-zA-Z]|\\\\[a-zA-Z]+)(?:\\([a-zA-Z0-9,+-]+\\))?', left) or not right:
        return None
    return right

def proportional_polynomial_equations(sp: Any, left: Any, right: Any) -> bool:
    symbols = sorted(left.free_symbols | right.free_symbols, key=str)
    if not symbols:
        return False
    try:
        if left.as_poly(*symbols) is None or right.as_poly(*symbols) is None:
            return False
        ratio = sp.cancel(left / right)
        return bool(not ratio.free_symbols and ratio != 0 and (ratio.is_finite is not False))
    except Exception:
        return False

def scalar_equivalence(candidate: Any, reference: Any) -> dict[str, Any]:
    if not symbolic_reference_is_supported(reference):
        return {'verdict': 'UNCERTAIN', 'method': 'symbolic_reference_outside_contract'}
    if not symbolic_reference_is_supported(candidate):
        return {'verdict': 'UNCERTAIN', 'method': 'symbolic_candidate_outside_contract'}
    left_text = normalize_answer(candidate)
    right_text = normalize_answer(reference)
    if left_text == right_text:
        return {'verdict': 'PASS', 'method': 'normalized_exact'}
    numeric = compare_answers(candidate, [reference], answer_type='numeric')
    if numeric['verdict'] in {'PASS', 'FAIL'}:
        return numeric
    left_assignment = simple_assignment_rhs(candidate)
    right_assignment = simple_assignment_rhs(reference)
    if left_assignment is not None and right_assignment is None:
        result = scalar_equivalence(left_assignment, reference)
        if result['verdict'] == 'PASS':
            return {'verdict': 'PASS', 'method': 'simple_assignment_rhs', 'component': result}
    if right_assignment is not None and left_assignment is None:
        result = scalar_equivalence(candidate, right_assignment)
        if result['verdict'] == 'PASS':
            return {'verdict': 'PASS', 'method': 'simple_assignment_rhs', 'component': result}
    try:
        import sympy as sp
        from sympy.parsing import sympy_parser as parser
        transformations = parser.standard_transformations + (parser.convert_xor, parser.implicit_multiplication_application)
        left = parse_expression(sp, parser, transformations, candidate)
        right = parse_expression(sp, parser, transformations, reference)
        difference = sp.simplify((left - right).rewrite(sp.acos, sp.asin))
        equality = True if difference == 0 else difference.equals(0)
        equivalent = equality is True
        proportional = bool(not equivalent and left_text.count('=') == 1 and (right_text.count('=') == 1) and proportional_polynomial_equations(sp, left, right))
        return {'verdict': 'PASS' if equivalent or proportional else 'FAIL' if equality is False else 'UNCERTAIN', 'method': 'proportional_polynomial_equation' if proportional else 'sympy_simplify', 'candidate': str(left), 'reference': str(right)}
    except Exception as exc:
        return {'verdict': 'UNCERTAIN', 'method': 'symbolic_parse_failed', 'error': f'{type(exc).__name__}: {exc}'}

def symbolic_equivalence(candidate: Any, reference: Any, *, problem: str='') -> dict[str, Any]:
    right_matrix = matrix_values(reference)
    if right_matrix is not None:
        left_matrix = matrix_values(candidate)
        vector_sequence = structured_values(candidate, problem=problem)
        reference_is_vector = len(right_matrix) == 1 or all((len(row) == 1 for row in right_matrix))
        if left_matrix is None and reference_is_vector and (vector_sequence is not None) and (vector_sequence[0] == 'sequence') and re.search('\\bvectors?\\b|\\u5411\\u91cf|\\u77e2\\u91cf', problem, re.I):
            reference_values = [cell for row in right_matrix for cell in row]
            if len(vector_sequence[1]) != len(reference_values):
                return {'verdict': 'FAIL', 'method': 'vector_cardinality_mismatch', 'candidate_count': len(vector_sequence[1]), 'reference_count': len(reference_values)}
            components = [scalar_equivalence(left, right) for left, right in zip(vector_sequence[1], reference_values)]
            verdicts = [item['verdict'] for item in components]
            return {'verdict': 'PASS' if all((value == 'PASS' for value in verdicts)) else 'FAIL' if 'FAIL' in verdicts else 'UNCERTAIN', 'method': 'vector_sequence_to_matrix_elementwise', 'components': components}
        if left_matrix is None:
            return {'verdict': 'UNCERTAIN', 'method': 'matrix_candidate_unparseable'}
        left_shape = (len(left_matrix), len(left_matrix[0]))
        right_shape = (len(right_matrix), len(right_matrix[0]))
        if left_shape != right_shape:
            return {'verdict': 'FAIL', 'method': 'matrix_shape_mismatch', 'candidate_shape': left_shape, 'reference_shape': right_shape}
        components = [scalar_equivalence(left, right) for left_row, right_row in zip(left_matrix, right_matrix) for left, right in zip(left_row, right_row)]
        verdicts = [item['verdict'] for item in components]
        verdict = 'PASS' if all((value == 'PASS' for value in verdicts)) else 'FAIL' if 'FAIL' in verdicts else 'UNCERTAIN'
        return {'verdict': verdict, 'method': 'matrix_elementwise', 'shape': right_shape, 'components': components}
    if not symbolic_reference_is_supported(reference, problem=problem):
        return {'verdict': 'UNCERTAIN', 'method': 'symbolic_reference_outside_contract'}
    if not symbolic_reference_is_supported(candidate, problem=problem):
        return {'verdict': 'UNCERTAIN', 'method': 'symbolic_candidate_outside_contract'}
    left = structured_values(candidate, problem=problem)
    right = structured_values(reference, problem=problem)
    if any((value is not None and value[0].startswith('interval:') for value in (left, right))):
        left = structured_values(candidate, problem=problem, interval=True)
        right = structured_values(reference, problem=problem, interval=True)
    if symbolic_solution_context(problem):
        if left is None and right is not None and (right[0] == 'set'):
            left = ('set', [str(candidate)])
        if right is None and left is not None and (left[0] == 'set'):
            right = ('set', [str(reference)])
    if left is not None or right is not None:
        if left is None or right is None:
            structure = left or right
            scalar = candidate if left is None else reference
            if structure[0] != 'set' and numeric_value(scalar) is not None:
                return {'verdict': 'FAIL', 'method': 'structured_numeric_scalar_mismatch'}
            return {'verdict': 'UNCERTAIN', 'method': 'structured_scalar_semantics_unresolved'}
        if left[0] != right[0]:
            verdict = 'FAIL' if left[0].startswith('interval:') and right[0].startswith('interval:') else 'UNCERTAIN'
            return {'verdict': verdict, 'method': 'structured_kind_mismatch'}
        if len(left[1]) != len(right[1]):
            return {'verdict': 'FAIL', 'method': 'structured_cardinality_mismatch', 'candidate_count': len(left[1]), 'reference_count': len(right[1])}
        if left[0] == 'sequence' or left[0].startswith('interval:'):
            components = [scalar_equivalence(a, b) for a, b in zip(left[1], right[1])]
            verdicts = [item['verdict'] for item in components]
            return {'verdict': 'PASS' if all((v == 'PASS' for v in verdicts)) else 'FAIL' if 'FAIL' in verdicts else 'UNCERTAIN', 'method': 'interval_endpoints' if left[0].startswith('interval:') else 'ordered_sequence', 'components': components}
        unmatched = list(right[1])
        components = []
        for candidate_item in left[1]:
            match = None
            uncertain = False
            for index, reference_item in enumerate(unmatched):
                result = scalar_equivalence(candidate_item, reference_item)
                if result['verdict'] == 'PASS':
                    match = index
                    components.append(result)
                    break
                uncertain = uncertain or result['verdict'] == 'UNCERTAIN'
            if match is None:
                return {'verdict': 'UNCERTAIN' if uncertain else 'FAIL', 'method': 'unordered_set', 'components': components}
            unmatched.pop(match)
        return {'verdict': 'PASS', 'method': 'unordered_set', 'components': components}
    return scalar_equivalence(candidate, reference)

def fallback_verify(request: dict[str, Any]) -> dict[str, Any]:
    candidate = request.get('candidate')
    references = request.get('reference_answers')
    if references is None:
        references = [request.get('reference_answer')]
    if not isinstance(references, list) or not references or candidate is None:
        return {'verdict': 'UNCERTAIN', 'details': 'missing expression'}
    problem = str(request.get('problem') or '')
    results = []
    for reference in references:
        result = symbolic_equivalence(candidate, reference, problem=problem)
        results.append(result)
        if result['verdict'] == 'PASS':
            return {'verdict': 'PASS', 'details': results}
    verdict = 'FAIL' if results and all((item['verdict'] == 'FAIL' for item in results)) else 'UNCERTAIN'
    candidate_text = str(candidate)
    if verdict == 'FAIL' and re.search('\\bangle\\b|\\u89d2', problem, re.I) and re.search('\\\\pi|(?<![A-Za-z])pi(?![A-Za-z])', candidate_text, re.I) and any((re.search('(?<!\\d)180(?!\\d)', str(reference)) for reference in references)):
        return {'verdict': 'UNCERTAIN', 'details': [*results, {'method': 'angle_unit_convention_requires_model_review'}]}
    return {'verdict': verdict, 'details': results}
_MATH_VERIFY: tuple[Any, Any] | None | bool = None
SYMBOLIC_FALLBACK_TIMEOUT_SECONDS = 20.0

def _symbolic_fallback_timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError('symbolic fallback exceeded its verifier deadline')

def parsed_values(parse: Any, value: Any) -> list[Any]:
    text = str(value).strip()
    if not text or len(text) > 8192:
        return []
    values = parse(text, parsing_timeout=3, raise_on_error=False)
    if values:
        return values
    return parse(f'\\boxed{{{text}}}', parsing_timeout=3, raise_on_error=False)

def math_verify_functions() -> tuple[Any, Any] | None:
    global _MATH_VERIFY
    if _MATH_VERIFY is False:
        return None
    if _MATH_VERIFY is None:
        try:
            from math_verify import parse, verify
        except Exception:
            _MATH_VERIFY = False
            return None
        _MATH_VERIFY = (parse, verify)
    return _MATH_VERIFY

def run_fallback(request: dict[str, Any]) -> dict[str, Any]:
    previous = signal.signal(signal.SIGALRM, _symbolic_fallback_timeout)
    signal.setitimer(signal.ITIMER_REAL, SYMBOLIC_FALLBACK_TIMEOUT_SECONDS)
    try:
        return fallback_verify(request)
    except Exception as exc:
        return {'verdict': 'UNCERTAIN', 'details': [{'method': 'math_equivalence_fail_closed', 'error': f'{type(exc).__name__}: {str(exc)[:1000]}'}]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)

def verify_request(request: dict[str, Any]) -> dict[str, Any]:
    candidate = request.get('candidate')
    references = request.get('reference_answers')
    if references is None:
        references = [request.get('reference_answer')]
    problem = str(request.get('problem') or '')
    structured_reference = isinstance(references, list) and any((symbolic_matrix_values(reference) is not None or symbolic_structured_values(reference, problem=problem) is not None for reference in references))
    structured_candidate = symbolic_matrix_values(candidate) is not None or symbolic_structured_values(candidate, problem=problem) is not None
    if candidate is not None and isinstance(references, list) and (not (structured_reference or structured_candidate)):
        try:
            functions = math_verify_functions()
            if functions is None:
                raise RuntimeError('math_verify unavailable')
            parse, verify = functions
            target = parsed_values(parse, candidate)
            if target:
                for reference in references:
                    gold = parsed_values(parse, reference)
                    if gold and verify(gold, target, strict=True, timeout_seconds=3, raise_on_error=False):
                        return {'verdict': 'PASS', 'details': [{'method': 'math_verify_0.9.0_strict', 'reference': str(reference)}]}
        except Exception:
            pass
    return run_fallback(request)

def main() -> int:
    result_stream = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    if '--jsonl-worker' in sys.argv[1:]:
        if math_verify_functions() is None:
            raise RuntimeError('formal math verifier requires math_verify')
        for line in sys.stdin:
            try:
                result = verify_request(json.loads(line))
            except Exception as exc:
                result = {'verdict': 'UNCERTAIN', 'details': {'method': 'math_worker_fail_closed', 'error': f'{type(exc).__name__}: {str(exc)[:1000]}'}}
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=result_stream, flush=True)
        return 0
    print(json.dumps(verify_request(json.load(sys.stdin)), ensure_ascii=False, sort_keys=True), file=result_stream, flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
