from __future__ import annotations
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
THINK_PROTOCOL_REVISION = 'yanchor_think_format_v1'
NATURAL_FINISH_REASONS = frozenset({'eos'})
CAP_FINISH_REASONS = frozenset({'length', 'max_tokens', 'max_new_tokens', 'max_output_tokens', 'prompt_length_cap', 'total_length_cap', 'capped'})
TIMEOUT_FINISH_REASONS = frozenset({'timeout', 'timeout_dropped', 'generation_timeout_dropped'})
THINK_IRREVERSIBLE_CAP_FAILURE_REASONS = frozenset({'redundant_think_open', 'repeated_think_close', 'empty_reasoning_before_think_close', 'think_close_text_token_mismatch'})

def think_protocol_status(response_text: str, finish_reason: str, response_ids: Sequence[int] | None=None, *, think_open_id: int | None=None, think_close_id: int | None=None, prompt_think_closed: bool=False, allow_empty_reasoning: bool=False) -> dict[str, Any]:
    text = str(response_text or '')
    finish = str(finish_reason or '').strip().lower()
    ids_available = response_ids is not None
    ids = np.asarray(response_ids if response_ids is not None else (), dtype=np.int64)
    literal_open_count = text.count('<think>')
    literal_close_count = text.count('</think>')
    token_open_count = int(np.count_nonzero(ids == int(think_open_id))) if ids_available and think_open_id is not None else None
    close_positions = np.flatnonzero(ids == int(think_close_id)).tolist() if ids_available and think_close_id is not None else None
    token_close_count = len(close_positions) if close_positions is not None else None
    opening_count = max(literal_open_count, 0 if token_open_count is None else token_open_count)
    closing_count = max(literal_close_count, 0 if token_close_count is None else token_close_count)

    def result(verdict: str, reason: str) -> dict[str, Any]:
        capped = finish in CAP_FINISH_REASONS
        irreversible_at_cap = bool(capped and verdict == 'FAIL' and (reason in THINK_IRREVERSIBLE_CAP_FAILURE_REASONS))
        return {'revision': THINK_PROTOCOL_REVISION, 'verdict': verdict, 'reason': reason, 'opening_count': opening_count, 'closing_count': closing_count, 'literal_opening_count': literal_open_count, 'literal_closing_count': literal_close_count, 'token_opening_count': token_open_count, 'token_closing_count': token_close_count, 'think_close_indices': close_positions, 'finish_reason': finish, 'finish_capped': capped, 'irreversible_at_cap': irreversible_at_cap, 'append_repairable_at_cap': bool(capped and (not irreversible_at_cap))}
    if finish in TIMEOUT_FINISH_REASONS:
        return result('EXCLUDED', 'generation_timeout_dropped')
    capped = finish in CAP_FINISH_REASONS
    natural = finish in NATURAL_FINISH_REASONS
    if not capped and (not natural):
        return result('EXCLUDED', 'generation_finish_unresolved')
    if token_close_count is not None and literal_close_count != token_close_count:
        return result('FAIL', 'think_close_text_token_mismatch')
    if opening_count:
        return result('FAIL', 'redundant_think_open')
    if closing_count > (0 if prompt_think_closed else 1):
        return result('FAIL', 'repeated_think_close')
    if closing_count == 0 and (not prompt_think_closed):
        if capped:
            return result('CAPPED_INCOMPLETE', 'generation_reached_cap')
        return result('FAIL', 'missing_think_close_at_natural_end')
    if prompt_think_closed:
        has_reasoning = True
        has_visible_answer = bool(text.strip())
    elif literal_close_count == 1:
        reasoning, visible = text.split('</think>', 1)
        visible = re.sub('[ \\t\\r\\n]*<\\|im_end\\|>[ \\t\\r\\n]*$', '', visible)
        has_reasoning = bool(reasoning.strip())
        has_visible_answer = bool(visible.strip())
    elif token_close_count == 1 and think_close_id is not None:
        close_index = int(np.flatnonzero(ids == int(think_close_id))[0])
        has_reasoning = close_index > 0
        has_visible_answer = close_index < len(ids) - 1
    else:
        return result('FAIL', 'think_close_text_token_mismatch')
    if not has_reasoning and (not allow_empty_reasoning):
        return result('FAIL', 'empty_reasoning_before_think_close')
    if not has_visible_answer:
        if capped:
            return result('CAPPED_INCOMPLETE', 'generation_reached_cap')
        return result('FAIL', 'empty_visible_answer_after_think_close')
    if capped:
        return result('CAPPED_INCOMPLETE', 'generation_reached_cap')
    return result('PASS', 'none')
PYTHON_FENCE_CONTRACT = 'visible_python_fence'
CODE_PROTOCOL_REVISION = 'visible_python_fence_append_repairable_open_prefix_v3'
CODE_IRREVERSIBLE_CAP_FAILURE_REASONS = frozenset({'empty_python_body', 'multiple_or_nested_fences', 'no_visible_python_fence', 'text_outside_python_fence'})
MECHANICAL_REPETITION_FINISH_REASONS = frozenset({'mechanical_repetition', 'repetition', 'repetition_detected'})
MECHANICAL_REPETITION_EARLY_STOP_REVISION = 'exact_tail_pattern_128to256_tokens_repeat8_first_unit_v2'
MECHANICAL_REPETITION_EARLY_STOP_MIN_PATTERN_TOKENS = 128
MECHANICAL_REPETITION_EARLY_STOP_MAX_PATTERN_TOKENS = 256
MECHANICAL_REPETITION_EARLY_STOP_MIN_REPETITIONS = 8

def first_repeated_unit_onset(response_ids: Sequence[int], end: int, maximum_period: int=256) -> int | None:
    for period in range(1, min(maximum_period, end // 2) + 1):
        if list(response_ids[end - period:end]) != list(response_ids[end - 2 * period:end - period]):
            continue
        start = end - 2 * period
        while start > 0 and response_ids[start - 1] == response_ids[start - 1 + period]:
            start -= 1
        if end - start >= max(16, 4 * period):
            return start + period
    return None

def mechanical_repetition_early_stop_patterns(tails: np.ndarray, lengths: np.ndarray, *, min_pattern_tokens=MECHANICAL_REPETITION_EARLY_STOP_MIN_PATTERN_TOKENS, max_pattern_tokens=MECHANICAL_REPETITION_EARLY_STOP_MAX_PATTERN_TOKENS, min_repetitions=MECHANICAL_REPETITION_EARLY_STOP_MIN_REPETITIONS) -> np.ndarray:
    tails = np.asarray(tails)
    lengths = np.minimum(np.asarray(lengths), tails.shape[1])
    matched = np.zeros(len(tails), dtype=np.int32)
    repeats = min_repetitions
    maximum = min(max_pattern_tokens, tails.shape[1] // repeats)
    periods = np.arange(min_pattern_tokens, maximum + 1)
    if not len(periods):
        return matched
    possible = (lengths[:, None] >= periods[None, :] * repeats) & (tails[:, -1:] == tails[:, -periods - 1])
    columns = periods > 1
    possible[:, columns] &= tails[:, -2:-1] == tails[:, -periods[columns] - 2]
    for column in np.flatnonzero(possible.any(axis=0)):
        period = int(periods[column])
        candidates = np.flatnonzero((matched == 0) & possible[:, column])
        if not len(candidates):
            continue
        blocks = tails[candidates, -period * repeats:].reshape(-1, repeats, period)
        exact = (blocks[:, :-1] == blocks[:, -1:]).all(axis=(1, 2))
        matched[candidates[exact]] = period
    return matched

def mechanical_repetition_early_stop_metrics(response_ids: Sequence[int], *, min_pattern_tokens: int=MECHANICAL_REPETITION_EARLY_STOP_MIN_PATTERN_TOKENS, max_pattern_tokens: int=MECHANICAL_REPETITION_EARLY_STOP_MAX_PATTERN_TOKENS, min_repetitions: int=MECHANICAL_REPETITION_EARLY_STOP_MIN_REPETITIONS) -> dict[str, Any]:
    if min_pattern_tokens <= 0 or max_pattern_tokens < min_pattern_tokens:
        raise ValueError('mechanical repetition pattern range is invalid')
    if min_repetitions < 2:
        raise ValueError('mechanical repetition requires at least two repeats')
    response_tokens = len(response_ids)
    values = np.asarray(response_ids[-max_pattern_tokens * min_repetitions:], dtype=np.int64)
    pattern_tokens = int(mechanical_repetition_early_stop_patterns(values[None, :], np.array([len(values)]), min_pattern_tokens=min_pattern_tokens, max_pattern_tokens=max_pattern_tokens, min_repetitions=min_repetitions)[0])
    if pattern_tokens:
        repeated_tokens = pattern_tokens * min_repetitions
        return {'revision': MECHANICAL_REPETITION_EARLY_STOP_REVISION, 'stop': True, 'response_tokens': response_tokens, 'pattern_tokens': pattern_tokens, 'minimum_repetitions': min_repetitions, 'repeated_tokens': repeated_tokens, 'repeated_tail_start': response_tokens - repeated_tokens, 'failure_onset_index': first_repeated_unit_onset(response_ids, response_tokens, pattern_tokens), 'confirmation_index': response_tokens - 1}
    return {'revision': MECHANICAL_REPETITION_EARLY_STOP_REVISION, 'stop': False, 'response_tokens': response_tokens, 'pattern_tokens': None, 'minimum_repetitions': min_repetitions, 'repeated_tokens': 0, 'repeated_tail_start': None, 'failure_onset_index': None, 'confirmation_index': None}

def mechanical_repetitive_tail_metrics(response_ids: Sequence[int], finish_reason: str, *, tail_tokens: int=256, require_exact_confirmation: bool=False) -> dict[str, Any]:
    values = response_ids
    finish = str(finish_reason).strip().lower()
    capped = finish in CAP_FINISH_REASONS
    generation_stopped_repetition = finish in MECHANICAL_REPETITION_FINISH_REASONS
    if tail_tokens < 16:
        raise ValueError('degenerate-tail window must contain at least 16 tokens')
    tail = list(map(int, values[-tail_tokens:]))
    fourgrams = [tuple(tail[index:index + 4]) for index in range(len(tail) - 3)]
    unique_token_ratio = len(set(tail)) / len(tail) if tail else 1.0
    repeated_fourgram_ratio = 1.0 - len(set(fourgrams)) / len(fourgrams) if fourgrams else 0.0
    eligible_tail = len(tail) >= 16
    exact = None
    if not generation_stopped_repetition and (require_exact_confirmation or not (eligible_tail and repeated_fourgram_ratio >= 0.9)):
        for end in dict.fromkeys((len(values), len(values) - 1, len(values) // 128 * 128)):
            if end >= 1024:
                candidate = mechanical_repetition_early_stop_metrics(values if end == len(values) else values[:end])
                if candidate['stop']:
                    exact = candidate
                    break
    repetitive_degenerate = bool(generation_stopped_repetition or exact is not None or (not require_exact_confirmation and eligible_tail and (repeated_fourgram_ratio >= 0.9)))
    onset: int | None = None
    confirmation: int | None = None
    if generation_stopped_repetition:
        confirmation = len(values) - 1 if len(values) else None
        onset = first_repeated_unit_onset(values, len(values))
    elif exact is not None:
        onset, confirmation = (exact['failure_onset_index'], exact['confirmation_index'])
    elif repetitive_degenerate:
        values = list(map(int, values))
        first_end = 15
        grams = Counter((tuple(values[index:index + 4]) for index in range(0, first_end - 2)))
        for end in range(first_end, len(values)):
            if end > first_end:
                added_start = end - 3
                grams[tuple(values[added_start:added_start + 4])] += 1
                if end + 1 > tail_tokens:
                    removed_start = end - tail_tokens
                    removed = tuple(values[removed_start:removed_start + 4])
                    grams[removed] -= 1
                    if grams[removed] == 0:
                        del grams[removed]
            width = min(end + 1, tail_tokens) - 3
            if 1.0 - len(grams) / width >= 0.9:
                confirmation = end
                onset = first_repeated_unit_onset(values, end + 1)
                break
        if confirmation is None:
            raise RuntimeError('mechanical failure lacks a threshold-crossing token')
    return {'capped': capped, 'response_tokens': len(values), 'tail_tokens': len(tail), 'unique_token_ratio': unique_token_ratio, 'repeated_fourgram_ratio': repeated_fourgram_ratio, 'generation_stopped_repetition': generation_stopped_repetition, 'repetitive_degenerate': repetitive_degenerate, 'degenerate': repetitive_degenerate, 'failure_onset_index': onset, 'confirmation_index': confirmation}

def _contract_name(output_contract: str | Mapping[str, Any]) -> str:
    if isinstance(output_contract, Mapping):
        return str(output_contract.get('output_contract') or '').strip().lower()
    return str(output_contract or '').strip().lower()

def _opening_fence_prefix_is_append_repairable(visible: str) -> bool:
    prefix = str(visible).lstrip()
    if not prefix:
        return True
    if '\n' in prefix:
        return False
    pending_crlf = prefix.endswith('\r') and '\r' not in prefix[:-1]
    if pending_crlf:
        prefix = prefix[:-1]
    elif '\r' in prefix:
        return False
    if len(prefix) < 3:
        return prefix == '```'[:len(prefix)]
    if not prefix.startswith('```'):
        return False
    language = prefix[3:]
    stripped = language.strip().lower()
    if pending_crlf or language != language.rstrip():
        return stripped in {'', 'py', 'python'}
    return any((value.startswith(stripped) for value in ('', 'py', 'python')))

def code_contract_status(response_text: str, response_ids: Sequence[int] | None, finish_reason: str, output_contract: str | Mapping[str, Any], *, eos_token_ids: set[int] | None=None, prompt_think_closed: bool=False) -> dict[str, Any]:
    contract = _contract_name(output_contract)
    finish = str(finish_reason or '').strip().lower()
    finish_natural = finish in NATURAL_FINISH_REASONS
    finish_capped = finish in CAP_FINISH_REASONS
    if contract != PYTHON_FENCE_CONTRACT:
        reason = 'output_contract_missing' if not contract else 'unsupported_output_contract'
        return {'revision': CODE_PROTOCOL_REVISION, 'applicable': True, 'contract_pass': False, 'contract_verdict': 'FAIL', 'contract_reason': reason, 'structure_reason': reason, 'visible_final': '', 'visible_code': '', 'opening_language': '', 'finish_natural': finish_natural, 'single_unterminated_python_fence': False, 'eos_after_contract': None}
    text = str(response_text or '')
    status: dict[str, Any] = {'revision': CODE_PROTOCOL_REVISION, 'applicable': True, 'contract_pass': False, 'contract_verdict': 'FAIL', 'contract_reason': '', 'structure_reason': '', 'visible_final': '', 'visible_code': '', 'opening_language': '', 'finish_natural': finish_natural, 'finish_capped': finish_capped, 'irreversible_at_cap': False, 'append_repairable_at_cap': finish_capped, 'single_unterminated_python_fence': False, 'eos_after_contract': None}

    def fail(reason: str, *, structure_reason: str | None=None) -> dict[str, Any]:
        structural = structure_reason or reason
        status['contract_reason'] = reason
        status['structure_reason'] = structural
        status['irreversible_at_cap'] = bool(finish_capped and structural in CODE_IRREVERSIBLE_CAP_FAILURE_REASONS)
        status['append_repairable_at_cap'] = bool(finish_capped and (not status['irreversible_at_cap']))
        if status['append_repairable_at_cap']:
            status['contract_verdict'] = 'CAPPED_INCOMPLETE'
        return status
    if '</think>' not in text and (not prompt_think_closed):
        if finish_capped:
            return fail('generation_open', structure_reason='generation_open')
        return fail('no_visible_python_fence')
    visible = text if prompt_think_closed else text.split('</think>', 1)[1]
    visible = re.sub('[ \\t\\r\\n]*<\\|im_end\\|>[ \\t\\r\\n]*$', '', visible).lstrip()
    status['visible_final'] = visible.strip()
    if any((marker in visible for marker in ('<|im_start|>', '<|assistant|>', '<|user|>', '<|system|>'))):
        return fail('text_outside_python_fence')
    opening = re.match('^```([^\\n`]*)\\r?\\n', visible)
    if opening is None or opening.group(1).strip().lower() not in {'', 'py', 'python'}:
        structure_reason = 'no_visible_python_fence'
        if finish_capped and _opening_fence_prefix_is_append_repairable(visible):
            return fail('generation_open', structure_reason='unterminated_python_fence')
        reason = 'generation_open' if not finish_natural else structure_reason
        return fail(reason, structure_reason=structure_reason)
    status['opening_language'] = opening.group(1).strip().lower()
    fences = [match.start() for match in re.finditer('```', visible)]
    if len(fences) == 1:
        status['visible_code'] = visible[opening.end():].strip()
        status['single_unterminated_python_fence'] = True
        structure_reason = 'unterminated_python_fence'
        reason = 'generation_open' if not finish_natural else structure_reason
        return fail(reason, structure_reason=structure_reason)
    if len(fences) != 2:
        return fail('multiple_or_nested_fences')
    closing_start = fences[1]
    code = visible[opening.end():closing_start]
    trailing = visible[closing_start + 3:]
    status['visible_code'] = code.strip()
    if trailing.strip():
        return fail('text_outside_python_fence')
    if not code.strip():
        return fail('empty_python_body')
    if not finish_natural:
        return fail('generation_open', structure_reason='none')
    if response_ids is not None and eos_token_ids is not None:
        ids = list(map(int, response_ids))
        eos_positions = [index for index, token in enumerate(ids) if token in eos_token_ids]
        status['eos_after_contract'] = bool(eos_positions and eos_positions[-1] == len(ids) - 1)
        if not status['eos_after_contract']:
            return fail('eos_before_contract_complete', structure_reason='none')
    status.update({'contract_pass': True, 'contract_verdict': 'PASS', 'contract_reason': 'none', 'structure_reason': 'none'})
    return status
