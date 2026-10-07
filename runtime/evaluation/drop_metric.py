# Adapted from AllenAI allennlp-models; modified for bundled DROP evaluation.
# Licensed under Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
from typing import List, Set, Tuple, Union
import string
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

def _remove_articles(text: str) -> str:
    regex = re.compile('\\b(a|an|the)\\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def _white_space_fix(text: str) -> str:
    return ' '.join(text.split())
EXCLUDE = set(string.punctuation)

def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return ''.join((ch for ch in text if ch not in EXCLUDE))
    else:
        return text

def _lower(text: str) -> str:
    return text.lower()

def _tokenize(text: str) -> List[str]:
    return re.split(' |-', text)

def _normalize_answer(text: str) -> str:
    parts = [_white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token))))) for token in _tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized

def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False

def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text

def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return (normalized_spans, token_bags)

def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores

def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = 2 * precision * recall / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1

def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if not gold_numbers or gold_numbers.intersection(predicted_numbers):
        return True
    return False

def get_metrics(predicted: Union[str, List[str], Tuple[str, ...]], gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return (exact_match, f1)
