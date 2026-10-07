from collections import defaultdict
import json
import math
from pathlib import Path
import re
import unicodedata
PERCEPTION = {'existence', 'count', 'position', 'color', 'posters', 'celebrity', 'scene', 'landmark', 'artwork', 'OCR'}
ANSWER_EXTRACTION_VERSION = 'visual-final-answer-v6'
REFERENCE_CORRECTIONS = json.loads((Path(__file__).resolve().parents[1] / 'configs/visual_reference_corrections.json').read_text())
ANSWER_MARKER = re.compile('(?im)(?:\\b(?:the\\s+)?(?:(?:final|correct|best)\\s+)?(?:answer|option|choice|code)\\s*(?:(?:is|would be|should be)\\s*[:=]?|[:=])\\s*|(?:最终答案|正确答案|正确的?选项|答案|故选|选择)\\s*(?:是|为)?\\s*[:：]?\\s*)([^\\n]+)')
CHOICE_COMMITMENT = re.compile('(?im)(?:\\b((?-i:[A-D]))\\s+is\\s+(?:the\\s+)?(?:correct|best|right|intended)\\s+(?:answer|option|choice)|\\b(?:option|choice)\\s+((?-i:[A-D]))\\s+is\\s+(?:the\\s+)?(?:correct|best|right|intended)\\b|\\bI\\s+(?:will\\s+)?(?:choose|select|pick)\\s+((?-i:[A-D]))\\b|\\b(?:relationship|caption|statement|appeal)\\b[^.\\n]*?\\s+(?:is|would be)\\s+((?-i:[A-D]))\\b|\\b(?:letter|option)\\s+(?:representing\\b[^.\\n]*?|corresponding\\b[^.\\n]*?)\\s+is\\s+((?-i:[A-D]))\\b)')

def clean_answer(text):
    value = unicodedata.normalize('NFKC', text).strip()
    value = value.replace('**', '').replace('`', '').replace('\\%', '%')
    value = re.sub('</?(?:answer|final_answer|options?)>', '', value, flags=re.I)
    value = re.sub('\\\\[\\[\\]()]', '', value)
    for match in reversed(list(re.finditer('\\\\(?:boxed|fbox|text|textbf|textrm|mathrm|mathbf|operatorname)\\s*\\{', value))):
        depth = 1
        for end in range(match.end(), len(value)):
            depth += (value[end] == '{') - (value[end] == '}')
            if depth == 0:
                value = value[:match.start()] + value[match.end():end] + value[end + 1:]
                break
    return value.strip().strip('$*').strip()

def choice_value(value, choices, explicit_option=False):
    value = clean_answer(value).strip(' \t\r\n.*"\':：。!！')
    if re.fullmatch('[\\[(]\\s*[A-Za-z]\\s*[\\])]', value):
        value = value[1:-1].strip()
    labeled = re.fullmatch('([A-Da-d])[.)、:：]\\s*(.+)', value)
    if labeled:
        content_keys = [key for key, answer in choices.items() if labeled[2].strip(' .*"\'。').casefold() == clean_answer(answer).strip(' .*"\'。').casefold()]
        if content_keys and labeled[1].upper() not in content_keys:
            return None
    if explicit_option:
        value = re.sub('^(?:option|choice|letter|选项)\\s*', '', value, flags=re.I)
        match = re.match('([A-Za-z])(?=$|[\\s.,:;()。:：])', value)
        if match:
            if re.match('\\s*(?:\\b(?:or|and)\\b|[&/]|或|和|、|[,，]\\s*[A-D](?![A-Za-z]))', value[match.end():], re.I):
                return None
            if match[1].upper() in choices:
                return match[1].upper()
    if value in choices:
        return value
    matching = [key for key, answer in choices.items() if value.casefold() == clean_answer(answer).strip(' .*"\'。').casefold()]
    if matching:
        return matching[0]
    if len(value) == 1 and value.upper() in choices:
        return value.upper()
    match = labeled
    if match and match[1].upper() in choices:
        option = clean_answer(choices[match[1].upper()]).strip(' .*"\'。')
        content = match[2].strip(' .*"\'。')
        if content == match[1].upper() or content.casefold() == option.casefold():
            return match[1].upper()
        if not re.search('\\b[A-D][.)、:：]\\s|\\b(?:or|and)\\s+[A-D]\\b', content):
            return match[1].upper()
    return None

def answer_markers(text):
    matches = []
    for match in ANSWER_MARKER.finditer(text):
        prefix = re.split('[.!?。\\n]', text[:match.start()])[-1]
        if re.search('\\b(?:if|whether|possible|assuming)\\b|如果|是否|假如', prefix, re.I):
            continue
        if re.search('[?？]', re.split('[.。\\n]', match[1])[0]):
            continue
        matches.append(match)
    return matches

def choice_answer(text, choices):
    value = clean_answer(text)
    if not value:
        return None
    blocks = re.split('\\n\\s*\\n', value)
    tail = blocks[-1].strip()
    if re.fullmatch('[A-Za-z](?:\\s*[,，/&、]\\s*[A-Za-z])+[.。]?', tail):
        return None
    structured = re.findall('["\\\']answer["\\\']\\s*:\\s*["\\\']([^"\\\']+)["\\\']', tail, re.I)
    if structured:
        candidates = [choice_value(item, choices, explicit_option=True) for item in structured]
        return candidates[0] if len(set(candidates)) == 1 else None
    direct = choice_value(value, choices)
    if direct is not None:
        return direct
    tail_marked = answer_markers(tail)
    if tail_marked:
        candidates = [choice_value(match[1], choices, explicit_option=True) for match in tail_marked]
        if all((candidate is not None for candidate in candidates)):
            return candidates[0] if len(set(candidates)) == 1 else None
        if len(candidates) > 1:
            return None
    tail_lines = tail.splitlines()
    preceding = [choice_value(line, choices) for line in tail_lines[:-1]]
    if '\n' not in tail or not any((item is not None for item in preceding)):
        terminal_text = tail.splitlines()[-1]
        committed_tail = re.fullmatch("I(?:'ll| will)?\\s+(?:answer|choose|select|pick|go with)\\s+(?:option\\s+)?([A-Za-z])[.!。]?", terminal_text, re.I)
        if committed_tail:
            return choice_value(committed_tail[1], choices, explicit_option=True)
        named = re.fullmatch('(?:option|choice|answer|选项|答案)[_\\s:：]*([A-Za-z])[.。]?', terminal_text, re.I)
        if named:
            return choice_value(named[1], choices, explicit_option=True)
        explicit = False
        marked = answer_markers(value)
        if marked and len(terminal_text) == 1 and terminal_text.islower():
            explicit = bool(re.match(re.escape(terminal_text) + '(?=$|[\\s.。])', marked[-1][1], re.I))
        terminal = choice_value(terminal_text, choices, explicit_option=explicit)
        if terminal is not None or re.fullmatch('[A-Za-z][.!。]?', terminal_text):
            return terminal
        depicted = re.fullmatch('The (?:diagram|image|picture) (?:depicts|shows|represents) ([A-D])[.。]?', terminal_text, re.I)
        if depicted:
            return depicted[1].upper() if depicted[1].upper() in choices else None
    marked = answer_markers(value)
    candidates = [choice_value(match[1], choices, explicit_option=True) for match in marked]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if candidates:
        return candidates[0] if len(set(candidates)) == 1 else None
    committed = []
    for match in CHOICE_COMMITMENT.finditer(value):
        if re.search('\\b(?:if|whether|possible|assuming)\\b|如果|是否', re.split('[.!?。\\n]', value[:match.start()])[-1], re.I):
            continue
        if re.match('\\s*(?:or|and|[,/&]|或|和)\\s*[A-D]\\b', value[match.end():], re.I):
            continue
        letter = next((group for group in match.groups() if group))
        diagram = match.lastindex == 5 and match[0].lower().startswith('letter')
        committed.append(choice_value(letter.lower() if diagram else letter, choices, explicit_option=not diagram))
    if committed:
        return committed[0] if len(set(committed)) == 1 else None
    if len(value.splitlines()) > 1:
        listed = [choice_value(line, choices) for line in value.splitlines()]
        listed = [candidate for candidate in listed if candidate is not None]
        if listed and len(set(listed)) == 1:
            return listed[0]
    return None

def short_answer(text):
    value = clean_answer(text)
    if not value:
        return ''
    tail = re.split('\\n\\s*\\n', value)[-1].strip()
    marked = answer_markers(tail)
    return (marked[-1][1] if marked else tail).strip().strip('"\'').strip()

def yes_no_answer(text):
    text = clean_answer(text)
    equivalents = {'true': 'yes', 'false': 'no', '是': 'yes', '否': 'no', '不是': 'no'}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    conclusion = re.split('\\b(?:therefore|thus|hence|in conclusion)\\s*,?\\s*', text, flags=re.I)[-1]
    for value in [short_answer(text), conclusion, *lines[-1:], *lines[:1]]:
        value = re.sub('^(?:(?:therefore|thus|so|in conclusion|hence)[,:]?\\s*)+', '', value, flags=re.I)
        if value.lower().strip('.。!！') in equivalents:
            return equivalents[value.lower().strip('.。!！')]
        match = re.match('^(yes|no|y|n)\\b', value, re.I)
        if match:
            if re.match('\\s*(?:or|and|/|或|和)\\s*(?:yes|no)\\b', value[match.end():], re.I):
                return None
            return 'yes' if match[1].lower() in ('yes', 'y') else 'no'
    return None
CHART_NUMBER = '[+-]?(?:\\d{1,3}(?:[, ]\\d{3})+(?:\\.\\d+)?|\\d+(?:\\.\\d+)?|\\.\\d+)(?:[eE][+-]?\\d+)?'
CHART_UNIT = '(?:(?:thousand|million|billion|trillion|Canadian|Australian|U\\.?S\\.?|US|USD|EUR|GBP|RMB|dollars?|euros?|pounds?|yuan|people|persons?|households?|tons?|tonnes?|kg|kilograms?|km|kWh|miles?|hours?|years?|units?|per|capita|k|t)\\b\\.?\\s*)*'

def chart_scalar(value):
    value = clean_answer(value).rstrip('.。').replace('−', '-')
    value = re.sub('^(?:approximately|approx\\.?|about|around|roughly)\\s+', '', value, flags=re.I)
    value = re.sub('\\\\frac\\s*\\{([^{}]+)\\}\\s*\\{([^{}]+)\\}', '\\1/\\2', value)
    ratio = re.fullmatch('(' + CHART_NUMBER + ')\\s*[:/]\\s*(' + CHART_NUMBER + ')', value)
    if ratio:
        denominator = float(ratio[2].replace(',', '').replace(' ', ''))
        return str(float(ratio[1].replace(',', '').replace(' ', '')) / denominator) if denominator else None
    match = re.fullmatch('(?:[$€£¥]\\s*)?(' + CHART_NUMBER + ')\\s*(%?)\\s*' + CHART_UNIT, value, re.I)
    if match:
        return match[1].replace(',', '').replace(' ', '') + match[2]
    return None

def chart_answer(text, question=''):
    value = short_answer(text).strip().rstrip('.。')
    value = re.sub('^(?:the|a|an)\\s+', '', value, flags=re.I)
    direct = chart_scalar(value)
    if direct is not None:
        return direct
    if not re.search('\\d', value):
        return value
    emphasis = re.findall('\\*\\*([^*\\n]+)\\*\\*', text)
    if emphasis:
        emphasized = chart_scalar(emphasis[-1])
        if emphasized is not None:
            return emphasized
    conclusions = re.split('\\b(?:Therefore|Thus|Hence|In conclusion)[,:]?\\s*', value, flags=re.I)
    conclusion = conclusions[-1].strip()
    count = re.match('^(' + CHART_NUMBER + ')\\s+(?:countries|colors|bars|categories|people|items|years)\\b', conclusion, re.I)
    if count and (len(conclusions) > 1 or re.search('\\bhow many\\b', question, re.I)):
        return chart_scalar(count[1])
    committed = re.findall('\\b(?:is|was|were|equals|gives|at)\\s+(?:(?:approximately|about|around)\\s+)?(' + CHART_NUMBER + '\\s*%?\\s*' + CHART_UNIT + ')(?=[.。,;]|$)', conclusion, re.I)
    if committed:
        scalar = chart_scalar(committed[-1].strip())
        if scalar is not None:
            return scalar
    return value

def chart_text(value):
    value = re.sub('\\bgrey\\b', 'gray', clean_answer(value).casefold()).strip(' []"\'.。')
    return tuple(sorted((item.strip(' \t\n"\'') for item in re.split('\\s+and\\s+|\\s*[,;]\\s*', value))))

def normalized_ocr(value, formula=False):
    value = clean_answer(value)
    if formula:
        value = re.sub('\\\\(?:quad|qquad|enspace|left|right)\\b|\\\\[,;! ]', '', value)
        value = re.sub('\\\\triangle\\b', '\\\\Delta', value)
        value = re.sub('\\\\perp\\b', '\\\\bot', value)
        value = re.sub('\\{\\s*([A-Za-z0-9])\\s*\\}', '\\1', value)
        return re.sub('\\s+', '', value)
    value = re.sub('(?<=\\.)\\s+(?=\\w)', '', value.casefold())
    return re.sub('\\s+', ' ', value).strip()

def score_prediction(row, final_answer):
    text = final_answer.strip()
    source = row['source']
    prediction = text
    extracted = bool(text)
    correction = REFERENCE_CORRECTIONS['by_id'].get(row.get('id'), {})
    family = f"{source}:{row.get('circular_id')}"
    correction = {**REFERENCE_CORRECTIONS['by_family'].get(family, {}), **correction}
    reference = correction.get('answer', row['answer'])
    details = {'answer_extraction_version': ANSWER_EXTRACTION_VERSION}
    if correction:
        details['reference_correction'] = correction['reason']
    if correction.get('exclude'):
        details['excluded_from_metric'] = correction['reason']
    if row['choices']:
        prediction = choice_answer(text, row['choices'])
        accepted = correction.get('choice_texts', [row['choices'][reference]])
        accepted = [*accepted, *(row['choices'][reference].replace(old, new) for old, new in correction.get('choice_alias_replacements', []))]
        score = prediction is not None and any((clean_answer(row['choices'][prediction]) == clean_answer(answer) for answer in accepted))
        extracted = prediction is not None
    elif source == 'ocrbench':

        def normalized(value):
            if row['dataset'] == 'HME100k':
                return value.strip().replace('\n', ' ').replace(' ', '')
            value = ''.join((chr(ord(c) - 65248) if 65281 <= ord(c) <= 65374 else ' ' if c == '\u3000' else c for c in value))
            return value.lower().strip().replace('\n', ' ')
        details['literal_ocr_score'] = float(any((normalized(answer) in normalized(text) for answer in row['answer']))) if text else 0.0
        formula = row['dataset'] == 'HME100k'
        if formula:
            expressions = re.findall('\\$\\$(.*?)\\$\\$|\\\\\\[(.*?)\\\\\\]|(?<!\\$)\\$(?!\\$)(.*?)\\$', text, re.S)
            candidate = next((part for part in expressions[-1] if part), '') if expressions else short_answer(text)
            score = bool(details['literal_ocr_score']) or any((normalized_ocr(answer, True) == normalized_ocr(candidate, True) for answer in reference)) if text else False
        else:
            score = any((normalized_ocr(answer) in normalized_ocr(text) for answer in reference)) if text else False
    elif source == 'chartqa':
        prediction = chart_answer(text, row.get('question', ''))

        def numeric(value):
            try:
                number = float(value.rstrip('%'))
                return number if math.isfinite(number) else None
            except ValueError:
                return None
        pred = numeric(prediction)
        score = False
        for answer in [*reference, *correction.get('answer_aliases', [])]:
            target_text = chart_answer(answer)
            target = numeric(target_text)
            current = pred
            if target_text.endswith('%') and target is not None:
                target /= 100
                if prediction.endswith('%') and current is not None:
                    current /= 100
            elif prediction.endswith('%') and current is not None and (target is not None):
                score |= abs(current / 100 - target) <= 0.05 * abs(target)
            score |= abs(current - target) <= 0.05 * abs(target) if current is not None and target is not None else chart_text(prediction) == chart_text(target_text)
    elif source in ('mme', 'pope'):
        prediction = yes_no_answer(text)
        extracted = prediction is not None
        score = prediction == row['answer'].lower().strip().replace('.', '')
    else:
        raise ValueError(source)
    return {'score': float(score), 'prediction': prediction, 'extraction_failed': not extracted, **details}

def aggregate_scores(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['source']].append(row)
    result = {}
    for source, values in groups.items():
        total = len(values)
        values = [row for row in values if not row.get('excluded_from_metric')]
        entry = {'responses': total, 'scored_responses': len(values), 'excluded_responses': total - len(values), 'accuracy': sum((r['score'] for r in values)) / len(values)}
        subsets = defaultdict(list)
        for row in values:
            subsets[row['subset']].append(row)
        entry['subsets'] = {key: {'responses': len(part), 'accuracy': sum((r['score'] for r in part)) / len(part)} for key, part in subsets.items()}
        if source.startswith('mmbench'):
            circular = defaultdict(list)
            for row in values:
                circular[row['circular_id']].append(row['score'])
            entry['circular_questions'] = len(circular)
            entry['circular_accuracy'] = sum((all(scores) for scores in circular.values())) / len(circular)
            entry['answer_matching'] = ANSWER_EXTRACTION_VERSION + '_no_judge'
        elif source == 'mme':
            categories = defaultdict(lambda: defaultdict(list))
            for row in values:
                categories[row['category']][row['pair_id']].append(row['score'])
            entry['category_scores'] = {key: sum((100 * sum(pair) / len(pair) + 100 * all(pair) for pair in pairs.values())) / len(pairs) for key, pairs in categories.items()}
            entry['perception_score'] = sum((v for k, v in entry['category_scores'].items() if k in PERCEPTION))
            entry['cognition_score'] = sum((v for k, v in entry['category_scores'].items() if k not in PERCEPTION))
        elif source == 'pope':
            for key, part in subsets.items():
                tp = sum((r['prediction'] == 'yes' and r['answer'].lower() == 'yes' and (not r['extraction_failed']) for r in part))
                predicted = sum((r['prediction'] == 'yes' and (not r['extraction_failed']) for r in part))
                positives = sum((r['answer'].lower() == 'yes' for r in part))
                precision, recall = (tp / predicted if predicted else 0, tp / positives if positives else 0)
                entry['subsets'][key].update(precision=precision, recall=recall, f1=2 * precision * recall / (precision + recall) if precision + recall else 0, yes_ratio=predicted / len(part))
        elif source == 'ocrbench':
            entry['score_out_of_1000'] = sum((r['score'] for r in values))
            entry['literal_score_out_of_1000'] = sum((r.get('literal_ocr_score', r['score']) for r in values))
            category = defaultdict(list)
            for row in values:
                category[row['category']].append(row['score'])
            entry['category_scores'] = {key: sum(part) for key, part in category.items()}
        result[source] = entry
    return result
