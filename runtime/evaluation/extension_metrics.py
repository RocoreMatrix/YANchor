import json

def score_prediction(row):
    from evaluation.drop_metric import get_metrics
    from scoring.verification import evaluation_answer_text, typed_declared_answer, compare_answers
    source = row['source']
    text = str(row['visible_response']) if row.get('visible_response') is not None else evaluation_answer_text(row['response'])
    metadata = json.loads(row['metadata_json'])
    if source in {'MuSR', 'LogiQA2'}:
        answer = typed_declared_answer(text, answer_type='choice', problem=row['prompt'])
        score = float(bool(answer and (not answer['ambiguous']) and (compare_answers(answer['candidate_answer'], [row['reference_answer']], answer_type='choice', problem=row['prompt'])['verdict'] == 'PASS')))
        return {'score': score, 'exact_match': score, 'f1': None}
    if source == 'DROP':
        declared = typed_declared_answer(text, answer_type='drop_official')
        if declared and declared['ambiguous']:
            return {'score': 0.0, 'exact_match': 0.0, 'f1': 0.0}
        prediction = declared['candidate_answer'] if declared else text
        if prediction.startswith('['):
            try:
                value = json.loads(prediction)
                if isinstance(value, list) and all((isinstance(s, str) for s in value)):
                    prediction = value
            except json.JSONDecodeError:
                pass
        metrics = [get_metrics(prediction, gold) for gold in metadata['reference_answers']]
        em = max((x[0] for x in metrics), default=0.0)
        f1 = float(max((x[1] for x in metrics), default=0.0))
        return {'score': f1, 'exact_match': em, 'f1': f1}
    raise ValueError(f'No scorer selected for {source}')
