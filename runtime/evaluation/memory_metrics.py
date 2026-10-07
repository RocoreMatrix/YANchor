import re

def parse_queries(text):
    from scoring.verification import extract_declared_answer
    parts = re.split('(?:^|\\n)\\s*Q([1-4])\\s*[:：]', text)
    answers = {}
    for i in range(1, len(parts) - 1, 2):
        body = re.split('(?:Evidence|依据)\\s*[:：]', parts[i + 1], maxsplit=1)[0]
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if lines:
            value = lines[-1].removeprefix('Answer:').strip().strip('`$ ')
            marked = extract_declared_answer(value, allow_hash_answer=True)
            answers[parts[i]] = marked['candidate_answer'] if marked else value
    return answers

def score_generation(case, text):
    from scoring.verification import compare_answers
    answers = parse_queries(text)
    for q, gold in case['answers'].items():
        value = answers.get(q, '')
        if re.fullmatch('K[0-9a-fA-F]{8}', gold) and re.fullmatch('K[0-9a-fA-F]{8}(?:\\s*(?:->|→)\\s*K[0-9a-fA-F]{8})+', value):
            answers[q] = re.split('\\s*(?:->|→)\\s*', value)[-1]
        elif re.fullmatch('-?\\d+(?:\\.\\d+)?', gold) and re.fullmatch('[\\d\\s.+*/()−-]+\\s*=\\s*-?\\d+(?:\\.\\d+)?', value):
            answers[q] = value.rsplit('=', 1)[1].strip()
    correct = {q: compare_answers(answers.get(q), [gold])['verdict'] == 'PASS' for q, gold in case['answers'].items()}
    return dict(query_predictions=answers, query_correct=correct, parsed_query_fraction=sum((q in answers for q in correct)) / len(correct), exact_match=sum(correct.values()) / len(correct), all_queries_correct=all(correct.values()))
