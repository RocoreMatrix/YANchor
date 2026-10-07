import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

class Platform(Enum):
    LEETCODE = 'leetcode'
    CODEFORCES = 'codeforces'
    ATCODER = 'atcoder'

class Difficulty(Enum):
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'

class TestType(Enum):
    STDIN = 'stdin'
    FUNCTIONAL = 'functional'

@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)

@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)
        self.public_test_cases = json.loads(self.public_test_cases)
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]
        try:
            self.private_test_cases = json.loads(self.private_test_cases)
        except:
            self.private_test_cases = json.loads(pickle.loads(zlib.decompress(base64.b64decode(self.private_test_cases.encode('utf-8')))))
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]
        self.metadata = json.loads(self.metadata)

    def get_evaluation_sample(self):
        return {'input_output': json.dumps({'inputs': [t.input for t in self.public_test_cases + self.private_test_cases], 'outputs': [t.output for t in self.public_test_cases + self.private_test_cases], 'fn_name': self.metadata.get('func_name', None)})}
