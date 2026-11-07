"""Reference solution: point the checks at solution/answers/ instead of answers/."""

from pathlib import Path


def solve(aws) -> None:
    aws.code_dir = Path(__file__).resolve().parent / "answers"
