"""The `labs` command, without a stack: list, show, and a clear message when MiniStack is down."""

from labkit.cli import main

from .test_catalog import LABS


def test_list(capsys):
    assert main(["--labs", str(LABS), "list"]) == 0
    out = capsys.readouterr().out
    assert "lambda-sqs-pipeline" in out and "beginner" in out


def test_show_prints_fidelity_and_questions(capsys):
    assert main(["--labs", str(LABS), "show", "lambda-sqs-pipeline"]) == 0
    out = capsys.readouterr().out
    assert "Real here" in out and "On real AWS" in out and "Interview questions" in out
    assert "visibility timeout" in out


def test_unknown_lab(capsys):
    assert main(["--labs", str(LABS), "show", "nope"]) == 2
    assert "no lab 'nope'" in capsys.readouterr().err


def test_check_when_stack_is_down(capsys):
    code = main(["--labs", str(LABS), "--endpoint", "http://127.0.0.1:9", "check", "lambda-sqs-pipeline"])
    assert code == 2
    assert "docker compose up -d ministack" in capsys.readouterr().err


def test_unknown_task(capsys, monkeypatch):
    monkeypatch.setattr("labkit.cli.health", lambda endpoint: {})
    assert main(["--labs", str(LABS), "check", "lambda-sqs-pipeline", "nope"]) == 2
    assert "no task 'nope'" in capsys.readouterr().err
