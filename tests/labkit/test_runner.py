"""The runner turns check functions into per-task results without stopping at the first failure."""

from labkit.catalog import Fidelity, Lab, Task
from labkit.runner import CheckFailed, run_checks


def _lab(tmp_path, tasks):
    return Lab(id="demo", title="Demo", level="beginner", minutes=5, skills=["x"], services=["sqs"],
               fidelity=Fidelity(real=["r"], differs_on_aws=["d"]), tasks=tasks,
               interview_questions=["q"], path=tmp_path)


def test_results_per_task(tmp_path):
    def ok(aws):
        return None

    def missing(aws):
        raise CheckFailed("queue 'orders' not found")

    def boom(aws):
        raise RuntimeError("connection refused")

    tasks = [Task(id="a", title="A", description="", check="ok"),
             Task(id="b", title="B", description="", check="missing"),
             Task(id="c", title="C", description="", check="boom")]
    results = run_checks(_lab(tmp_path, tasks), {"ok": ok, "missing": missing, "boom": boom}, aws=None)
    assert [(r.task.id, r.passed) for r in results] == [("a", True), ("b", False), ("c", False)]
    assert results[1].message == "queue 'orders' not found"
    assert "RuntimeError" in results[2].message and "connection refused" in results[2].message


def test_only_selected_task(tmp_path):
    calls = []
    tasks = [Task(id="a", title="A", description="", check="a"), Task(id="b", title="B", description="", check="b")]
    checks = {"a": lambda aws: calls.append("a"), "b": lambda aws: calls.append("b")}
    results = run_checks(_lab(tmp_path, tasks), checks, aws=None, only="b")
    assert calls == ["b"] and [r.task.id for r in results] == ["b"]
