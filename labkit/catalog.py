"""Labs on disk. Each lab is a folder under ``labs/`` holding ``lab.toml``
(what to build and what to expect in an interview), ``README.md`` (the
scenario), ``check.py`` (one function per task, each given an ``Aws``) and
``solution/solve.py`` (a reference solution with ``solve(aws)``)."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class LabError(Exception):
    pass


class Fidelity(BaseModel):
    """How close the local stack is to the real cloud for this lab, so a
    learner never walks into an interview with the wrong idea."""

    model_config = ConfigDict(extra="forbid")
    real: list[str]  # what actually runs locally
    stand_in: list[str] = Field(default_factory=list)  # accepted but not enforced, or emulated
    differs_on_aws: list[str]  # what to expect on the real service


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    description: str
    check: str  # name of a function in check.py
    hint: str = ""


class Lab(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    level: Literal["beginner", "intermediate", "advanced"]
    minutes: int
    skills: list[str]  # lowercase skill names a resume or job posting would use
    services: list[str]
    fidelity: Fidelity
    tasks: list[Task]
    interview_questions: list[str]
    path: Path

    @model_validator(mode="after")
    def _unique_task_ids(self) -> "Lab":
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        return self


def load_lab(path: Path) -> Lab:
    toml = path / "lab.toml"
    try:
        data = tomllib.loads(toml.read_text())
        lab = Lab(**data, path=path)
    except (OSError, tomllib.TOMLDecodeError, ValidationError, TypeError) as e:
        raise LabError(f"{toml}: {e}") from e
    if lab.id != path.name:
        raise LabError(f"{toml}: id {lab.id!r} must match its folder name {path.name!r}")
    return lab


def load_catalog(root: Path) -> list[Lab]:
    return [load_lab(p) for p in sorted(root.iterdir()) if (p / "lab.toml").is_file()]


def find_lab(root: Path, lab_id: str) -> Lab:
    for lab in load_catalog(root):
        if lab.id == lab_id:
            return lab
    raise LabError(f"no lab {lab_id!r}; `labs list` shows them")


def _module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise LabError(f"{path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checks(lab: Lab) -> dict[str, Callable]:
    """Public functions named ``check_<name>`` in check.py, keyed by name."""
    module = _module(lab.path / "check.py", f"labs_{lab.id.replace('-', '_')}_check")
    return {name.removeprefix("check_"): fn for name, fn in vars(module).items()
            if name.startswith("check_") and callable(fn)}


def load_solution(lab: Lab) -> Callable:
    module = _module(lab.path / "solution" / "solve.py", f"labs_{lab.id.replace('-', '_')}_solve")
    return module.solve
