"""`labs`: list and read labs, check your work against the local stack, reset it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aws import Aws, health, reset_stack
from .catalog import LabError, find_lab, load_catalog, load_checks, load_solution
from .runner import run_checks
from .suggest import suggest

DEFAULT_LABS = Path(__file__).resolve().parent.parent / "labs"


def _bullets(title: str, items: list[str]) -> list[str]:
    return [f"\n{title}:", *(f"  - {i}" for i in items)] if items else []


def cmd_list(args) -> int:
    for lab in load_catalog(args.labs):
        print(f"{lab.id:28} {lab.level:13} {lab.minutes:>3} min  {lab.title}")
    return 0


def cmd_show(args) -> int:
    lab = find_lab(args.labs, args.lab)
    lines = [f"{lab.title}  ({lab.level}, ~{lab.minutes} min)", f"Skills: {', '.join(lab.skills)}", "\nTasks:"]
    lines += [f"  {n}. [{t.id}] {t.title}\n     {t.description}" for n, t in enumerate(lab.tasks, 1)]
    lines += _bullets("Real here", lab.fidelity.real)
    lines += _bullets("Stand-ins here", lab.fidelity.stand_in)
    lines += _bullets("On real AWS", lab.fidelity.differs_on_aws)
    lines += _bullets("Interview questions", lab.interview_questions)
    print("\n".join(lines))
    return 0


def _stack(args) -> Aws:
    try:
        health(args.endpoint)
    except OSError:
        raise LabError(f"no local stack at {args.endpoint}; start it with `docker compose up -d ministack`")
    return Aws(args.endpoint)


def cmd_check(args) -> int:
    lab = find_lab(args.labs, args.lab)
    if args.task and args.task not in {t.id for t in lab.tasks}:
        raise LabError(f"no task {args.task!r} in {lab.id}; tasks: {', '.join(t.id for t in lab.tasks)}")
    aws = _stack(args)
    results = run_checks(lab, load_checks(lab), aws, only=args.task)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {r.task.id:12} {r.task.title}")
        if not r.passed:
            print(f"      {r.message}")
            if r.task.hint and args.hint:
                print(f"      hint: {r.task.hint}")
    done = sum(r.passed for r in results)
    print(f"\n{done}/{len(results)} passed" + ("" if done == len(results) or args.hint else "  (add --hint for hints)"))
    return 0 if done == len(results) else 1


def cmd_suggest(args) -> int:
    text = " ".join(args.text) if args.text else sys.stdin.read()
    found = suggest(load_catalog(args.labs), text)
    if not found:
        print("No lab matches those skills yet.")
        return 1
    for s in found:
        print(f"{s.lab.id:28} {s.lab.level:13} matches: {', '.join(s.matched)}")
    return 0


def cmd_reset(args) -> int:
    _stack(args)
    reset_stack(args.endpoint)
    print("Local stack reset: every service is empty.")
    return 0


def cmd_solve(args) -> int:
    lab = find_lab(args.labs, args.lab)
    load_solution(lab)(_stack(args))
    print(f"Applied the reference solution for {lab.id}. Run `labs check {lab.id}`.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="labs", description=__doc__)
    p.add_argument("--labs", type=Path, default=DEFAULT_LABS, help=argparse.SUPPRESS)
    p.add_argument("--endpoint", default=None, help="local stack URL (default $LABS_ENDPOINT or http://localhost:4566)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the labs").set_defaults(fn=cmd_list)
    s = sub.add_parser("show", help="tasks, what is real, interview questions")
    s.add_argument("lab")
    s.set_defaults(fn=cmd_show)
    c = sub.add_parser("check", help="check your work")
    c.add_argument("lab")
    c.add_argument("task", nargs="?")
    c.add_argument("--hint", action="store_true", help="show hints for failing tasks")
    c.set_defaults(fn=cmd_check)
    g = sub.add_parser("suggest", help="labs for the skills in a resume or job posting (text or stdin)")
    g.add_argument("text", nargs="*")
    g.set_defaults(fn=cmd_suggest)
    sub.add_parser("reset", help="wipe the local stack").set_defaults(fn=cmd_reset)
    v = sub.add_parser("solve", help="apply the reference solution (spoiler)")
    v.add_argument("lab")
    v.set_defaults(fn=cmd_solve)
    args = p.parse_args(argv)
    args.endpoint = Aws(args.endpoint).endpoint
    try:
        return args.fn(args)
    except LabError as e:
        print(f"labs: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
