"""The evals from the command line::

    uv run python -m evals [--mode rules|llm|jev|hybrid|all] [--live] [--tags fast-path,history]
                           [--limit N] [--report PATH] [--advisor] [--concurrency N]

Offline by default, every mode (``--mode all``): the rules router on the whole
dataset, scripted models for llm/jev/hybrid (the plumbing, not the models),
and the advisor suite. ``--live`` reads the environment (and ``.env``), builds
the real models, and refuses to start without the keys the chosen modes need.
``--advisor`` adds the advisor suite to a single mode.

Prints the pydantic-evals table per mode and the thresholds, writes a Markdown
report (default ``evals/reports/<UTC timestamp>-<mode>.md``, ``-live`` added
for live runs) and exits 1 when a full-dataset run falls below THRESHOLDS in
``evals/harness.py``, 2 on a usage error or a missing key."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evals.cases import CASES
from evals.evaluators import ROUTING_CHECKS
from evals.harness import (
    HARD,
    JEV_USD_PER_MILLION_INPUT,
    MODES,
    RULES_KNOWN_FAILURES,
    THRESHOLD_METRICS,
    AdvisorRun,
    CaseResult,
    Rate,
    RoutingRun,
    RoutingSummary,
    live_settings,
    missing_advisor_keys,
    missing_live_keys,
    percentile,
    run_advisor,
    run_routing,
    select_cases,
)
from r3.settings import ControllerSettings

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
LIVE_CONCURRENCY = 4
METRIC_NAMES = {"intent_accuracy": "Intent accuracy", "params_pass_rate": "Params pass rate",
                "action_pass_rate": "Action pass rate"}
CONFIDENCE_BUCKETS = ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals", description=(
        "Evaluate the R3 Controller: routing per router mode, and the advisor's answers (--mode all, --advisor)."))
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all",
                        help="router mode to evaluate; 'all' (default) runs all four and the advisor suite")
    parser.add_argument("--live", action="store_true",
                        help="use real models from the environment/.env instead of scripted ones (needs keys)")
    parser.add_argument("--tags", default="", help="only cases with any of these comma-separated tags")
    parser.add_argument("--limit", type=int, default=None, help="only the first N (matching) cases")
    parser.add_argument("--report", default=None,
                        help="Markdown report path (default: evals/reports/<UTC timestamp>-<mode>.md)")
    parser.add_argument("--advisor", action="store_true", help="also run the advisor suite")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"cases in flight at once (default: 1 offline, {LIVE_CONCURRENCY} live)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    args.command = shlex.join(["uv", "run", "python", "-m", "evals", *argv])
    modes = list(MODES) if args.mode == "all" else [args.mode]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    for flag, value in (("--limit", args.limit), ("--concurrency", args.concurrency)):
        if value is not None and value < 1:
            print(f"{flag} must be at least 1", file=sys.stderr)
            return 2
    cases = select_cases(CASES, tags, args.limit)
    if not cases:
        print(f"No case carries any of the tags {', '.join(tags)}.", file=sys.stderr)
        return 2
    with_advisor = args.advisor or args.mode == "all"
    if args.live:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
        problems = [f"--mode {mode} needs {' and '.join(missing)}"
                    for mode in modes if (missing := missing_live_keys(mode, live_settings(mode)))]
        if with_advisor and (missing := missing_advisor_keys(ControllerSettings.from_env())):
            problems.append(f"the advisor suite needs {' and '.join(missing)}")
        if problems:
            print("Refusing to run live: " + "; ".join(problems) + ".\nSet the keys in the environment or in .env, "
                  "or drop --live for the offline run.", file=sys.stderr)
            return 2
    return asyncio.run(_run(args, modes, cases, tags, with_advisor))


async def _run(args: argparse.Namespace, modes: list[str], cases: list[Any], tags: list[str],
               with_advisor: bool) -> int:
    started = datetime.now(timezone.utc)
    # Offline cases run one at a time, so a turn's latency is its own and not the queue's.
    concurrency = args.concurrency or (LIVE_CONCURRENCY if args.live else 1)
    progress = args.live and sys.stdout.isatty()
    runs: list[RoutingRun] = []
    for mode in modes:
        run = await run_routing(mode, live=args.live, cases=cases, concurrency=concurrency,  # type: ignore[arg-type]
                                progress=progress)
        _print_routing(run)
        runs.append(run)
    advisor = None
    if with_advisor:
        advisor = await run_advisor(live=args.live, concurrency=concurrency, progress=progress)
        _print_advisor(advisor)
    path = Path(args.report) if args.report else REPORTS_DIR / (
        f"{started:%Y%m%dT%H%M%SZ}-{args.mode}{'-live' if args.live else ''}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(runs, advisor, started=started, live=args.live, tags=tags, limit=args.limit,
                                    command=args.command), encoding="utf-8")
    violations = [v for run in runs for v in run.summary.violations()] + (advisor.violations() if advisor else [])
    print(f"\nReport: {path}")
    if violations:
        print("Below threshold:\n  " + "\n  ".join(violations))
        return 1
    gated = any(run.summary.full_dataset for run in runs) or (advisor is not None and not advisor.live)
    print("All thresholds met." if gated else "Subset run: thresholds not applied.")
    return 0


# --- console -----------------------------------------------------------------------------


def _label(value: Any) -> str:
    if isinstance(value, dict):  # the Averages row: the share of each label
        return ", ".join(f"{k} {p:.0%}" for k, p in sorted(value.items(), key=lambda kv: -kv[1]))
    return str(value)


def _table_width() -> int | None:
    return None if sys.stdout.isatty() else 160


def _print_routing(run: RoutingRun) -> None:
    s = run.summary
    run.report.print(include_reasons=True, width=_table_width(),
                     label_configs={"intent": {"value_formatter": _label}, "router": {"value_formatter": _label}})
    rates, hard = s.rates(), s.rates(hard=True)
    scope = "full dataset" if s.full_dataset else "subset: thresholds not applied"
    print(f"\nThresholds for {s.mode} ({'live' if s.live else 'offline'}, {scope}; "
          f"{sum(not r.hard for r in s.results)} scored cases, {sum(r.hard for r in s.results)} hard reported apart):")
    floors = s.thresholds()
    for metric in THRESHOLD_METRICS:
        floor = floors.get(metric)
        status = _status(rates[metric], floor, s.full_dataset)
        shown = f"{floor:.2f}" if floor is not None else "-"
        print(f"  {metric:18} {rates[metric]!s:>16}   floor {shown:>4}   {status:9} hard {hard[metric]}")


def _print_advisor(run: AdvisorRun) -> None:
    run.report.print(include_reasons=True, width=_table_width())
    print(f"\nAdvisor ({'live' if run.live else 'offline'}, {run.model}):")
    for name, rate in run.rates().items():
        print(f"  {name:18} {rate}")


def _status(rate: Rate, floor: float | None, gated: bool) -> str:
    if rate.value is None:
        return "n/a"
    if floor is None or not gated:
        return "not gated"
    return "pass" if rate.value + 1e-9 >= floor else "BELOW"


# --- Markdown ------------------------------------------------------------------------------


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _pct(rate: Rate) -> str:
    return str(rate) if rate.total else "n/a"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else (f"{value:,.0f} ms" if value >= 100 else f"{value:.2f} ms")


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _verdict(summary: RoutingSummary) -> str:
    if not summary.full_dataset:
        return "not gated (subset)"
    return "pass" if not summary.violations() else "BELOW THRESHOLD"


def _expected_text(result: CaseResult) -> str:
    e = result.expected
    parts = [e.intent]
    if e.params:
        parts.append(", ".join(f"{k}={v}" for k, v in e.params.items()))
    if e.action:
        parts.append(f"action={e.action}")
    return "; ".join(parts)


def _got_text(result: CaseResult) -> str:
    if result.decision is None:
        return f"error: {result.error}"
    params = {k: v for k, v in result.decision.params.model_dump().items() if v not in (None, "", [], False)}
    shown = ", ".join(f"{k}={v}" for k, v in params.items())
    return "; ".join(p for p in (result.decision.intent, shown, f"action={result.action_kind}") if p)


def _mark(result: CaseResult, check: str) -> str:
    if check not in result.checks:
        return "–"
    return "✓" if result.checks[check] else "✗"


def render_markdown(runs: Sequence[RoutingRun], advisor: AdvisorRun | None, *, started: datetime, live: bool,
                    tags: Sequence[str] = (), limit: int | None = None, command: str = "") -> str:
    """The report: a summary across modes, then per mode the thresholds,
    per-intent accuracy, the confusion matrix, router usage and fallbacks,
    Jev's confidence, latency, tokens and cost, the failures and every case;
    then the advisor suite."""
    out: list[str] = []
    what = ", ".join(run.summary.mode for run in runs) or "advisor"
    out += [f"# R3 Controller evals: {what} ({'live' if live else 'offline'})", ""]
    first = runs[0].summary if runs else None
    if first is not None:
        n, n_hard = len(first.results), sum(r.hard for r in first.results)
        subset = "" if first.full_dataset else (
            f" (subset of {len(CASES)}: " + ", ".join(p for p in (f"tags {', '.join(tags)}" if tags else "",
                                                               f"limit {limit}" if limit else "") if p) + ")")
        out += [f"- Run: {started:%Y-%m-%d %H:%M} UTC, {'live models from the environment' if live else 'offline'}",
                f"- Dataset: `evals/cases.py`, {n} cases{subset}: {n - n_hard} scored, {n_hard} tagged `{HARD}` "
                "and reported apart",
                "- Each turn: `route_message` with the test database's facets, then `plan_action` against "
                "`StubDB` (one new AI/ML job, #7 Senior ML Engineer at Acme AI) and `FakeRuns`",
                "- Floors: `THRESHOLDS` in `evals/harness.py` (measured rate minus 0.03, rounded down to 0.05); "
                "they gate full-dataset runs only", ""]
    if len(runs) > 1:
        out += _cross_mode(runs)
    for run in runs:
        out += _mode_section(run)
    if advisor is not None:
        out += _advisor_section(advisor)
    command = command or f"uv run python -m evals --mode {what if len(runs) == 1 else 'all'}{' --live' if live else ''}"
    out += ["---", "", f"Produced by `{command}`.", ""]
    return "\n".join(out)


def _cross_mode(runs: Sequence[RoutingRun]) -> list[str]:
    out = ["## Across modes", "",
           "| Mode | Intent (scored) | Params | Action | Intent (hard) | Fallback to rules | Hybrid to LLM "
           "| Jev confidence (mean) | Latency mean / p95 | Tokens per turn (in / out) | Est. cost per turn | Verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        s = run.summary
        rates, hard, fallback = s.rates(), s.rates(hard=True), s.fallback_rates()
        confidences = s.jev_confidences()
        latencies = s.latencies()
        tokens_in, tokens_out = _tokens(s)
        per_turn = (f"{tokens_in / len(s.results):,.0f} / {tokens_out / len(s.results):,.0f}"
                    if s.live and s.results else "n/a (offline)")
        cost = f"${s.jev_cost_usd() / len(s.results):.8f}" if s.live and "jev" in s.usage() else "n/a"
        mean_conf = _mean(confidences)
        out.append("| " + " | ".join([
            s.mode, _pct(rates["intent_accuracy"]), _pct(rates["params_pass_rate"]), _pct(rates["action_pass_rate"]), _pct(hard["intent_accuracy"]),
            _pct(fallback["to_rules"]) if s.mode != "rules" else "n/a",
            _pct(fallback["to_llm"]) if "to_llm" in fallback else "n/a",
            f"{mean_conf:.2f} (n={len(confidences)})" if mean_conf is not None else "n/a",
            f"{_ms(_mean(latencies))} / {_ms(percentile(latencies, 0.95))}", per_turn, cost, _verdict(s),
        ]) + " |")
    out.append("")
    if not any(run.summary.live for run in runs):
        out += ["Offline, llm/jev/hybrid run on scripted models that answer with the expected route: their rows "
                "prove the plumbing, and the numbers that measure the models come from `--live`.", ""]
    return out


def _tokens(summary: RoutingSummary) -> tuple[int, int]:
    usage = summary.usage()
    return (sum(u.get("input_tokens", 0) for u in usage.values()),
            sum(u.get("output_tokens", 0) for u in usage.values()))


def _mode_section(run: RoutingRun) -> list[str]:
    s = run.summary
    rates, hard, floors = s.rates(), s.rates(hard=True), s.thresholds()
    scored_n, hard_n = sum(not r.hard for r in s.results), sum(r.hard for r in s.results)
    models = "; ".join(f"{role}: `{model}`" for role, model in run.models.items()) or "none (keyword rules)"
    out = [f"## Mode: {s.mode} ({'live' if s.live else 'offline'})", "",
           f"Models: {models}. Verdict: **{_verdict(s)}**."
           + (" Floors for live model modes are provisional (the rules' floors) until a live run is measured."
              if s.live and s.mode != "rules" else ""), "",
           f"| Metric | Scored ({scored_n}) | Floor | Status | Hard ({hard_n}) |", "|---|---|---|---|---|"]
    for metric in THRESHOLD_METRICS:
        floor = floors.get(metric)
        out.append(f"| {METRIC_NAMES[metric]} | {_pct(rates[metric])} | "
                   f"{f'{floor:.2f}' if floor is not None else '-'} | {_status(rates[metric], floor, s.full_dataset)} "
                   f"| {_pct(hard[metric])} |")
    out += ["", "Params and action rates count only the cases that state that expectation.", ""]

    out += ["### Per-intent accuracy", "", "| Expected intent | Scored | Hard |", "|---|---|---|"]
    for intent, (scored, hard_rate) in s.per_intent().items():
        if scored.total or hard_rate.total:
            out.append(f"| {intent} | {_pct(scored)} | {_pct(hard_rate)} |")
    out.append("")

    if s.confusion is not None:
        labels = s.confusion.class_labels
        out += ["### Confusion matrix", "", "All cases, hard included; rows are the expected intent, columns the "
                "router's. Off-diagonal cells are misroutes.", "",
                "| expected \\ predicted | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
        for label, row in zip(labels, s.confusion.matrix, strict=True):
            cells = [f"**{v}**" if label == col and v else str(v) for col, v in zip(labels, row, strict=True)]
            out.append(f"| **{label}** | " + " | ".join(cells) + " |")
        out.append("")

    out += _router_usage(s)
    out += _latency_and_cost(s)
    out += _failures(s)
    out += _all_cases(s)
    return out


def _router_usage(s: RoutingSummary) -> list[str]:
    routers = s.routers()
    total = len(s.results)
    out = ["### Router usage", "", "| Decided by | Turns | Share |", "|---|---|---|"]
    for router, count in routers.most_common():
        out.append(f"| {router} | {count} | {count / total:.1%} |")
    out.append("")
    if s.mode == "rules":
        out += ["The rules decide every turn: no model, no fallback.", ""]
    else:
        fallback = s.fallback_rates()
        model_turns = len(s.model_turns())
        lines = [f"Model turns (everything but the fast path): {model_turns} of {total}.",
                 f"Fallback to the rules after a model failure: {_pct(fallback['to_rules'])}."]
        if "to_llm" in fallback:
            lines.append(f"Hybrid, sent to the LLM because Jev was below `R3_JEV_CONFIDENCE` or failed: "
                         f"{_pct(fallback['to_llm'])}.")
        out += [" ".join(lines), ""]
    confidences = s.jev_confidences()
    if confidences:
        out += ["Jev's intent confidence, per Jev answer (including answers hybrid set aside), and the intent "
                "accuracy of the turns in each band:", "",
                "| Confidence | Answers | Intent accuracy |", "|---|---|---|"]
        for low, high in CONFIDENCE_BUCKETS:
            band = [r for r in s.results if r.jev_confidence is not None and low <= r.jev_confidence < high]
            rate = Rate()
            for r in band:
                rate.add(r.predicted == r.expected.intent)
            out.append(f"| {low:.1f}–{min(high, 1.0):.1f} | {len(band)} | {_pct(rate)} |")
        out += ["", f"Mean {_mean(confidences):.3f}, min {min(confidences):.3f}, max {max(confidences):.3f}."]
        decided = [r for r in s.results if r.decision is not None and r.decision.confidence is not None]
        right = sum(r.predicted == r.expected.intent for r in decided)
        for scalar in s.scalars:  # the precision-recall AUC says something only when Jev was sometimes wrong
            out.append(f"{scalar.title}: {scalar.value:.3f}." if 0 < right < len(decided) else
                       f"{scalar.title}: not informative (Jev decided {len(decided)} turns, {right} correctly).")
        out.append("")
    return out


def _latency_and_cost(s: RoutingSummary) -> list[str]:
    everything, model_turns = s.latencies(), s.latencies(model_turns_only=True)
    out = ["### Latency", "", "Wall time of `route_message` per turn (the fast path included in 'all turns').", "",
           "| Turns | n | Mean | p95 | Max |", "|---|---|---|---|---|"]
    for name, values in (("All turns", everything), ("Model turns", model_turns)):
        if values:
            out.append(f"| {name} | {len(values)} | {_ms(_mean(values))} | {_ms(percentile(values, 0.95))} "
                       f"| {_ms(max(values))} |")
    out.append("")
    usage = s.usage()
    out += ["### Tokens and cost", ""]
    if not s.live:
        requests = sum(u.get("requests", 0) for u in usage.values())
        out += [f"Offline: no tokens spent. Model requests to the scripted models: {requests} "
                f"({requests / max(1, len(s.model_turns())):.2f} per model turn)." if s.mode != "rules" else
                "The rules router makes no model request.", ""]
        return out
    if not usage:
        return out + ["No model request was made.", ""]
    out += ["| Model | Requests | Input tokens | Output tokens | Per turn (in / out) | Est. cost |",
            "|---|---|---|---|---|---|"]
    turns = max(1, len(s.results))
    for role, used in usage.items():
        cost = (f"${used['input_tokens'] * JEV_USD_PER_MILLION_INPUT / 1_000_000:.6f} "
                f"(${JEV_USD_PER_MILLION_INPUT}/M input, output free)" if role == "jev"
                else "price unknown: tokens only")
        out.append(f"| {role} | {used['requests']} | {used['input_tokens']:,} | {used['output_tokens']:,} | "
                   f"{used['input_tokens'] / turns:,.0f} / {used['output_tokens'] / turns:,.0f} | {cost} |")
    out.append("")
    return out


def _failures(s: RoutingSummary) -> list[str]:
    failures = s.failures()
    if not failures:
        return ["### Failures", "", "None.", ""]
    out = [f"### Failures ({len(failures)})", "",
           "| Case | Tags | Message | Expected | Got | Router: reason | Failed checks |",
           "|---|---|---|---|---|---|---|"]
    for r in failures:
        known = " (known)" if s.known(r) else ""
        failed = "; ".join(f"{check}: {r.reasons.get(check, '')}".rstrip(": ") for check, ok in r.checks.items()
                           if not ok)
        reason = f"{r.router}: {r.decision.reason}" if r.decision else r.router
        out.append(f"| {_cell(r.name)}{known} | {_cell(', '.join(r.tags))} | {_cell(r.text)} | "
                   f"{_cell(_expected_text(r))} | {_cell(_got_text(r))} | {_cell(reason)} | {_cell(failed)} |")
    out.append("")
    if s.mode == "rules":
        stale = [name for name in RULES_KNOWN_FAILURES if name in {r.name for r in s.results}
                 and name not in {r.name for r in failures}]
        if stale:
            out += [f"Known failures that now pass (remove them from `RULES_KNOWN_FAILURES`): {', '.join(stale)}.", ""]
        new = [r.name for r in failures if not s.known(r)]
        out += [f"New failures (not in `RULES_KNOWN_FAILURES`): {', '.join(new) if new else 'none'}.", ""]
    return out


def _all_cases(s: RoutingSummary) -> list[str]:
    out = ["### All cases", "", "✓ passed, ✗ failed, – not stated by the case.", "",
           "| Case | Tags | Expected | Predicted | Router | " + " | ".join(ROUTING_CHECKS) + " |",
           "|---|---|---|---|---|" + "---|" * len(ROUTING_CHECKS)]
    for r in s.results:
        marks = " | ".join(_mark(r, check) for check in ROUTING_CHECKS)
        out.append(f"| {_cell(r.name)} | {_cell(', '.join(r.tags))} | {r.expected.intent} | {r.predicted} "
                   f"| {r.router} | {marks} |")
    out.append("")
    return out


def _advisor_section(run: AdvisorRun) -> list[str]:
    out = [f"## Advisor ({'live' if run.live else 'offline'})", "",
           f"Model: `{run.model}`. Judges: {', '.join(run.judges) or 'none (offline: deterministic checks only)'}. "
           "Each case scripts the route, so the router is not under test; the action comes from `plan_action` "
           "against `StubDB` and `FakeRuns`.", "",
           "| Check | Pass rate |", "|---|---|"]
    out += [f"| {name} | {rate} |" for name, rate in run.rates().items()]
    out.append("")
    failures = run.failures()
    if failures:
        out += [f"Failures ({len(failures)}):", "", "| Case | Check | Reason |", "|---|---|---|"]
        out += [f"| {_cell(case)} | {_cell(check)} | {_cell(reason)} |" for case, check, reason in failures]
        out.append("")
    out += ["| Case | Action | Events | Answer |", "|---|---|---|---|"]
    tokens_in = tokens_out = 0
    for rc in run.report.cases:
        o = rc.output
        tokens_in += o["usage"].get("input_tokens", 0)
        tokens_out += o["usage"].get("output_tokens", 0)
        answer = o["answer"] or f"(error: {o['error']})"
        out.append(f"| {_cell(rc.name)} | {o['action_kind']} | {_cell(' '.join(o['events']))} | "
                   f"{_cell(answer[:200] + ('…' if len(answer) > 200 else ''))} |")
    out.append("")
    if run.live:
        turns = max(1, len(run.report.cases))
        out += [f"Advisor tokens: {tokens_in:,} in / {tokens_out:,} out ({tokens_in / turns:,.0f} / "
                f"{tokens_out / turns:,.0f} per turn; price unknown: tokens only).", ""]
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
