"""Run the agent with the real model on every fixture and report the eval metrics.

Usage: python evals/run_evals.py [--fixtures kitchen_sink clean ...] [--workers 3]
                                 [--no-llm-matcher]

Prints every match decision, a table per fixture and overall, and saves the results to
evals/results/<timestamp>.json. Calls the real LLM, so it costs money.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from dva.cli import build_deps
from dva.config import Settings, configure_logging, load_settings
from dva.llm import create_chat_model, invoke_structured
from dva.orchestrator.graph import run_batch
from dva.state import Finding

sys.path.insert(0, str(Path(__file__).parent))
from metrics import (  # noqa: E402
    Issue,
    RunRecord,
    mapping_accuracy,
    match_findings,
    meets_target,
    ratio,
    summarize,
)

MATCHER_PROMPT = """\
You audit a data validation agent. Answer "yes" only if the finding describes the same
problem as the ground-truth issue (same field or column and the same kind of problem;
the count may be approximate or expressed differently). Otherwise answer "no".

Finding: field={field}; issue_type={issue_type}; count={count}; detail={detail}
Ground-truth issue: field={gt_field}; issue_type={gt_type}; expected count={gt_count}"""


class Verdict(BaseModel):
    answer: Literal["yes", "no"]
    reason: str


class Matcher:
    """LLM judge for findings that code matching left unmatched; counts its tokens."""

    def __init__(self, settings: Settings) -> None:
        self._model = create_chat_model(settings, "eval_matcher")
        self._retries = settings.llm.structured_output_retries
        self.tokens = 0

    def __call__(self, finding: Finding, issue: Issue) -> bool:
        prompt = MATCHER_PROMPT.format(
            field=finding.field, issue_type=finding.issue_type.value, count=finding.count,
            detail=finding.detail, gt_field=issue.field, gt_type=issue.issue_type,
            gt_count=issue.expected,
        )  # fmt: skip
        messages = [SystemMessage("Answer with the structured verdict only."), HumanMessage(prompt)]
        result = invoke_structured(self._model, Verdict, messages, retries=self._retries)
        self.tokens += result.tokens_used
        return result.parsed.answer == "yes"


def run_fixture(name: str, settings: Settings, stamp: str, use_matcher: bool) -> RunRecord:
    """Run one fixture end to end and score it. Writes under workspace/<session>/."""
    folder = settings.paths.fixtures_dir / name
    gt = json.loads((folder / "ground_truth.json").read_text())
    session = f"eval-{name.replace('_', '-')}-{stamp}"[:64]
    events: list[tuple[str, dict[str, Any]]] = []
    deps = build_deps(settings, None).model_copy(
        update={"recorder": lambda tool, data: events.append((tool, data))})  # fmt: skip
    record = RunRecord(fixture=name, session=session)
    started = time.monotonic()
    try:
        state = run_batch(deps, session, [str(folder / f) for f in gt["files"].values()])
    except Exception as exc:  # an eval must report crashes, not stop on them
        return record.model_copy(update={"error": f"{type(exc).__name__}: {exc}",
                                         "latency_s": time.monotonic() - started})  # fmt: skip
    record.latency_s = round(time.monotonic() - started, 1)
    return score(record, state, gt, events, Matcher(settings) if use_matcher else None)


Events = list[tuple[str, dict[str, Any]]]


def score(record: RunRecord, state: dict[str, Any], gt: dict[str, Any], events: Events,
          matcher: Matcher | None) -> RunRecord:  # fmt: skip
    """Fill in the metrics for one run from its final state and tool events."""
    issues = [Issue(**i) for i in gt["expected_issues"]]
    findings = state.get("findings") or []
    decisions, found, correct = match_findings(findings, issues, matcher)
    mapped, total = mapping_accuracy(state, gt["expected_field_mapping"])
    eda = [data for tool, data in events if tool == "run_eda"]
    first_pass = [ok for _, data in events for ok in data.get("first_pass", [])]
    return record.model_copy(update={
        "stop_reason": state.get("stop_reason"), "report_ref": state.get("report_ref"),
        "mapping_correct": mapped, "mapping_total": total,
        "issues_found": len(found), "issues_total": len(issues),
        "findings_correct": correct, "findings_total": len(findings),
        "eda_ok": sum(bool(d.get("ok")) for d in eda), "eda_total": len(eda),
        "first_pass_ok": sum(first_pass), "first_pass_total": len(first_pass),
        "tool_calls": state.get("tool_calls") or 0, "tokens": state.get("tokens_used") or 0,
        "matcher_tokens": matcher.tokens if matcher else 0, "decisions": decisions,
    })  # fmt: skip


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def print_decisions(records: list[RunRecord]) -> None:
    print("\nMatch decisions")
    for record in records:
        for d in record.decisions:
            target = d.issue or "no ground-truth issue"
            print(f"  {record.fixture:18} {d.method:5} {'MATCH' if d.matched else 'miss ':5} "
                  f"{d.finding}  ->  {target}")  # fmt: skip


def _row(r: RunRecord) -> str:
    mapping = _pct(ratio(r.mapping_correct, r.mapping_total))
    recall = _pct(ratio(r.issues_found, r.issues_total))
    precision = _pct(ratio(r.findings_correct, r.findings_total))
    outcome = r.error or r.stop_reason
    return (f"{r.fixture:18} {mapping:>5} {recall:>6} {precision:>5} "
            f"{r.eda_ok}/{r.eda_total:<3} {r.first_pass_ok}/{r.first_pass_total:<3} "
            f"{r.tool_calls:>5} {r.tokens:>7} {r.latency_s:>6.1f}  {outcome}")  # fmt: skip


def print_table(records: list[RunRecord], overall: dict[str, Any]) -> None:
    header = f"{'fixture':18} {'map':>5} {'recall':>6} {'prec':>5} {'eda':>5} {'1st':>5} " \
             f"{'calls':>5} {'tokens':>7} {'secs':>6}  stop"  # fmt: skip
    print("\n" + header + "\n" + "-" * len(header))
    for r in records:
        print(_row(r))
    print("\nOverall")
    for name in ("field_mapping_accuracy", "issue_recall", "issue_precision",
                 "eda_execution_success", "first_pass_schema_rate", "cap_hit_rate"):  # fmt: skip
        verdict = {True: "PASS", False: "FAIL", None: "report only"}[
            meets_target(name, overall[name])
        ]
        print(f"  {name:24} {_pct(overall[name]):>6}  {verdict}")
    for name in ("tool_calls", "latency_s", "tokens"):
        print(f"  {name:24} mean {overall[name]['mean']:.1f}, max {overall[name]['max']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures", nargs="*", help="fixture names (default: all)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-llm-matcher", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    configure_logging(settings)
    names = args.fixtures or sorted(p.name for p in settings.paths.fixtures_dir.iterdir()
                                    if (p / "ground_truth.json").is_file())  # fmt: skip
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda n: run_fixture(n, settings, stamp, not args.no_llm_matcher),
                                names))  # fmt: skip
    overall = summarize(records)
    print_decisions(records)
    print_table(records, overall)
    out = settings.paths.eval_results_dir / f"{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"overall": overall, "runs": [r.model_dump() for r in records]},
                              indent=2))  # fmt: skip
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
