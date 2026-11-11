#!/usr/bin/env python3
# Copyright (C) 2026 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare ROS2Connector benchmark runs of a base and a head ref.

Usage:
    python scripts/bench_compare.py --base base-*.json --head head-*.json \\
        [--summary "$GITHUB_STEP_SUMMARY"]

Each file is the JSON written by tests/benchmarks/ros2_connector_bench.py --out.
Repeated runs of one ref are reduced to per-metric medians. Every metric is
classified by the first matching rule in RULES:

- exact:  counts that must not get worse (leaked entities, drops, errors,
          deliveries). Fails when head is worse than base at all.
- ratio:  steady-state latencies and throughput (run-to-run CV <= ~10 %).
          Fails when head/base crosses the threshold, warns at WARN_RATIO.
- report: noisy metrics (cold start, first call, p99, wall times). Shown, never fail.
- ignore: bookkeeping (sample counts, error texts).

Metrics present on only one side are reported as added or removed and never
fail, so the benchmark can gain or lose metrics without breaking the job. A
benchmark that errors on head but not on base fails.

Exit code: 1 if any metric failed, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

Kind = Literal["exact", "ratio", "report", "ignore"]
Status = Literal["fail", "warn", "ok", "info", "added", "removed"]

WARN_RATIO = 1.10


@dataclass(frozen=True)
class Rule:
    pattern: str
    kind: Kind
    # True: larger values are better (throughput, deliveries).
    higher_is_better: bool = False
    # ratio only: allowed worsening factor, e.g. 1.25 means head may be 25 % worse.
    threshold: float = 1.25
    # ratio only: absolute differences below this never fail (sub-ms jitter).
    min_abs: float = 0.0


# First match wins. Patterns are fnmatch globs over flattened dotted metric names.
RULES: list[Rule] = [
    # bookkeeping
    Rule("*.n", "ignore"),
    Rule("*.sent", "ignore"),
    Rule("*.first_error", "ignore"),
    Rule("*.first", "ignore"),
    Rule("*.skipped", "ignore"),
    Rule("leak.register_unregister", "ignore"),
    # noisy: cold start, first call, fresh clients, tails, wall times (04 §4.5: CV 50-137 %)
    Rule("*elapsed_s", "report"),
    Rule("*.cold.*", "report"),
    Rule("*first_call*", "report"),
    Rule("*.fresh.*", "report"),
    Rule("*first_feedback*", "report"),
    Rule("*.p99_ms", "report"),
    Rule("*.done", "report"),
    # The current dispatcher is lossy and its latency is bimodal: A/A runs of one commit gave
    # 31 vs 22 of 100 delivered at 100 Hz and p50 0.4 vs 5.9 ms. Promote delivered, dropped and
    # inversions to exact and latency.p50_ms to ratio (1.5x, 1 ms floor) once the R2/R3
    # dispatch rewrite is stable (rai3-plan/04 §4.5).
    Rule("dispatch_*.delivered", "report"),
    Rule("dispatch_*.dropped", "report"),
    Rule("dispatch_*.inversions", "report"),
    Rule("dispatch_*.latency.*", "report"),
    # A cache read at ~1.5M/s; varies 1.4x between runs.
    Rule("receive_message.warm.*", "report"),
    # exact: correctness and leaks
    Rule("*.errors", "exact"),
    Rule("*.dropped", "exact"),
    Rule("*.inversions", "exact"),
    Rule("*.delivered", "exact", higher_is_better=True),
    Rule("leak.*.subscriptions", "exact"),
    Rule("leak.*.publishers", "exact"),
    Rule("leak.*.clients", "exact"),
    Rule("leak.*.waitables", "exact"),
    # ratio: steady state
    # Executor pools grow lazily, so thread counts differ by a few between runs; a leak per call
    # adds hundreds.
    Rule("leak.*.threads", "ratio", min_abs=10),
    Rule("leak.*.rss_kb", "ratio", min_abs=20_000),
    Rule("start_action.accepted.p50_ms", "ratio", threshold=1.5, min_abs=1.0),
    # The benchmark's slow callbacks sleep 50 ms; waiting for one shows up as +50 ms.
    Rule("shutdown.duration_ms", "ratio", min_abs=25.0),
    Rule("service_call.concurrent_8x2.p50_ms", "ratio", threshold=1.5, min_abs=0.5),
    Rule(
        "service_call.concurrent_8x2.per_s",
        "ratio",
        higher_is_better=True,
        threshold=1.5,
    ),
    Rule("*.p50_ms", "ratio", min_abs=0.05),
    Rule("*.per_s", "ratio", higher_is_better=True),
    # wall times of whole phases (leak.*_s): depend on timeouts and retries
    Rule("*_s", "report"),
    # anything new defaults to report-only until someone classifies it
    Rule("*", "report"),
]


def classify(metric: str, rules: list[Rule] = RULES) -> Rule:
    for rule in rules:
        if fnmatchcase(metric, rule.pattern):
            return rule
    return Rule("*", "report")


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{path}."))
        elif value is not None and not isinstance(value, bool):
            out[path] = value
    return out


def load_runs(paths: list[Path]) -> dict[str, Any]:
    """Per-metric median over runs.

    A text value (a benchmark error) is kept only if most runs have it, so one flaky run
    does not count as an error.
    """
    samples: dict[str, list[Any]] = {}
    for path in paths:
        results = json.loads(path.read_text()).get("results", {})
        for key, value in flatten(results).items():
            samples.setdefault(key, []).append(value)
    merged: dict[str, Any] = {}
    for key, values in samples.items():
        numbers = [v for v in values if isinstance(v, (int, float))]
        if numbers:
            merged[key] = statistics.median(numbers)
        elif 2 * len(values) > len(paths):
            merged[key] = values[0]
    return merged


@dataclass(frozen=True)
class Row:
    metric: str
    kind: Kind
    base: Any
    head: Any
    status: Status
    note: str = ""


def _worse_factor(base: float, head: float, higher_is_better: bool) -> float:
    """How many times worse head is than base (1.0 = equal, > 1 = worse)."""
    if higher_is_better:
        return base / head if head else float("inf")
    return head / base


def compare_metric(metric: str, base: Any, head: Any, rules: list[Rule] = RULES) -> Row:
    if metric == "error" or metric.endswith(".error"):
        if head is not None and base is None:
            return Row(metric, "exact", base, head, "fail", "new benchmark error")
        if head is None:
            return Row(metric, "exact", base, head, "removed", "error gone on head")
        return Row(metric, "exact", base, head, "info", "errors on both refs")
    rule = classify(metric, rules)
    if base is None:
        return Row(metric, rule.kind, base, head, "added")
    if head is None:
        return Row(metric, rule.kind, base, head, "removed")
    numeric = isinstance(base, (int, float)) and isinstance(head, (int, float))
    if rule.kind in ("report", "ignore") or not numeric:
        return Row(metric, rule.kind, base, head, "info")
    if rule.kind == "exact":
        worse = head < base if rule.higher_is_better else head > base
        return Row(metric, "exact", base, head, "fail" if worse else "ok")
    # ratio
    if base == 0:
        return Row(metric, "ratio", base, head, "info", "base is 0")
    factor = _worse_factor(base, head, rule.higher_is_better)
    note = f"{factor:.2f}x"
    if abs(head - base) < rule.min_abs:
        return Row(metric, "ratio", base, head, "ok", note)
    if factor > rule.threshold:
        return Row(metric, "ratio", base, head, "fail", f"{note} > {rule.threshold}x")
    if factor > WARN_RATIO:
        return Row(metric, "ratio", base, head, "warn", note)
    return Row(metric, "ratio", base, head, "ok", note)


def compare(
    base: dict[str, Any], head: dict[str, Any], rules: list[Rule] = RULES
) -> list[Row]:
    return [
        compare_metric(key, base.get(key), head.get(key), rules)
        for key in sorted(base.keys() | head.keys())
    ]


ICONS: dict[Status, str] = {
    "fail": ":x:",
    "warn": ":warning:",
    "ok": ":white_check_mark:",
    "info": "",
    "added": ":new:",
    "removed": ":heavy_minus_sign:",
}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")[:80]


def _table(rows: list[Row]) -> list[str]:
    lines = ["| metric | class | base | head | | note |", "|---|---|---:|---:|---|---|"]
    for r in rows:
        lines.append(
            f"| `{r.metric}` | {r.kind} | {_fmt(r.base)} | {_fmt(r.head)} "
            f"| {ICONS[r.status]} | {r.note} |"
        )
    return lines


def render(rows: list[Row], title: str = "Connector benchmark: head vs base") -> str:
    visible = [r for r in rows if r.kind != "ignore"]
    count = {s: sum(r.status == s for r in visible) for s in ICONS}
    lines = [
        f"### {title}",
        "",
        f"{count['fail']} failed, {count['warn']} warned, {count['ok']} ok, "
        f"{count['added']} added, {count['removed']} removed "
        f"(medians over repeated runs; ratio fails above the per-metric threshold, "
        f"warns above {WARN_RATIO}x).",
        "",
    ]
    flagged = [r for r in visible if r.status in ("fail", "warn")]
    if flagged:
        lines += _table(flagged) + [""]
    lines += ["<details><summary>All metrics</summary>", ""]
    lines += _table(visible) + ["", "</details>", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", nargs="+", type=Path, required=True)
    parser.add_argument("--head", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        default=os.environ.get("GITHUB_STEP_SUMMARY") or None,
        help="append the Markdown report here (default: $GITHUB_STEP_SUMMARY)",
    )
    args = parser.parse_args(argv)
    rows = compare(load_runs(args.base), load_runs(args.head))
    report = render(rows)
    print(report)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(report + "\n")
    return 1 if any(r.status == "fail" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
