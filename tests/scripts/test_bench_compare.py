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

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "bench_compare.py"
_spec = importlib.util.spec_from_file_location("bench_compare", SCRIPT)
assert _spec and _spec.loader
bc = importlib.util.module_from_spec(_spec)
sys.modules["bench_compare"] = bc  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(bc)


@pytest.mark.parametrize(
    ("metric", "kind"),
    [
        ("publish.string.per_s", "ratio"),
        ("publish.image_vga.p50_ms", "ratio"),
        ("dispatch_hri_30hz.latency.p50_ms", "report"),
        ("receive_message.warm.per_s", "report"),
        ("service_call.reused.p50_ms", "ratio"),
        ("start_action.result.p50_ms", "ratio"),
        ("shutdown.duration_ms", "ratio"),
        ("leak.after_actions.rss_kb", "ratio"),
        ("leak.after_actions.threads", "ratio"),
        ("dispatch_100hz.delivered", "report"),
        ("dispatch_hri_30hz.dropped", "report"),
        ("dispatch_1khz.inversions", "report"),
        ("other_bench.delivered", "exact"),
        ("other_bench.dropped", "exact"),
        ("leak.after_service_calls.clients", "exact"),
        ("leak.after_actions.waitables", "exact"),
        ("service_call.concurrent_8x2.errors", "exact"),
        ("leak.service_call_errors.errors", "exact"),
        ("receive_message.cold.p50_ms", "report"),
        ("receive_message.cold.per_s", "report"),
        ("service_call.first_call.p50_ms", "report"),
        ("service_call.fresh.p50_ms", "report"),
        ("service_call.fresh.per_s", "report"),
        ("dispatch_1khz.latency.p99_ms", "report"),
        ("service_call.reused.p99_ms", "report"),
        ("start_action.first_feedback.p50_ms", "report"),
        ("leak.service_calls_s", "report"),
        ("publish.elapsed_s", "report"),
        ("some_future_bench.mystery", "report"),
        ("publish.string.n", "ignore"),
        ("dispatch_100hz.sent", "ignore"),
        ("leak.action_errors.first", "ignore"),
        ("hooks.skipped", "ignore"),
    ],
)
def test_classification(metric, kind):
    assert bc.classify(metric).kind == kind


def status(metric, base, head):
    return bc.compare_metric(metric, base, head).status


def test_ratio_latency_thresholds():
    metric = "service_call.reused.p50_ms"
    assert status(metric, 1.0, 1.05) == "ok"
    assert status(metric, 1.0, 1.2) == "warn"
    assert status(metric, 1.0, 1.3) == "fail"
    assert status(metric, 1.0, 0.5) == "ok"


def test_ratio_throughput_is_higher_better():
    metric = "publish.string.per_s"
    assert status(metric, 1000, 1100) == "ok"
    assert status(metric, 1000, 850) == "warn"
    assert status(metric, 1000, 700) == "fail"
    assert status(metric, 1000, 0) == "fail"


def test_ratio_per_metric_threshold():
    assert status("service_call.concurrent_8x2.p50_ms", 4.0, 5.6) == "warn"
    assert status("service_call.concurrent_8x2.p50_ms", 4.0, 6.4) == "fail"


def test_ratio_ignores_tiny_absolute_changes():
    # 0.08 ms -> 0.12 ms is 1.5x but below the 0.05 ms jitter floor.
    assert status("publish.image_vga.p50_ms", 0.08, 0.12) == "ok"
    assert status("publish.image_vga.p50_ms", 0.08, 0.2) == "fail"


def test_ratio_with_zero_base_is_informational():
    assert status("service_call.reused.p50_ms", 0, 5.0) == "info"


def test_jitter_floors_from_the_aa_run():
    # Values from an A/A run of one commit on Lyrical; none of these is a regression.
    assert status("shutdown.duration_ms", 3.3, 7.5) == "ok"
    assert status("shutdown.duration_ms", 3.3, 55.0) == "fail"
    assert status("start_action.accepted.p50_ms", 2.0, 2.9) == "ok"
    assert status("start_action.accepted.p50_ms", 2.0, 4.0) == "fail"
    assert status("leak.after_actions.threads", 37.5, 42) == "ok"
    assert status("leak.after_actions.threads", 37, 237) == "fail"


def test_exact_lower_is_better():
    assert status("leak.after_service_calls.clients", 2, 2) == "ok"
    assert status("leak.after_service_calls.clients", 200, 2) == "ok"
    assert status("leak.after_service_calls.clients", 2, 3) == "fail"


def test_exact_higher_is_better():
    assert status("other_bench.delivered", 100, 100) == "ok"
    assert status("other_bench.delivered", 100, 99) == "fail"


def test_report_never_fails():
    assert status("receive_message.cold.p99_ms", 500, 3000) == "info"
    assert status("dispatch_1khz.latency.p99_ms", 26, 228) == "info"


def test_missing_metrics_are_reported_not_failed():
    rows = bc.compare(
        {"old.p50_ms": 1.0, "kept.p50_ms": 1.0},
        {"new.p50_ms": 9.0, "kept.p50_ms": 1.0},
    )
    by_metric = {r.metric: r.status for r in rows}
    assert by_metric == {
        "kept.p50_ms": "ok",
        "new.p50_ms": "added",
        "old.p50_ms": "removed",
    }


def test_benchmark_errors():
    assert status("publish.error", None, "TimeoutError: x") == "fail"
    assert status("publish.error", "TimeoutError: x", None) == "removed"
    assert status("publish.error", "TimeoutError: x", "TimeoutError: y") == "info"


def test_non_numeric_value_is_informational():
    assert status("service_call.reused.p50_ms", "n/a", 1.0) == "info"


def write_run(path: Path, results: dict) -> Path:
    path.write_text(json.dumps({"meta": {}, "results": results}))
    return path


def test_load_runs_takes_medians_and_flattens(tmp_path):
    runs = [
        write_run(
            tmp_path / f"r{i}.json",
            {"svc": {"p50_ms": v, "first_error": None}, "ok": True},
        )
        for i, v in enumerate([1.0, 9.0, 2.0])
    ]
    assert bc.load_runs(runs) == {"svc.p50_ms": 2.0}


def test_load_runs_keeps_errors_seen_in_most_runs(tmp_path):
    def runs(errors):
        return [
            write_run(
                tmp_path / f"e{i}.json",
                {"svc": {"error": e}} if e else {"svc": {"p50_ms": 1.0}},
            )
            for i, e in enumerate(errors)
        ]

    assert "svc.error" not in bc.load_runs(runs(["boom", None, None]))
    assert bc.load_runs(runs(["boom", "boom", None]))["svc.error"] == "boom"


def test_main_exit_code_and_summary(tmp_path):
    base = write_run(tmp_path / "base.json", {"svc": {"p50_ms": 1.0, "n": 10}})
    good = write_run(tmp_path / "good.json", {"svc": {"p50_ms": 1.02, "n": 10}})
    bad = write_run(tmp_path / "bad.json", {"svc": {"p50_ms": 2.0, "n": 10}})
    summary = tmp_path / "summary.md"

    assert (
        bc.main(["--base", str(base), "--head", str(good), "--summary", str(summary)])
        == 0
    )
    assert (
        bc.main(["--base", str(base), "--head", str(bad), "--summary", str(summary)])
        == 1
    )
    text = summary.read_text()
    assert text.count("### Connector benchmark") == 2
    assert "`svc.p50_ms`" in text
    assert "svc.n" not in text  # ignored metrics are hidden
    assert "1 failed" in text
