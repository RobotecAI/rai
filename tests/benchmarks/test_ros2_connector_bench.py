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

"""Runs the ROS2Connector benchmark. Report-only: it fails only if the script
does. Deselected by default; run it with `pytest -m benchmark tests/benchmarks -s`.

RAI_BENCH_OUT sets the JSON path, RAI_BENCH_ARGS passes extra script arguments
(e.g. "--quick --compare baseline.json").
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).with_name("ros2_connector_bench.py")


@pytest.mark.benchmark
@pytest.mark.timeout(1200)
def test_ros2_connector_benchmark(tmp_path):
    out = os.environ.get("RAI_BENCH_OUT") or str(tmp_path / "bench.json")
    args = shlex.split(os.environ.get("RAI_BENCH_ARGS", ""))
    subprocess.run([sys.executable, str(BENCH), "--out", out, *args], check=True)
    results = json.loads(Path(out).read_text())["results"]
    print(json.dumps(results, indent=1))
