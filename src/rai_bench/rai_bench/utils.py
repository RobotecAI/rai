# Copyright (C) 2025 Robotec.AI
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

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.chat_models.base import BaseChatModel
from rai.initialization import get_llm_model_direct


def parse_tool_calling_benchmark_args():
    parser = argparse.ArgumentParser(description="Run the Tool Calling Agent Benchmark")
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use for benchmarking",
        required=True,
    )
    parser.add_argument("--vendor", type=str, help="Vendor of the model", required=True)
    parser.add_argument(
        "--extra-tool-calls",
        type=int,
        help="Number of extra tools calls agent can make and still pass the task",
        default=0,
    )
    parser.add_argument(
        "--complexities",
        type=str,
        nargs="+",
        choices=["easy", "medium", "hard"],
        default=["easy", "medium", "hard"],
        help="Complexity levels of tasks to include in the benchmark",
    )
    parser.add_argument(
        "--prompt-detail",
        type=str,
        nargs="+",
        choices=["brief", "descriptive"],
        default=["brief", "descriptive"],
        help="Prompt detail level to include in the benchmark",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        nargs="+",
        choices=[0, 2, 5],
        default=[0, 2, 5],
        help="Number of examples in system prompt for few-shot prompting",
    )
    parser.add_argument(
        "--task-types",
        type=str,
        nargs="+",
        choices=[
            "basic",
            "manipulation",
            "custom_interfaces",
            "spatial_reasoning",
        ],
        default=[
            "basic",
            "manipulation",
            "custom_interfaces",
            "spatial_reasoning",
        ],
        help="Types of tasks to include in the benchmark",
    )
    now = datetime.now()
    parser.add_argument(
        "--out-dir",
        type=str,
        default=f"src/rai_bench/rai_bench/experiments/tool_calling/{now.strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Output directory for results and logs",
    )
    return parser.parse_args()


def parse_manipulation_o3de_benchmark_args():
    parser = argparse.ArgumentParser(description="Run the Manipulation O3DE Benchmark")
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use for benchmarking",
        required=True,
    )
    parser.add_argument("--vendor", type=str, help="Vendor of the model", required=True)
    parser.add_argument(
        "--o3de-config-path",
        type=str,
        default="src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml",
        help="Path to the O3DE configuration file",
    )
    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        choices=["trivial", "easy", "medium", "hard", "very_hard"],
        default=["trivial", "easy", "medium", "hard", "very_hard"],
        help="Difficulty levels to include in the benchmark",
    )
    now = datetime.now()
    parser.add_argument(
        "--out-dir",
        type=str,
        default=f"src/rai_bench/rai_bench/experiments/o3de_manipulation/{now.strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Output directory for results and logs",
    )
    return parser.parse_args()


def define_benchmark_logger(out_dir: Path, level: int = logging.INFO) -> logging.Logger:
    log_file = out_dir / "benchmark.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    bench_logger = logging.getLogger("Benchmark logger")
    for handler in bench_logger.handlers:
        bench_logger.removeHandler(handler)
    bench_logger.setLevel(level)
    bench_logger.addHandler(file_handler)

    return bench_logger


def get_llm_for_benchmark(model_name: str, vendor: str, **kwargs: Any) -> BaseChatModel:
    if vendor == "ollama":
        llm = get_llm_model_direct(
            model_name=model_name, vendor=vendor, keep_alive=20, **kwargs
        )
    else:
        llm = get_llm_model_direct(model_name=model_name, vendor=vendor, **kwargs)
    return llm


def get_llm_model_name(llm: BaseChatModel) -> str:
    """Get the actual model name from any LLM, regardless of vendor"""

    # Try common attribute names in order of preference
    for attr in ["model", "model_name", "deployment_name"]:
        if hasattr(llm, attr):
            value = getattr(llm, attr)
            if value:
                return str(value)

    # Fallback to vendor name if model name not found
    return llm.get_name()
