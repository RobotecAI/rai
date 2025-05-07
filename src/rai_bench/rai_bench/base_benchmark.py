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

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field


class BenchmarkSummary(BaseModel):
    model_name: str = Field(..., description="Name of the LLM.")
    success_rate: float = Field(
        ..., description="Percentage of successfully completed tasks."
    )
    avg_time: float = Field(..., description="Average time taken across all tasks.")
    total_extra_tool_calls_used: int = Field(
        ..., description="Total number of extra tool calls used in this Task"
    )
    total_tasks: int = Field(..., description="Total number of executed tasks.")


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(
        self,
        model_name: str,
        results_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the benchmark with model information and output locations.

        Parameters
        ----------
        model_name : str
            Name of the LLM model being benchmarked.
        results_dir : Path
            Directory path where benchmark results will be stored.
        logger : logging.Logger | None, optional
            Logger instance for tracking benchmark execution. If None,
            a default logger will be created.
        """
        self.model_name = model_name
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results_filename = results_dir / "results.csv"
        self.summary_filename = results_dir / "results_summary.csv"

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    @staticmethod
    def csv_initialize(filename: Path, base_model_cls: type[BaseModel]) -> None:
        """Initialize a CSV file based on a Pydantic model class.

        Parameters
        ----------
        filename : Path
            Filename of the CSV file.
        base_model_cls : type[BaseModel]
            Pydantic model class to be used for creating the columns in the CSV file.
        """
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=base_model_cls.__annotations__.keys()
            )
            writer.writeheader()

    @staticmethod
    def csv_writerow(filename: Path, base_model_instance: BaseModel) -> None:
        """Write a single row to a CSV file based on a Pydantic model instance contents,
        ensuring that multiline strings are converted to one-line strings by replacing newlines.

        Parameters
        ----------
        filename : Path
            Filename of the CSV file.
        base_model_instance : BaseModel
            Pydantic model instance which contains the data to be written to the CSV file.
        """
        row = base_model_instance.model_dump()

        for key, value in row.items():
            if isinstance(value, str):
                # Replace newline characters with a single space so they don't break csv
                row[key] = " ".join(value.split())

        with open(filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=base_model_instance.__annotations__.keys()
            )
            writer.writerow(row)

    @abstractmethod
    def run_next(self, agent: CompiledStateGraph) -> None:
        """Run the next task/scenario of the benchmark.

        Parameters
        ----------
        agent : CompiledStateGraph
            LangChain tool calling agent.
        """
        pass

    @abstractmethod
    def compute_and_save_summary(self) -> None:
        """Compute summary statistics and save them to the summary file."""
        pass

    # TODO (jm) this can be probably same for all benchmark in the future
