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

from pathlib import Path

from rai_bench import (
    define_benchmark_logger,
    parse_tool_calling_benchmark_args,
)
from rai_bench.agents import ConversationalAgentFactory, agent_factory
from rai_bench.tool_calling_agent import (
    get_tasks,
    run_benchmark,
)
from rai_bench.utils import get_llm_for_benchmark

if __name__ == "__main__":
    args = parse_tool_calling_benchmark_args()
    experiment_dir = Path(args.out_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    bench_logger = define_benchmark_logger(out_dir=experiment_dir)

    tasks = get_tasks(
        extra_tool_calls=args.extra_tool_calls,
        complexities=args.complexities,
        task_types=args.task_types,
        n_shots=args.n_shots,
        prompt_detail=args.prompt_detail,
    )
    for task in tasks:
        task.set_logger(bench_logger)

    llm = get_llm_for_benchmark(
        model_name=args.model_name,
        vendor=args.vendor,
    )

    run_benchmark(
        llm=llm,
        out_dir=experiment_dir,
        tasks=tasks,
        bench_logger=bench_logger,
    )
