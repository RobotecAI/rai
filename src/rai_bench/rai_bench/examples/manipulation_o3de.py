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

from rai_bench import define_benchmark_logger, parse_manipulation_o3de_benchmark_args
from rai_bench.manipulation_o3de import get_scenarios, run_benchmark
from rai_bench.utils import get_llm_for_benchmark

if __name__ == "__main__":
    args = parse_manipulation_o3de_benchmark_args()
    experiment_dir = Path(args.out_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    bench_logger = define_benchmark_logger(out_dir=experiment_dir)

    # import ready scenarios
    scenarios = get_scenarios(logger=bench_logger, levels=args.levels)

    llm = get_llm_for_benchmark(
        model_name=args.model_name,
        vendor=args.vendor,
    )
    run_benchmark(
        llm=llm,
        out_dir=experiment_dir,
        o3de_config_path=args.o3de_config_path,
        scenarios=scenarios,
        bench_logger=bench_logger,
    )
