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

from rai_bench.manipulation_o3de.benchmark import (
    run_benchmark,
)
from rai_bench.manipulation_o3de.predefined.scenarios import (
    trivial_scenarios,
)
from rai_bench.utils import define_benchmark_logger, parse_benchmark_args

if __name__ == "__main__":
    args = parse_benchmark_args()
    experiment_dir = Path(args.out_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    bench_logger = define_benchmark_logger(out_dir=experiment_dir)

    configs_dir = "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/"
    connector_path = configs_dir + "o3de_config.yaml"
    ### import ready scenarios
    t_scenarios = trivial_scenarios(logger=bench_logger)
    # e_scenarios = easy_scenarios(
    #     logger=bench_logger
    # )
    # m_scenarios = medium_scenarios(
    #      logger=bench_logger
    # )
    # h_scenarios = hard_scenarios(
    #      logger=bench_logger
    # )
    # vh_scenarios = very_hard_scenarios(
    #      logger=bench_logger
    # )

    all_scenarios = t_scenarios
    run_benchmark(
        model_name=args.model_name,
        vendor=args.vendor,
        out_dir=experiment_dir,
        o3de_config_path=connector_path,
        scenarios=all_scenarios,
        bench_logger=bench_logger,
    )
