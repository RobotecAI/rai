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
from datetime import datetime

import rai_bench.examples.manipulation_o3de.main as manipulation_o3de_bench
import rai_bench.examples.tool_calling_agent.main as tool_calling_agent_bench

if __name__ == "__main__":
    models_name = ["llama3.2", "qwen2.5:7b"]
    vendors = ["ollama", "ollama"]
    benchmarks = ["tool_calling_agent"]
    repeats = 1

    now = datetime.now()
    out_dir = (
        f"src/rai_bench/rai_bench/experiments/run_{now.strftime('%Y-%m-%d_%H-%M-%S')}/"
    )

    if len(models_name) != len(vendors):
        raise ValueError("Number of passed models must match number of passed vendors")
    else:
        for benchmark in benchmarks:
            for i, model_name in enumerate(models_name):
                for u in range(repeats):
                    curr_out_dir = out_dir + benchmark + "/" + model_name + "/" + str(u)
                    # try:
                    if benchmark == "tool_calling_agent":
                        tool_calling_agent_bench.run_benchmark(
                            model_name=model_name,
                            vendor=vendors[i],
                            out_dir=curr_out_dir,
                        )
                    elif benchmark == "manipulation_o3de":
                        manipulation_o3de_bench.run_benchmark(
                            model_name=model_name,
                            vendor=vendors[i],
                            out_dir=curr_out_dir,
                        )
                    else:
                        print(f"No benchmark named: {benchmark}")
                    # except Exception as e:
                    #     print(
                    #         f"Failed to run {benchmark} benchmark for {model_name}, vendor: {vendors[i]}, execution number: {u + 1}, because: {str(e)}"
                    #     )
