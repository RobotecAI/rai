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
import time
from datetime import datetime
from pathlib import Path
from typing import List

import rclpy
from langchain.tools import BaseTool
from rai.agents.conversational_agent import create_conversational_agent
from rai.communication.ros2.connectors import ROS2Connector
from rai.initialization import get_llm_model_direct
from rai.tools.ros2 import (
    GetObjectPositionsTool,
    GetROS2ImageTool,
    GetROS2TopicsNamesAndTypesTool,
    MoveToPointTool,
)
from rai_open_set_vision.tools import GetGrabbingPointTool

from rai_bench.examples.manipulation_o3de.scenarios import (
    trivial_scenarios,
)
from rai_bench.manipulation_o3de.benchmark import ManipulationO3DEBenchmark
from rai_sim.o3de.o3de_bridge import (
    O3DEngineArmManipulationBridge,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Tool Calling Agent Benchmark")
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use for benchmarking",
    )
    parser.add_argument(
        "--vendor", type=str, default=None, help="Vendor of the model (optional)"
    )
    now = datetime.now()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"src/rai_bench/rai_bench/experiments/o3de_manipulation/{now.strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Output directory for results and logs",
    )
    return parser.parse_args()


def run_benchmark(model_name: str, vendor: str, out_dir: str):
    rclpy.init()
    connector = ROS2Connector()
    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

    # define model
    llm = get_llm_model_direct(model_name=model_name, vendor=vendor)

    # define tools
    tools: List[BaseTool] = [
        GetObjectPositionsTool(
            connector=connector,
            target_frame="panda_link0",
            source_frame="RGBDCamera5",
            camera_topic="/color_image5",
            depth_topic="/depth_image5",
            camera_info_topic="/color_camera_info5",
            get_grabbing_point_tool=GetGrabbingPointTool(connector=connector),
        ),
        MoveToPointTool(connector=connector, manipulator_frame="panda_link0"),
        GetROS2ImageTool(connector=connector),
        GetROS2TopicsNamesAndTypesTool(connector=connector),
    ]
    # define loggers
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_file = f"{out_dir}/benchmark.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    bench_logger = logging.getLogger("Benchmark logger")
    bench_logger.setLevel(logging.INFO)
    bench_logger.addHandler(file_handler)

    agent_logger = logging.getLogger("Agent logger")
    agent_logger.setLevel(logging.INFO)
    agent_logger.addHandler(file_handler)

    configs_dir = "src/rai_bench/rai_bench/examples/manipulation_o3de/configs/"
    connector_path = configs_dir + "o3de_config.yaml"
    #### Create scenarios manually
    # load different scenes
    # one_carrot_simulation_config = O3DExROS2SimulationConfig.load_config(
    #     base_config_path=Path(configs_dir + "scene1.yaml"),
    #     connector_config_path=Path(connector_path),
    # )
    # multiple_carrot_simulation_config = O3DExROS2SimulationConfig.load_config(
    #     base_config_path=Path(configs_dir + "scene2.yaml"),
    #     connector_config_path=Path(connector_path),
    # )
    # red_cubes_simulation_config = O3DExROS2SimulationConfig.load_config(
    #     base_config_path=Path(configs_dir + "scene3.yaml"),
    #     connector_config_path=Path(connector_path),
    # )
    # multiple_cubes_simulation_config = O3DExROS2SimulationConfig.load_config(
    #     base_config_path=Path(configs_dir + "scene4.yaml"),
    #     connector_config_path=Path(connector_path),
    # )
    # # combine different scene configs with the tasks to create various scenarios
    # scenarios = [
    #     Scenario(
    #         task=GrabCarrotTask(logger=bench_logger),
    #         simulation_config=one_carrot_simulation_config,
    #         simulation_config_path=configs_dir + "scene1.yaml",
    #     ),
    #     Scenario(
    #         task=GrabCarrotTask(logger=bench_logger),
    #         simulation_config=multiple_carrot_simulation_config,
    #         simulation_config_path=configs_dir + "scene2.yaml",
    #     ),
    #     Scenario(
    #         task=PlaceCubesTask(logger=bench_logger),
    #         simulation_config=red_cubes_simulation_config,
    #         simulation_config_path=configs_dir + "scene3.yaml",
    #     ),
    #     Scenario(
    #         task=PlaceCubesTask(logger=bench_logger),
    #         simulation_config=multiple_cubes_simulation_config,
    #         simulation_config_path=configs_dir + "scene4.yaml",
    #     ),
    # ]

    ### import ready scenarios
    t_scenarios = trivial_scenarios(
        configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
    )
    # e_scenarios = easy_scenarios(
    #     configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
    # )
    # m_scenarios = medium_scenarios(
    #     configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
    # )
    # h_scenarios = hard_scenarios(
    #     configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
    # )
    # vh_scenarios = very_hard_scenarios(
    #     configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
    # )

    all_scenarios = t_scenarios
    o3de = O3DEngineArmManipulationBridge(connector, logger=agent_logger)
    try:
        # define benchamrk
        benchmark = ManipulationO3DEBenchmark(
            model_name=model_name,
            simulation_bridge=o3de,
            scenarios=all_scenarios,
            logger=bench_logger,
            results_dir=Path(out_dir),
        )
        for scenario in all_scenarios:
            agent = create_conversational_agent(
                llm, tools, scenario.task.system_prompt, logger=agent_logger
            )
            benchmark.run_next(agent=agent)
            o3de.reset_arm()
            time.sleep(0.2)  # admire the end position for a second ;)

        bench_logger.info(
            "==============================================================="
        )
        bench_logger.info("ALL SCENARIOS DONE. BENCHMARK COMPLETED!")
        bench_logger.info(
            "==============================================================="
        )
    finally:
        connector.shutdown()
        o3de.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(model_name=args.model_name, vendor=args.vendor, out_dir=args.out_dir)
