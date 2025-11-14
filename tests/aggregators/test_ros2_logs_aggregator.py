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

from dataclasses import dataclass

from langchain_core.messages import HumanMessage  # type: ignore[import-untyped]
from rai.aggregators.ros2.aggregators import (
    ROS2LogsAggregator,  # type: ignore[import-untyped]
)


@dataclass
class DummyLog:
    level: int
    name: str
    function: str
    msg: str


def test_ros2_logs_aggregator_deduplicates_and_clears_buffer():
    aggregator = ROS2LogsAggregator()

    repeated_log = DummyLog(
        level=30, name="demo_node", function="do_work", msg="System warming up"
    )
    unique_log = DummyLog(
        level=40, name="demo_node", function="do_work", msg="System failure detected"
    )

    aggregator(repeated_log)
    aggregator(DummyLog(**repeated_log.__dict__))
    aggregator(unique_log)

    summary = aggregator.get()

    assert isinstance(summary, HumanMessage)
    assert (
        summary.content
        == "Logs summary: ['[demo_node] [WARNING] [do_work] System warming up', 'Log above repeated 1 times']"
    )
    assert aggregator.get_buffer() == []


def test_ros2_logs_aggregator_str():
    aggregator = ROS2LogsAggregator()
    assert str(aggregator) == "ROS2LogsAggregator(len=0)"
    aggregator(
        DummyLog(
            level=30, name="demo_node", function="do_work", msg="System warming up"
        )
    )
    assert str(aggregator) == "ROS2LogsAggregator(len=1)"


def test_ros2_logs_aggregator_overflow():
    aggregator = ROS2LogsAggregator(max_size=2)
    for i in range(10):
        aggregator(
            DummyLog(
                level=30,
                name="demo_node",
                function="do_work",
                msg=f"System warming up: {i}",
            )
        )
    assert len(aggregator.get_buffer()) == 2
    assert aggregator.get_buffer()[0].msg == "System warming up: 8"
    assert aggregator.get_buffer()[1].msg == "System warming up: 9"
