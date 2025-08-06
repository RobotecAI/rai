from typing import List

from langchain_core.tools import BaseTool

from rai_bench.tool_calling_agent.interfaces import Task, TaskArgs
from rai_bench.tool_calling_agent.mocked_tools import (
    MockGetROS2TopicsNamesAndTypesTool,
    MockReceiveROS2MessageTool,
)
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)


# This Task will check if robot can receive msessage from specified topic
class GetROS2RobotPositionTask(Task):
    complexity = "easy"
    type = "custom"

    @property
    def available_tools(self) -> List[BaseTool]:
        # define topics that will be seen by agent
        TOPICS = [
            "/robot_position",
            "/attached_collision_object",
            "/clock",
            "/collision_object",
        ]

        TOPICS_STRING = [
            "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
            "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
            "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
            "topic: /robot_position\n type: sensor_msgs/msg/RobotPosition",
        ]
        # define which tools will be available for agent
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPICS_STRING
            ),
            MockReceiveROS2MessageTool(available_topics=TOPICS),
        ]

    def get_system_prompt(self) -> str:
        return "You are a ROS 2 expert that want to solve tasks. You have access to various tools that allow you to query the ROS 2 system."

    def get_base_prompt(self) -> str:
        return "Get the position of the robot."

    def get_prompt(self) -> str:
        # Create versions for different levels
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can discover what topics are currently active."
            )

    @property
    def optional_tool_calls_number(self) -> int:
        # Listing topics before getting any message is fine
        return 1


# define subtask
receive_robot_pos_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_position"},
    expected_optional_args={
        "timeout_sec": int  # if there is not exact value expected, you can pass type
    },
)
# use OrderedCallValidator as there is only 1 subtask to check
topics_ord_val = OrderedCallsValidator(subtasks=[receive_robot_pos_subtask])


# optionally pass number of extra tool calls
args = TaskArgs(extra_tool_calls=0)
super_task = GetROS2RobotPositionTask(validators=[topics_ord_val], task_args=args)

##### Now you can run it in benchmark ##################
if __name__ == "__main__":
    from pathlib import Path

    from rai_bench import (
        define_benchmark_logger,
    )
    from rai_bench.tool_calling_agent import (
        run_benchmark,
    )
    from rai_bench.utils import get_llm_for_benchmark

    experiment_dir = Path("src/rai_bench/rai_bench/experiments/custom_task")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    bench_logger = define_benchmark_logger(out_dir=experiment_dir)

    super_task.set_logger(bench_logger)

    llm = get_llm_for_benchmark(
        model_name="gpt-4o",
        vendor="openai",
    )

    run_benchmark(
        llm=llm,
        out_dir=experiment_dir,
        tasks=[super_task],
        bench_logger=bench_logger,
    )
