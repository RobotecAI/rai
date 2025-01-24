# Copyright (C) 2024 Robotec.AI
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


import time
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import sensor_msgs.msg
from langchain.tools import BaseTool
from langchain.tools.render import render_text_description_and_args
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.graph import CompiledGraph
from rai.agents.state_based import Report, State, create_state_based_agent
from rai.messages import HumanMultimodalMessage
from rai.ros2_apis import Ros2ActionsAPI, Ros2TopicsAPI
from rai.tools.ros.native import Ros2BaseTool
from rai.tools.ros.utils import convert_ros_img_to_base64
from rai.utils.model_initialization import get_llm_model, get_tracing_callbacks
from rai.utils.ros import NodeDiscovery
from rai.utils.ros_async import get_future_result
from rai.utils.ros_executors import MultiThreadedExecutorFixed
from rai.utils.ros_logs import create_logs_parser
from rclpy.action.server import ActionServer, GoalResponse, ServerGoalHandle
from rclpy.node import Node
from std_srvs.srv import Trigger

from rai_interfaces.action import Task as TaskAction

WHOAMI_SYSTEM_PROMPT_TEMPLATE = """
    Constitution:
    {constitution}

    Identity:
    {identity}

    {prompt}
"""


def append_whoami_info_to_prompt(
    node: Node, prompt: str, robot_description_package: Optional[str] = None
) -> str:
    if robot_description_package is None:
        node.get_logger().warning(
            "Robot description package not set, using empty identity and constitution."
        )
        return WHOAMI_SYSTEM_PROMPT_TEMPLATE.format(
            constitution="",
            identity="",
            prompt=prompt,
        )

    constitution_service = node.create_client(
        Trigger,
        "rai_whoami_constitution_service",
    )
    identity_service = node.create_client(Trigger, "rai_whoami_identity_service")

    while not constitution_service.wait_for_service(timeout_sec=1.0):
        node.get_logger().info("Constitution service not available, waiting again...")

    while not identity_service.wait_for_service(timeout_sec=1.0):
        node.get_logger().info("Identity service not available, waiting again...")

    constitution_future = constitution_service.call_async(Trigger.Request())
    constitution_response: Optional[Trigger.Response] = get_future_result(
        constitution_future
    )
    constitution_message = (
        "" if constitution_response is None else constitution_response.message
    )

    identity_future = identity_service.call_async(Trigger.Request())
    identity_response: Optional[Trigger.Response] = get_future_result(identity_future)
    identity_message = "" if identity_response is None else identity_response.message

    system_prompt = WHOAMI_SYSTEM_PROMPT_TEMPLATE.format(
        constitution=constitution_message,
        identity=identity_message,
        prompt=prompt,
    )

    node.get_logger().info("System prompt initialized")
    node.get_logger().debug(f"System prompt:\n{system_prompt}")
    return system_prompt


def append_tools_text_description_to_prompt(prompt: str, tools: List[BaseTool]) -> str:
    if len(tools) == 0:
        return prompt

    return f"""{prompt}

    Use the tooling provided to gather information about the environment:

    {render_text_description_and_args(tools)}
    """


class RaiBaseNode(Node):
    def __init__(
        self,
        allowlist: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # ---------- ROS configuration ----------
        self.callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

        # ---------- ROS helpers ----------
        self.ros_discovery_info = NodeDiscovery(self, allowlist=allowlist)
        self.async_action_client = Ros2ActionsAPI(self)
        self.topics_handler = Ros2TopicsAPI(
            self, self.callback_group, self.ros_discovery_info
        )
        self.ros_discovery_info.add_setter(self.topics_handler.set_ros_discovery_info)

    # ------------- ros2 topics interface -------------
    def get_raw_message_from_topic(self, topic: str, timeout_sec: int = 5) -> Any:
        return self.topics_handler.get_raw_message_from_topic(topic, timeout_sec)

    # ------------- ros2 actions interface -------------
    def run_action(
        self, action_name: str, action_type: str, action_goal_args: Dict[str, Any]
    ):
        return self.async_action_client.run_action(
            action_name, action_type, action_goal_args
        )

    def get_task_result(self) -> str:
        return self.async_action_client.get_task_result()

    def is_task_complete(self) -> bool:
        return self.async_action_client.is_task_complete()

    @property
    def action_feedback(self) -> Any:
        return self.async_action_client.action_feedback

    def cancel_task(self) -> Union[str, bool]:
        return self.async_action_client.cancel_task()

    # ------------- other methods -------------
    def spin(self):
        executor = MultiThreadedExecutorFixed()
        executor.add_node(self)
        executor.spin()


def parse_task_goal(ros_action_goal: TaskAction.Goal) -> Dict[str, Any]:
    return dict(
        task=ros_action_goal.task,
        description=ros_action_goal.description,
        priority=ros_action_goal.priority,
    )


class RaiStateBasedLlmNode(RaiBaseNode):
    AGENT_RECURSION_LIMIT = 500
    STATE_UPDATE_PERIOD = 5.0

    def __init__(
        self,
        system_prompt: str,
        observe_topics: Optional[List[str]] = None,
        observe_postprocessors: Optional[Dict[str, Callable[[Any], Any]]] = None,
        allowlist: Optional[List[str]] = None,
        tools: Optional[List[Type[BaseTool]]] = None,
        logs_parser_type: Literal["llm", "rai_state_logs"] = "rai_state_logs",
        *args,
        **kwargs,
    ):
        super().__init__(
            node_name="rai_node",
            allowlist=allowlist,
            *args,
            **kwargs,
        )

        # ---------- Robot State ----------
        self.state_topics = observe_topics if observe_topics is not None else []
        self.state_postprocessors = (
            observe_postprocessors if observe_postprocessors is not None else dict()
        )
        self._initialize_robot_state_interfaces(self.state_topics)
        self.state_update_timer = self.create_timer(
            self.STATE_UPDATE_PERIOD,
            self.state_update_callback,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )
        self.state_dict = dict()

        # ---------- Task Queue ----------
        self.task_action_server = ActionServer(
            self,
            TaskAction,
            "perform_task",
            execute_callback=self.agent_loop,
            callback_group=self.callback_group,
            goal_callback=self.goal_callback,
        )
        # Node is busy when task is executed. Only 1 task is allowed
        self.busy = False
        self.current_task = None

        # ---------- LLM Agents ----------
        self.tools = self._initialize_tools(tools) if tools is not None else []
        self.system_prompt = self._initialize_system_prompt(system_prompt)
        self.llm_app: CompiledGraph = create_state_based_agent(
            llm=get_llm_model(model_type="complex_model"),
            tools=self.tools,
            state_retriever=self.get_robot_state,
            logger=self.get_logger(),
        )

        self.simple_llm = get_llm_model(model_type="simple_model")
        self.logs_parser = create_logs_parser(
            logs_parser_type,
            self,
            callback_group=self.callback_group,
            llm=self.simple_llm,
        )

    def summarize_logs(self) -> str:
        return self.logs_parser.summarize()

    def _initialize_tools(self, tools: List[Type[BaseTool]]):
        initialized_tools: List[BaseTool] = list()
        for tool_cls in tools:
            if issubclass(tool_cls, Ros2BaseTool):
                tool = tool_cls(node=self)
            else:
                tool = tool_cls()

            initialized_tools.append(tool)
        return initialized_tools

    def _initialize_system_prompt(self, prompt: str):
        system_prompt = append_whoami_info_to_prompt(self, prompt)
        system_prompt = append_tools_text_description_to_prompt(
            system_prompt, self.tools
        )
        return system_prompt

    def _initialize_robot_state_interfaces(self, topics: List[str]):
        for topic in topics:
            self.topics_handler.create_subscription_by_topic_name(topic)

    def goal_callback(self, _) -> GoalResponse:
        """Accept or reject a client request to begin an action."""
        response = GoalResponse.REJECT if self.busy else GoalResponse.ACCEPT
        self.get_logger().info(f"Received goal request. Response: {response}")
        return response

    async def agent_loop(self, goal_handle: ServerGoalHandle):
        self.busy = True
        try:
            action_request: TaskAction.Goal = goal_handle.request
            task: Dict[str, Any] = parse_task_goal(
                action_request
            )  # TODO(boczekbartek): base model and json

            self.get_logger().info(f"Received task: {task}")

            # ---- LLM Task Handling ----
            self.get_logger().debug(f'This is system prompt: "{self.system_prompt}"')

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=f"Robot intefaces: {self.ros_discovery_info.dict()}"
                ),
                HumanMessage(content=f"Task: {task}"),
            ]
            self.current_task = task

            payload = State(messages=messages)

            state = None
            for state in self.llm_app.stream(
                payload,
                {
                    "recursion_limit": self.AGENT_RECURSION_LIMIT,
                    "callbacks": get_tracing_callbacks(),
                },
            ):

                graph_node_name = list(state.keys())[0]
                if graph_node_name == "reporter":
                    continue

                msg = state[graph_node_name]["messages"][-1]
                if isinstance(msg, HumanMultimodalMessage):
                    last_msg = msg.text
                elif isinstance(msg, BaseMessage):
                    if isinstance(msg.content, list):
                        assert len(msg.content) == 1
                        last_msg = msg.content[0].get("text", "")
                    else:
                        last_msg = msg.content
                else:
                    raise ValueError(f"Unexpected type of message: {type(msg)}")

                # TODO(boczekbartek): Find a better way to create meaningful feedback
                last_msg = self.simple_llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "You are an experienced reporter deployed on a autonomous robot. "  # type: ignore
                                "Your task is to summarize the message in a way that is easy for other agents to understand. "
                                "Do not use markdown formatting. Keep it short and concise. If the message is empty, please return empty string ('')."
                            )
                        ),
                        HumanMessage(content=last_msg),
                    ]
                ).content

                if len(str(last_msg)) > 0 and graph_node_name != "state_retriever":
                    feedback_msg = TaskAction.Feedback()
                    feedback_msg.current_status = f"{graph_node_name}: {last_msg}"

                    goal_handle.publish_feedback(feedback_msg)

            # ---- Share Action Result ----
            if state is None:
                raise ValueError("No output from LLM")

            graph_node_name = list(state.keys())[0]
            if graph_node_name != "reporter":
                raise ValueError(f"Unexpected output llm node: {graph_node_name}")

            report = state["reporter"]["messages"][
                -1
            ]  # TODO define graph more strictly not as dict key

            if not isinstance(report, Report):
                raise ValueError(f"Unexpected type of agent output: {type(report)}")

            if report.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()

            result = TaskAction.Result()
            result.success = report.success
            result.report = report.outcome

            self.get_logger().info(f"Finished task:\n{result}")

            return result
        finally:
            self.busy = False
            self.current_task = None

    def state_update_callback(self):
        self.get_logger().info("Updating state.")
        state_dict = dict()
        state_dict["current_task"] = self.current_task

        ts = time.perf_counter()
        try:
            state_dict["logs_summary"] = self.summarize_logs()
        except Exception as e:
            self.get_logger().error(f"Error summarizing logs: {e}")
            state_dict["logs_summary"] = ""
        te = time.perf_counter() - ts
        self.get_logger().info(f"Logs summary retrieved in: {te:.2f}")
        self.get_logger().debug(f"{state_dict=}")

        if self.topics_handler.last_subscription_msgs_buffer is None:
            self.state_dict = state_dict
            return

        for t in self.state_topics:
            if t not in self.topics_handler.last_subscription_msgs_buffer:
                msg = "No message yet"
                state_dict[t] = msg
                continue

            ts = time.perf_counter()
            msg = self.topics_handler.last_subscription_msgs_buffer[t]
            if t in self.state_postprocessors:
                msg = self.state_postprocessors[t](msg)
            te = time.perf_counter() - ts
            self.get_logger().info(f"Topic '{t}' postprocessed in: {te:.2f}")

            state_dict[t] = msg

        self.state_dict = state_dict
        self.get_logger().info("State updated.")

    def get_robot_state(self) -> Dict[str, str]:
        return self.state_dict


def describe_ros_image(
    msg: sensor_msgs.msg.Image,
) -> Dict[Literal["camera_image_summary"], str]:
    PROMPT = """Please describe the image in 2 sentences max 150 chars."""
    small_llm = get_llm_model(model_type="simple_model")
    base64_image = convert_ros_img_to_base64(msg)
    llm_msg = HumanMultimodalMessage(content=PROMPT, images=[base64_image])
    output = small_llm.invoke([llm_msg])
    return {"camera_image_summary": str(output.content)}
