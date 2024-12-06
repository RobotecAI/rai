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


import functools
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
import sensor_msgs.msg
from action_msgs.msg import GoalStatus
from langchain.tools import BaseTool
from langchain.tools.render import render_text_description_and_args
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.graph import CompiledGraph
from rclpy.action.client import ActionClient
from rclpy.action.server import ActionServer, GoalResponse, ServerGoalHandle
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.topic_endpoint_info import TopicEndpointInfo
from std_srvs.srv import Trigger

import rai.utils.ros
from rai.agents.state_based import Report, State, create_state_based_agent
from rai.messages import HumanMultimodalMessage
from rai.tools.ros.native import Ros2BaseTool
from rai.tools.ros.utils import convert_ros_img_to_base64, import_message_from_str
from rai.utils.model_initialization import get_llm_model, get_tracing_callbacks
from rai.utils.ros import NodeDiscovery
from rai.utils.ros_async import get_future_result
from rai.utils.ros_logs import create_logs_parser
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


def ros2_build_msg(msg_type: str, msg_args: Dict[str, Any]) -> Tuple[object, Type]:
    """
    Import message and create it. Return both ready message and message class.

    msgs args can have two formats:
    { "goal" : {arg 1 : xyz, ... } or {arg 1 : xyz, ... }
    """

    msg_cls: Type = rosidl_runtime_py.utilities.get_interface(msg_type)
    msg = msg_cls.Goal()

    if "goal" in msg_args:
        msg_args = msg_args["goal"]
    rosidl_runtime_py.set_message.set_message_fields(msg, msg_args)
    return msg, msg_cls


class AsyncRos2ActionClient:
    def __init__(self, node: rclpy.node.Node):
        self.node = node

        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status: Optional[int] = None
        self.client: Optional[ActionClient] = None
        self.action_feedback: Optional[Any] = None

    def get_logger(self):
        return self.node.get_logger()

    def run_action(
        self, action_name: str, action_type: str, action_goal_args: Dict[str, Any]
    ):
        if not self.is_task_complete():
            raise AssertionError(
                "Another ros2 action is currently running and parallel actions are not supported. Please wait until the previous action is complete before starting a new one. You can also cancel the current one."
            )

        if action_name[0] != "/":
            action_name = "/" + action_name
            self.get_logger().info(f"Action name corrected to: {action_name}")

        try:
            goal_msg, msg_cls = ros2_build_msg(action_type, action_goal_args)
        except Exception as e:
            return f"Failed to build message: {e}"

        self.client = ActionClient(self.node, msg_cls, action_name)
        self.msg_cls = msg_cls

        retries = 0
        while not self.client.wait_for_server(timeout_sec=1.0):
            retries += 1
            if retries > 5:
                raise Exception(
                    f"Action server '{action_name}' is not available. Make sure `action_name` is correct..."
                )
            self.get_logger().info(
                f"'{action_name}' action server not available, waiting..."
            )

        self.get_logger().info(f"Sending action message: {goal_msg}")

        send_goal_future = self.client.send_goal_async(
            goal_msg, self._feedback_callback
        )
        self.get_logger().info("Action goal sent!")

        self.goal_handle = get_future_result(send_goal_future)

        if not self.goal_handle:
            raise Exception(f"Action '{action_name}' not sent to server")

        if not self.goal_handle.accepted:
            raise Exception(f"Action '{action_name}' not accepted by server")

        self.result_future = self.goal_handle.get_result_async()
        self.get_logger().info("Action sent!")
        return f"{action_name} started successfully with args: {action_goal_args}"

    def get_task_result(self) -> str:
        if not self.is_task_complete():
            return "Task is not complete yet"

        def parse_status(status: int) -> str:
            return {
                GoalStatus.STATUS_SUCCEEDED: "succeeded",
                GoalStatus.STATUS_ABORTED: "aborted",
                GoalStatus.STATUS_CANCELED: "canceled",
                GoalStatus.STATUS_ACCEPTED: "accepted",
                GoalStatus.STATUS_CANCELING: "canceling",
                GoalStatus.STATUS_EXECUTING: "executing",
                GoalStatus.STATUS_UNKNOWN: "unknown",
            }.get(status, "unknown")

        result = self.result_future.result()

        self.destroy_client()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            msg = f"Result succeeded: {result.result}"
            self.get_logger().info(msg)
            return msg
        else:
            str_status = parse_status(result.status)
            error_code_str = self.parse_error_code(result.result.error_code)
            msg = f"Result {str_status}, because of: error_code={result.result.error_code}({error_code_str}), error_msg={result.result.error_msg}"
            self.get_logger().info(msg)
            return msg

    def parse_error_code(self, code: int) -> str:
        name_to_code = self.msg_cls.Result.__prepare__(
            name="", bases=""
        )  # arguments are not used
        code_to_name = {v: k for k, v in name_to_code.items()}
        return code_to_name.get(code, "UNKNOWN")

    def _feedback_callback(self, msg):
        self.get_logger().info(f"Received ros2 action feedback: {msg}")
        self.action_feedback = msg

    def is_task_complete(self):
        if not self.result_future:
            # task was cancelled or completed
            return True

        result = get_future_result(self.result_future, timeout_sec=0.10)
        if result is not None:
            self.status = result.status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().debug(
                    f"Task with failed with status code: {self.status}"
                )
                return True
        else:
            self.get_logger().info("There is no result")
            # Timed out, still processing, not complete yet
            return False

        self.get_logger().info("Task succeeded!")
        return True

    def cancel_task(self) -> Union[str, bool]:
        self.get_logger().info("Canceling current task.")
        try:
            if self.result_future and self.goal_handle:
                future = self.goal_handle.cancel_goal_async()
                result = get_future_result(future, timeout_sec=1.0)
                return "Failed to cancel result" if result is None else True
            return True
        finally:
            self.destroy_client()

    def destroy_client(self):
        if self.client:
            self.client.destroy()


class Ros2TopicsHandler:
    def __init__(
        self,
        node: rclpy.node.Node,
        callback_group: rclpy.callback_groups.CallbackGroup,
        ros_discovery_info: NodeDiscovery,
    ) -> None:
        self.node = node
        self.qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            liveliness=LivelinessPolicy.AUTOMATIC,
        )
        self.callback_group = callback_group
        self.last_subscription_msgs_buffer = dict()
        self.qos_profile_cache: Dict[str, QoSProfile] = dict()

        self.ros_discovery_info = ros_discovery_info

    def get_logger(self):
        return self.node.get_logger()
    
    def adapt_requests_to_offers(
        self, publisher_info: List[TopicEndpointInfo]
    ) -> QoSProfile:
        if not publisher_info:
            return QoSProfile(depth=1)

        num_endpoints = len(publisher_info)
        reliability_reliable_count = 0
        durability_transient_local_count = 0

        for endpoint in publisher_info:
            profile = endpoint.qos_profile
            if profile.reliability == ReliabilityPolicy.RELIABLE:
                reliability_reliable_count += 1
            if profile.durability == DurabilityPolicy.TRANSIENT_LOCAL:
                durability_transient_local_count += 1

        request_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            liveliness=LivelinessPolicy.AUTOMATIC,
        )

        # Set reliability based on publisher offers
        if reliability_reliable_count == num_endpoints:
            request_qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            if reliability_reliable_count > 0:
                self.get_logger().warning(
                    "Some, but not all, publishers are offering RELIABLE reliability. "
                    "Falling back to BEST_EFFORT as it will connect to all publishers. "
                    "Some messages from Reliable publishers could be dropped."
                )
            request_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Set durability based on publisher offers
        if durability_transient_local_count == num_endpoints:
            request_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        else:
            if durability_transient_local_count > 0:
                self.get_logger().warning(
                    "Some, but not all, publishers are offering TRANSIENT_LOCAL durability. "
                    "Falling back to VOLATILE as it will connect to all publishers. "
                    "Previously-published latched messages will not be retrieved."
                )
            request_qos.durability = DurabilityPolicy.VOLATILE

        return request_qos

    def create_subscription_by_topic_name(self, topic):
        if self.has_subscription(topic):
            self.get_logger().warning(
                f"Subscription to {topic} already exists. To override use destroy_subscription_by_topic_name first"
            )
            return
       
        msg_type = self.get_msg_type(topic)
        
        if topic not in self.qos_profile_cache:
            self.get_logger().debug(f"Getting qos profile for topic: {topic}")
            qos_profile = self.adapt_requests_to_offers(
                self.node.get_publishers_info_by_topic(topic)
            )
            self.qos_profile_cache[topic] = qos_profile
        else:
            self.get_logger().debug(f"Using cached qos profile for topic: {topic}")
            qos_profile = self.qos_profile_cache[topic]

        topic_callback = functools.partial(
            self.generic_state_subscriber_callback, topic
        )

        self.node.create_subscription(
            msg_type,
            topic,
            callback=topic_callback,
            callback_group=self.callback_group,
            qos_profile=qos_profile,
        )
    
    def get_msg_type(self, topic: str, n_tries: int = 5) -> Any:
        """Sometimes node fails to do full discovery, therefore we need to retry"""
        for _ in range(n_tries):
            if topic in self.ros_discovery_info.topics_and_types:
                msg_type = self.ros_discovery_info.topics_and_types[topic]
                return import_message_from_str(msg_type)
            else:
                # Wait for next discovery cycle
                self.get_logger().info(f"Waiting for topic: {topic}")
                if self.ros_discovery_info:
                    time.sleep(self.ros_discovery_info.period_sec)
                else:
                    time.sleep(1.0)
        raise KeyError(f"Topic {topic} not found")

    def set_ros_discovery_info(self, new_ros_discovery_info: NodeDiscovery):
        self.ros_discovery_info = new_ros_discovery_info

    def get_raw_message_from_topic(self, topic: str, timeout_sec: int = 5) -> Any:
        self.get_logger().debug(f"Getting msg from topic: {topic}")

        ts = time.perf_counter()

        if topic not in self.ros_discovery_info.topics_and_types:
            raise KeyError(
                f"Topic {topic} not found. Available topics: {self.ros_discovery_info.topics_and_types.keys()}"
            )

        if topic in self.last_subscription_msgs_buffer:
            self.get_logger().info("Returning cached message")
            return self.last_subscription_msgs_buffer[topic]
        else:
            self.create_subscription_by_topic_name(topic)
            try:
                msg = self.last_subscription_msgs_buffer.get(topic, None)
                while msg is None and time.perf_counter() - ts < timeout_sec:
                    msg = self.last_subscription_msgs_buffer.get(topic, None)
                    self.get_logger().info("Waiting for message...")
                    time.sleep(0.1)

                success = msg is not None

                if success:
                    self.get_logger().debug(
                        f"Received message of type {type(msg)} from topic {topic}"
                    )
                    return msg
                else:
                    error = f"No message received in {timeout_sec} seconds from topic {topic}"
                    self.get_logger().error(error)
                    return error
            finally:
                self.destroy_subscription_by_topic_name(topic)

    def generic_state_subscriber_callback(self, topic_name: str, msg: Any):
        self.get_logger().debug(
            f"Received message of type {type(msg)} from topic {topic_name}"
        )
        self.last_subscription_msgs_buffer[topic_name] = msg

    def has_subscription(self, topic: str) -> bool:
        for sub in self.node._subscriptions:
            if sub.topic == topic:
                return True
        return False

    def destroy_subscription_by_topic_name(self, topic: str):
        self.last_subscription_msgs_buffer.clear()
        for sub in self.node._subscriptions:
            if sub.topic == topic:
                self.node.destroy_subscription(sub)


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
        self.async_action_client = AsyncRos2ActionClient(self)
        self.topics_handler = Ros2TopicsHandler(
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
        executor = rai.utils.ros.MultiThreadedExecutorFixed()
        executor.add_node(self)
        executor.spin()
        rclpy.shutdown()


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
