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
#
import json
from typing import Literal, Optional

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import rclpy.time
import std_msgs.msg
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener, TransformStamped

from rai.messages.multimodal import HumanMultimodalMessage
from rai.tools.ros.utils import convert_ros_img_to_base64
from rai.utils.model_initialization import get_llm_model


class AnomalyReport(BaseModel):
    is_anomaly: bool = Field(
        ..., description="True if the robot is in an anomaly state."
    )
    description: str = Field(..., description="Description of the anomaly, else empty.")
    anomaly_type: Literal["robot", "environment"] = Field(
        ...,
        description="Type of the anomaly. Is to connected to the robot or environment",
    )
    distance: Literal["close", "medium", "far"] = Field(
        ..., description="Distance from the robot."
    )


class Node(rclpy.node.Node):
    SYSTEM_PROMPT = """You are the inspection robot. You will be given 1 or 2 camera images.
    Your task is to analyze the camera image and report dangerous and unexpected situations.
    Take into account the robot and sourrounding environment.
    """
    DEFAULT_FREQ = 2.0
    DEFAULT_CAMERA_TOPIC = "/base/camera_image_color"

    def __init__(self):
        super().__init__("quadruped_inspection_node")

        self._declate_parameters()
        self.system_prompt = self.initialize_system_prompt(self.SYSTEM_PROMPT)
        self._initialize_llm_agent()

        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data,
        )
        inspection_freq = (
            self.get_parameter("inspection_freq").get_parameter_value().double_value
        )
        self.llm_interaction_timer = self.create_timer(
            inspection_freq,
            self.llm_interaction,
        )

        self.anomaly_publisher = self.create_publisher(
            String, "/anomalies", qos_profile=rclpy.qos.qos_profile_services_default
        )

        self.last_image_b64: Optional[str] = None
        self.last_processed_image_b64: Optional[str] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _declate_parameters(self):
        self.declare_parameter(
            "image_topic",
            self.DEFAULT_CAMERA_TOPIC,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera topic for inspection",
            ),
        )
        self.declare_parameter(
            "inspection_freq",
            self.DEFAULT_FREQ,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Frequency of LLM interaction",
            ),
        )

    def initialize_system_prompt(self, prompt: str) -> str:
        constitution_service = self.create_client(
            Trigger,
            "rai_whoami_constitution_service",
        )
        identity_service = self.create_client(Trigger, "rai_whoami_identity_service")
        try:
            while not constitution_service.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    "Constitution service not available, waiting again..."
                )

            while not identity_service.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    "Identity service not available, waiting again..."
                )

            constitution_request = Trigger.Request()

            constitution_future = constitution_service.call_async(constitution_request)
            rclpy.spin_until_future_complete(self, constitution_future)
            constitution_response = constitution_future.result()

            identity_request = Trigger.Request()

            identity_future = identity_service.call_async(identity_request)
            rclpy.spin_until_future_complete(self, identity_future)
            identity_response = identity_future.result()

            system_prompt = f"""
            Constitution:
            {constitution_response.message}

            Identity:
            {identity_response.message}

            {prompt}
            """

            self.get_logger().info(f"System prompt initialized: {system_prompt}")
            return system_prompt
        finally:
            constitution_service.destroy()
            identity_service.destroy()

    def _initialize_llm_agent(self):
        self.llm = get_llm_model(model_type="complex_model")
        self.agent = self.llm.with_structured_output(AnomalyReport)

    def llm_interaction(self):
        self.get_logger().info("LLM interaction")
        image = self.last_image_b64
        if image is None:
            self.get_logger().warning("No image received")
            return

        # if image == self.last_processed_image_b64:
        #     self.get_logger().info("No new image received")
        #     return

        transform_stamped = self.tf_buffer.lookup_transform(
            "base/odom", "base/", rclpy.time.Time()
        )

        images = (
            [image]
            if self.last_processed_image_b64 is None
            else [image, self.last_processed_image_b64]
        )
        response: AnomalyReport = self.agent.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMultimodalMessage(
                    content="Please analyze this image", images=images
                ),
            ]
        )  # type: ignore

        self.get_logger().info(f"Current pose: {transform_stamped}")
        self.get_logger().info(f"Anomaly report: {response}")
        if response.is_anomaly:
            msg = std_msgs.msg.String()
            msg.data = self.format_response(response, transform_stamped)
            self.anomaly_publisher.publish(msg)
        self.last_processed_image_b64 = image

    def image_callback(self, msg):
        self.last_image_b64 = convert_ros_img_to_base64(msg)

    def format_response(self, response: AnomalyReport, pose: TransformStamped):
        d = response.model_dump(mode="json")
        del d["is_anomaly"]
        d["tf"] = message_to_ordereddict(pose)
        return json.dumps(d)


def main():
    rclpy.init()
    node = Node()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
