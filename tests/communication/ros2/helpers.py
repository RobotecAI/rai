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

import threading
import time
from typing import Generator, List, Tuple

import numpy as np
import pytest
import rclpy
from cv_bridge import CvBridge
from nav2_msgs.action import NavigateToPose
from pydub import AudioSegment
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster, TransformStamped

from rai_interfaces.msg import HRIMessage


class HRIMessagePublisher(Node):
    def __init__(self, topic: str):
        """
        Initialize the HRIMessagePublisher node.
        
        This constructor sets up the ROS 2 node with the name "test_hri_message_publisher". It creates a publisher for HRIMessage objects on the specified topic with a queue size of 10, starts a timer that calls the publish_message callback every 0.1 seconds, and initializes a CvBridge instance for converting image formats.
          
        Parameters:
            topic (str): The ROS topic on which HRIMessage objects will be published.
        """
        super().__init__("test_hri_message_publisher")
        self.publisher = self.create_publisher(HRIMessage, topic, 10)
        self.timer = self.create_timer(0.1, self.publish_message)
        self.cv_bridge = CvBridge()

    def publish_message(self) -> None:
        """
        Publishes an HRIMessage containing a converted image, silent audio, and a greeting text.
        
        This method creates an HRIMessage instance and populates its fields by:
        - Converting a 100x100 black NumPy array into a ROS image message using CvBridge and assigning it to the `images` list.
        - Generating a one-second silent audio segment with AudioSegment and assigning it to the `audios` list.
        - Setting the `text` field to "Hello, HRI!".
        
        The constructed message is then published using the node's publisher.
        
        Returns:
            None
        """
        msg = HRIMessage()
        image = self.cv_bridge.cv2_to_imgmsg(np.zeros((100, 100, 3), dtype=np.uint8))
        msg.images = [image]
        msg.audios = [AudioSegment.silent(duration=1000)]
        msg.text = "Hello, HRI!"
        self.publisher.publish(msg)


class HRIMessageSubscriber(Node):
    def __init__(self, topic: str):
        """
        Initialize a HRIMessageSubscriber node with a subscription to HRIMessage messages.
        
        This constructor initializes a ROS 2 node with the name "test_hri_message_subscriber" and creates a subscription
        to receive HRIMessage objects from the specified topic. The subscription uses a queue size of 10 and directs incoming
        messages to the `handle_test_message` callback. An empty list is also initialized to store all received HRIMessage instances.
        
        Parameters:
            topic (str): The topic name from which to subscribe for HRIMessage messages.
        """
        super().__init__("test_hri_message_subscriber")
        self.subscription = self.create_subscription(
            HRIMessage, topic, self.handle_test_message, 10
        )
        self.received_messages: List[HRIMessage] = []

    def handle_test_message(self, msg: HRIMessage) -> None:
        """
        Callback for handling a received HRIMessage.
        
        Appends the received HRIMessage to the internal list of messages.
        
        Parameters:
            msg (HRIMessage): The HRIMessage instance received from the subscribed topic.
        
        Returns:
            None
        """
        self.received_messages.append(msg)


class ServiceServer(Node):
    def __init__(self, service_name: str):
        """
        Initialize the ServiceServer node with the given service name.
        
        This constructor creates a ROS 2 service server node named "test_service_server" by setting up a service using the SetBool service type. The service is identified by the provided service name and uses the 'handle_test_service' callback to process incoming requests.
        
        Parameters:
            service_name (str): The name of the service that the node will offer.
        """
        super().__init__("test_service_server")
        self.srv = self.create_service(SetBool, service_name, self.handle_test_service)

    def handle_test_service(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        response.success = True
        response.message = "Test service called"
        return response


class ImagePublisher(Node):
    def __init__(self, topic: str):
        super().__init__("test_image_publisher")
        self.publisher = self.create_publisher(Image, topic, 10)  # type: ignore
        self.timer = self.create_timer(0.1, self.publish_image)  # type: ignore

    def publish_image(self) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()  # type: ignore
        msg.header.frame_id = "test_frame"  # type: ignore
        msg.height = 100
        msg.width = 100
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 300
        msg.data = np.zeros((100, 100, 3), dtype=np.uint8).tobytes()  # type: ignore
        self.publisher.publish(msg)


class MessageReceiver(Node):
    def __init__(self, topic: str):
        super().__init__("test_message_receiver")
        self.subscription = self.create_subscription(
            String, topic, self.handle_test_message, 10
        )
        self.received_messages: List[String] = []

    def handle_test_message(self, msg: String) -> None:
        self.received_messages.append(msg)


class ActionServer_(Node):
    def __init__(self, action_name: str):
        super().__init__("test_action_server")
        self.action_server = ActionServer(
            self,
            action_type=NavigateToPose,
            action_name=action_name,
            execute_callback=self.handle_test_action,
            goal_callback=self.goal_accepted,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

    def handle_test_action(
        self, goal_handle: ServerGoalHandle
    ) -> NavigateToPose.Result:
        for i in range(1, 11):
            if goal_handle.is_cancel_requested:
                print("Cancel detected in execute callback")
                goal_handle.canceled()
                result = NavigateToPose.Result()
                result.error_code = 3
                return result
            feedback_msg = NavigateToPose.Feedback(distance_remaining=10.0 / i)
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.01)

        goal_handle.succeed()

        result = NavigateToPose.Result()
        result.error_code = NavigateToPose.Result.NONE
        return result

    def goal_accepted(self, goal_handle: ServerGoalHandle) -> GoalResponse:
        self.get_logger().info("Got goal, accepting")
        return GoalResponse.ACCEPT

    def cancel_callback(self, cancel_request) -> CancelResponse:
        self.get_logger().info("Got cancel request")
        return CancelResponse.ACCEPT


class MessagePublisher(Node):
    def __init__(self, topic: str):
        super().__init__("test_message_publisher")
        self.publisher = self.create_publisher(String, topic, 10)
        self.timer = self.create_timer(0.1, self.publish_message)

    def publish_message(self) -> None:
        msg = String()
        msg.data = "Hello, ROS2!"
        self.publisher.publish(msg)


class TransformPublisher(Node):
    def __init__(self, topic: str):
        super().__init__("test_transform_publisher")
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.publish_transform)
        self.frame_id = "base_link"
        self.child_frame_id = "map"

    def publish_transform(self) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()  # type: ignore
        msg.header.frame_id = self.frame_id  # type: ignore
        msg.child_frame_id = self.child_frame_id  # type: ignore
        msg.transform.translation.x = 1.0  # type: ignore
        msg.transform.translation.y = 2.0  # type: ignore
        msg.transform.translation.z = 3.0  # type: ignore
        msg.transform.rotation.x = 0.0  # type: ignore
        msg.transform.rotation.y = 0.0  # type: ignore
        msg.transform.rotation.z = 0.0  # type: ignore
        msg.transform.rotation.w = 1.0  # type: ignore
        self.tf_broadcaster.sendTransform(msg)


def multi_threaded_spinner(
    nodes: List[Node],
) -> Tuple[List[MultiThreadedExecutor], List[threading.Thread]]:
    executors: List[MultiThreadedExecutor] = []
    executor_threads: List[threading.Thread] = []
    for node in nodes:
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executors.append(executor)
    for executor in executors:
        executor_thread = threading.Thread(target=executor.spin)
        executor_thread.daemon = True
        executor_thread.start()
        executor_threads.append(executor_thread)
    return executors, executor_threads


def shutdown_executors_and_threads(
    executors: List[MultiThreadedExecutor], threads: List[threading.Thread]
) -> None:
    # First shutdown executors
    for executor in executors:
        executor.shutdown()
    # Small delay to allow executors to finish pending operations
    time.sleep(0.5)
    # Then join threads with a timeout
    for thread in threads:
        thread.join(timeout=2.0)


@pytest.fixture(scope="function")
def ros_setup() -> Generator[None, None, None]:
    rclpy.init()
    yield
    rclpy.shutdown()
