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

import logging
import uuid
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

from rai.communication.ros2.api import (
    ConfigurableROS2TopicAPI,
    TopicConfig,
)
from rai.communication.ros2.connectors.base import ROS2BaseConnector
from rai.communication.ros2.messages import ROS2HRIMessage

try:
    import rai_interfaces.msg
except ImportError:
    logging.warning("rai_interfaces is not installed, ROS 2 HRIMessage will not work.")


class ROS2HRIConnector(ROS2BaseConnector[ROS2HRIMessage]):
    def __init__(
        self,
        node_name: str = f"rai_ros2_hri_connector_{str(uuid.uuid4())[-12:]}",
        targets: List[Union[str, Tuple[str, TopicConfig]]] = [],
        sources: List[Union[str, Tuple[str, TopicConfig]]] = [],
    ):
        configured_targets = [
            target[0] if isinstance(target, tuple) else target for target in targets
        ]
        configured_sources = [
            source[0] if isinstance(source, tuple) else source for source in sources
        ]
        self.configured_targets = configured_targets
        self.configured_sources = configured_sources

        _targets = [
            (
                target
                if isinstance(target, tuple)
                else (target, TopicConfig(is_subscriber=False))
            )
            for target in targets
        ]
        _sources = [
            (
                source
                if isinstance(source, tuple)
                else (source, TopicConfig(is_subscriber=True))
            )
            for source in sources
        ]
        super().__init__(node_name=node_name)
        self._topic_api = ConfigurableROS2TopicAPI(self._node)
        self._configure_publishers(_targets)
        self._configure_subscribers(_sources)

    def _configure_publishers(self, targets: List[Tuple[str, TopicConfig]]):
        for target in targets:
            self._topic_api.configure_publisher(target[0], target[1])

    def _configure_subscribers(self, sources: List[Tuple[str, TopicConfig]]):
        for source in sources:
            self._topic_api.configure_subscriber(source[0], source[1])

    def send_message(self, message: ROS2HRIMessage, target: str, **kwargs):
        self._topic_api.publish_configured(
            topic=target,
            msg_content=message.to_ros2_dict(),
        )

    def receive_message(
        self,
        source: str,
        timeout_sec: float = 1.0,
        *,
        message_author: Literal["human", "ai"] = "human",
        msg_type: Optional[str] = None,
        auto_topic_type: bool = True,
        **kwargs: Any,
    ) -> ROS2HRIMessage:
        msg = self._topic_api.receive(
            topic=source,
            timeout_sec=timeout_sec,
            auto_topic_type=auto_topic_type,
        )
        if not isinstance(msg, rai_interfaces.msg.HRIMessage):
            raise ValueError(
                f"Received message is not of type rai_interfaces.msg.HRIMessage, got {type(msg)}"
            )
        return ROS2HRIMessage.from_ros2(msg, message_author)

    def create_service(
        self,
        service_name: str,
        on_request: Callable,
        on_done: Optional[Callable] = None,
        *,
        service_type: str,
        **kwargs: Any,
    ) -> str:
        return self._service_api.create_service(
            service_name=service_name,
            callback=on_request,
            on_done=on_done,
            service_type=service_type,
            **kwargs,
        )

    def create_action(
        self,
        action_name: str,
        generate_feedback_callback: Callable,
        *,
        action_type: str,
        **kwargs: Any,
    ) -> str:
        return self._actions_api.create_action_server(
            action_name=action_name,
            action_type=action_type,
            execute_callback=generate_feedback_callback,
            **kwargs,
        )

    def shutdown(self):
        self._executor.shutdown()
        self._thread.join()
        self._actions_api.shutdown()
        self._topic_api.shutdown()
        self._node.destroy_node()
