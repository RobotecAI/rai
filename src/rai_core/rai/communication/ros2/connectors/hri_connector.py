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

import importlib.util
import logging
import uuid
from typing import Any, Callable, Literal, Optional

from rclpy.qos import QoSProfile

from rai.communication import HRIConnector
from rai.communication.ros2.connectors.base import ROS2BaseConnector
from rai.communication.ros2.messages import ROS2HRIMessage

if importlib.util.find_spec("rai_interfaces") is None:
    logging.warning(
        "This feature is based on rai_interfaces. Make sure rai_interfaces is installed."
    )


class ROS2HRIConnector(ROS2BaseConnector[ROS2HRIMessage], HRIConnector[ROS2HRIMessage]):
    """ROS2-specific implementation of the HRIConnector for multimodal human-robot interaction.

    This connector provides functionality for exchanging multimodal messages (text, images, audio)
    between humans and robots through ROS2 topics. It combines the capabilities of ROS2BaseConnector
    and HRIConnector to handle ROS2-specific HRI communication.

    Parameters
    ----------
    node_name : str, optional
        Name of the ROS2 node. If not provided, generates a unique name with UUID.

    Notes
    -----
    Message Format:
        Messages are exchanged using the `rai_interfaces/msg/HRIMessage` ROS2 message type,
        which supports:
        - Text content
        - Images (as base64-encoded strings)
        - Audio (as base64-encoded WAV files)
        - Message metadata (author, communication ID, sequence numbers)

    Dependencies:
        Requires the `rai_interfaces` ROS2 package to be installed for proper operation.
    """

    def __init__(
        self,
        node_name: str = f"rai_ros2_hri_connector_{str(uuid.uuid4())[-12:]}",
        executor_type: Literal["single_threaded", "multi_threaded"] = "multi_threaded",
    ):
        """Initialize the ROS2HRIConnector.

        Parameters
        ----------
        node_name : str, optional
            Name of the ROS2 node. If not provided, generates a unique name with UUID.
        """
        super().__init__(node_name=node_name, executor_type=executor_type)

    def send_message(
        self,
        message: ROS2HRIMessage,
        target: str,
        *,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,
        **kwargs,
    ):
        """Send a multimodal HRI message to a ROS2 topic.

        Parameters
        ----------
        message : ROS2HRIMessage
            The multimodal message to send.
        target : str
            The target ROS2 topic name.
        qos_profile : Optional[QoSProfile], optional
            Custom QoS profile to use, by default None.
        auto_qos_matching : bool, optional
            Whether to automatically match QoS profiles, by default True.
        **kwargs : Any
            Additional keyword arguments.

        Notes
        -----
        The message is automatically converted to the ROS2 `rai_interfaces/msg/HRIMessage` format
        before being published to the topic.
        """
        self._topic_api.publish(
            topic=target,
            msg_content=message.to_ros2_dict(),
            msg_type="rai_interfaces/msg/HRIMessage",
            auto_qos_matching=auto_qos_matching,
            qos_profile=qos_profile,
        )

    def register_callback(
        self,
        source: str,
        callback: Callable[[ROS2HRIMessage], None],
        raw: bool = False,
        *,
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,
        **kwargs: Any,
    ) -> str:
        """Register a callback for receiving multimodal HRI messages.

        Parameters
        ----------
        source : str
            The ROS2 topic to subscribe to.
        callback : Callable[[ROS2HRIMessage], None]
            The callback function to execute when a message is received.
        raw : bool, optional
            Whether to pass raw messages to the callback, by default False.
        msg_type : Optional[str], optional
            The ROS2 message type, by default None (uses rai_interfaces/msg/HRIMessage).
        qos_profile : Optional[QoSProfile], optional
            Custom QoS profile to use, by default None.
        auto_qos_matching : bool, optional
            Whether to automatically match QoS profiles, by default True.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str
            The callback ID.

        Notes
        -----
        If msg_type is not provided, it defaults to 'rai_interfaces/msg/HRIMessage'.
        The callback will receive ROS2HRIMessage objects with converted content
        (base64-encoded images and audio are automatically decoded).
        """
        if msg_type is None:
            msg_type = "rai_interfaces/msg/HRIMessage"
        return super().register_callback(
            source,
            callback,
            raw,
            msg_type=msg_type,
            qos_profile=qos_profile,
            auto_qos_matching=auto_qos_matching,
            **kwargs,
        )

    def general_callback_preprocessor(self, message: Any) -> ROS2HRIMessage:
        """Preprocess a raw ROS2 HRI message into a ROS2HRIMessage.

        Parameters
        ----------
        message : Any
            The raw ROS2 message (rai_interfaces/msg/HRIMessage).

        Returns
        -------
        ROS2HRIMessage
            The preprocessed message with decoded content.

        Notes
        -----
        This method:
        1. Converts the raw ROS2 message to a ROS2HRIMessage
        2. Sets the message author to "human" by default
        3. Decodes base64-encoded images and audio content
        """
        return ROS2HRIMessage.from_ros2(message, message_author="human")
