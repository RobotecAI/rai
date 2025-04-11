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

from typing import Any, Generic, TypeVar

from .base_connector import BaseConnector, BaseMessage


class ARIMessage(BaseMessage):
    """Base message type for Agent-Robot Interface (ARI) communication.

    This class serves as a marker class and defines the contract for all ARI messages.
    Inherit from this class to create specific ARI message types.
    """

    payload: Any


T = TypeVar("T", bound=ARIMessage)


class ARIConnector(Generic[T], BaseConnector[T]):
    """
    Base class for Agent-Robot Interface (ARI) connectors.

    This class provides the foundation for implementing connectors that facilitate
    communication between software agents and robots. It ensures type safety by
    accepting only ARIMessage-based messages.

    Usage:
        Inherit from this class to implement specific ARI connectors, such as
        ROS2-based or custom protocol-based connectors.   Base class for Robot-Robot Interaction (RRI) connectors.
    """
