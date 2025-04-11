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

import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseMessage(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


T = TypeVar("T", bound=BaseMessage)


class BaseConnector(Generic[T]):
    """Base class for communication connectors in the RAI framework.

    This class provides a generic interface for different communication protocols.
    While the interface is designed to be flexible, it's important to note that:
    1. Only the message type 'T' is strictly enforced through generics
    2. Not all communication protocols may support every method in this interface
    3. Implementations should focus on supporting the methods that make sense
       for their specific communication protocol
    4. The interface is intentionally broad to accommodate various communication
       patterns (messages, services, actions) but concrete implementations
       may choose to support only a subset of these features
    5. Subclasses should prioritize implementing correct behavior for their
       specific communication protocol over maintaining exact parameter signatures.
       The method signatures in this base class serve as a guideline rather than
       a strict contract.
    6. Each method can be extended through **kwargs to support protocol-specific
       parameters and configurations
    7. Connectors based on this class are NOT meant to be interchangeable.
       Different communication protocols (e.g., ROS 2, MQTT, HTTP) have different
       requirements, capabilities, and parameter needs. While they share a common
       interface structure, the actual parameters and behaviors will vary
       significantly between implementations.

    The generic type T represents the message type that the connector will handle,
    which must be a subclass of BaseMessage.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _generate_handle(self) -> str:
        return str(uuid4())

    @abstractmethod
    def send_message(self, message: T, target: str, **kwargs: Any):
        """Implements publish pattern.

        Sends a message to one or more subscribers. The target parameter
        can be used to specify the destination or topic.
        """
        pass

    @abstractmethod
    def receive_message(self, source: str, timeout_sec: float, **kwargs: Any) -> T:
        """Implements subscribe pattern.

        Receives a message from a publisher. The source parameter
        can be used to specify the source or topic to subscribe to.
        """
        pass

    @abstractmethod
    def service_call(
        self, message: T, target: str, timeout_sec: float, **kwargs: Any
    ) -> BaseMessage:
        """Implements request-response pattern.

        Sends a request and waits for a response. The target parameter
        specifies the service endpoint to call.
        """
        pass

    @abstractmethod
    def create_service(
        self,
        service_name: str,
        on_request: Callable,
        on_done: Optional[Callable] = None,
        **kwargs: Any,
    ) -> str:
        """Sets up a service endpoint for handling requests.

        Creates a service that can receive and process requests.
        The on_request callback handles incoming requests,
        and on_done (if provided) is called when the service is terminated.
        """
        pass

    @abstractmethod
    def create_action(
        self,
        action_name: str,
        generate_feedback_callback: Callable,
        **kwargs: Any,
    ) -> str:
        """Sets up an action endpoint for long-running operations.

        Creates an action that can be started and monitored.
        The generate_feedback_callback is used to provide progress updates
        during the action's execution.
        """
        pass

    @abstractmethod
    def start_action(
        self,
        action_data: Optional[T],
        target: str,
        on_feedback: Callable,
        on_done: Callable,
        timeout_sec: float,
        **kwargs: Any,
    ) -> str:
        """Initiates a long-running operation with feedback.

        Starts an action and provides callbacks for feedback and completion.
        The on_feedback callback receives progress updates,
        and on_done is called when the action completes.
        """
        pass

    @abstractmethod
    def terminate_action(self, action_handle: str, **kwargs: Any):
        """Cancels an ongoing action.

        Stops the execution of a previously started action.
        The action_handle identifies which action to terminate.
        """
        pass

    @abstractmethod
    def shutdown(self):
        """Shuts down the connector and releases all resources."""
        pass
