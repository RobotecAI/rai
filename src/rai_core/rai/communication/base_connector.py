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
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ConnectorException(Exception):
    """Base exception for all connector exceptions."""

    pass


class BaseMessage(BaseModel):
    payload: Any = Field(
        default=None,
        description="Payload is meant for non-validated data if such data is present.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    model_config = ConfigDict(arbitrary_types_allowed=True)


T = TypeVar("T", bound=BaseMessage)


class ParametrizedCallback(BaseModel, Generic[T]):
    # Callback is of type T if raw is False, otherwise it is of type Any
    callback: Callable[[T | Any], None]
    raw: bool


class BaseConnector(Generic[T]):
    def __init__(self, callback_max_workers: int = 4):
        self.callback_max_workers = callback_max_workers
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registered_callbacks: Dict[str, List[ParametrizedCallback[T]]] = (
            defaultdict(list)
        )
        self.callback_executor = ThreadPoolExecutor(
            max_workers=self.callback_max_workers
        )

    def _generate_handle(self) -> str:
        return str(uuid4())

    def send_message(self, message: T, target: str, **kwargs: Any) -> None:
        """Implements publish pattern.

        Sends a message to one or more subscribers. The target parameter
        can be used to specify the destination or topic.

        Raises:
            ConnectorException: If the message cannot be sent.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def receive_message(self, source: str, timeout_sec: float, **kwargs: Any) -> T:
        """Implements subscribe pattern.

        Receives a message from a publisher. The source parameter
        can be used to specify the source or topic to subscribe to.

        Raises:
            ConnectorException: If the message cannot be received.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def register_callback(
        self,
        source: str,
        callback: Callable[[T | Any], None],
        raw: bool = False,
        **kwargs: Any,
    ) -> None:
        """Implements register callback.

        Registers a callback to be called when a message is received from a source.
        If raw is False, the callback will receive a T object.
        If raw is True, the callback will receive the raw message.

        Raises:
            ConnectorException: If the callback cannot be registered.
        """
        self.registered_callbacks[source].append(
            ParametrizedCallback(callback=callback, raw=raw)
        )

    def _safe_callback_wrapper(self, callback: Callable[[T], None], message: T) -> None:
        """Safely execute a callback with error handling.

        Args:
            callback: The callback function to execute
            message: The message to pass to the callback
        """
        try:
            callback(message)
        except Exception as e:
            self.logger.error(f"Error in callback: {str(e)}")

    def general_callback(self, source: str, message: Any) -> None:
        """General callback for all messages.
        Use through functools.partial to pass source."""
        processed_message = self.general_callback_preprocessor(message)
        for parametrized_callback in self.registered_callbacks.get(source, []):
            payload = message if parametrized_callback.raw else processed_message
            self.callback_executor.submit(
                self._safe_callback_wrapper, parametrized_callback.callback, payload
            )

    def general_callback_preprocessor(self, message: Any) -> T:
        """Preprocessor for general callback used to transform any message to a BaseMessage."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    def service_call(
        self, message: T, target: str, timeout_sec: float, **kwargs: Any
    ) -> BaseMessage:
        """Implements request-response pattern.

        Sends a request and waits for a response. The target parameter
        specifies the service endpoint to call.

        Raises:
            ConnectorException: If the service call cannot be made.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

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

        Raises:
            ConnectorException: If the service cannot be created.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

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

        Raises:
            ConnectorException: If the action cannot be created.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

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

        Raises:
            ConnectorException: If the action cannot be started.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def terminate_action(self, action_handle: str, **kwargs: Any) -> Any:
        """Cancels an ongoing action.

        Stops the execution of a previously started action.
        The action_handle identifies which action to terminate.

        Raises:
            ConnectorException: If the action cannot be terminated.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def shutdown(self):
        """Shuts down the connector and releases all resources."""
        self.callback_executor.shutdown(wait=True)
