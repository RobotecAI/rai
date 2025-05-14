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
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    get_args,
)
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ConnectorException(Exception):
    """Base exception for all connector exceptions."""

    pass


class BaseMessage(BaseModel):
    """Base class for all messages in the connector system.

    Attributes
    ----------
    payload : Any, optional
        Payload is meant for non-validated data if such data is present.
    metadata : Dict[str, Any], optional
        Dictionary containing message metadata.
    timestamp : float, optional
        Timestamp of message creation.
    """

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
    id: str = Field(default_factory=lambda: str(uuid4()))


class BaseConnector(Generic[T]):
    def __init__(self, callback_max_workers: int = 4):
        self.callback_max_workers = callback_max_workers
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registered_callbacks: Dict[str, Dict[str, ParametrizedCallback[T]]] = (
            defaultdict(dict)
        )
        self.callback_id_mapping: Dict[str, tuple[str, ParametrizedCallback[T]]] = {}
        self.callback_executor = ThreadPoolExecutor(
            max_workers=self.callback_max_workers
        )

        if not hasattr(self, "__orig_bases__"):
            self.__orig_bases__ = {}
            raise ConnectorException(
                f"Error while instantiating {str(self.__class__)}: "
                "Message type T derived from BaseMessage needs to be provided"
                " e.g. Connector[MessageType]()"
            )
        self.T_class: Type[T] = get_args(self.__orig_bases__[-1])[0]

    def _generate_handle(self) -> str:
        return str(uuid4())

    def send_message(self, message: T, target: str, **kwargs: Optional[Any]) -> None:
        """Implements publish pattern.

        Sends a message to one or more subscribers. The target parameter
        can be used to specify the destination or topic.

        Parameters
        ----------
        message : T
            The message to send.
        target : str
            The destination or topic to send the message to.
        **kwargs : Optional[Any]
            Additional keyword arguments.

        Raises
        ------
        ConnectorException
            If the message cannot be sent.
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def receive_message(
        self, source: str, timeout_sec: float, **kwargs: Optional[Any]
    ) -> T:
        """Implements subscribe pattern.

        Receives a message from a publisher. The source parameter
        can be used to specify the source or topic to subscribe to.

        Parameters
        ----------
        source : str
            The source or topic to receive the message from.
        timeout_sec : float
            Timeout in seconds for receiving the message.
        **kwargs : Optional[Any]
            Additional keyword arguments.

        Returns
        -------
        T
            The received message.

        Raises
        ------
        ConnectorException
            If the message cannot be received.
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def register_callback(
        self,
        source: str,
        callback: Callable[[T | Any], None],
        raw: bool = False,
        **kwargs: Optional[Any],
    ) -> str:
        """Implements register callback.

        Registers a callback to be called when a message is received from a source.
        If raw is False, the callback will receive a T object.
        If raw is True, the callback will receive the raw message.

        Parameters
        ----------
        source : str
            The source to register the callback for.
        callback : Callable[[T | Any], None]
            The callback function to register.
        raw : bool, optional
            Whether to pass raw message to callback, by default False.
        **kwargs : Optional[Any]
            Additional keyword arguments.

        Returns
        -------
        str
            The ID of the registered callback.

        Raises
        ------
        ConnectorException
            If the callback cannot be registered.
        """
        parametrized_callback = ParametrizedCallback[T](callback=callback, raw=raw)
        self.registered_callbacks[source][parametrized_callback.id] = (
            parametrized_callback
        )
        self.callback_id_mapping[parametrized_callback.id] = (
            source,
            parametrized_callback,
        )
        return parametrized_callback.id

    def unregister_callback(self, callback_id: str) -> None:
        """Unregisters a callback from a source.

        Parameters
        ----------
        callback_id : str
            The id of the callback to unregister.

        Raises
        ------
        ConnectorException
            If the callback cannot be unregistered.
        """
        if callback_id not in self.callback_id_mapping:
            raise ConnectorException(f"Callback with id {callback_id} not found.")

        source, _ = self.callback_id_mapping[callback_id]
        del self.registered_callbacks[source][callback_id]
        del self.callback_id_mapping[callback_id]

    def _safe_callback_wrapper(self, callback: Callable[[T], None], message: T) -> None:
        """Safely execute a callback with error handling.

        Parameters
        ----------
        callback : Callable[[T], None]
            The callback function to execute.
        message : T
            The message to pass to the callback.
        """
        try:
            callback(message)
        except Exception as e:
            self.logger.error(f"Error in callback: {str(e)}")

    def general_callback(self, source: str, message: Any) -> None:
        processed_message = self.general_callback_preprocessor(message)
        for parametrized_callback in self.registered_callbacks.get(source, {}).values():
            payload = message if parametrized_callback.raw else processed_message
            self.callback_executor.submit(
                self._safe_callback_wrapper, parametrized_callback.callback, payload
            )

    def general_callback_preprocessor(self, message: Any) -> T:
        """Preprocessor for general callback used to transform any message to a BaseMessage.

        Parameters
        ----------
        message : Any
            The message to preprocess.

        Returns
        -------
        T
            The preprocessed message.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def service_call(
        self, message: T, target: str, timeout_sec: float, **kwargs: Optional[Any]
    ) -> BaseMessage:
        """Implements request-response pattern.

        Sends a request and waits for a response. The target parameter
        specifies the service endpoint to call.

        Parameters
        ----------
        message : T
            The request message to send.
        target : str
            The service endpoint to call.
        timeout_sec : float
            Timeout in seconds for the service call.
        **kwargs : Optional[Any]
            Additional keyword arguments.

        Returns
        -------
        BaseMessage
            The response message.

        Raises
        ------
        ConnectorException
            If the service call cannot be made.
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def create_service(
        self,
        service_name: str,
        on_request: Callable,
        on_done: Optional[Callable] = None,
        **kwargs: Optional[Any],
    ) -> str:
        """Sets up a service endpoint for handling requests.

        Creates a service that can receive and process requests.
        The on_request callback handles incoming requests,
        and on_done (if provided) is called when the service is terminated.

        Parameters
        ----------
        service_name : str
            Name of the service to create.
        on_request : Callable
            Callback function to handle incoming requests.
        on_done : Optional[Callable], optional
            Callback function called when service is terminated, by default None.
        **kwargs : Optional[Any]
            Additional keyword arguments.

        Returns
        -------
        str
            The handle of the created service.

        Raises
        ------
        ConnectorException
            If the service cannot be created.
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def create_action(
        self,
        action_name: str,
        generate_feedback_callback: Callable,
        **kwargs: Optional[Any],
    ) -> str:
        """Sets up an action endpoint for long-running operations.

        Creates an action that can be started and monitored.
        The generate_feedback_callback is used to provide progress updates
        during the action's execution.

        Parameters
        ----------
        action_name : str
            Name of the action to create.
        generate_feedback_callback : Callable
            Callback function to generate feedback during action execution.
        **kwargs : Optional[Any]
            Additional keyword arguments.

        Returns
        -------
        str
            The handle of the created action.

        Raises
        ------
        ConnectorException
            If the action cannot be created.
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def start_action(
        self,
        action_data: Optional[T],
        target: str,
        on_feedback: Callable,
        on_done: Callable,
        timeout_sec: float,
        **kwargs: Optional[Any],
    ) -> str:
        """Initiates a long-running operation with feedback.

        Starts an action and provides callbacks for feedback and completion.
        The on_feedback callback receives progress updates,
        and on_done is called when the action completes.

        Parameters
        ----------
        action_data : Optional[T]
            Data to pass to the action.
        target : str
            The action endpoint to start.
        on_feedback : Callable
            Callback function to receive progress updates.
        on_done : Callable
            Callback function called when action completes.
        timeout_sec : float
            Timeout in seconds for the action.
        **kwargs : Optional[Any]
            Additional keyword arguments.

        Returns
        -------
        str
            The handle of the started action.

        Raises
        ------
        ConnectorException
            If the action cannot be started.
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def terminate_action(self, action_handle: str, **kwargs: Optional[Any]) -> Any:
        """Cancels an ongoing action.

        Stops the execution of a previously started action.
        The action_handle identifies which action to terminate.

        Parameters
        ----------
        action_handle : str
            The handle of the action to terminate.
        **kwargs : Optional[Any]
            Additional keyword arguments.

        Returns
        -------
        Any
            Result of the termination operation.

        Raises
        ------
        ConnectorException
            If the action cannot be terminated.
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def shutdown(self):
        """Shuts down the connector and releases all resources."""
        self.callback_executor.shutdown(wait=True)
