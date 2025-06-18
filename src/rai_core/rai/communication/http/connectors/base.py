from typing import Any, Callable, Optional, TypeVar

from rai.communication.base_connector import BaseConnector, BaseMessage
from rai.communication.http.messages import HTTPMessage

T = TypeVar("T", bound=HTTPMessage)


class HTTPBaseConnector(BaseConnector[T]):
    def send_message(self, message: T, target: str, **kwargs: Optional[Any]) -> None:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def receive_message(
        self, source: str, timeout_sec: float, **kwargs: Optional[Any]
    ) -> T:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def _safe_callback_wrapper(self, callback: Callable[[T], None], message: T) -> None:
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
        raise NotImplementedError("This method should be implemented by the subclass.")

    def service_call(
        self, message: T, target: str, timeout_sec: float, **kwargs: Optional[Any]
    ) -> BaseMessage:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def create_service(
        self,
        service_name: str,
        on_request: Callable,
        on_done: Optional[Callable] = None,
        **kwargs: Optional[Any],
    ) -> str:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def create_action(
        self,
        action_name: str,
        generate_feedback_callback: Callable,
        **kwargs: Optional[Any],
    ) -> str:
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
        raise NotImplementedError("This method should be implemented by the subclass.")

    def terminate_action(self, action_handle: str, **kwargs: Optional[Any]) -> Any:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def shutdown(self):
        """Shuts down the connector and releases all resources."""
        self.callback_executor.shutdown(wait=True)
