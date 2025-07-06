from typing import Any, Callable, Optional, TypeVar

from rai.communication.base_connector import BaseConnector, BaseMessage
from rai.communication.http.messages import HTTPMessage
from rai.communication.http.api import HTTPAPI, HTTPAPIError, HTTPConnectorMode

T = TypeVar("T", bound=HTTPMessage)


class HTTPBaseConnector(BaseConnector[T]):
    def __init__(
        self,
        mode: HTTPConnectorMode = HTTPConnectorMode.client,
        host: str = "localhost",
        port: int = 8080,
    ):
        super().__init__()

        self._api = HTTPAPI(mode, host, port)
        self._api.run()
        self._services = []

    def send_message(self, message: T, target: str, **kwargs: Optional[Any]) -> None:
        self._api.send_request(
            message.method,
            target,
            None,
            payload=message.payload,
            headers=message.headers,
        )

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
        payload, status = self._api.send_request(
            message.method,
            target,
            timeout_sec,
            payload=message.payload,
            headers=message.headers,
        )
        ret = BaseMessage(payload=payload, metadata={"status": status})
        return ret

    def create_service(
        self,
        service_name: str,
        on_request: Callable,
        on_done: Optional[Callable] = None,
        *,
        method: str,
        **kwargs: Optional[Any],
    ) -> str:
        id_str = f"{method.upper()}_{service_name}"
        if id_str in self._services:
            raise HTTPAPIError(
                f"Service {service_name} already has a {method.upper()} handler"
            )
        self._api.add_route(method, service_name, on_request)
        return id_str

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
        self._api.shutdown()
        self.callback_executor.shutdown(wait=True)
