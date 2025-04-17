from typing import Any, Callable, Literal, Optional

from rai.communication.hri_connector import HRIConnector, HRIMessage, HRIPayload


class CLIHRIMessage(HRIMessage):
    def __init__(
        self,
        payload: HRIPayload,
        message_author: Literal["ai", "human"],
        communication_id: Optional[str] = None,
        seq_no: int = 0,
        seq_end: bool = False,
    ):
        super().__init__(payload, {}, message_author, communication_id, seq_no, seq_end)


class CLIHRIConnector(HRIConnector[CLIHRIMessage]):
    def __init__(
        self,
        targets: list[str] = [],
        sources: list[str] = [],
    ):
        super().__init__(targets, sources)

    def send_message(self, message: CLIHRIMessage, target: str, **kwargs):
        print(f"Sending message to {target}: {message}")

    def receive_message(
        self, source: str, timeout_sec: float, **kwargs: Any
    ) -> CLIHRIMessage:
        input_text = input(f"Enter message from {source}: ")
        return CLIHRIMessage(
            payload=HRIPayload(text=input_text),
            message_author="human",
            communication_id=None,
            seq_no=0,
            seq_end=False,
        )

    def service_call(
        self, message: CLIHRIMessage, target: str, timeout_sec: float, **kwargs: Any
    ) -> CLIHRIMessage:
        print(f"Service call to {target} with message: {message}")
        response_text = input("Enter response from service: ")
        return CLIHRIMessage(
            payload=HRIPayload(text=response_text),
            message_author="ai",
            communication_id=None,
            seq_no=0,
            seq_end=False,
        )

    def create_service(
        self,
        service_name: str,
        on_request: Callable,
        on_done: Optional[Callable] = None,
        **kwargs: Any,
    ) -> str:
        print(f"Creating service {service_name}")
        return service_name

    def create_action(
        self,
        action_name: str,
        generate_feedback_callback: Callable,
        **kwargs: Any,
    ) -> str:
        print(f"Creating action {action_name}")
        return action_name

    def start_action(
        self,
        action_data: Optional[CLIHRIMessage],
        target: str,
        on_feedback: Callable,
        on_done: Callable,
        timeout_sec: float,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("CLI does not support actions.")

    def terminate_action(self, action_handle: str, **kwargs: Any):
        raise NotImplementedError("CLI does not support actions.")
