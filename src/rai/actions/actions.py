import base64
import difflib
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    from rai.scenario_engine.scenario_engine import ScenarioRunner


class Action:
    def __init__(self, logging_level: int = logging.INFO, separate_thread: bool = True):
        self.logging_level = logging_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.logging_level)
        self.separate_thread = separate_thread

    def run(self, runner: "ScenarioRunner") -> None:
        raise NotImplementedError()


class MessageAdminAction(Action):
    def __init__(self, message: Dict[str, Any]):
        self.message = message

    def run(self, runner: "ScenarioRunner") -> None:
        logging.critical(f"Message: {self.message}")


class SoundAlarmAction(Action):
    def __init__(self, alarm: str):
        self.alarm = alarm

    def run(self, runner: "ScenarioRunner") -> None:
        logging.critical(f"Alarm: {self.alarm}")


class SendEmailAction(Action):
    def __init__(self, email: str, logging_level: int = logging.INFO):
        super().__init__(logging_level=logging_level)
        self.email = email

    def run(self, runner: "ScenarioRunner") -> None:
        from rai.communication.communication import EmailSender

        email_sender = EmailSender(smtp_server="", smtp_port=587)
        email_sender.send_email(
            self.email,
            runner.history[-1].content,
            runner.get_html(runner.history, runner.ai_vendor),
        )


class SendStopSignalAction(Action):
    def __init__(self, signal: str):
        self.signal = signal

    def run(self, runner: "ScenarioRunner") -> None:
        logging.critical(f"Signal: {self.signal}")


class EventReportSaver(Action):
    def __init__(
        self,
        namespace: str,
        position: Callable[[], Any],
        image_idx: int = 0,
        action_idx: int = -1,
    ):
        super().__init__()
        self.namespace = namespace
        self.image_idx = image_idx
        self.action_idx = action_idx
        self.position = position

    def run(self, runner: "ScenarioRunner") -> None:
        folder_path = Path(runner.logs_dir) / "events"
        folder_path.mkdir(parents=True, exist_ok=True)

        image = runner.history[self.image_idx].images[0]
        image = base64.b64decode(image)
        with open(folder_path / "image.png", "wb") as image_file:
            image_file.write(image)

        action = runner.history[self.action_idx].content
        with open(folder_path / "action.txt", "w") as action_file:
            action_file.write(action)

        with open(folder_path / "position.txt", "w") as position_file:
            position_file.write(str(self.position))


class Wait(Action):
    def __init__(self, seconds: float):
        super().__init__(separate_thread=False)
        self.seconds = seconds

    def run(self, runner: "ScenarioRunner") -> None:
        time.sleep(self.seconds)


class CallCommand(Action):
    def __init__(self, action_to_command: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.action_to_command = action_to_command

    def run(self, runner: "ScenarioRunner") -> None:
        last_message = runner.history[-1].content.replace("`", "")
        try:
            command = self.action_to_command[last_message]
        except KeyError:
            closest_match = difflib.get_close_matches(
                last_message, self.action_to_command.keys()
            )
            if len(closest_match) == 0:
                raise ValueError(
                    f"Requested action {last_message} not found. No close match found."
                )
            runner.logger.info(
                f"Requested action {last_message} not found. Closest match: {closest_match[0]}"
            )
            command = self.action_to_command[closest_match[0]]

        runner.logger.info("Requested command: " + command)
        output = subprocess.run(
            command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        status = "Success" if output.returncode == 0 else "Failed"
        runner.logger.info(f"Requested command status: {status}")
