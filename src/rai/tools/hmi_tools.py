import subprocess
import time

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool


class PlayVoiceMessageToolInput(BaseModel):
    """Input for the play_voice_message tool."""

    content: str = Field(..., description="The content of the voice message")


class PlayVoiceMessageTool(BaseTool):
    """Output a voice message"""

    name: str = "play_voice_message"
    description: str = (
        "A tool for sending voice messages. "
        "Useful for sending audio content as messages. "
        "Input should be the content of the voice message."
    )

    args_schema = PlayVoiceMessageToolInput

    def _run(self, content: str):
        """plays the voice message."""
        filepath = f"outputs/{time.time_ns()}.wav"
        content = content.replace("'", "")
        subprocess.run(
            f"echo {content} | piper   --model en_US-libritts-high   --output_file {filepath}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["play", filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return "Voice message sent."


class WaitForSecondsToolInput(BaseModel):
    """Input for the wait_for_seconds tool."""

    seconds: int = Field(..., description="The number of seconds to wait")


class WaitForSecondsTool(BaseTool):
    """Wait for a specified number of seconds"""

    name: str = "wait_for_seconds"
    description: str = (
        "A tool for waiting. "
        "Useful for pausing execution for a specified number of seconds. "
        "Input should be the number of seconds to wait."
    )

    args_schema = WaitForSecondsToolInput

    def _run(self, seconds: int):
        """Waits for the specified number of seconds."""
        time.sleep(seconds)
        return f"Waited for {seconds} seconds."
