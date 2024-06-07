import subprocess
import time

from langchain_core.pydantic_v1 import BaseModel, Field


class send_voice_message(BaseModel):
    """Output a voice message"""

    content: str = Field(..., description="The content of the voice message")

    def run(self):
        """Gets the current map from the specified topic."""

        filepath = f"outputs/{time.time_ns()}.wav"
        content = self.content.replace("'", "")
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


class wait_for_seconds(BaseModel):
    """Wait for a specified number of seconds"""

    seconds: int = Field(..., description="The number of seconds to wait")

    def run(self):
        """Waits for the specified number of seconds."""
        time.sleep(self.seconds)
        return f"Waited for {self.seconds} seconds."
