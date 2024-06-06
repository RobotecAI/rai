import subprocess
import time

from langchain_core.pydantic_v1 import BaseModel, Field


class send_voice_message(BaseModel):
    """Output a voice message"""

    content: str = Field(..., description="The content of the voice message")

    def run(self):
        """Gets the current map from the specified topic."""

        filepath = f"outputs/{time.time_ns()}.wav"

        p = subprocess.check_output(
            f"echo {self.content} | piper   --model en_US-libritts-high   --output_file {filepath}",
            shell=True,
        )
        subprocess.run(["play", filepath])
        return filepath
