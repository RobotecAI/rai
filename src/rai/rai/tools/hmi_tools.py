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
#

import time
from typing import Type

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from rai.communication.communication import EmailSender


class PlayVoiceMessageToolInput(BaseModel):
    """Input for the PlayVoiceMessageTool tool."""

    content: str = Field(..., description="The content of the voice message")


class PlayVoiceMessageTool(BaseTool):
    """Output a voice message"""

    name: str = "PlayVoiceMessageTool"
    description: str = (
        "A tool for sending voice messages. "
        "Useful for sending audio content as messages. "
        "Input should be the content of the voice message."
    )

    args_schema: Type[PlayVoiceMessageToolInput] = PlayVoiceMessageToolInput

    def _run(self, content: str):
        raise NotImplementedError


class WaitForSecondsToolInput(BaseModel):
    """Input for the WaitForSecondsTool tool."""

    seconds: int = Field(..., description="The number of seconds to wait")


class WaitForSecondsTool(BaseTool):
    """Wait for a specified number of seconds"""

    name: str = "WaitForSecondsTool"
    description: str = (
        "A tool for waiting. "
        "Useful for pausing execution for a specified number of seconds. "
        "Input should be the number of seconds to wait."
    )

    args_schema: Type[WaitForSecondsToolInput] = WaitForSecondsToolInput

    def _run(self, seconds: int):
        """Waits for the specified number of seconds."""
        time.sleep(seconds)
        return f"Waited for {seconds} seconds."


class SendEmailToolInput(BaseModel):
    """Input for the SendEmailToAdminTool tool."""

    recipient: str = Field(..., description="The email address of the recipient.")
    subject: str = Field(
        ..., description="The subject of the email. Should be very short."
    )
    content: str = Field(
        ..., description="The content of the email. Should be short and concise."
    )


class SendEmailTool(BaseTool):
    """Send an email to the admin"""

    name: str = "SendEmailToAdminTool"
    description: str = (
        "A tool for sending emails to the admin. "
        "Useful for sending notifications to the admin. "
        "Input should be the subject and content of the email."
    )

    args_schema: Type[SendEmailToolInput] = SendEmailToolInput

    def _run(self, recipient: str, subject: str, content: str):
        """Sends an email to the admin."""
        email_sender = EmailSender(smtp_server="", smtp_port=587)
        email_sender.send_email(
            recipient_email=recipient, subject=subject, message=content
        )
        return "Email sent to admin."
