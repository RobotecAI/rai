# Copyright (C) 2026 Kajetan Rachwał
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

from typing import Any, Callable, Optional

from rai.communication import HRIConnector
from rai.communication.http.api import HTTPConnectorMode
from rai.communication.http.connectors.base import HTTPBaseConnector
from rai.communication.http.messages import HTTPHRIMessage


class HTTPHRIConnector(HTTPBaseConnector[HTTPHRIMessage], HRIConnector[HTTPHRIMessage]):
    """HTTP connector specialised for multimodal Human-Robot Interaction messages.

    Serialises :class:`~rai.communication.http.messages.HTTPHRIMessage` to JSON
    (with base64-encoded images and audio) for transport and reconstructs them
    on the receiving side.

    Parameters
    ----------
    host : str
        Hostname used when running in server or client_server mode.
    port : int
        Port used when running in server or client_server mode.
    mode : HTTPConnectorMode
        Operating mode: client, server, or client_server.
    """

    def __init__(
        self,
        host: str,
        port: int,
        mode: HTTPConnectorMode = HTTPConnectorMode.client,
    ):
        super().__init__(host=host, port=port, mode=mode)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def message_to_payload(message: HTTPHRIMessage) -> dict:
        return {
            "text": message.text,
            "images": [HTTPHRIMessage._image_to_base64(img) for img in message.images],
            "audios": [HTTPHRIMessage._audio_to_base64(aud) for aud in message.audios],
            "message_author": message.message_author,
            "communication_id": message.communication_id,
            "seq_no": message.seq_no,
            "seq_end": message.seq_end,
        }

    @staticmethod
    def payload_to_message(data: dict) -> HTTPHRIMessage:
        return HTTPHRIMessage(
            text=data.get("text", ""),
            images=[
                HTTPHRIMessage._base64_to_image(img)
                for img in data.get("images", [])
            ],
            audios=[
                HTTPHRIMessage._base64_to_audio(aud)
                for aud in data.get("audios", [])
            ],
            message_author=data.get("message_author", "human"),
            communication_id=data.get("communication_id"),
            seq_no=data.get("seq_no", 0),
            seq_end=data.get("seq_end", False),
        )
