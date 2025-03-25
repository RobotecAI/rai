# Copyright (C) 2025 Robotec.AI
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
import threading
from typing import List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

from rai.communication.hri_connector import HRIConnector, HRIMessage


class HRICallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        connectors: dict[str, HRIConnector[HRIMessage]],
        aggregate_chunks: bool = False,
        splitting_chars: Optional[List[str]] = None,
        max_buffer_size: int = 200,
        logger: Optional[logging.Logger] = None,
    ):
        self.connectors = connectors
        self.aggregate_chunks = aggregate_chunks
        self.splitting_chars = splitting_chars or ["\n", ".", "!", "?"]
        self.chunks_buffer = ""
        self.max_buffer_size = max_buffer_size
        self._buffer_lock = threading.Lock()
        self.logger = logger or logging.getLogger(__name__)
        self.current_conversation_id = None

    def _should_split(self, token: str) -> bool:
        return token in self.splitting_chars

    def _send_all_targets(self, tokens: str):
        self.logger.info(
            f"Sending {len(tokens)} tokens to {len(self.connectors)} connectors"
        )
        for connector_name, connector in self.connectors.items():
            try:
                connector.send_all_targets(
                    AIMessage(content=tokens), self.current_conversation_id
                )
                self.logger.debug(f"Sent {len(tokens)} tokens to {connector_name}")
            except Exception as e:
                self.logger.error(
                    f"Failed to send {len(tokens)} tokens to {connector_name}: {e}"
                )

    def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs):
        if token == "":
            return
        self.current_conversation_id = str(run_id)
        if self.aggregate_chunks:
            with self._buffer_lock:
                self.chunks_buffer += token
                if len(self.chunks_buffer) < self.max_buffer_size:
                    if self._should_split(token):
                        self._send_all_targets(self.chunks_buffer)
                        self.chunks_buffer = ""
                else:
                    self._send_all_targets(self.chunks_buffer)
                    self.chunks_buffer = ""
        else:
            self._send_all_targets(token)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs,
    ):
        self.current_conversation_id = str(run_id)
        if self.aggregate_chunks and self.chunks_buffer:
            with self._buffer_lock:
                self._send_all_targets(self.chunks_buffer)
                self.chunks_buffer = ""
