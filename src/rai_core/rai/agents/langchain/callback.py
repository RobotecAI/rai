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
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

from rai.communication.hri_connector import HRIConnector, HRIMessage


class HRICallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        connectors: Dict[str, HRIConnector[HRIMessage]],
        aggregate_chunks: bool = False,
        splitting_chars: Optional[List[str]] = None,
        max_buffer_size: int = 200,
        logger: Optional[logging.Logger] = None,
        stream_response: bool = True,
    ):
        self.connectors = connectors
        self.aggregate_chunks = aggregate_chunks
        self.stream_response = stream_response
        self.splitting_chars = splitting_chars or ["\n", ".", "!", "?"]
        self.chunks_buffer = ""
        self.max_buffer_size = max_buffer_size
        self._buffer_lock = threading.Lock()
        self.logger = logger or logging.getLogger(__name__)
        self.current_conversation_id = None
        self.current_chunk_id = 0
        self.working = False
        self.hit_on_llm_new_token = False

    def _should_split(self, token: str) -> bool:
        return token in self.splitting_chars

    def _send_all_targets(self, tokens: str, done: bool = False):
        for target, connector in self.connectors.items():
            self.logger.info(f"Sending {len(tokens)} tokens to target: {target}")
            try:
                to_send: HRIMessage = connector.build_message(
                    AIMessage(content=tokens),
                    self.current_conversation_id,
                    self.current_chunk_id,
                    done,
                )
                connector.send_message(to_send, target)
                self.logger.debug(f"Sent {len(tokens)} tokens to hri_connector.")
            except Exception as e:
                self.logger.error(
                    f"Failed to send {len(tokens)} tokens to hri_connector: {e}"
                )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self.working = True

    def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs):
        self.hit_on_llm_new_token = True
        if token == "" or not self.stream_response:
            return
        if self.current_conversation_id != str(run_id):
            self.current_conversation_id = str(run_id)
            self.current_chunk_id = 0
        if self.aggregate_chunks:
            with self._buffer_lock:
                self.chunks_buffer += token
                if len(self.chunks_buffer) < self.max_buffer_size:
                    if self._should_split(token):
                        self._send_all_targets(self.chunks_buffer)
                        self.chunks_buffer = ""
                        self.current_chunk_id += 1
                else:
                    self._send_all_targets(self.chunks_buffer)
                    self.chunks_buffer = ""
                    self.current_chunk_id += 1
        else:
            self._send_all_targets(token)
            self.current_chunk_id += 1

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs,
    ):
        self.current_conversation_id = str(run_id)
        if self.stream_response and not self.hit_on_llm_new_token:
            self.logger.error(
                (
                    "No tokens were sent to the callback handler. "
                    "LLM did not stream response. "
                    "Is your BaseChatModel configured to stream? "
                    "Sending generated text as a single message."
                )
            )
            msg = response.generations[0][0].message
            self._send_all_targets(msg.content, done=True)
        elif not self.stream_response:
            msg = response.generations[0][0].message
            self._send_all_targets(msg.content, done=True)
        elif self.aggregate_chunks and self.chunks_buffer:
            with self._buffer_lock:
                self._send_all_targets(self.chunks_buffer, done=True)
                self.chunks_buffer = ""
        self.working = False
