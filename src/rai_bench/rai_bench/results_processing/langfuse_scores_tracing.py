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

from typing import List
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langfuse.callback import CallbackHandler
from rai.initialization import get_tracing_callbacks


class ScoreTracingHandler:
    """
    Class to handle sending scores to tracing backends.
    """

    # TODO (mkotynia) handle grouping single benchmark scores to sessions
    # TODO (mkotynia) trace and send more metadata?
    @staticmethod
    def get_callbacks() -> List[BaseCallbackHandler]:
        return get_tracing_callbacks()

    @staticmethod
    def send_score(
        callback: BaseCallbackHandler,
        run_id: UUID,
        score: float,
        errors: List[List[str]] | None = None,
    ) -> None:
        comment = (
            "; ".join(", ".join(error_group) for error_group in errors)
            if errors
            else ""
        )
        if isinstance(callback, CallbackHandler):
            callback.langfuse.score(
                trace_id=str(run_id),
                name="tool calls result",
                value=score,
                comment=comment,
            )
            return None
        if isinstance(callback, LangChainTracer):
            callback.client.create_feedback(
                run_id=run_id,
                key="tool calls result",
                score=score,
                comment=comment,
            )
            return None
        raise NotImplementedError(
            f"Callback {callback} of type {callback.__class__.__name__} not supported"
        )
