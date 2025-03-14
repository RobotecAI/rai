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

from langchain_core.callbacks.base import BaseCallbackHandler
from langfuse.callback import CallbackHandler
from rai.utils.model_initialization import get_tracing_callbacks


class ScoreTracingHandler:
    """
    Class to handle sending scores to tracing backends.
    """

    # TODO (mkotynia) currently only for langfuse, to handle langsmith tracing as well
    # TODO (mkotynia) handle grouping single benchmark scores to sessions (supported by langfuse)
    # TODO (mkotynia) trace and send more data - e.g. execution time
    @staticmethod
    def get_callbacks() -> List[BaseCallbackHandler]:
        return get_tracing_callbacks()

    @staticmethod
    def get_trace_id(callback: BaseCallbackHandler):
        if isinstance(callback, CallbackHandler):
            return callback.get_trace_id()
        raise NotImplementedError(
            f"Callback {callback} of type {callback.__class__.__name__} not supported"
        )

    @staticmethod
    def send_score(
        callback: BaseCallbackHandler, trace_id: str, success: bool, errors: List[str]
    ) -> None:
        if isinstance(callback, CallbackHandler):
            callback.langfuse.score(
                trace_id=trace_id,
                name="tool calls result",
                value=float(success),
                comment="; ".join(errors) if errors else "",
            )
            return
        raise NotImplementedError(
            f"Callback {callback} of type {callback.__class__.__name__} not supported"
        )
