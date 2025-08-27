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
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from rai.initialization import get_tracing_callbacks

logger = logging.getLogger(__name__)


def invoke_llm_with_tracing(
    llm: BaseChatModel,
    messages: List[BaseMessage],
    config: Optional[RunnableConfig] = None,
) -> Any:
    """
    Invoke an LLM with enhanced tracing callbacks.

    This function automatically adds tracing callbacks (like Langfuse) to LLM calls
    within LangGraph nodes, solving the callback propagation issue.

    Tracing is controlled by config.toml. If the file is missing, no tracing is applied.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to invoke
    messages : List[BaseMessage]
        Messages to send to the LLM
    config : Optional[RunnableConfig]
        Existing configuration (may contain some callbacks)

    Returns
    -------
    Any
        The LLM response
    """
    tracing_callbacks = get_tracing_callbacks()

    if len(tracing_callbacks) == 0:
        # No tracing callbacks available, use config as-is
        return llm.invoke(messages, config=config)

    # Create enhanced config with tracing callbacks
    enhanced_config = config.copy() if config else {}

    # Add tracing callbacks to existing callbacks
    existing_callbacks = config.get("callbacks", []) if config else []

    if hasattr(existing_callbacks, "handlers"):
        # Merge with existing CallbackManager
        all_callbacks = existing_callbacks.handlers + tracing_callbacks
    elif isinstance(existing_callbacks, list):
        all_callbacks = existing_callbacks + tracing_callbacks
    else:
        all_callbacks = tracing_callbacks

    enhanced_config["callbacks"] = all_callbacks

    return llm.invoke(messages, config=enhanced_config)
