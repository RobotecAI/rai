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

from .agents import ReActAgent
from .initialization import (
    get_embeddings_model,
    get_llm_model,
    get_llm_model_config_and_vendor,
    get_llm_model_direct,
    get_tracing_callbacks,
)

__all__ = [
    "ReActAgent",
    "get_embeddings_model",
    "get_llm_model",
    "get_llm_model_config_and_vendor",
    "get_llm_model_direct",
    "get_tracing_callbacks",
    "run",
]


def __getattr__(name: str):
    if name == "run":  # lazy, so that `import rai` stays cheap
        from rai.agents.runner import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
