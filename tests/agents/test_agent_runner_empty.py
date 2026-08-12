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

import pytest
from rai.agents.runner import AgentRunner, run_agents, wait_for_shutdown


@pytest.mark.parametrize(
    "fn",
    [
        lambda: run_agents([]),
        lambda: wait_for_shutdown([]),
        lambda: AgentRunner([]),
        lambda: run_agents(None),  # type: ignore[arg-type]
    ],
)
def test_agent_helpers_reject_empty_agents(fn):
    with pytest.raises(ValueError, match="agents must be a non-empty list"):
        fn()
