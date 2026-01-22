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
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


def create_agent():
    """Create and configure the manipulation agent (v2).

    This is a thin wrapper around manipulation_common.create_agent(version="v2").
    See manipulation_common.py for documentation on parameter configuration.
    """
    from manipulation_common import create_agent as _create_agent

    agent, _ = _create_agent(version="v2")
    return agent


def main():
    agent = create_agent()
    messages: List[BaseMessage] = []

    while True:
        prompt = input("Enter a prompt: ")
        messages.append(HumanMessage(content=prompt))
        output = agent.invoke({"messages": messages})
        output["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
