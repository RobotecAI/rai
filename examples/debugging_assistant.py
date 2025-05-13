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

import streamlit as st
from rai import get_llm_model
from rai.agents.langchain.core import create_conversational_agent
from rai.frontend import run_streamlit_app
from rai.tools.ros2 import ROS2CLIToolkit


@st.cache_resource
def initialize_agent():
    llm = get_llm_model(model_type="complex_model", streaming=True)
    agent = create_conversational_agent(
        llm,
        ROS2CLIToolkit().get_tools(),
        system_prompt="""You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions. Retrieve as much information from the ROS 2 system as possible.
                """,
    )
    return agent


st.set_page_config(
    page_title="ROS 2 Debugging Assistant",
    page_icon=":robot:",
)


def main():
    run_streamlit_app(
        initialize_agent(),
        page_title="ROS 2 Debugging Assistant",
        initial_message="Hi! I am a ROS 2 assistant. How can I help you?",
    )


if __name__ == "__main__":
    main()
