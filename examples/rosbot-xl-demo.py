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

import uuid
from typing import List, cast

import rclpy
import streamlit as st
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain import (
    ReActAgent,
    ReActAgentState,
)
from rai.communication.ros2 import ROS2Connector
from rai.frontend.streamlit import run_streamlit_app
from rai.tools.ros2 import (
    GetObjectPositionsTool,
    GetROS2ImageConfiguredTool,
    GetROS2TransformConfiguredTool,
    Nav2Toolkit,
)
from rai.tools.time import WaitForSecondsTool
from rai_perception.tools import GetGrabbingPointTool

from rai_whoami import EmbodimentInfo

# Set page configuration first
st.set_page_config(
    page_title="RAI ROSBotXL Demo",
    page_icon=":robot:",
)


@st.cache_resource
def initialize_agent() -> Runnable[ReActAgentState, ReActAgentState]:
    rclpy.init()
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/rosbotxl_embodiment.json"
    )

    connector = ROS2Connector(executor_type="multi_threaded", use_sim_time=True)
    tools: List[BaseTool] = [
        GetROS2TransformConfiguredTool(
            connector=connector,
            source_frame="map",
            target_frame="base_link",
            timeout_sec=5.0,
        ),
        GetROS2ImageConfiguredTool(
            connector=connector,
            topic="/camera/camera/color/image_raw",
        ),
        WaitForSecondsTool(),
        GetObjectPositionsTool(
            connector=connector,
            target_frame="map",
            source_frame="sensor_frame",
            camera_topic="/camera/camera/color/image_raw",
            depth_topic="/camera/camera/depth/image_rect_raw",
            camera_info_topic="/camera/camera/color/camera_info",
            get_grabbing_point_tool=GetGrabbingPointTool(
                connector=connector,
            ),
        ),
        *Nav2Toolkit(connector=connector).get_tools(),
    ]

    agent = ReActAgent(
        target_connectors={},  # empty dict, since we're using the agent in direct mode
        llm=get_llm_model("complex_model", streaming=True),
        system_prompt=embodiment_info.to_langchain(),
        tools=tools,
    ).agent
    connector.node.declare_parameter("conversion_ratio", 1.0)

    return cast(Runnable[ReActAgentState, ReActAgentState], agent)


def _get_tracing_decorators():
    """Get tracing decorators based on config.toml settings."""
    from rai.initialization.model_initialization import load_config

    try:
        config = load_config()
        if config.tracing.langfuse.use_langfuse:
            from langfuse import observe, propagate_attributes
            return observe, propagate_attributes
    except Exception:
        pass

    # Return no-op decorators if langfuse is not enabled
    def noop_decorator(func):
        return func

    from contextlib import nullcontext

    def noop_context_manager(**kwargs):
        return nullcontext()

    return noop_decorator, noop_context_manager


observe, propagate_attributes = _get_tracing_decorators()


def generate_session_id(app_name: str) -> str:
    """Generate a session ID using app name and UUID."""
    session_uuid = str(uuid.uuid4())
    return f"{app_name}-{session_uuid}"


@observe()
def main():
    app_name = "rosbot-xl-demo"
    session_id = generate_session_id(app_name)
    with propagate_attributes(session_id=session_id):
        # All nested observations automatically inherit session_id
        run_streamlit_app(
            initialize_agent(),
            "RAI ROSBotXL Demo",
            "Hi! I am a ROSBotXL robot. What can I do for you?",
        )

if __name__ == "__main__":
    main()
