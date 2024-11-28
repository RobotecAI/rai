import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rai.agents.conversational_agent import create_conversational_agent
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.tools.ros.debugging import (
    ros2_action,
    ros2_interface,
    ros2_node,
    ros2_service,
    ros2_topic,
)
from rai.utils.model_initialization import get_llm_model


def initialize_graph():
    llm = get_llm_model(model_type="complex_model", streaming=True)
    agent = create_conversational_agent(
        llm,
        [ros2_topic, ros2_interface, ros2_node, ros2_service, ros2_action],
        system_prompt="You are a helpful assistant that can answer questions about ROS 2.",
    )
    return agent


st.title("ROS 2 Debugging Assistant")
st.markdown("---")

st.sidebar.header("Tool Calls History")

if "graph" not in st.session_state:
    graph = initialize_graph()
    st.session_state["graph"] = graph

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="Hi! I am a ROS 2 assistant. How can I help you?")
    ]

prompt = st.chat_input()
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        if msg.content:
            st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.sidebar.expander(f"Tool: {msg.name}", expanded=False):
            st.code(msg.content, language="json")


if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        response = streamlit_invoke(
            st.session_state["graph"], st.session_state.messages, [st_callback]
        )
