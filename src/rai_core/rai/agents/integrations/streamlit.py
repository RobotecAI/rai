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

import base64
import inspect
from typing import Any, Callable, Dict, List, TypeVar

import cv2
import numpy as np
import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# code inspired by (mostly copied, some changes were applied and might be updated in the future)
# https://github.com/shiv248/Streamlit-x-LangGraph-Cookbooks/tree/b8e623bdc9821fc1cf581607454dae1afc054df2/tool_calling_via_callback


# Define a function to create a callback handler for Streamlit that updates the UI dynamically
def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that updates the provided Streamlit container with new tokens.
    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered.
    Returns:
        BaseCallbackHandler: An instance of a callback handler configured for Streamlit.
    """

    # Define a custom callback handler class for managing and displaying stream events in Streamlit
    class StreamHandler(BaseCallbackHandler):
        """
        Custom callback handler for Streamlit that updates a Streamlit container with new tokens.
        """

        def __init__(
            self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
        ):
            """
            Initializes the StreamHandler with a Streamlit container and optional initial text.
            Args:
                container (st.delta_generator.DeltaGenerator): The Streamlit container where text will be rendered.
                initial_text (str): Optional initial text to start with in the container.
            """
            self.container = container  # The Streamlit container to update
            self.thoughts_placeholder = (
                self.container.container()
            )  # container to hold tool_call renders
            self.tool_output_placeholder = None  # placeholder for the output of the tool call to be in the expander
            self.token_placeholder = self.container.empty()  # for token streaming
            self.text = (
                initial_text  # The text content to display, starting with initial text
            )

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            """
            Callback method triggered when a new token is received (e.g., from a language model).
            Args:
                token (str): The new token received.
                **kwargs: Additional keyword arguments.
            """
            self.text += token  # Append the new token to the existing text
            self.token_placeholder.write(self.text)

        def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
        ) -> None:
            """
            Run when the tool starts running.
            Args:
                serialized (Dict[str, Any]): The serialized tool.
                input_str (str): The input string.
                kwargs (Any): Additional keyword arguments.
            """
            with self.thoughts_placeholder:
                status_placeholder = st.empty()  # Placeholder to show the tool's status
                with status_placeholder.status("Calling Tool...", expanded=True) as s:
                    st.write("Called tool: ", serialized["name"])
                    st.write("tool description: ", serialized["description"])
                    st.write("tool input: ")
                    st.code(input_str)  # Display the input data sent to the tool
                    st.write("tool output: ")
                    # Placeholder for tool output that will be updated later below
                    self.tool_output_placeholder = st.empty()
                    s.update(
                        label="Completed Calling Tool!", expanded=False
                    )  # Update the status once done

        def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
            """
            Run when the tool ends.
            Args:
                output (Any): The output from the tool.
                kwargs (Any): Additional keyword arguments.
            """
            # We assume that `on_tool_end` comes after `on_tool_start`, meaning output_placeholder exists
            if self.tool_output_placeholder:
                with self.tool_output_placeholder.container():
                    st.code(output.content)  # Display the tool's output

                    if output.artifact is not None:
                        if "images" in output.artifact:  # Display available images
                            for image in output.artifact["images"]:
                                image_cv2 = cv2.imdecode(
                                    np.frombuffer(base64.b64decode(image), np.uint8),
                                    cv2.IMREAD_COLOR,
                                )
                                st.image(
                                    image_cv2, channels="BGR", use_container_width=True
                                )

    # Define a type variable for generic type hinting in the decorator, to maintain
    # input function and wrapped function return type
    fn_return_type = TypeVar("fn_return_type")

    # Decorator function to add the Streamlit execution context to a function
    def add_streamlit_context(
        fn: Callable[..., fn_return_type],
    ) -> Callable[..., fn_return_type]:
        """
        Decorator to ensure that the decorated function runs within the Streamlit execution context.
        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated.
        Returns:
            Callable[..., fn_return_type]: The decorated function that includes the Streamlit context setup.
        """
        ctx = (
            get_script_run_ctx()
        )  # Retrieve the current Streamlit script execution context

        def wrapper(*args, **kwargs) -> fn_return_type:
            """
            Wrapper function that adds the Streamlit context and then calls the original function.
            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.
            Returns:
                fn_return_type: The result from the original function.
            """
            add_script_run_ctx(
                ctx=ctx
            )  # Add the Streamlit context to the current execution
            return fn(*args, **kwargs)  # Call the original function with its arguments

        return wrapper

    # Create an instance of the custom StreamHandler with the provided Streamlit container
    st_cb = StreamHandler(parent_container)

    # Iterate over all methods of the StreamHandler instance
    for method_name, method_func in inspect.getmembers(
        st_cb, predicate=inspect.ismethod
    ):
        if method_name.startswith("on_"):  # Identify callback methods
            setattr(
                st_cb, method_name, add_streamlit_context(method_func)
            )  # Wrap and replace the method

    # Return the fully configured StreamHandler instance with the context-aware callback methods
    return st_cb


def streamlit_invoke(
    graph: Runnable[Any, Any], messages: List[BaseMessage], callables: List[Callable]
):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    return graph.invoke(
        {"messages": messages},
        config=RunnableConfig({"callbacks": callables, "recursion_limit": 100}),
    )
