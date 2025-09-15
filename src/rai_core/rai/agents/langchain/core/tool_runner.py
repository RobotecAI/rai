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


import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.prebuilt.tool_node import msg_content_output
from langgraph.utils.runnable import RunnableCallable
from pydantic import ValidationError

from rai.messages import MultimodalArtifact, ToolMultimodalMessage, store_artifacts


class ToolRunner(RunnableCallable):
    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(self._func, name=name, tags=tags, trace=False)
        self.logger = logger or logging.getLogger(__name__)
        self.tools_by_name: Dict[str, BaseTool] = {}
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            self.tools_by_name[tool_.name] = tool_

    def get_messages(self, input: dict[str, Any]) -> List:
        """Get fields from from input that will be processed."""
        return input.get("messages", [])

    def update_input_with_outputs(
        self, input: dict[str, Any], outputs: List[Any]
    ) -> None:
        """Update input with tool outputs."""
        input["messages"].extend(outputs)

    def _func(self, input: dict[str, Any], config: RunnableConfig) -> Any:
        config["max_concurrency"] = (
            1  # TODO(maciejmajek): use better mechanism for task queueing
        )
        messages = self.get_messages(input)
        if not messages:
            raise ValueError("No message found in input")

        message = messages[-1]
        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        def run_one(call: ToolCall):
            self.logger.info(f"Running tool: {call['name']}, args: {call['args']}")
            artifact = None

            try:
                ts = time.perf_counter()
                output = self.tools_by_name[call["name"]].invoke(call, config)  # type: ignore
                te = time.perf_counter() - ts
                self.logger.info(
                    f"Tool {call['name']} completed in {te:.2f} seconds. Tool output: {str(output.content)[:100]}{'...' if len(str(output.content)) > 100 else ''}"
                )
                self.logger.debug(
                    f"Tool {call['name']} output: \n\n{str(output.content)}"
                )
            except ValidationError as e:
                errors = e.errors()
                for error in errors:
                    error.pop(
                        "url"
                    )  # get rid of the  https://errors.pydantic.dev/... url

                error_message = f"""
                                    Validation error in tool {call["name"]}:
                                    {e.title}
                                    Number of errors: {e.error_count()}
                                    Errors:
                                    {json.dumps(errors, indent=2)}
                                """
                self.logger.info(error_message)
                output = ToolMessage(
                    content=error_message,
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )
            except Exception as e:
                self.logger.info(f'Error in "{call["name"]}", error: {e}')
                output = ToolMessage(
                    content=f"Failed to run tool. Error: {e}",
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )

            if output.artifact is not None:
                artifact = output.artifact
                if not isinstance(artifact, dict):
                    raise ValueError(
                        "Artifact must be a dictionary with optional keys: 'images', 'audios'"
                    )

                artifact = cast(MultimodalArtifact, artifact)
                store_artifacts(output.tool_call_id, [artifact])

            if artifact is not None and (
                len(artifact.get("images", [])) > 0
                or len(artifact.get("audios", [])) > 0
            ):  # multimodal case, we currently support images and audios artifacts
                return ToolMultimodalMessage(
                    content=msg_content_output(output.content),
                    name=call["name"],
                    tool_call_id=call["id"],
                    images=artifact.get("images", []),
                    audios=artifact.get("audios", []),
                )

            return output

        with get_executor_for_config(config) as executor:
            raw_outputs = [*executor.map(run_one, message.tool_calls)]
            outputs: List[Any] = []
            for raw_output in raw_outputs:
                if isinstance(raw_output, ToolMultimodalMessage):
                    outputs.extend(
                        raw_output.postprocess()
                    )  # openai please allow tool messages with images!
                else:
                    outputs.append(raw_output)

            # because we can't answer an aiMessage with an alternating sequence of tool and human messages
            # we sort the messages by type so that the tool messages are sent first
            # for more information see implementation of ToolMultimodalMessage.postprocess
            outputs.sort(key=lambda x: x.__class__.__name__, reverse=True)

            self.update_input_with_outputs(input, outputs)
            return input


class SubAgentToolRunner(ToolRunner):
    """ToolRunner that works with 'step_messages' key used by subagents"""

    def get_messages(self, input: dict[str, Any]) -> List:
        return input.get("step_messages", [])

    def update_input_with_outputs(
        self, input: dict[str, Any], outputs: List[Any]
    ) -> None:
        input["messages"].extend(outputs)
        input["step_messages"].extend(outputs)
