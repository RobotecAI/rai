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

import random
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def get_tool_schema_info(tool: BaseTool) -> dict:
    """Extract tool information for system prompt."""
    schema_info = {"name": tool.name, "description": tool.description, "args": {}}

    if hasattr(tool, "args_schema") and tool.args_schema:
        schema = tool.args_schema.model_json_schema()
        properties = schema.get("properties", {})

        for arg_name, arg_info in properties.items():
            schema_info["args"][arg_name] = {
                "type": arg_info.get("type", "unknown"),
                "description": arg_info.get("description", "No description available"),
            }

    return schema_info


def generate_coding_agent_system_prompt(tools: List[BaseTool]) -> str:
    """Generate system prompt for coding agent with tool information."""
    base_prompt = """
You have access to a Python interpreter tool that you must use for all code execution and generation. You should always use this tool when the user asks to do something.

You can use the following methods that are available as global variables in your Python environment:

"""

    tool_descriptions = []
    for tool in tools:
        schema_info = get_tool_schema_info(tool)
        tool_desc = f"**{schema_info['name']}**:\n"
        tool_desc += f"- Description: {schema_info['description']}\n"
        tool_desc += (
            f"- Usage: Use `{schema_info['name']}._run()` method statement to execute\n"
        )

        if schema_info["args"]:
            tool_desc += "- Arguments:\n"
            for arg_name, arg_info in schema_info["args"].items():
                tool_desc += (
                    f"  - {arg_name} ({arg_info['type']}): {arg_info['description']}\n"
                )

            # Generate proper example with actual argument names
            example_args = []
            for arg_name, arg_info in schema_info["args"].items():
                if arg_info["type"] == "string":
                    example_args.append(f'{arg_name}="example_value"')
                elif arg_info["type"] in ["number", "integer"]:
                    example_args.append(f"{arg_name}={random.randint(0, 9)}")
                elif arg_info["type"] == "boolean":
                    example_args.append(f"{arg_name}=True")
                else:
                    example_args.append(f"{arg_name}=value")

            example_call = (
                f"result = {schema_info['name']}._run({', '.join(example_args)})\n"
                "print(result)"
            )
        else:
            tool_desc += "- Arguments: None\n"
            example_call = f"result = {schema_info['name']}._run()\nprint(result)"

        tool_desc += f"- Example:\n```python\n{example_call}\n```\n"
        tool_descriptions.append(tool_desc)

    tools_section = "\n".join(tool_descriptions)

    instructions = """
IMPORTANT INSTRUCTIONS:
1. Always use the python_interpreter tool for code execution
2. The objects listed above are available as global variables in your Python environment
3. The variables defined in the code become available as global variables in your Python environment in future code execution
4. ABSOLUTELY NEVER stop your work until the task is COMPLETED. Don't wait for user approval.
5. Remember to import required libraries before using them in the code (e.g. `import numpy as np`, `import math`)

Remember: All code must be executed through the python_interpreter tool, and you can use the available tools within that environment by calling their _run() methods.
"""

    return base_prompt + tools_section + instructions


class CodeExecutionInterrupted(Exception):
    pass


class CodeExecutor:
    def __init__(self):
        self.current_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self.exception: Optional[Exception] = None

    def trace_function(self, frame: Any, event: str, arg: Any) -> Any:
        if self.should_stop.is_set():
            raise CodeExecutionInterrupted("Code execution was interrupted")
        return self.trace_function

    def execute(self, code: str, globals_dict: dict) -> None:
        try:
            with redirect_stderr(StringIO()):
                sys.settrace(self.trace_function)
            exec(code, globals_dict)
        except Exception as e:
            self.exception = e
        finally:
            with redirect_stderr(StringIO()):
                sys.settrace(None)

    def stop(self):
        if self.current_thread and self.current_thread.is_alive():
            self.should_stop.set()
            self.current_thread.join()

    def is_running(self):
        return self.current_thread and self.current_thread.is_alive()


class PythonInterpreterInput(BaseModel):
    code: str = Field(description="The Python code to execute")


class PythonInterpreterWithTools(BaseTool):
    """Python interpreter tool with exposed tools as global variables."""

    name: str = "python_interpreter"
    description: str = (
        "Python interpreter that executes Python code. Use this tool for all code execution. "
        "The variables defined in the code become available as global variables in your Python environment. "
        "Additional tools are available as global variables that can be called using their _run() method."
    )

    tools: List[BaseTool] = Field(
        default_factory=list, description="Tools to expose in Python environment"
    )
    args_schema: Type[PythonInterpreterInput] = PythonInterpreterInput
    exec_global_scope: dict = Field(default_factory=dict, exclude=True)
    executor: Optional[CodeExecutor] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, tools: Optional[List[BaseTool]] = None, **kwargs):
        super().__init__(tools=tools or [], **kwargs)

    def model_post_init(self, __context: Any) -> None:
        """Called after model initialization."""
        self._setup_execution_environment()

    def _setup_execution_environment(self):
        """Setup the execution environment with tools as global variables."""
        self.exec_global_scope = {
            "__builtins__": __builtins__,
            "get_defined_vars": self.get_defined_vars,
            "has_defined_vars": self.has_defined_vars,
            "clear_defined_vars": self.clear_defined_vars,
        }

        # Add tools as global variables
        for tool in self.tools:
            self.exec_global_scope[tool.name] = tool

        self.executor = CodeExecutor()

    def get_defined_vars(self):
        """Lists variables defined by the user in the execution scope."""
        builtin_modules = set(sys.modules.keys())
        initial_keys = {
            "__builtins__",
            "get_defined_vars",
            "has_defined_vars",
            "clear_defined_vars",
        }
        # Add tool names to initial keys
        initial_keys.update(tool.name for tool in self.tools)

        result = []
        for key, value in self.exec_global_scope.items():
            if key not in initial_keys and key not in builtin_modules:
                var_type = type(value).__name__
                try:
                    str_value = str(value)
                    if len(str_value) > 20:
                        content = str_value[:20] + "..."
                    else:
                        content = str_value
                except Exception:
                    content = "<error converting to string>"

                result.append(f"Name: {key}\nType: {var_type}\nContent: {content}\n")

        return "\n".join(result)

    def has_defined_vars(self):
        """Checks if any user-defined variables exist in the execution scope."""
        builtin_modules = set(sys.modules.keys())
        initial_keys = {
            "__builtins__",
            "get_defined_vars",
            "has_defined_vars",
            "clear_defined_vars",
        }
        initial_keys.update(tool.name for tool in self.tools)

        for key in self.exec_global_scope:
            if key not in initial_keys and key not in builtin_modules:
                return True
        return False

    def clear_defined_vars(self):
        """Removes all user-defined variables from the execution scope."""
        builtin_modules = set(sys.modules.keys())
        initial_keys = {
            "__builtins__",
            "get_defined_vars",
            "has_defined_vars",
            "clear_defined_vars",
        }
        initial_keys.update(tool.name for tool in self.tools)

        keys_to_remove = [
            key
            for key in list(self.exec_global_scope.keys())
            if key not in initial_keys and key not in builtin_modules
        ]

        for key in keys_to_remove:
            del self.exec_global_scope[key]

        return len(keys_to_remove)

    def _run(self, code: str) -> str:
        """Execute Python code with tools available as global variables."""
        exception = ""
        self.executor.exception = None

        try:
            f = StringIO()
            with redirect_stdout(f):
                self.executor.should_stop.clear()
                # thread = threading.Thread(
                #     target=self.executor.execute, args=(code, self.exec_global_scope)
                # )
                # self.executor.current_thread = thread
                # thread.start()
                # thread.join()
                self.executor.execute(code, self.exec_global_scope)

                if self.executor.exception:
                    raise self.executor.exception
        except Exception as e:
            output = f.getvalue()
            try:
                tb = e.__traceback__
                while tb and tb.tb_next:
                    tb = tb.tb_next

                error_type = e.__class__.__name__
                error_msg = str(e)
                file_name = tb.tb_frame.f_code.co_filename if tb else "unknown"
                actual_line_number = tb.tb_lineno if tb else 0

                executed_code_lines = code.split("\n")
                first_executed_line = (
                    executed_code_lines[0] if executed_code_lines else ""
                )
                executed_line_number = 1

                if "<string>" in file_name:
                    error_location = f"at line {actual_line_number} of executed code"
                    if 1 <= actual_line_number <= len(executed_code_lines):
                        first_executed_line = executed_code_lines[
                            actual_line_number - 1
                        ]
                        executed_line_number = actual_line_number
                    exception = f"<message>{error_type}: {error_msg}</message>"
                else:
                    error_location = f"in {file_name}, line {actual_line_number}"
                    orig_tb = e.__traceback__
                    while orig_tb:
                        frame = orig_tb.tb_frame
                        if frame.f_code.co_filename == "<string>":
                            executed_line_number = frame.f_lineno
                            if 1 <= executed_line_number <= len(executed_code_lines):
                                first_executed_line = executed_code_lines[
                                    executed_line_number - 1
                                ]
                            break
                        orig_tb = orig_tb.tb_next
                    exception = (
                        f"<message>{error_type} {error_location}: {error_msg}</message>"
                    )

                exception += f"\n<executed_code_line_number>{executed_line_number}</executed_code_line_number>"
                exception += (
                    f"\n<executed_code_line>{first_executed_line}</executed_code_line>"
                )
            except Exception as e:
                exception = f"<message>Error: {str(e)}</message>"
                exception += (
                    "\n<executed_code_line_number>1</executed_code_line_number>"
                )
                first_line = code.split("\n")[0] if code and "\n" in code else code
                exception += f"\n<executed_code_line>{first_line}</executed_code_line>"

            return f"<python_interpreter_result>\n<std_out>\n{output}</std_out>\n<exception>\n{exception}</exception>\n</python_interpreter_result>"
        else:
            output = f.getvalue()
            return f"<python_interpreter_result>\n<std_out>\n{output}</std_out>\n<exception>\n</exception>\n</python_interpreter_result>"


def create_coding_agent_with_tools(
    tools: List[BaseTool], llm, embodiment_prompt: str = ""
) -> tuple:
    """
    Create a coding agent with Python interpreter and exposed tools.

    Returns:
        tuple: (agent, system_prompt) where agent has python_interpreter as the only tool
               and system_prompt includes tool documentation
    """
    from rai.agents.langchain.core import create_conversational_agent

    # Create Python interpreter tool with exposed tools
    python_tool = PythonInterpreterWithTools(tools=tools)

    # Generate system prompt
    coding_prompt = generate_coding_agent_system_prompt(tools)
    full_prompt = (
        embodiment_prompt + "\n\n" + coding_prompt
        if embodiment_prompt
        else coding_prompt
    )
    # full_prompt += "!!!IMPORTANT!!! /no_think"
    full_prompt += """
<examples>
    <example>
        # Prompt: move all objects 5 cm forward
        # Code:
        move_to_point._run(x=0.0, y=0.5, z=0.59) # move the arm to not obstruct objects with the arm
        positions = get_object_positions._run("object")
        target_position = (positions[0].x + 0.05, positions[0].y, positions[0].z)
        for pos in positions:
            move_object_from_to._run(x=pos.x, y=pos.y, z=pos.z, x1=target_position[0], y1=target_position[1], z1=target_position[2])
    </example>
</examples>
"""

    # Create agent with only the Python interpreter tool
    agent = create_conversational_agent(
        llm=llm,
        tools=[python_tool],
        system_prompt=full_prompt,
    )

    return agent, full_prompt
