# find all python classes that inherit from pydantic.BaseTool in src/ catalog

import importlib.util
import inspect
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool


def get_all_tool_classes() -> set[BaseTool]:
    """Recursively find all classes that inherit from pydantic.BaseModel in src/rai/rai/tools"""
    tools = []
    tools_path = Path("src/rai/rai/tools")

    # Recursively find all .py files
    for py_file in tools_path.rglob("*.py"):
        if py_file.name.startswith("_"):  # Skip __init__.py and similar
            continue

        try:
            # Manual module loading since files aren't in __init__
            module_name = py_file.stem
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseTool)
                    and obj != BaseTool
                ):
                    tools.append(obj)
        except Exception as e:
            print(f"Failed to process {py_file}: {e}")

    return set(tools)


@pytest.mark.parametrize("tool_class", get_all_tool_classes())
def test_tool_input_args_compatibility(tool_class: BaseTool):
    tool = tool_class
    tool_run_annotations = tool._run.__annotations__
    if "return" in tool_run_annotations:
        tool_run_annotations.pop("return")
    if "args" in tool_run_annotations and "kwargs" in tool_run_annotations:
        print(
            f"Tool {tool_class} has *args or **kwargs, the _run method is most likely still an abstractmethod"
        )
        pytest.xfail(
            reason="Tool has *args or **kwargs, the _run method is most likely still an abstractmethod"
        )
    if "args_schema" not in tool.__annotations__:
        print(f"Tool {tool_class} has no args_schema")
        pytest.xfail(reason="Tool has no args_schema")

    if len(tool.__annotations__["args_schema"].__args__) != 1:
        raise NotImplementedError(f"Tool {tool_class} has ambiguous args_schema")

    tool_input_annotations = (
        tool.__annotations__["args_schema"].__args__[0].__annotations__
    )
    assert tool_run_annotations == tool_input_annotations
