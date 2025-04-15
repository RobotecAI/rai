import pytest

from rai_bench.tool_calling_agent.interfaces import SubTaskValidationError
from rai_bench.tool_calling_agent.validators import (
    NotOrderedCallsValidator,
    OrderedCallsValidator,
)


# Mock tool call
class ToolCall:
    def __init__(self, name="test_tool", arguments=None):
        self.name = name
        self.arguments = arguments or {}


class DummySubTask:
    def __init__(self, name="test_subtask", specific_tool=None, should_pass=True):
        self.name = name
        self.specific_tool = specific_tool
        self.should_pass = should_pass

    def validate(self, tool_call):
        if not self.should_pass:
            raise SubTaskValidationError(f"Validation failed for {self.name}")

        if self.specific_tool and tool_call.name != self.specific_tool:
            raise SubTaskValidationError(
                f"Expected tool {self.specific_tool}, got {tool_call.name}"
            )

        return True

    def dump(self):
        return {"name": self.name, "specific_tool": self.specific_tool}


class TestOrderedCallsValidator:
    def test_init_with_empty_subtasks(self):
        with pytest.raises(ValueError, match="Validator must have at least 1 subtask"):
            OrderedCallsValidator(subtasks=[])

    def test_validate_empty_tool_calls(self):
        subtasks = [DummySubTask("task1")]
        validator = OrderedCallsValidator(subtasks=subtasks)

        success, remaining = validator.validate(tool_calls=[])

        assert not success
        assert remaining == []
        assert validator.get_all_validation_errors() == [
            "Not a single tool call to validate"
        ]

    def test_validate_successful_one_task(self):
        subtasks = [DummySubTask("task1")]
        validator = OrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall()]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert remaining == []
        assert validator.get_all_validation_errors() == []

    def test_validate_successful_multiple_subtasks(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
            DummySubTask("task3", specific_tool="tool3"),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)
        tool_calls = [
            ToolCall(name="tool1"),
            ToolCall(name="tool2"),
            ToolCall(name="tool3"),
        ]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert remaining == []
        assert validator.get_all_validation_errors() == []

    def test_validate_successful_excess_toolcalls(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
            DummySubTask("task3", specific_tool="tool3"),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)
        tool_calls = [
            ToolCall(name="tool1"),
            ToolCall(name="tool2"),
            ToolCall(name="tool3"),
            ToolCall(name="extra_tool"),
        ]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert len(remaining) == 1
        assert remaining[0].name == "extra_tool"
        assert validator.get_all_validation_errors() == []

    def test_validate_successful_after_couple_toolcalls(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
            DummySubTask("task3", specific_tool="tool3"),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)
        tool_calls = [
            ToolCall(name="extra_tool"),
            ToolCall(name="extra_tool2"),
            ToolCall(name="extra_tool3"),
            ToolCall(name="tool1"),
            ToolCall(name="tool2"),
            ToolCall(name="tool3"),
        ]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert remaining == []
        assert validator.get_all_validation_errors() == []

    def test_validate_failure_wrong_order(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall(name="tool2"), ToolCall(name="tool1")]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert len(validator.get_all_validation_errors()) == 1

    def test_validate_missing_subtasks(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
            DummySubTask("task3", specific_tool="tool3"),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall(name="tool1"), ToolCall(name="tool2")]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert validator.get_all_validation_errors() == [
            "Not all subtasks were completed in given tool calls."
        ]

    def test_validate_subtask_failed(self):
        subtasks = [
            DummySubTask("task1", should_pass=True),
            DummySubTask("task2", should_pass=False),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)

        tool_calls = [ToolCall(name="tool1"), ToolCall(name="tool2")]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert validator.get_all_validation_errors() == [
            "Not all subtasks were completed in given tool calls."
        ]


class TestNotOrderedCallsValidator:
    def test_init(self):
        subtasks = [DummySubTask("task1"), DummySubTask("task2")]
        validator = NotOrderedCallsValidator(subtasks=subtasks)

        assert validator.subtasks == subtasks

    def test_validate_empty_tool_calls(self):
        subtasks = [DummySubTask("task1")]
        validator = NotOrderedCallsValidator(subtasks=subtasks)

        success, remaining = validator.validate(tool_calls=[])

        assert not success
        assert remaining == []
        assert validator.get_all_validation_errors() == [
            "Not a single tool call to validate"
        ]

    def test_validate_successful_single_task(self):
        subtasks = [DummySubTask("task1", specific_tool="tool1")]
        validator = NotOrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall(name="tool1")]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert remaining == []
        assert validator.get_all_validation_errors() == []

    def test_validate_successful_out_of_order(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
        ]
        validator = NotOrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall(name="tool2"), ToolCall(name="tool1")]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert remaining == []
        assert validator.errors_queue.empty()

    def test_validate_with_extra_tool_calls(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
        ]
        validator = NotOrderedCallsValidator(subtasks=subtasks)
        tool_calls = [
            ToolCall(name="tool1"),
            ToolCall(name="extra_tool"),
            ToolCall(name="tool2"),
            ToolCall(name="another_extra"),
        ]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert remaining == []
        assert validator.errors_queue.empty()

    def test_validate_missing_subtask(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
        ]
        validator = NotOrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall(name="tool1")]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert (
            "Not all subtasks were completed"
            in validator.get_all_validation_errors()[0]
        )

    def test_validate_all_subtasks_fail(self):
        subtasks = [
            DummySubTask("task1", should_pass=False),
            DummySubTask("task2", should_pass=False),
        ]
        validator = NotOrderedCallsValidator(subtasks=subtasks)

        tool_calls = [ToolCall(), ToolCall()]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
