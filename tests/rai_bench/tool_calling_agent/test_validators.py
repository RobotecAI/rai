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


from typing import Any, Dict, Sequence

import pytest

from rai_bench.tool_calling_agent.interfaces import SubTaskValidationError, Validator
from rai_bench.tool_calling_agent.validators import (
    NotOrderedCallsValidator,
    OrderedCallsValidator,
)


# Mock tool call
class ToolCall:
    def __init__(self, name: str = "test_tool", arguments=None):
        self.name = name
        self.arguments = arguments or {}


class DummySubTask:
    def __init__(
        self,
        name: str = "test_subtask",
        specific_tool: str | None = None,
        outcomes: Sequence[bool] | None = None,
    ):
        super().__init__()
        self.name = name
        self.specific_tool = specific_tool
        # list of bools if subtask passed or not for given validate iteration
        self._outcomes = iter(outcomes) if outcomes is not None else None

    def validate(self, tool_call: Dict[str, Any]) -> bool:
        if self.specific_tool and tool_call.name != self.specific_tool:
            raise SubTaskValidationError(
                f"Expected tool {self.specific_tool}, got {tool_call.name}"
            )

        if self._outcomes is not None:
            try:
                should_pass = next(self._outcomes)
            except StopIteration:
                # if run out, default to True
                should_pass = True
        else:
            should_pass = True

        if not should_pass:
            raise SubTaskValidationError(f"error in {self.name}")

        return True

    @property
    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "specific_tool": self.specific_tool}


def assert_dumped(
    validator: Validator,
    *,
    expected_type: str,
    expected_passed: bool,
    expected_extra_calls: int,
    expected_subtasks_passed: list[bool],
    expected_errors_counts: list[int] | None = None,
):
    """Verify if results dumped after every scenario are valid"""
    result = validator.dump_results()
    assert result.type == expected_type
    assert result.passed is expected_passed
    assert result.extra_tool_calls_used == expected_extra_calls

    actual_passed = [st.passed for st in result.subtasks]
    assert actual_passed == expected_subtasks_passed

    if expected_errors_counts is not None:
        actual_errors = [len(st.errors) for st in result.subtasks]
        assert actual_errors == expected_errors_counts

    return result


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
        assert validator.subtasks_errors[0] == []  # No specific subtask errors
        assert validator.passed is False
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=False,
            expected_extra_calls=0,
            expected_subtasks_passed=[False],
            expected_errors_counts=[0],
        )

    def test_validate_successful_one_task(self):
        subtasks = [DummySubTask("task1")]
        validator = OrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall()]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert remaining == []
        assert validator.subtasks_errors[0] == []
        assert validator.subtasks_passed[0] is True
        assert validator.passed is True
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=True,
            expected_extra_calls=0,
            expected_subtasks_passed=[True],
            expected_errors_counts=[0],
        )

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
        assert all(errors == [] for errors in validator.subtasks_errors)
        assert all(validator.subtasks_passed)
        assert validator.passed is True
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=True,
            expected_extra_calls=0,
            expected_subtasks_passed=[True, True, True],
            expected_errors_counts=[0, 0, 0],
        )

    def test_validate_successful_excess_tool_calls(self):
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
        assert all(errors == [] for errors in validator.subtasks_errors)
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=True,
            expected_extra_calls=0,
            expected_subtasks_passed=[True, True, True],
            expected_errors_counts=[0, 0, 0],
        )

    def test_validate_successful_with_excess_tool_calls_2(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)
        tool_calls = [
            ToolCall(name="tool1"),
            ToolCall(name="extra_tool"),
            ToolCall(name="tool2"),
            ToolCall(name="another_extra"),
        ]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert len(remaining) == 1
        assert remaining[0].name == "another_extra"
        assert len(validator.subtasks_errors[0]) == 0
        assert len(validator.subtasks_errors[1]) == 1
        assert all(validator.subtasks_passed)
        assert validator.passed is True
        assert validator.extra_calls_used == 1

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=True,
            expected_extra_calls=1,
            expected_subtasks_passed=[True, True],
            expected_errors_counts=[0, 1],
        )

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
        # first task should have 3 errors as wrong tools are given
        assert len(validator.subtasks_errors[0]) == 3
        assert len(validator.subtasks_errors[1]) == 0
        assert len(validator.subtasks_errors[2]) == 0
        assert "Expected tool tool1, got extra_tool" in validator.subtasks_errors[0][0]
        assert validator.subtasks_passed[0] is True
        assert validator.subtasks_passed[1] is True
        assert validator.subtasks_passed[2] is True
        assert validator.passed is True
        assert validator.extra_calls_used == 3

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=True,
            expected_extra_calls=3,
            expected_subtasks_passed=[True, True, True],
            expected_errors_counts=[3, 0, 0],
        )

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
        assert len(validator.subtasks_errors[0]) == 1
        assert len(validator.subtasks_errors[1]) == 0
        assert "Expected tool tool1, got tool2" in validator.subtasks_errors[0][0]
        assert (
            validator.subtasks_passed[0] is True
        )  # the 1st will pass on the 2nd tool call
        assert validator.subtasks_passed[1] is False
        assert validator.passed is False
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=False,
            expected_extra_calls=0,
            expected_subtasks_passed=[True, False],
            expected_errors_counts=[1, 0],
        )

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
        assert validator.subtasks_passed[0] is True
        assert validator.subtasks_passed[1] is True
        assert validator.subtasks_passed[2] is False
        assert len(validator.subtasks_errors[0]) == 0
        assert len(validator.subtasks_errors[1]) == 0
        assert len(validator.subtasks_errors[2]) == 0
        assert validator.passed is False
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=False,
            expected_extra_calls=0,
            expected_subtasks_passed=[True, True, False],
            expected_errors_counts=[0, 0, 0],
        )

    def test_validate_subtask_failed(self):
        subtasks = [
            DummySubTask("task1"),
            DummySubTask("task2", outcomes=[False]),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)

        tool_calls = [ToolCall(name="tool1"), ToolCall(name="tool2")]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert validator.subtasks_passed[0] is True
        assert validator.subtasks_passed[1] is False
        assert len(validator.subtasks_errors[0]) == 0
        assert len(validator.subtasks_errors[1]) == 1
        assert "error in task2" in validator.subtasks_errors[1][0]
        assert validator.passed is False
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=False,
            expected_extra_calls=0,
            expected_subtasks_passed=[True, False],
            expected_errors_counts=[0, 1],
        )

    def test_validate_extra_calls_when_subtask_fails(self):
        subtasks = [
            DummySubTask("task1"),
            DummySubTask("task2", outcomes=5 * [False]),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)

        tool_calls = [
            ToolCall(name="tool1"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
        ]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert validator.subtasks_passed[0] is True
        assert validator.subtasks_passed[1] is False
        assert len(validator.subtasks_errors[1]) == 5
        assert "error in task2" in validator.subtasks_errors[1][0]
        assert validator.passed is False
        assert validator.extra_calls_used == 4

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=False,
            expected_extra_calls=4,
            expected_subtasks_passed=[True, False],
            expected_errors_counts=[0, 5],
        )

    def test_validate_extra_calls_when_subtask_eventually_passes(self):
        subtasks = [
            DummySubTask("task1"),
            DummySubTask("task2", outcomes=5 * [False]),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)

        tool_calls = [
            ToolCall(name="tool1"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
        ]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert remaining == []
        assert validator.subtasks_passed[0] is True
        assert validator.subtasks_passed[1] is True
        assert len(validator.subtasks_errors[1]) == 5
        assert "error in task2" in validator.subtasks_errors[1][0]
        assert validator.passed is True
        assert validator.extra_calls_used == 5

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=True,
            expected_extra_calls=5,
            expected_subtasks_passed=[True, True],
            expected_errors_counts=[0, 5],
        )

    def test_validate_reset(self):
        subtasks = [
            DummySubTask("task1"),
            DummySubTask("task2", outcomes=10 * [False]),
        ]
        validator = OrderedCallsValidator(subtasks=subtasks)

        tool_calls = [
            ToolCall(name="tool1"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
            ToolCall(name="tool2"),
        ]
        # additional call
        validator.validate(tool_calls=tool_calls)
        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert validator.subtasks_passed[0] is True
        assert validator.subtasks_passed[1] is False
        assert len(validator.subtasks_errors[1]) == 5
        assert "error in task2" in validator.subtasks_errors[1][0]
        assert validator.passed is False
        assert validator.extra_calls_used == 4

        assert_dumped(
            validator,
            expected_type="ordered",
            expected_passed=False,
            expected_extra_calls=4,
            expected_subtasks_passed=[True, False],
            expected_errors_counts=[0, 5],
        )


class TestNotOrderedCallsValidator:
    def test_init_with_empty_subtasks(self):
        with pytest.raises(ValueError, match="Validator must have at least 1 subtask"):
            NotOrderedCallsValidator(subtasks=[])

    def test_validate_empty_tool_calls(self):
        subtasks = [DummySubTask("task1")]
        validator = NotOrderedCallsValidator(subtasks=subtasks)

        success, remaining = validator.validate(tool_calls=[])

        assert not success
        assert remaining == []
        assert len(validator.subtasks_errors[0]) == 0
        assert validator.subtasks_passed[0] is False
        assert validator.passed is False
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="not ordered",
            expected_passed=False,
            expected_extra_calls=0,
            expected_subtasks_passed=[False],
            expected_errors_counts=[0],
        )

    def test_validate_successful_single_task(self):
        subtasks = [DummySubTask("task1", specific_tool="tool1")]
        validator = NotOrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall(name="tool1")]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert success
        assert remaining == []
        assert len(validator.subtasks_errors[0]) == 0
        assert validator.subtasks_passed[0] is True
        assert validator.passed is True
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="not ordered",
            expected_passed=True,
            expected_extra_calls=0,
            expected_subtasks_passed=[True],
            expected_errors_counts=[0],
        )

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
        assert len(validator.subtasks_errors[0]) == 0
        assert len(validator.subtasks_errors[1]) == 0
        assert all(validator.subtasks_passed)
        assert validator.passed is True
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="not ordered",
            expected_passed=True,
            expected_extra_calls=0,
            expected_subtasks_passed=[True, True],
            expected_errors_counts=[0, 0],
        )

    def test_validate_with_excess_tool_calls(self):
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
        assert len(remaining) == 1
        assert remaining[0].name == "another_extra"
        assert len(validator.subtasks_errors[0]) == 0
        assert len(validator.subtasks_errors[1]) == 1
        assert all(validator.subtasks_passed)
        assert validator.passed is True
        assert validator.extra_calls_used == 1

        assert_dumped(
            validator,
            expected_type="not ordered",
            expected_passed=True,
            expected_extra_calls=1,
            expected_subtasks_passed=[True, True],
            expected_errors_counts=[0, 1],
        )

    def test_validate_missing_subtask(self):
        subtasks = [
            DummySubTask("task1", specific_tool="tool1"),
            DummySubTask("task2", specific_tool="tool2"),
        ]
        validator = NotOrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall(name="tool2")]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert len(validator.subtasks_errors[0]) == 0
        assert len(validator.subtasks_errors[1]) == 0
        assert validator.subtasks_passed[0] is False
        assert validator.subtasks_passed[1] is True
        assert validator.passed is False
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="not ordered",
            expected_passed=False,
            expected_extra_calls=0,
            expected_subtasks_passed=[False, True],
            expected_errors_counts=[0, 0],
        )

    def test_validate_all_subtasks_fail(self):
        subtasks = [
            DummySubTask("task1", outcomes=[False, False]),
            DummySubTask("task2", outcomes=[False, False]),
        ]
        validator = NotOrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall(), ToolCall()]

        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert all(not passed for passed in validator.subtasks_passed)
        assert len(validator.subtasks_errors[0]) == 2
        assert len(validator.subtasks_errors[1]) == 2
        assert validator.passed is False
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="not ordered",
            expected_passed=False,
            expected_extra_calls=0,
            expected_subtasks_passed=[False, False],
            expected_errors_counts=[2, 2],
        )

    def test_validate_reset(self):
        subtasks = [
            DummySubTask("task1", outcomes=4 * [False]),
            DummySubTask("task2", outcomes=4 * [False]),
        ]
        validator = NotOrderedCallsValidator(subtasks=subtasks)
        tool_calls = [ToolCall(), ToolCall()]

        # additional call
        validator.validate(tool_calls=tool_calls)
        success, remaining = validator.validate(tool_calls=tool_calls)

        assert not success
        assert remaining == []
        assert all(not passed for passed in validator.subtasks_passed)
        assert len(validator.subtasks_errors[0]) == 2
        assert len(validator.subtasks_errors[1]) == 2
        assert validator.passed is False
        assert validator.extra_calls_used == 0

        assert_dumped(
            validator,
            expected_type="not ordered",
            expected_passed=False,
            expected_extra_calls=0,
            expected_subtasks_passed=[False, False],
            expected_errors_counts=[2, 2],
        )
