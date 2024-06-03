from enum import Enum
from typing import Any, Callable, Dict


class RequirementSeverity(Enum):
    MANDATORY = 0  # Raises an exception if not met
    OPTIONAL = 1  # Issues a warning if not met


class Requirement:
    def __init__(
        self, severity: RequirementSeverity, rule: Callable[[Dict[str, Any]], bool]
    ):
        self.severity = severity
        self.rule = rule

    def __call__(self, text: Dict[str, Any]):
        return self.rule(text)


class MessageLengthRequirement(Requirement):
    def __init__(
        self, severity: RequirementSeverity, max_length: int, min_length: int = 0
    ):
        rule: Callable[[Dict[str, Any]], bool] = (
            lambda x: min_length <= len(x["content"]) <= max_length
        )
        self.max_length = max_length
        super().__init__(severity=severity, rule=rule)
