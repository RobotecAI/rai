# Copyright (C) 2025 Julia Jia
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

from typing import Any, Optional


class PerceptionError(Exception):
    """Base exception for all perception-related errors."""

    pass


class PerceptionValidationError(PerceptionError):
    """Exception raised for input validation failures beyond Pydantic validation."""

    def __init__(
        self,
        validation_rule: str,
        input_value: Any,
        valid_range: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.validation_rule = validation_rule
        self.input_value = input_value
        self.valid_range = valid_range
        self.suggestion = suggestion
        message = f"Validation failed: {validation_rule}"
        if valid_range:
            message += f" (valid range: {valid_range})"
        if suggestion:
            message += f". {suggestion}"
        super().__init__(message)
