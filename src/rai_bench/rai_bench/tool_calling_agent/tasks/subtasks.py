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

from typing import Any, Dict, Optional

from langchain_core.messages import ToolCall

from rai_bench.tool_calling_agent.interfaces import SubTask


class CheckArgsToolCallSubTask(SubTask):
    def __init__(
        self,
        expected_tool_name: str,
        expected_args: Dict[str, Any],
        expected_optional_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.expected_tool_name = expected_tool_name
        self.expected_args = expected_args
        self.expected_optional_args = expected_optional_args

    def validate(
        self,
        tool_call: ToolCall,
    ) -> bool:
        return self._check_tool_call(
            tool_call=tool_call,
            expected_name=self.expected_tool_name,
            expected_args=self.expected_args,
            expected_optional_args=self.expected_optional_args,
        )

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "expected_tool_name": self.expected_tool_name,
            "expected_args": self.expected_args,
            "expected_optional_args": self.expected_optional_args,
        }


class CheckTopicFieldsToolCallSubTask(SubTask):
    def __init__(
        self,
        expected_tool_name: str,
        expected_topic: str,
        expected_message_type: str,
        expected_fields: Dict[str, Any],
    ):
        super().__init__()
        self.expected_tool_name = expected_tool_name
        self.expected_fields = expected_fields
        self.expected_topic = expected_topic
        self.expected_message_type = expected_message_type

    def validate(
        self,
        tool_call: ToolCall,
    ) -> bool:
        for field, value in self.expected_fields.items():
            if not self._check_topic_tool_call_field(
                tool_call=tool_call,
                expected_name=self.expected_tool_name,
                expected_topic=self.expected_topic,
                expected_message_type=self.expected_message_type,
                field_path=field,
                expected_value=value,
            ):
                return False
        return True

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "expected_tool_name": self.expected_tool_name,
            "expected_topic": self.expected_topic,
            "expected_message_type": self.expected_message_type,
            "expected_fields": self.expected_fields,
        }


class CheckServiceFieldsToolCallSubTask(SubTask):
    def __init__(
        self,
        expected_tool_name: str,
        expected_service: str,
        expected_service_type: str,
        expected_fields: Dict[str, Any],
    ):
        super().__init__()
        self.expected_tool_name = expected_tool_name
        self.expected_fields = expected_fields
        self.expected_service = expected_service
        self.expected_service_type = expected_service_type

    def validate(
        self,
        tool_call: ToolCall,
    ) -> bool:
        for field, value in self.expected_fields.items():
            if not self._check_service_tool_call_field(
                tool_call=tool_call,
                expected_name=self.expected_tool_name,
                expected_service=self.expected_service,
                expected_service_type=self.expected_service_type,
                field_path=field,
                expected_value=value,
            ):
                return False
        return True

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "expected_tool_name": self.expected_tool_name,
            "expected_service": self.expected_service,
            "expected_service_type": self.expected_service_type,
            "expected_fields": self.expected_fields,
        }


class CheckActionFieldsToolCallSubTask(SubTask):
    def __init__(
        self,
        expected_tool_name: str,
        expected_action: str,
        expected_action_type: str,
        expected_fields: Dict[str, Any],
    ):
        super().__init__()
        self.expected_tool_name = expected_tool_name
        self.expected_fields = expected_fields
        self.expected_action = expected_action
        self.expected_action_type = expected_action_type

    def validate(
        self,
        tool_call: ToolCall,
    ) -> bool:
        for field, value in self.expected_fields.items():
            if not self._check_action_tool_call_field(
                tool_call=tool_call,
                expected_name=self.expected_tool_name,
                expected_action=self.expected_action,
                expected_action_type=self.expected_action_type,
                field_path=field,
                expected_value=value,
            ):
                return False
        return True

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "expected_tool_name": self.expected_tool_name,
            "expected_action": self.expected_action,
            "expected_action_type": self.expected_action_type,
            "expected_fields": self.expected_fields,
        }
