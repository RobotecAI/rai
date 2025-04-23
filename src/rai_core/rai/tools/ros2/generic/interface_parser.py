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


import typing

import rosidl_adapter
import rosidl_adapter.parser
import rosidl_runtime_py.convert
from rosidl_adapter.parser import (
    ACTION_REQUEST_RESPONSE_SEPARATOR,
    SERVICE_REQUEST_RESPONSE_SEPARATOR,
    Constant,
    MessageSpecification,
    parse_message_string,
)


class InterfaceTextLine:
    """A convenience class for a single text line in an interface file."""

    def __init__(
        self,
        pkg_name: str,
        msg_name: str,
        line_text: str,
    ):
        if line_text in (
            SERVICE_REQUEST_RESPONSE_SEPARATOR,
            ACTION_REQUEST_RESPONSE_SEPARATOR,
        ):
            msg_spec = None
        else:
            msg_spec = parse_message_string(
                pkg_name=pkg_name,
                msg_name=msg_name,
                message_string=line_text,
            )
            if len(msg_spec.fields) > 1:  # type: ignore
                raise ValueError("'line_text' must be only one line")
        self._msg_spec: MessageSpecification | None = msg_spec
        self._raw_line_text = line_text

    def __str__(self) -> str:
        return self._raw_line_text

    def is_comment(self) -> bool:
        return bool(self._msg_spec) and self._msg_spec.annotations["comment"]  # type: ignore

    def is_trailing_comment(self) -> bool:
        return self._is_field_trailing_comment() or self._is_constant_trailing_comment()

    def _is_field_trailing_comment(self) -> bool:
        return self._field and self._field.annotations["comment"]  # type: ignore

    def _is_constant_trailing_comment(self) -> bool:
        return self._constant and self._constant.annotations["comment"]  # type: ignore

    @property
    def nested_type(self) -> typing.Optional[str]:
        if self._field and self._is_nested():
            interface_type: str = str(self._field.type)
            if self._field.type.is_array:  # type: ignore
                interface_type = interface_type[: interface_type.find("[")]
            return interface_type.replace("/", "/msg/")

    @property
    def trailing_comment(self) -> typing.Optional[str]:
        if self._is_field_trailing_comment():
            return self._field.annotations["comment"][0]  # type: ignore
        elif self._is_constant_trailing_comment():
            return self._constant.annotations["comment"][0]  # type: ignore
        else:
            return None

    @property
    def _field(self) -> rosidl_adapter.parser.Field | None:
        if self._msg_spec and self._msg_spec.fields:  # type: ignore
            return self._msg_spec.fields[0]  # type: ignore

    @property
    def _constant(self) -> Constant | None:
        if self._msg_spec and self._msg_spec.constants:  # type: ignore
            return self._msg_spec.constants[0]  # type: ignore

    def _is_nested(self) -> bool:
        if self._msg_spec and self._msg_spec.fields and self._field:  # type: ignore
            return "/" in str(self._field.type)
        else:
            return False


def _get_interface_lines(
    interface_identifier: str,
) -> typing.Iterable[InterfaceTextLine]:
    parts: typing.List[str] = interface_identifier.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid name '{interface_identifier}'. Expected three parts separated by '/'"
        )
    pkg_name, _, msg_name = parts

    file_path = rosidl_runtime_py.get_interface_path(interface_identifier)
    with open(file_path) as file_handler:
        for line in file_handler:
            yield InterfaceTextLine(
                pkg_name=pkg_name,
                msg_name=msg_name,
                line_text=line.rstrip(),
            )


def _render_interface_line(
    line: InterfaceTextLine, is_show_comments: bool, indent_level: int
) -> str:
    text = str(line)
    if not is_show_comments:
        if not text or line.is_comment():
            return ""
        elif line.is_trailing_comment():
            if line.trailing_comment:
                comment_start_idx = text.find(line.trailing_comment)
                text = text[: comment_start_idx - 1].strip()
    if text:
        indent_string = indent_level * "\t"
        return f"{indent_string}{text}"
    return ""


def render_interface_string(
    interface_identifier: str,
    is_show_comments: bool = True,
    is_show_nested_comments: bool = False,
    indent_level: int = 0,
) -> str:
    lines: typing.List[str] = []
    for line in _get_interface_lines(interface_identifier):
        rendered = _render_interface_line(
            line, is_show_comments=is_show_comments, indent_level=indent_level
        )
        if rendered.strip():
            lines.append(rendered)
        if line.nested_type:
            nested_rendered = render_interface_string(
                line.nested_type,
                is_show_comments=is_show_nested_comments,
                is_show_nested_comments=is_show_nested_comments,
                indent_level=indent_level + 1,
            )
            if nested_rendered.strip():
                lines.append(nested_rendered)

    return "\n".join(lines)
