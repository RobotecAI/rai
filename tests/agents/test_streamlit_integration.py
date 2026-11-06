# Copyright (C) 2026 Robotec.AI
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
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke


def test_streams_tokens_and_matches_tool_calls_by_run_id():
    parent = MagicMock()
    tools_area, answer = parent.container.return_value, parent.empty.return_value
    camera, pose = MagicMock(name="camera"), MagicMock(name="pose")
    tools_area.status.side_effect = [camera, pose]
    cb = get_streamlit_cb(parent)

    for token in ["Hello", " robot"]:
        cb.on_llm_new_token(token, run_id=uuid4())
    assert answer.markdown.call_args.args[0].startswith("Hello robot")

    camera_id, pose_id = uuid4(), uuid4()
    cb.on_tool_start({"name": "camera"}, "{}", run_id=camera_id)
    cb.on_tool_start({"name": "pose", "description": "d"}, "{}", run_id=pose_id)
    image = base64.b64encode(b"\x89PNG fake").decode()
    cb.on_tool_end(
        ToolMessage(content="frame", artifact={"images": [image]}, tool_call_id="1"),
        run_id=camera_id,
    )
    cb.on_tool_end("x=1", run_id=pose_id)

    camera.image.assert_called_once_with(b"\x89PNG fake", width="stretch")
    pose.image.assert_not_called()
    assert camera.update.call_args.kwargs["state"] == "complete"
    assert pose.update.call_args.kwargs["state"] == "complete"


def test_streamlit_invoke_requires_list():
    with pytest.raises(TypeError):
        streamlit_invoke(MagicMock(), [], ())
