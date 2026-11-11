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

from unittest.mock import patch

from rai_interfaces.srv import EmbodimentInfo as EmbodimentInfoSrv
from rai_whoami.agents.ros2 import embodiment_info_agent
from rai_whoami.models import EmbodimentInfo


def test_handle_request_with_unset_fields(tmp_path):
    """Fields missing from info.json are None; the response gets empty values."""
    (tmp_path / "generated").mkdir()
    EmbodimentInfo(description="A robot.").to_directory(tmp_path / "generated")
    with patch.object(embodiment_info_agent, "ROS2Connector"):
        agent = embodiment_info_agent.ROS2EmbodimentInfoAgent(
            root_dir=str(tmp_path), service_name="rai_whoami_embodiment_info"
        )
    # Newer ROS generators make field validation opt-in.
    if hasattr(EmbodimentInfoSrv.Response, "_check_fields"):
        response = EmbodimentInfoSrv.Response(check_fields=True)
    else:
        response = EmbodimentInfoSrv.Response()

    result = agent.handle_request(EmbodimentInfoSrv.Request(), response)

    assert result is response
    assert response.description == "A robot."
    assert list(response.rules) == []
    assert list(response.capabilities) == []
    assert list(response.behaviors) == []
    assert list(response.images) == []
