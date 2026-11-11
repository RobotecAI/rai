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
# See the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import pytest
from rai_perception.scripts import run_perception_services
from rai_perception.scripts.run_perception_services import main

from tests.rai_perception.conftest import make_mock_connector

MODULE = "rai_perception.scripts.run_perception_services"


@pytest.fixture
def patched():
    connectors = [make_mock_connector(), make_mock_connector()]
    with (
        patch(f"{MODULE}.ROS2Runtime") as runtime,
        patch(f"{MODULE}.ROS2Connector", side_effect=connectors) as connector,
        patch(f"{MODULE}.DetectionService") as detection,
        patch(f"{MODULE}.SegmentationService") as segmentation,
        patch(f"{MODULE}.run") as run,
    ):
        yield MagicMock(
            runtime=runtime,
            connector=connector,
            connectors=connectors,
            detection=detection,
            segmentation=segmentation,
            run=run,
        )


def test_main_starts_both_services_on_their_own_nodes(patched):
    main()

    names = [c.args[0] for c in patched.runtime.call_args_list]
    assert names == ["detection_service", "segmentation_service"]
    detection_connector, segmentation_connector = patched.connectors
    patched.detection.assert_called_once_with(ros2_connector=detection_connector)
    patched.segmentation.assert_called_once_with(ros2_connector=segmentation_connector)
    patched.detection.return_value.run.assert_called_once()
    patched.segmentation.return_value.run.assert_called_once()
    assert [a.service for a in patched.run.call_args.args] == [
        patched.detection.return_value,
        patched.segmentation.return_value,
    ]


@pytest.mark.parametrize("env, expected", [(None, True), ("false", False)])
def test_legacy_service_names_follow_the_environment(
    patched, monkeypatch, env, expected
):
    if env is not None:
        monkeypatch.setenv("ENABLE_LEGACY_SERVICE_NAMES", env)
    main()
    for connector in patched.connectors:
        assert connector.params["enable_legacy_service_names"] is expected


def test_stop_on_shutdown_stops_the_service():
    service = MagicMock()
    run_perception_services._StopOnShutdown(service).teardown()
    service.stop.assert_called_once()
