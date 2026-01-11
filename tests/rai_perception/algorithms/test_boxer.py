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

from rai_perception.algorithms.boxer import Box, GDBoxer
from rclpy.time import Time
from vision_msgs.msg import Detection2D

from tests.rai_perception.algorithms.test_base_boxer import TestGDBoxerBase


class TestBox:
    """Test cases for Box class."""

    def test_box_initialization(self):
        """Test Box initialization."""
        box = Box((50.0, 50.0), 40.0, 40.0, "dinosaur", 0.9)

        assert box.center == (50.0, 50.0)
        assert box.size_x == 40.0
        assert box.size_y == 40.0
        assert box.phrase == "dinosaur"
        assert box.confidence == 0.9

    def test_box_to_detection_msg(self):
        """Test Box conversion to Detection2D message."""
        box = Box((50.0, 50.0), 40.0, 40.0, "dinosaur", 0.9)

        class_dict = {"dinosaur": 0, "dragon": 1}
        timestamp = Time()

        detection = box.to_detection_msg(class_dict, timestamp)

        assert isinstance(detection, Detection2D)
        assert detection.bbox.center.position.x == 50.0
        assert detection.bbox.center.position.y == 50.0
        assert detection.bbox.size_x == 40.0
        assert detection.bbox.size_y == 40.0
        assert detection.results[0].hypothesis.class_id == "dinosaur"
        assert detection.results[0].hypothesis.score == 0.9
        assert detection.header.stamp == timestamp.to_msg()


class TestGDBoxer(TestGDBoxerBase):
    """Test cases for algorithms.boxer.GDBoxer class."""

    def get_boxer_class(self):
        """Return the GDBoxer class from algorithms."""
        return GDBoxer

    def get_patch_path(self, target):
        """Return patch path for algorithms module."""
        patch_map = {
            "Model": "rai_perception.algorithms.boxer.Model",
            "CvBridge": "rai_perception.algorithms.boxer.CvBridge",
            "logging.getLogger": "rai_perception.algorithms.boxer.logging.getLogger",
        }
        return patch_map[target]


class TestGDBoxerViaAgent(TestGDBoxerBase):
    """Test cases for GDBoxer via GroundingDinoAgent - verifies agent delegates correctly."""

    def get_boxer_class(self):
        """Return the GDBoxer class (extracted from agent's service)."""
        # Agents use services which use algorithms, so we test the algorithm through the agent
        return GDBoxer

    def get_patch_path(self, target):
        """Return patch path for algorithms module (delegation target)."""
        patch_map = {
            "Model": "rai_perception.algorithms.boxer.Model",
            "CvBridge": "rai_perception.algorithms.boxer.CvBridge",
            "logging.getLogger": "rai_perception.algorithms.boxer.logging.getLogger",
        }
        return patch_map[target]

    def test_gdboxer_initialization(self, tmp_path, mock_connector):
        """Test GDBoxer initialization via agent - verifies agent sets up boxer correctly."""
        from rai_perception.agents.grounding_dino import GroundingDinoAgent

        from tests.rai_perception.agents.test_base_vision_agent import cleanup_agent
        from tests.rai_perception.test_helpers import (
            get_detection_weights_path,
            patch_detection_agent_dependencies,
        )
        from tests.rai_perception.test_mocks import MockGDBoxer

        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_agent_dependencies(
            mock_connector, MockGDBoxer, weights_path
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            # Verify agent correctly sets up service with boxer
            assert agent._service._boxer is not None
            assert isinstance(agent._service._boxer, MockGDBoxer)
            assert str(agent._service._boxer.weights_path) == str(weights_path)

            cleanup_agent(agent)

    def test_gdboxer_get_boxes(self, tmp_path, mock_connector):
        """Test GDBoxer get_boxes via agent - verifies agent delegates to boxer correctly."""
        from rai_perception.agents.grounding_dino import GroundingDinoAgent

        from rai_interfaces.srv import RAIGroundingDino
        from tests.rai_perception.agents.test_base_vision_agent import cleanup_agent
        from tests.rai_perception.conftest import setup_mock_clock
        from tests.rai_perception.test_helpers import (
            create_detection_request,
            get_detection_weights_path,
            patch_detection_agent_dependencies,
        )
        from tests.rai_perception.test_mocks import MockGDBoxer

        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_agent_dependencies(
            mock_connector, MockGDBoxer, weights_path
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            # Test via service callback (which uses boxer)
            request = create_detection_request("dinosaur, dragon")
            response = RAIGroundingDino.Response()

            setup_mock_clock(agent)

            result = agent._service._classify_callback(request, response)

            # Verify boxer behavior through agent
            assert len(result.detections.detections) == 2
            assert result.detections.detection_classes == ["dinosaur", "dragon"]

            cleanup_agent(agent)

    def test_gdboxer_get_boxes_empty(self, tmp_path, mock_connector):
        """Test GDBoxer get_boxes with no detections via agent."""
        from rai_perception.agents.grounding_dino import GroundingDinoAgent

        from rai_interfaces.srv import RAIGroundingDino
        from tests.rai_perception.agents.test_base_vision_agent import cleanup_agent
        from tests.rai_perception.conftest import setup_mock_clock
        from tests.rai_perception.test_helpers import (
            create_detection_request,
            get_detection_weights_path,
            patch_detection_agent_dependencies,
        )
        from tests.rai_perception.test_mocks import EmptyBoxer

        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_agent_dependencies(
            mock_connector, EmptyBoxer, weights_path
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            request = create_detection_request("dinosaur")
            response = RAIGroundingDino.Response()

            setup_mock_clock(agent)

            result = agent._service._classify_callback(request, response)

            assert len(result.detections.detections) == 0
            assert result.detections.detection_classes == ["dinosaur"]

            cleanup_agent(agent)
