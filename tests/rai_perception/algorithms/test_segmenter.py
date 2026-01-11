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

"""Unit tests for GDSegmenter algorithm."""

from rai_perception.algorithms.segmenter import GDSegmenter

from tests.rai_perception.algorithms.test_base_segmenter import TestGDSegmenterBase


class TestGDSegmenter(TestGDSegmenterBase):
    """Test cases for algorithms.segmenter.GDSegmenter class."""

    def get_segmenter_class(self):
        """Return the GDSegmenter class from algorithms."""
        return GDSegmenter

    def get_patch_path(self, target):
        """Return patch path for algorithms module."""
        patch_map = {
            "build_sam2": "rai_perception.algorithms.segmenter.build_sam2",
            "SAM2ImagePredictor": "rai_perception.algorithms.segmenter.SAM2ImagePredictor",
            "convert_ros_img_to_ndarray": "rai_perception.algorithms.segmenter.convert_ros_img_to_ndarray",
        }
        return patch_map[target]


class TestGDSegmenterViaAgent(TestGDSegmenterBase):
    """Test cases for GDSegmenter via GroundedSamAgent - verifies agent delegates correctly."""

    def get_segmenter_class(self):
        """Return the GDSegmenter class (extracted from agent's service)."""
        # Agents use services which use algorithms, so we test the algorithm through the agent
        return GDSegmenter

    def get_patch_path(self, target):
        """Return patch path for algorithms module (delegation target)."""
        patch_map = {
            "build_sam2": "rai_perception.algorithms.segmenter.build_sam2",
            "SAM2ImagePredictor": "rai_perception.algorithms.segmenter.SAM2ImagePredictor",
            "convert_ros_img_to_ndarray": "rai_perception.algorithms.segmenter.convert_ros_img_to_ndarray",
        }
        return patch_map[target]

    def test_gdsegmenter_initialization(self, tmp_path, mock_connector):
        """Test GDSegmenter initialization via agent - verifies agent sets up segmenter correctly."""
        from rai_perception.agents.grounded_sam import GroundedSamAgent

        from tests.rai_perception.agents.test_base_vision_agent import cleanup_agent
        from tests.rai_perception.test_helpers import (
            get_segmentation_weights_path,
            patch_segmentation_agent_dependencies,
        )
        from tests.rai_perception.test_mocks import MockGDSegmenter

        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_agent_dependencies(
            mock_connector, MockGDSegmenter, weights_path
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            # Verify agent correctly sets up service with segmenter
            assert agent._service._segmenter is not None
            assert isinstance(agent._service._segmenter, MockGDSegmenter)
            assert str(agent._service._segmenter.weights_path) == str(weights_path)

            cleanup_agent(agent)

    def test_gdsegmenter_get_segmentation(self, tmp_path, mock_connector):
        """Test GDSegmenter get_segmentation via agent - verifies agent delegates correctly."""
        from rai_perception.agents.grounded_sam import GroundedSamAgent

        from rai_interfaces.srv import RAIGroundedSam
        from tests.rai_perception.agents.test_base_vision_agent import cleanup_agent
        from tests.rai_perception.test_helpers import (
            create_segmentation_request,
            create_test_detection2d,
            get_segmentation_weights_path,
            patch_segmentation_agent_dependencies,
        )
        from tests.rai_perception.test_mocks import MockGDSegmenter

        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_agent_dependencies(
            mock_connector, MockGDSegmenter, weights_path
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            # Test via service callback (which uses segmenter)
            detection1 = create_test_detection2d(30.0, 30.0, 40.0, 40.0)
            detection2 = create_test_detection2d(75.0, 75.0, 30.0, 30.0)
            request = create_segmentation_request([detection1, detection2])
            response = RAIGroundedSam.Response()

            result = agent._service._segment_callback(request, response)

            # Verify segmenter behavior through agent
            assert len(result.masks) == 2
            assert result is response

            cleanup_agent(agent)

    def test_gdsegmenter_get_segmentation_empty_bboxes(self, tmp_path, mock_connector):
        """Test GDSegmenter get_segmentation with empty bbox list via agent."""
        from rai_perception.agents.grounded_sam import GroundedSamAgent

        from rai_interfaces.srv import RAIGroundedSam
        from tests.rai_perception.agents.test_base_vision_agent import cleanup_agent
        from tests.rai_perception.test_helpers import (
            create_segmentation_request,
            get_segmentation_weights_path,
            patch_segmentation_agent_dependencies,
        )
        from tests.rai_perception.test_mocks import EmptySegmenter

        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_agent_dependencies(
            mock_connector, EmptySegmenter, weights_path
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            request = create_segmentation_request()
            response = RAIGroundedSam.Response()

            result = agent._service._segment_callback(request, response)

            assert len(result.masks) == 0

            cleanup_agent(agent)
