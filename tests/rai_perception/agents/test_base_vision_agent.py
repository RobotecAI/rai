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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import rclpy
from rai_perception.agents.base_vision_agent import BaseVisionAgent

from tests.rai_perception.conftest import create_valid_weights_file


class MockBaseVisionAgent(BaseVisionAgent):
    """Mock implementation of BaseVisionAgent with required attributes."""

    WEIGHTS_URL = "https://example.com/test_weights.pth"
    WEIGHTS_FILENAME = "test_weights.pth"

    def run(self):
        """Dummy implementation of abstract run method for testing."""
        pass


def get_weights_path(tmp_path: Path) -> Path:
    """Helper to get the standard weights path for testing.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to the weights file
    """
    return tmp_path / "vision" / "weights" / "test_weights.pth"


def create_agent_with_weights(
    tmp_path: Path, weights_path: Path
) -> MockBaseVisionAgent:
    """Helper to create an agent with weights path set.

    Args:
        tmp_path: Temporary directory path
        weights_path: Path to weights file

    Returns:
        Configured MockBaseVisionAgent instance
    """
    agent = MockBaseVisionAgent(weights_root_path=str(tmp_path), ros2_name="test_agent")
    agent.weights_path = weights_path
    return agent


def cleanup_agent(agent: MockBaseVisionAgent) -> None:
    """Helper to clean up agent and ROS2 context.

    Args:
        agent: Agent instance to clean up
    """
    agent.stop()
    if rclpy.ok():
        rclpy.shutdown()


def extract_output_path_from_wget_args(args) -> Path:
    """Helper to extract output path from wget subprocess args.

    Args:
        args: Arguments passed to subprocess.run (args[0] is the command list)

    Returns:
        Path object for the output file
    """
    output_path_str = args[0][3]  # -O argument is at index 3
    return Path(output_path_str)


class TestVisionWeightsDownload:
    """Test cases for BaseVisionAgent._download_weights method."""

    def setup_method(self):
        """Initialize ROS2 before tests to prevent auto-initialization warning."""
        if not rclpy.ok():
            rclpy.init()

    def teardown_method(self):
        """Clean up ROS2 context after each test."""
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def test_download_weights_success(self, tmp_path):
        """Test successful weight download."""
        weights_path = get_weights_path(tmp_path)

        # check whether file doesn't exist before download
        assert not weights_path.exists()

        def mock_wget(*args, **kwargs):
            # Simulate wget creating the file
            output_path = extract_output_path_from_wget_args(args)
            create_valid_weights_file(output_path)
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=mock_wget) as mock_run:
            agent = create_agent_with_weights(tmp_path, weights_path)

            mock_run.assert_called_once_with(
                [
                    "wget",
                    "https://example.com/test_weights.pth",
                    "-O",
                    str(weights_path),
                    "--progress=dot:giga",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # Verify file exists after download
            assert weights_path.exists()

            cleanup_agent(agent)


class TestBaseVisionAgentInit:
    """Test cases for BaseVisionAgent.__init__ method."""

    def setup_method(self):
        """Initialize ROS2 before tests to prevent auto-initialization warning."""
        if not rclpy.ok():
            rclpy.init()

    def teardown_method(self):
        """Clean up ROS2 context after each test."""
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def test_init_without_weights_filename(self):
        """Test that ValueError is raised when WEIGHTS_FILENAME is not set."""

        class InvalidAgent(BaseVisionAgent):
            WEIGHTS_FILENAME = ""

            def run(self):
                """Dummy implementation of abstract run method."""
                pass

        with pytest.raises(ValueError, match="WEIGHTS_FILENAME is not set"):
            InvalidAgent()

    def test_init_with_path_string(self, tmp_path):
        """Test initialization with string path."""
        weights_path = tmp_path / "vision" / "weights" / "test_weights.pth"
        create_valid_weights_file(weights_path)

        agent = MockBaseVisionAgent(weights_root_path=str(tmp_path), ros2_name="test")
        assert agent.weights_root_path == tmp_path
        assert agent.weights_path == weights_path

        agent.stop()
        if rclpy.ok():
            rclpy.shutdown()

    def test_init_with_path_object(self, tmp_path):
        """Test initialization with Path object."""
        weights_path = tmp_path / "vision" / "weights" / "test_weights.pth"
        create_valid_weights_file(weights_path)

        agent = MockBaseVisionAgent(weights_root_path=tmp_path, ros2_name="test")
        assert agent.weights_root_path == tmp_path
        assert agent.weights_path == weights_path

        agent.stop()
        if rclpy.ok():
            rclpy.shutdown()

    def test_init_with_existing_file(self, tmp_path):
        """Test initialization when weights file already exists."""
        weights_path = tmp_path / "vision" / "weights" / "test_weights.pth"
        create_valid_weights_file(weights_path)

        with patch("subprocess.run") as mock_run:
            agent = MockBaseVisionAgent(
                weights_root_path=str(tmp_path), ros2_name="test_agent"
            )
            # Should not call download since file exists
            mock_run.assert_not_called()

        agent.stop()
        if rclpy.ok():
            rclpy.shutdown()


class TestLoadModelWithErrorHandling:
    """Test cases for BaseVisionAgent._load_model_with_error_handling method."""

    def setup_method(self):
        """Initialize ROS2 before tests to prevent auto-initialization warning."""
        if not rclpy.ok():
            rclpy.init()

    def teardown_method(self):
        """Clean up ROS2 context after each test."""
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def test_load_model_success(self, tmp_path):
        """Test successful model loading."""
        weights_path = tmp_path / "vision" / "weights" / "test_weights.pth"
        create_valid_weights_file(weights_path)

        class MockModel:
            def __init__(self, weights_path):
                self.weights_path = weights_path

        agent = MockBaseVisionAgent(weights_root_path=str(tmp_path), ros2_name="test")
        agent.weights_path = weights_path

        model = agent._load_model_with_error_handling(MockModel)
        assert model.weights_path == weights_path

        agent.stop()
        if rclpy.ok():
            rclpy.shutdown()

    def test_load_model_corrupted_weights(self, tmp_path):
        """Test model loading with corrupted weights triggers redownload."""
        weights_path = tmp_path / "vision" / "weights" / "test_weights.pth"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(b"corrupted")

        call_count = 0

        class MockModel:
            def __init__(self, weights_path):
                nonlocal call_count
                call_count += 1
                self.weights_path = weights_path
                if call_count == 1:
                    raise RuntimeError("PytorchStreamReader failed")

        def mock_wget(*args, **kwargs):
            output_path_str = args[0][3]
            output_path = Path(output_path_str)
            output_path.write_bytes(b"0" * (2 * 1024 * 1024))
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=mock_wget):
            agent = MockBaseVisionAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )
            agent.weights_path = weights_path

            model = agent._load_model_with_error_handling(MockModel)
            assert model.weights_path == weights_path
            assert call_count == 2  # Called twice: once fails, once succeeds

        agent.stop()
        if rclpy.ok():
            rclpy.shutdown()

    def test_load_model_other_runtime_error(self, tmp_path):
        """Test that non-corruption RuntimeErrors are re-raised."""
        weights_path = tmp_path / "vision" / "weights" / "test_weights.pth"
        create_valid_weights_file(weights_path)

        class MockModel:
            def __init__(self, weights_path):
                raise RuntimeError("Some other error")

        agent = MockBaseVisionAgent(weights_root_path=str(tmp_path), ros2_name="test")
        agent.weights_path = weights_path

        with pytest.raises(RuntimeError, match="Some other error"):
            agent._load_model_with_error_handling(MockModel)

        agent.stop()
        if rclpy.ok():
            rclpy.shutdown()


class TestBaseVisionAgentMethods:
    """Test cases for other BaseVisionAgent methods."""

    def setup_method(self):
        """Initialize ROS2 before tests to prevent auto-initialization warning."""
        if not rclpy.ok():
            rclpy.init()

    def teardown_method(self):
        """Clean up ROS2 context after each test."""
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def test_stop(self, tmp_path):
        """Test stop method shuts down ROS2 connector."""
        weights_path = tmp_path / "vision" / "weights" / "test_weights.pth"
        create_valid_weights_file(weights_path)

        agent = MockBaseVisionAgent(weights_root_path=str(tmp_path), ros2_name="test")
        agent.weights_path = weights_path

        with patch.object(agent.ros2_connector, "shutdown") as mock_shutdown:
            agent.stop()
            mock_shutdown.assert_called_once()

        if rclpy.ok():
            rclpy.shutdown()
