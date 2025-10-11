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

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from rai_open_set_vision.agents.base_vision_agent import BaseVisionAgent


class MockBaseVisionAgent(BaseVisionAgent):
    """Mock implementation of BaseVisionAgent with required attributes."""
    WEIGHTS_URL = "https://example.com/test_weights.pth"
    WEIGHTS_FILENAME = "test_weights.pth"
    
    def run(self):
        """Dummy implementation of abstract run method for testing."""
        pass


class TestVisionWeightsDownload:
    """Test cases for BaseVisionAgent._download_weights method."""

    def test_download_weights_success(self, tmp_path):
        """Test successful weight download."""
        # make sure we have multiple levels of directories
        weights_path = tmp_path / "vision" / "weights" / "test_weights.pth"
        
        # check whether file doesn't exist before download
        assert not weights_path.exists()
        
        def mock_wget(*args, **kwargs):
            # Simulate wget creating the file
            output_path = args[0][3]  # -O argument is at index 3
            output_path.write_text("downloaded weights content")
            return MagicMock(returncode=0)
        
        with patch('subprocess.run', side_effect=mock_wget) as mock_run:
            agent = MockBaseVisionAgent(weights_root_path=str(tmp_path), ros2_name="test_agent")
            agent.weights_path = weights_path
            
            mock_run.assert_called_once_with([
                "wget",
                "https://example.com/test_weights.pth",
                "-O",
                weights_path,  
                "--progress=dot:giga",
            ])
            
            # Verify file exists after download
            assert weights_path.exists()
            
            # Clean up ROS2 node
            agent.stop()
            
            
    def test_download_weights_failure(self, tmp_path):
        """Test weight download failure raises exception."""
        weights_path = tmp_path / "vision" / "weights" / "test_weights.pth"
        
        with patch('subprocess.run') as mock_run:
            # First call succeeds (during initialization), second call fails
            mock_run.side_effect = [
                MagicMock(returncode=0),  # Initial download succeeds
                subprocess.CalledProcessError(1, "wget")  # Explicit call fails
            ]
            
            agent = MockBaseVisionAgent(weights_root_path=str(tmp_path), ros2_name="test_agent")
            agent.weights_path = weights_path
            
            with pytest.raises(Exception, match="Could not download weights"):
                agent._download_weights()
            
            # Clean up ROS2 node
            agent.stop()

