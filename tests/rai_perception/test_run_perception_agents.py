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

from rai_perception.scripts.run_perception_agents import main


class TestRunPerceptionAgents:
    """Test cases for run_perception_agents.main function."""

    def test_main_initializes_agents(self):
        """Test that main function initializes both agents."""
        with (
            patch("rai_perception.scripts.run_perception_agents.rclpy") as mock_rclpy,
            patch(
                "rai_perception.scripts.run_perception_agents.GroundingDinoAgent"
            ) as mock_dino,
            patch(
                "rai_perception.scripts.run_perception_agents.GroundedSamAgent"
            ) as mock_sam,
            patch(
                "rai_perception.scripts.run_perception_agents.wait_for_shutdown"
            ) as mock_wait,
        ):
            mock_dino_instance = MagicMock()
            mock_sam_instance = MagicMock()
            mock_dino.return_value = mock_dino_instance
            mock_sam.return_value = mock_sam_instance

            main()

            mock_rclpy.init.assert_called_once()
            mock_dino.assert_called_once()
            mock_sam.assert_called_once()
            mock_dino_instance.run.assert_called_once()
            mock_sam_instance.run.assert_called_once()
            mock_wait.assert_called_once_with([mock_dino_instance, mock_sam_instance])
            mock_rclpy.shutdown.assert_called_once()

    def test_main_calls_agents_in_order(self):
        """Test that agents are called in the correct order."""
        call_order = []

        def track_dino_init(*args, **kwargs):
            call_order.append("dino_init")
            mock_instance = MagicMock()
            mock_instance.run.side_effect = lambda: call_order.append("dino_run")
            return mock_instance

        def track_sam_init(*args, **kwargs):
            call_order.append("sam_init")
            mock_instance = MagicMock()
            mock_instance.run.side_effect = lambda: call_order.append("sam_run")
            return mock_instance

        with (
            patch("rai_perception.scripts.run_perception_agents.rclpy"),
            patch(
                "rai_perception.scripts.run_perception_agents.GroundingDinoAgent",
                side_effect=track_dino_init,
            ),
            patch(
                "rai_perception.scripts.run_perception_agents.GroundedSamAgent",
                side_effect=track_sam_init,
            ),
            patch("rai_perception.scripts.run_perception_agents.wait_for_shutdown"),
        ):
            main()

            # Verify order: init dino, init sam, run dino, run sam
            assert call_order == ["dino_init", "sam_init", "dino_run", "sam_run"]

    def test_main_handles_shutdown(self):
        """Test that main properly shuts down ROS2."""
        with (
            patch("rai_perception.scripts.run_perception_agents.rclpy") as mock_rclpy,
            patch(
                "rai_perception.scripts.run_perception_agents.GroundingDinoAgent"
            ) as mock_dino,
            patch(
                "rai_perception.scripts.run_perception_agents.GroundedSamAgent"
            ) as mock_sam,
            patch("rai_perception.scripts.run_perception_agents.wait_for_shutdown"),
        ):
            mock_dino.return_value = MagicMock()
            mock_sam.return_value = MagicMock()

            main()

            mock_rclpy.init.assert_called_once()
            mock_rclpy.shutdown.assert_called_once()
