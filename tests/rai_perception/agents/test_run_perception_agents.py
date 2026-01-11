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

import warnings
from unittest.mock import patch

from rai_perception.scripts.run_perception_agents import main


class TestRunPerceptionAgents:
    """Test cases for run_perception_agents.main function."""

    def test_main_issues_deprecation_warning(self):
        """Test that main function issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch(
                "rai_perception.scripts.run_perception_services.main"
            ) as mock_services_main:
                main()
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "deprecated" in str(w[0].message).lower()
                mock_services_main.assert_called_once()

    def test_main_delegates_to_services(self):
        """Test that main function delegates to run_perception_services.main."""
        with patch(
            "rai_perception.scripts.run_perception_services.main"
        ) as mock_services_main:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main()
                mock_services_main.assert_called_once()
