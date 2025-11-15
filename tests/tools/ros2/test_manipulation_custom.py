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

import logging
import sys
from unittest.mock import patch

import pytest


def test_rai_perception_import_error_handling(caplog):
    """Test that ImportError for rai_perception.tools is handled gracefully with a warning."""
    # Remove the module from cache to allow re-import
    module_name = "rai.tools.ros2.manipulation.custom"
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    # Mock the import to raise ImportError for rai_perception.tools
    original_import = __import__
    
    def mock_import(name, *args, **kwargs):
        if name == "rai_perception.tools":
            raise ImportError("No module named 'rai_perception'")
        return original_import(name, *args, **kwargs)
    
    with patch("builtins.__import__", side_effect=mock_import):
        with caplog.at_level(logging.WARNING):
            # Import should succeed despite the ImportError for rai_perception
            import rai.tools.ros2.manipulation.custom  # noqa: F401
            
            # Check that the warning was logged
            assert any(
                "rai-perception is not installed, GetGrabbingPointTool will not work" in record.message
                for record in caplog.records
            )

