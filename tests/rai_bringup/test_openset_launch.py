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

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Import the launch file directly by path
launch_file_path = (
    Path(__file__).parent.parent.parent
    / "src"
    / "rai_bringup"
    / "launch"
    / "openset.launch.py"
)
spec = importlib.util.spec_from_file_location("openset_launch", launch_file_path)
openset_launch = importlib.util.module_from_spec(spec)
sys.modules["openset_launch"] = openset_launch
spec.loader.exec_module(openset_launch)
generate_launch_description = openset_launch.generate_launch_description


class TestOpensetLaunch:
    """Test cases for openset.launch.py"""

    @pytest.mark.timeout(10)
    def test_launch_process_starts_without_immediate_crash(self):
        """Test that the launch process starts without immediately crashing (catches basic errors such as incorrect path to the Python script)."""
        process = subprocess.Popen(
            ["ros2", "launch", "rai_bringup", "openset.launch.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait a few seconds and verify process hasn't crashed
            time.sleep(3)
            poll_result = process.poll()

            # If poll() returns None, process is still running (good)
            # If it returns a code, process has exited (likely an error)
            if poll_result is not None:
                stdout, stderr = process.communicate(timeout=1)
                error_msg = (
                    f"Launch process exited prematurely with code {poll_result}\n"
                )
                if stderr:
                    error_msg += f"STDERR:\n{stderr}"
                if stdout:
                    error_msg += f"\nSTDOUT:\n{stdout}"
                assert False, error_msg

        finally:
            # Terminate the launch process
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
