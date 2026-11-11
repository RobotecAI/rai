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

import subprocess
import sys


def test_import_rai_is_lazy():
    code = (
        "import sys, rai\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "('langchain', 'langchain_core', 'langgraph', 'rclpy', 'langchain_openai', "
        "'pydantic_settings')]\n"
        "assert not heavy, heavy\n"
        "assert callable(rai.get_llm)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
