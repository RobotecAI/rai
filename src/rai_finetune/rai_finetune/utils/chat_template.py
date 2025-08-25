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

"""Chat template handler for consistent message formatting"""

import logging
import os
from typing import Any, Dict, List, Optional

from jinja2 import Template

logger = logging.getLogger(__name__)


class ChatTemplate:
    """Simple chat template handler for consistent message formatting"""

    def __init__(self, template_name: str = "chatml"):
        self.template_name = template_name
        self.template = None

    def load_chatml_template(self) -> str:
        """Load chatml.jinja template for the given model"""

        try:
            # Look for chatml.jinja in common locations
            possible_paths = [
                "chatml.jinja",
                "templates/chatml.jinja",
                "utils/templates/chatml.jinja",
                os.path.join(os.path.dirname(__file__), "templates", "chatml.jinja"),
            ]

            # Try to load chatml.jinja template
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        template = self.load_jinja_template(path)
                        if template:
                            return template
                    except Exception as e:
                        logger.warning(f"Failed to load template from {path}: {e}")
                        continue
                    break
            return None
        except Exception as e:
            logger.error(f"Error loading chatml.jinja template: {e}")
            return None

    def load_jinja_template(self, template_path: str) -> str:
        """Load template from .jinja file"""
        try:
            with open(template_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {template_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading template file {template_path}: {e}")

    def format_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        """Format messages using the selected template"""
        if not self.template:
            raise ValueError("No template loaded")

        # Create context for Jinja2 template
        context = {
            "messages": messages,
            "system_prompt": system_prompt,
            "tools": tools or [],
            "add_generation_prompt": add_generation_prompt,
        }

        # Render template using Jinja2
        template = Template(self.template)
        return template.render(**context)
