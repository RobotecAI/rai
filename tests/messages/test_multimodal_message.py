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


from rai.messages import HumanMultimodalMessage


class TestMultimodalMessage:
    """Test the MultimodalMessage class and expected behaviors."""

    def test_human_multimodal_message_text_simple(self):
        """Test text() method with simple text content."""
        msg = HumanMultimodalMessage(content="Hello world")
        assert msg.text() == "Hello world"
        assert isinstance(msg.text(), str)

    def test_human_multimodal_message_text_with_images(self):
        """Test text() method with text and images."""
        # Use a small valid base64 image (1x1 pixel PNG)
        valid_base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        msg = HumanMultimodalMessage(
            content="Look at this image", images=[valid_base64_image]
        )
        assert msg.text() == "Look at this image"
        # Should only return text type blocks, not image content
        assert valid_base64_image not in msg.text()
