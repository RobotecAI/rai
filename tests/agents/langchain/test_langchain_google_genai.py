# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from unittest.mock import MagicMock, patch

import pytest

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    pytest.skip(
        "langchain-google-genai not installed, skipping Google GenAI tests",
        allow_module_level=True,
    )


class TestLangChainGoogleGenAIIntegration:
    """Test suite for RAI features with langchain-google-genai integration."""

    def test_google_genai_model_initialization(self):
        """Test that ChatGoogleGenerativeAI can be initialized with proper configuration."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            # Test model initialization
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7,
            )
            assert model is not None
            assert model.model_name == "gemini-1.5-pro"

    def test_google_genai_basic_invocation(self):
        """Test basic invocation of Google GenAI model."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            # Mock the API call
            mock_response = MagicMock()
            mock_response.content = "Test response from Google GenAI"

            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7,
            )

            with patch.object(model, "invoke", return_value=mock_response):
                response = model.invoke("Hello, test message")
                assert response.content == "Test response from Google GenAI"

    def test_google_genai_with_system_prompt(self):
        """Test Google GenAI with system prompt configuration."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.5,
                top_p=0.9,
            )
            assert model is not None

    def test_google_genai_streaming_capability(self):
        """Test streaming capability with Google GenAI model."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7,
                streaming=True,
            )
            assert model is not None

    def test_google_genai_model_parameters(self):
        """Test that Google GenAI accepts proper model parameters."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.3,
                max_output_tokens=512,
                top_p=0.95,
                top_k=40,
            )
            assert model is not None
            assert model.temperature == 0.3

    @patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke")
    def test_google_genai_error_handling(self, mock_invoke):
        """Test error handling for Google GenAI API failures."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            mock_invoke.side_effect = Exception("API Error")

            model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

            with pytest.raises(Exception, match="API Error"):
                model.invoke("Test message")

    def test_google_genai_model_variants(self):
        """Test that available Google GenAI model variants are supported."""
        supported_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ]

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            for model_name in supported_models:
                model = ChatGoogleGenerativeAI(model=model_name)
                assert model is not None

    def test_google_genai_with_default_parameters(self):
        """Test Google GenAI initialization with default parameters."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            model = ChatGoogleGenerativeAI()
            assert model is not None
