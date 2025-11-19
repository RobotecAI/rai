# Copyright (C) 2024 Robotec.AI
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

import os
from functools import partial
from typing import Dict, List

import numpy as np
import requests
import streamlit as st
import tomli
import tomli_w
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import logging
import importlib.util


def get_sound_devices(
    reinitialize: bool = False, output: bool = False
) -> List[Dict[str, str | int]]:
    import sounddevice as sd

    if reinitialize:
        sd._terminate()
        sd._initialize()
    devices: List[Dict[str, str | int]] = sd.query_devices()
    if output:
        recording_devices = [
            device for device in devices if device.get("max_output_channels", 0) > 0
        ]
    else:
        recording_devices = [
            device for device in devices if device.get("max_input_channels", 0) > 0
        ]
    return recording_devices


def welcome():
    st.title("Welcome to RAI Configurator! 👋")
    st.markdown(
        """
    This wizard will help you set up your RAI environment step by step:
    1. Configure your AI models and vendor
    2. Set up model tracing and monitoring
    3. Configure speech recognition (ASR) (if installed)
    4. Set up text-to-speech (TTS) (if installed)
    5. Enable additional features
    6. Review and save your configuration

    Let's get started!
    """
    )

    st.button("Begin Configuration →", on_click=next_step)


def model_selection():
    st.title("Model Configuration")
    st.info(
        """
    This step configures which AI models will be used by RAI's agents. Different models have different capabilities and costs:
    - Simple models are faster and cheaper, used for basic tasks
    - Complex models are more capable but slower, used for complex reasoning
    - Embedding models convert text into numerical representations for memory and search
    """
    )

    def on_vendor_change():
        st.session_state.vendor_index = ["openai", "aws", "ollama"].index(
            st.session_state.vendor
        )

    vendor = st.selectbox(
        "Which AI vendor would you like to use?",
        ["openai", "aws", "ollama"],
        placeholder="Select vendor",
        key="vendor",
        index=st.session_state.get("vendor_index", 0),
        on_change=on_vendor_change,
    )

    if vendor:
        if vendor == "openai":
            st.write(
                f"Check out available {vendor} models [here](https://platform.openai.com/docs/models)"
            )
            simple_model = st.text_input(
                "Model for simple tasks",
                value=st.session_state["config"]["openai"]["simple_model"],
                key="simple_model",
            )
            complex_model = st.text_input(
                "Model for complex tasks",
                value=st.session_state["config"]["openai"]["complex_model"],
                key="complex_model",
            )
            embeddings_model = st.text_input(
                "Embeddings model",
                value=st.session_state["config"]["openai"]["embeddings_model"],
                key="embeddings_model",
            )

            def on_openai_compatible_api_change():
                st.session_state.use_openai_compatible_api = (
                    st.session_state.openai_compatible_api_checkbox
                )

            if "use_openai_compatible_api" not in st.session_state:
                st.session_state.use_openai_compatible_api = False

            use_openai_compatible_api = st.checkbox(
                "Use OpenAI compatible API",
                value=st.session_state.use_openai_compatible_api,
                key="openai_compatible_api_checkbox",
                on_change=on_openai_compatible_api_change,
            )
            st.session_state.use_openai_compatible_api = use_openai_compatible_api

            if use_openai_compatible_api:
                st.info(
                    "Used for OpenAI compatible endpoints, e.g. Ollama, vLLM... Make sure to specify `OPENAI_API_KEY` environment variable based on vendor's specification."
                )
                openai_api_base_url = st.text_input(
                    "OpenAI API base URL",
                    value=st.session_state["config"]["openai"]["base_url"],
                    key="openai_api_base_url",
                )
            else:
                openai_api_base_url = st.session_state["config"]["openai"]["base_url"]
            st.session_state.config["openai"] = {
                "simple_model": simple_model,
                "complex_model": complex_model,
                "embeddings_model": embeddings_model,
                "base_url": openai_api_base_url,
            }

        elif vendor == "aws":
            st.write(
                f"Check out available {vendor} models [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)"
            )
            col1, col2 = st.columns(2)

            with col1:
                simple_model = st.text_input(
                    "Model for simple tasks",
                    value=st.session_state["config"]["aws"]["simple_model"],
                )
                complex_model = st.text_input(
                    "Model for complex tasks",
                    value=st.session_state["config"]["aws"]["complex_model"],
                )
                embeddings_model = st.text_input(
                    "Embeddings model",
                    value=st.session_state["config"]["aws"]["embeddings_model"],
                )
                st.session_state.config["aws"]["simple_model"] = simple_model
                st.session_state.config["aws"]["complex_model"] = complex_model
                st.session_state.config["aws"]["embeddings_model"] = embeddings_model

            with col2:
                aws_bedrock_region = st.text_input(
                    "AWS Bedrock region",
                    value=st.session_state["config"]["aws"]["region_name"],
                )
                st.session_state.config["aws"]["region_name"] = aws_bedrock_region

        elif vendor == "ollama":
            st.write(
                f"Check out available {vendor} models [here](https://ollama.com/models)"
            )
            base_url = st.text_input(
                "Ollama base URL",
                value=st.session_state["config"]["ollama"]["base_url"],
            )
            simple_model = st.text_input(
                "Model for simple tasks",
                value=st.session_state["config"]["ollama"]["simple_model"],
            )
            complex_model = st.text_input(
                "Model for complex tasks",
                value=st.session_state["config"]["ollama"]["complex_model"],
            )
            embeddings_model = st.text_input(
                "Embeddings model",
                value=st.session_state["config"]["ollama"]["embeddings_model"],
            )
            st.session_state.config["ollama"] = {
                "base_url": base_url,
                "simple_model": simple_model,
                "complex_model": complex_model,
                "embeddings_model": embeddings_model,
            }

    st.subheader("Multivendor configuration (Advanced)")
    st.write(
        "If you have access to multiple vendors, you can configure the models to use different vendors."
    )

    def on_advanced_config_change():
        st.session_state.use_advanced_config = st.session_state.advanced_config_checkbox

    if "use_advanced_config" not in st.session_state:
        st.session_state.use_advanced_config = False

    use_advanced_config = st.checkbox(
        "Use advanced configuration",
        value=st.session_state.use_advanced_config,
        key="advanced_config_checkbox",
        on_change=on_advanced_config_change,
    )
    st.session_state.use_advanced_config = use_advanced_config

    advanced_config = st.container()
    if use_advanced_config:
        with advanced_config:

            def on_model_vendor_change(model_type: str):
                st.session_state.config["vendor"][f"{model_type}_model"] = (
                    st.session_state[f"{model_type}_vendor_select"]
                )

            simple_model_vendor = st.selectbox(
                "Simple model vendor",
                options=["openai", "aws", "ollama"],
                index=["openai", "aws", "ollama"].index(
                    st.session_state.config["vendor"]["simple_model"]
                ),
                key="simple_vendor_select",
                on_change=lambda: on_model_vendor_change("simple"),
            )

            complex_model_vendor = st.selectbox(
                "Complex model vendor",
                options=["openai", "aws", "ollama"],
                index=["openai", "aws", "ollama"].index(
                    st.session_state.config["vendor"]["complex_model"]
                ),
                key="complex_vendor_select",
                on_change=lambda: on_model_vendor_change("complex"),
            )

            embeddings_model_vendor = st.selectbox(
                "Embeddings model vendor",
                options=["openai", "aws", "ollama"],
                index=["openai", "aws", "ollama"].index(
                    st.session_state.config["vendor"]["embeddings_model"]
                ),
                key="embeddings_vendor_select",
                on_change=lambda: on_model_vendor_change("embeddings"),
            )
    else:
        st.session_state.config["vendor"] = {
            "simple_model": vendor,
            "complex_model": vendor,
            "embeddings_model": vendor,
        }

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        st.button("Next →", on_click=next_step)


def tracing():
    st.title("Tracing Configuration")
    st.info(
        """
    Tracing helps you monitor, debug, and analyze your AI assistant's conversations and performance.
    Both frameworks offer similar capabilities, but have different pricing models:
    - Langfuse is open-source and free to use (recommended)
    - LangSmith is a paid service from LangChain

    You can enable either or both services based on your needs.
    """
    )

    def on_langfuse_change():
        st.session_state.config["tracing"]["langfuse"]["use_langfuse"] = (
            st.session_state.langfuse_checkbox
        )

    def on_langfuse_host_change():
        st.session_state.config["tracing"]["langfuse"]["host"] = (
            st.session_state.langfuse_host_input
        )

    def on_langsmith_change():
        st.session_state.config["tracing"]["langsmith"]["use_langsmith"] = (
            st.session_state.langsmith_checkbox
        )

    # Ensure tracing config exists
    if "tracing" not in st.session_state.config:
        st.session_state.config["tracing"] = {}

    # Langfuse configuration
    st.subheader("Langfuse Configuration")
    langfuse_enabled = st.checkbox(
        "Enable Langfuse",
        value=st.session_state.config["tracing"]["langfuse"]["use_langfuse"],
        key="langfuse_checkbox",
        on_change=on_langfuse_change,
    )

    if langfuse_enabled:
        st.info(
            """
        Please ensure you have the following environment variables set:
        - `LANGFUSE_SECRET_KEY="sk-lf-..."`
        - `LANGFUSE_PUBLIC_KEY="pk-lf-..."`

        Find setup instructions [here](https://langfuse.com/docs/deployment/local)
        """
        )

        langfuse_host = st.text_input(
            "Langfuse Host",
            value=st.session_state.config["tracing"]["langfuse"]["host"],
            key="langfuse_host_input",
            on_change=on_langfuse_host_change,
        )

    # Langsmith configuration
    st.subheader("LangSmith Configuration")
    langsmith_enabled = st.checkbox(
        "Enable LangSmith",
        value=st.session_state.config["tracing"]["langsmith"]["use_langsmith"],
        key="langsmith_checkbox",
        on_change=on_langsmith_change,
    )

    if langsmith_enabled:
        st.info(
            """
        Please ensure you have the following environment variable set:
        - `LANGCHAIN_API_KEY`

        Find setup instructions [here](https://docs.smith.langchain.com/)
        """
        )

        # Store in config
        if "tracing" not in st.session_state.config:
            st.session_state.config["tracing"] = {}
        st.session_state.config["tracing"]["langsmith"] = {
            "use_langsmith": langsmith_enabled
        }

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        st.button("Next →", on_click=next_step)


def asr():
    import sounddevice as sd
    from rai_s2s.asr import TRANSCRIBE_MODELS

    def on_recording_device_change():
        st.session_state.config["asr"]["recording_device_name"] = (
            st.session_state.recording_device_select
        )

    def on_asr_vendor_change():
        st.session_state.config["asr"]["transcription_model"] = (
            st.session_state.asr_vendor_select
        )

    def on_model_name_change():
        st.session_state.config["asr"]["transcription_model_name"] = (
            st.session_state.model_name_input
        )

    def on_language_change():
        st.session_state.config["asr"]["language"] = st.session_state.language_input

    def on_silence_grace_change():
        st.session_state.config["asr"]["silence_grace_period"] = (
            st.session_state.silence_grace_input
        )

    def on_vad_threshold_change():
        st.session_state.config["asr"]["vad_threshold"] = (
            st.session_state.vad_threshold_input
        )

    def on_wake_word_change():
        st.session_state.config["asr"]["use_wake_word"] = (
            st.session_state.wake_word_checkbox
        )

    def on_wake_word_model_change():
        st.session_state.config["asr"]["wake_word_model"] = (
            st.session_state.wake_word_model_input
        )

    def on_wake_word_model_name_change():
        st.session_state.config["asr"]["wake_word_model_name"] = (
            st.session_state.wake_word_model_name_input
        )

    def on_wake_word_threshold_change():
        st.session_state.config["asr"]["wake_word_threshold"] = (
            st.session_state.wake_word_threshold_input
        )

    # Ensure asr config exists
    if "asr" not in st.session_state.config:
        st.session_state.config["asr"] = {}

    st.title("Speech Recognition Configuration")
    st.info(
        """
    Speech recognition (ASR - Automatic Speech Recognition) converts spoken words into text. This allows your assistant to understand voice input:
    - Local ASR uses Whisper and runs on your computer (recommended with GPU)
    - Device selection determines which microphone is used for voice input
    """
    )

    recording_devices = get_sound_devices()
    currently_selected_device_name = st.session_state.config.get("asr", {}).get(
        "recording_device_name", ""
    )
    try:
        device_index = [device["name"] for device in recording_devices].index(
            currently_selected_device_name
        )
    except ValueError:
        device_index = None

    recording_device_name = st.selectbox(
        "Default recording device",
        [device["name"] for device in recording_devices],
        placeholder="Select device",
        index=device_index,
        key="recording_device_select",
        on_change=on_recording_device_change,
    )

    refresh_devices = st.button("Refresh devices")
    if refresh_devices:
        recording_devices = get_sound_devices(reinitialize=True)

    if recording_device_name == "default":
        st.info(
            """
        If you're experiencing audio issues and device_name is set to 'default', try specifying the exact device name instead, as this often resolves the problem.
        """
        )

    # Get the current vendor from config and convert to display name
    current_vendor = st.session_state.config.get("asr", {}).get(
        "transciption_model", TRANSCRIBE_MODELS[0]
    )

    asr_vendor = st.selectbox(
        "Choose your ASR vendor",
        TRANSCRIBE_MODELS,
        placeholder="Select vendor",
        index=TRANSCRIBE_MODELS.index(current_vendor),
        key="asr_vendor_select",
        on_change=on_asr_vendor_change,
    )

    if asr_vendor == "OpenAI":
        st.info(
            """
        OpenAI ASR uses the OpenAI API. Make sure to set `OPENAI_API_KEY` environment variable.
        """
        )
    else:
        st.info(
            f"""
        {asr_vendor} is recommended to use when Nvidia GPU is available.
        """
        )

    # Add ASR parameters
    st.subheader("ASR Parameters")

    model_name = st.text_input(
        "Model name",
        value=st.session_state.config.get("asr", {}).get("model_name", "tiny"),
        help="Particular model architecture of the provided type, e.g. 'tiny'",
        key="model_name_input",
        on_change=on_model_name_change,
    )

    language = st.text_input(
        "Language code",
        value=st.session_state.config.get("asr", {}).get("language", "en"),
        help="Language code for the ASR model (e.g., 'en' for English)",
        key="language_input",
        on_change=on_language_change,
    )

    silence_grace_period = st.number_input(
        "Silence grace period (seconds)",
        value=float(
            st.session_state.config.get("asr", {}).get("silence_grace_period", 1.0)
        ),
        min_value=0.1,
        help="Grace period in seconds after silence to stop recording",
        key="silence_grace_input",
        on_change=on_silence_grace_change,
    )

    vad_threshold = st.slider(
        "VAD threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.config.get("asr", {}).get("vad_threshold", 0.5)),
        help="Threshold for voice activity detection",
        key="vad_threshold_input",
        on_change=on_vad_threshold_change,
    )

    use_wake_word = st.checkbox(
        "Use wake word detection",
        value=st.session_state.config.get("asr", {}).get("use_wake_word", False),
        key="wake_word_checkbox",
        on_change=on_wake_word_change,
    )

    if use_wake_word:
        wake_word_model = st.text_input(
            "Wake word model",
            value=st.session_state.config.get("asr", {}).get("wake_word_model", ""),
            help="Wake word model type to use",
            key="wake_word_model_input",
            on_change=on_wake_word_model_change,
        )

        wake_word_model = st.text_input(
            "Wake word model name",
            value=st.session_state.config.get("asr", {}).get(
                "wake_word_model_name", ""
            ),
            help="Specific wake word model to use",
            key="wake_word_model_name_input",
            on_change=on_wake_word_model_name_change,
        )

        wake_word_threshold = st.slider(
            "Wake word threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(
                st.session_state.config.get("asr", {}).get("wake_word_threshold", 0.5)
            ),
            help="Threshold for wake word detection",
            key="wake_word_threshold_input",
            on_change=on_wake_word_threshold_change,
        )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        st.button("Next →", on_click=next_step)


def tts():
    from rai_s2s.tts import TTS_MODELS

    def on_tts_vendor_change():
        st.session_state.config["tts"]["vendor"] = st.session_state.tts_vendor_select

    def on_voice_change():
        st.session_state.config["tts"]["voice"] = st.session_state.tts_voice_input

    def on_sound_device_change():
        st.session_state.config["tts"]["speaker_device_name"] = (
            st.session_state.sound_device_select
        )

    # Ensure tts config exists
    if "tts" not in st.session_state.config:
        st.session_state.config["tts"] = {}

    st.title("Text to Speech Configuration")
    st.info(
        """
    Text to Speech (TTS) converts your assistant's text responses into spoken words:
    - ElevenLabs provides high-quality, natural-sounding voices (requires API key)
    - KokoroTTS in ONNX format runs locally on your computer.
    - OpenTTS runs locally on your computer with no API costs (requires Docker)
    """
    )

    sound_devices = get_sound_devices(output=True)
    currently_selected_device_name = st.session_state.config.get("tts", {}).get(
        "speaker_device_name", ""
    )
    try:
        device_index = [device["name"] for device in sound_devices].index(
            currently_selected_device_name
        )
    except ValueError:
        device_index = None

    recording_device_name = st.selectbox(
        "Default speaker device",
        [device["name"] for device in sound_devices],
        placeholder="Select device",
        index=device_index,
        key="sound_device_select",
        on_change=on_sound_device_change,
    )

    refresh_devices = st.button("Refresh devices")
    if refresh_devices:
        recording_devices = get_sound_devices(reinitialize=True, output=True)

    if recording_device_name == "default":
        st.info(
            """
        If you're experiencing audio issues and device_name is set to 'default', try specifying the exact device name instead, as this often resolves the problem.
        """
        )
    # Get the current vendor from config and convert to display name
    current_vendor = st.session_state.config.get("tts", {}).get("vendor", TTS_MODELS[0])

    tts_vendor = st.selectbox(
        "Choose your TTS vendor",
        TTS_MODELS,
        placeholder="Select vendor",
        index=TTS_MODELS.index(current_vendor),
        key="tts_vendor_select",
        on_change=on_tts_vendor_change,
    )

    if tts_vendor == "ElevenLabs":
        st.info(
            """
        Please ensure you have the following environment variable set:
        ```sh
        export ELEVENLABS_API_KEY="..."
        ```

        To get your API key, follow the instructions [here](https://elevenlabs.io/docs/api-reference/getting-started)
        """
        )
    elif tts_vendor == "OpenTTS":
        st.info(
            """
        Please ensure you have the Docker container running:
        ```sh
        docker run -it -p 5500:5500 synesthesiam/opentts:en
        ```

        To learn more about OpenTTS, visit [here](https://github.com/synesthesiam/opentts)
        """
        )
    elif tts_vendor == "KokoroTTS":
        st.info(
            """
        To learn more about KokoroTTS, visit [here](https://huggingface.co/hexgrad/Kokoro-82M)
        """
        )

    model_name = st.text_input(
        "Voice",
        value=st.session_state.config.get("asr", {}).get("voice", ""),
        help="Voice compatible with selected vendor. If left empty RAI will select a deafault value.",
        key="tts_voice_input",
        on_change=on_voice_change,
    )

    st.info(
        """
    Some speakers enter power-saving mode when inactive.
    Enabling this option will keep the speaker active, reducing audio playback latency.
    """
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        st.button("Next →", on_click=next_step)


def additional_features():
    st.title("Additional Features Configuration")
    st.info(
        """
    These optional features enhance your assistant's capabilities but require additional dependencies.
    Follow the installation instructions below for the features you want to use.
    """
    )

    # Perception Instructions
    st.subheader("Perception (Visual Understanding)")
    st.markdown(
        """
    Perception provides visual understanding through Grounding DINO and Grounded SAM models.

    To install Perception dependencies, run:
    ```bash
    poetry install --with perception
    ```

    This will install:
    - Grounding DINO for object detection
    - Grounded SAM for segmentation
    - Required CUDA dependencies
    """
    )

    # NOMAD Instructions
    st.subheader("NOMAD (Navigation)")
    st.markdown(
        """
    NOMAD enables navigation capabilities using transformer-based image processing.

    To install NOMAD dependencies, run:
    ```bash
    poetry install --with nomad
    ```

    This will install:
    - NOMAD navigation transformer
    - Required image processing libraries
    """
    )

    st.info(
        "⚠️ Note: These features require significant disk space and may need a GPU for optimal performance."
    )

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        st.button("Next →", on_click=next_step)


def review_and_save():
    st.title("Review & Save Configuration")
    st.write(
        """
    This is the final step where you can:
    - Review all your configuration settings
    - Test the configuration to ensure everything works
    - Save the settings to a file that your assistant will use
    """
    )
    st.info(
        "The configuration contains default values for each setting, even if you didn't set them."
    )

    # Display current configuration
    st.subheader("Current Configuration")
    toml_string = tomli_w.dumps(st.session_state.config)
    st.code(toml_string, language="toml")

    if st.button("Test Configuration"):
        progress = st.progress(0.0)
        test_results = {}

        def create_chat_model(model_type: str):
            vendor_name = st.session_state.config["vendor"][f"{model_type}_model"]
            model_name = st.session_state.config[vendor_name][f"{model_type}_model"]

            if vendor_name == "openai":
                base_url = st.session_state.config["openai"]["base_url"]
                return ChatOpenAI(model=model_name, base_url=base_url)
            elif vendor_name == "aws":
                return ChatBedrock(model_id=model_name)
            elif vendor_name == "ollama":
                return ChatOllama(
                    model=model_name,
                    base_url=st.session_state.config["ollama"]["base_url"],
                )
            raise ValueError(f"Unknown vendor: {vendor_name}")

        def test_chat_model(model_type: str) -> bool:
            try:
                model = create_chat_model(model_type)
                answer = model.invoke("Say hello!")
                return answer.content is not None
            except Exception as e:
                st.error(f"{model_type.title()} model error: {e}")
                return False

        def test_simple_model() -> bool:
            return test_chat_model("simple")

        def test_complex_model() -> bool:
            return test_chat_model("complex")

        def test_embeddings_model():
            try:
                embeddings_model_vendor_name = st.session_state.config["vendor"][
                    "embeddings_model"
                ]
                if embeddings_model_vendor_name == "openai":
                    embeddings_model = OpenAIEmbeddings(
                        model=st.session_state.config["openai"]["embeddings_model"]
                    )
                elif embeddings_model_vendor_name == "aws":
                    embeddings_model = BedrockEmbeddings(
                        model_id=st.session_state.config["aws"]["embeddings_model"]
                    )
                elif embeddings_model_vendor_name == "ollama":
                    embeddings_model = OllamaEmbeddings(
                        model=st.session_state.config["ollama"]["embeddings_model"],
                        base_url=st.session_state.config["ollama"]["base_url"],
                    )
                embeddings_answer = embeddings_model.embed_query("Say hello!")
                return embeddings_answer is not None
            except Exception as e:
                st.error(f"Embeddings model error: {e}")
                return False

        def test_langfuse():
            use_langfuse = st.session_state.config["tracing"]["langfuse"][
                "use_langfuse"
            ]
            if not use_langfuse:
                return True
            return bool(os.getenv("LANGFUSE_SECRET_KEY")) and bool(
                os.getenv("LANGFUSE_PUBLIC_KEY")
            )

        def test_langsmith():
            use_langsmith = st.session_state.config["tracing"]["langsmith"][
                "use_langsmith"
            ]
            if not use_langsmith:
                return True
            return bool(os.getenv("LANGCHAIN_API_KEY"))

        def test_tts():
            vendor = st.session_state.config["tts"]["vendor"]
            if vendor == "ElevenLabs":
                try:
                    from elevenlabs import ElevenLabs

                    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                    output = client.generate(text="Hello, world!")
                    output = list(output)
                    return True
                except Exception as e:
                    st.error(f"TTS error: {e}")
                return False
            elif vendor == "OpenTTS":
                try:
                    params = {
                        "voice": "glow-speak:en-us_mary_ann",
                        "text": "Hello, world!",
                    }
                    response = requests.get(
                        "http://localhost:5500/api/tts", params=params
                    )
                    if response.status_code == 200:
                        return True
                except Exception as e:
                    st.error(f"TTS error: {e}")
                return False
            elif vendor == "KokoroTTS":
                try:
                    from rai_s2s.tts import KokoroTTS

                    model = KokoroTTS()
                    model.get_speech(text="A")
                    return True
                except Exception as e:
                    st.error(f"TTS error: {e}")
                return False

        def test_recording_device(device_name: str):
            import sounddevice as sd

            devices = sd.query_devices()
            index = [device["name"] for device in devices].index(device_name)
            sample_rate = int(devices[index]["default_samplerate"])
            try:
                recording = sd.rec(
                    device=index,
                    frames=sample_rate,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                )
                sd.wait()
                if np.sum(np.abs(recording)) == 0:
                    return False
                return True
            except Exception as e:
                st.error(f"Recording device error: {e}")
                return False

        # TODO: Add ASR test
        # TODO: Move tests to a separate file in tests/

        # Run tests

        tests = [
            (test_simple_model, "Simple Model"),
            (test_complex_model, "Complex Model"),
            (test_embeddings_model, "Embeddings Model"),
            (test_langfuse, "Langfuse"),
            (test_langsmith, "LangSmith"),
        ]
        if st.session_state.features["s2s"]:
            tests.extend(
                [
                    (test_tts, "TTS"),
                    (
                        partial(
                            test_recording_device,
                            st.session_state.config["asr"]["recording_device_name"],
                        ),
                        "Recording Device",
                    ),
                ]
            )
        progress.progress(0.0, "Running tests...")
        for i, (test, name) in enumerate(tests):
            test_results[name] = test()
            progress.progress((1 + i) / len(tests), f"Testing {name}...")

        # Display results in a table
        st.subheader("Test Results")

        # Create a two-column table using streamlit columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Component**")
            for component in test_results.keys():
                st.write(component)
        with col2:
            st.markdown("**Status**")
            for result in test_results.values():
                st.write("✅ Pass" if result else "❌ Fail")

        # Overall success message
        if all(test_results.values()):
            st.success("All tests passed! You can save the configuration now.")
        else:
            st.error("Some tests failed. Please check the errors above.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        if st.button("Save Configuration"):
            # Save configuration to TOML file
            with open("config.toml", "wb") as f:
                tomli_w.dump(st.session_state.config, f)
            st.success("Configuration saved successfully!")


@st.cache_data
def setup_steps():
    step_names = ["👋 Welcome", "🤖 Model Selection", "📊 Tracing"]
    step_render = [welcome, model_selection, tracing]

    if importlib.util.find_spec("rai_s2s") is None:
        logging.warning(
            "Skipping speech recognition, rai_s2s not installed - install `poetry install --with s2s`"
        )
        st.session_state.features["s2s"] = False
    else:
        st.session_state.features["s2s"] = True

        try:
            from rai_s2s.asr import TRANSCRIBE_MODELS

            step_names.append("🎙️ Speech Recognition")
            step_render.append(asr)
        except ImportError as e:
            st.session_state.features["s2s"] = False
            logging.warning(f"Skipping speech recognition. {e}")

        try:
            from rai_s2s.tts import TTS_MODELS

            step_names.append("🔊 Text to Speech")
            step_render.append(tts)
        except ImportError as e:
            st.session_state.features["s2s"] = False
            logging.warning(f"Skipping text to speech. {e}")

    step_names.extend(
        [
            "🎯 Additional Features",
            "✅ Review & Save",
        ]
    )
    step_render.extend([additional_features, review_and_save])

    steps = dict(enumerate(step_names))
    step_renderer = dict(enumerate(step_render))
    return steps, step_renderer


# Initialize session state for tracking steps if not exists
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "features" not in st.session_state:
    st.session_state.features = {}
if "config" not in st.session_state:
    # Load initial config from TOML file
    try:
        with open("config.toml", "rb") as f:
            st.session_state.config = tomli.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("config.toml not found. Please recreate it.")

# Sidebar progress tracker
st.sidebar.title("Configuration Progress")
steps, step_renderer = setup_steps()

# Replace the existing step display with clickable elements
for step_num, step_name in steps.items():
    if step_num == st.session_state.current_step:
        # Current step is bold and has an arrow
        if st.sidebar.button(
            step_name, key=f"step_{step_num}", use_container_width=True
        ):
            st.session_state.current_step = step_num
    else:
        # Other steps are clickable but not highlighted
        if st.sidebar.button(
            step_name, key=f"step_{step_num}", use_container_width=True
        ):
            st.session_state.current_step = step_num


# Navigation buttons
def next_step():
    st.session_state.current_step = st.session_state.current_step + 1


def prev_step():
    st.session_state.current_step = st.session_state.current_step - 1


# Main content based on current step
step_renderer[st.session_state.current_step]()
