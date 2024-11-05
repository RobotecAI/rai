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
from typing import Dict, List

import sounddevice as sd
import streamlit as st
import tomli
import tomli_w
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize session state for tracking steps if not exists
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "config" not in st.session_state:
    # Load initial config from TOML file
    try:
        with open("config.toml", "rb") as f:
            st.session_state.config = tomli.load(f)
    except FileNotFoundError:
        st.session_state.config = {}

# Sidebar progress tracker
st.sidebar.title("Configuration Progress")
steps = {
    1: "👋 Welcome",
    2: "🤖 Model Selection",
    3: "📊 Tracing",
    4: "🎙️ Speech Recognition",
    5: "🔊 Text to Speech",
    6: "🎯 Additional Features",
    7: "✅ Review & Save",
}

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
if st.session_state.current_step == 1:
    st.title("Welcome to RAI Configurator! 👋")
    st.markdown(
        """
    This wizard will help you set up your RAI environment step by step:
    1. Configure your AI models and vendor
    2. Set up model tracing and monitoring
    3. Configure speech recognition (ASR)
    4. Set up text-to-speech (TTS)
    5. Enable additional features
    6. Review and save your configuration

    Let's get started!
    """
    )

    st.button("Begin Configuration →", on_click=next_step)

elif st.session_state.current_step == 2:
    st.title("Model Configuration")
    st.info(
        """
    This step configures which AI models will be used by your assistant. Different models have different capabilities and costs:
    - Simple models are faster and cheaper, used for basic tasks
    - Complex models are more capable but slower, used for complex reasoning
    - Embedding models convert text into numerical representations for memory and search
    """
    )

    vendor = st.selectbox(
        "Which AI vendor would you like to use?",
        ["openai", "aws", "ollama"],
        placeholder="Select vendor",
        key="vendor",
    )

    if vendor:
        # Store vendor in config
        if vendor == "openai":
            simple_model = st.text_input(
                "Model for simple tasks", value="gpt-4o-mini", key="simple_model"
            )
            complex_model = st.text_input(
                "Model for complex tasks",
                value="gpt-4o-2024-08-06",
                key="complex_model",
            )
            embeddings_model = st.text_input(
                "Embeddings model",
                value="text-embedding-ada-002",
                key="embeddings_model",
            )

        elif vendor == "aws":
            col1, col2 = st.columns(2)

            with col1:
                simple_model = st.text_input(
                    "Model for simple tasks",
                    value="anthropic.claude-3-haiku-20240307-v1:0",
                )
                complex_model = st.text_input(
                    "Model for complex tasks",
                    value="anthropic.claude-3-5-sonnet-20240620-v1:0",
                )
                embeddings_model = st.text_input(
                    "Embeddings model", value="amazon.titan-embed-text-v1"
                )

            with col2:
                aws_bedrock_region = st.text_input(
                    "AWS Bedrock region", value="us-east-1"
                )

        elif vendor == "ollama":
            base_url = st.text_input("Ollama base URL", value="http://localhost:11434")
            simple_model = st.text_input("Model for simple tasks", value="llama3.2")
            complex_model = st.text_input(
                "Model for complex tasks", value="llama3.1:70b"
            )
            embeddings_model = st.text_input("Embeddings model", value="llama3.2")

        st.subheader("Multivendor configuration (Advanced)")

        st.write(
            "If you have access to multiple vendors, you can configure the models to use different vendors."
        )
        use_advanced_config = st.checkbox("Use advanced configuration", value=False)
        advanced_config = st.container()
        if use_advanced_config:
            with advanced_config:
                models_col, vendor_col = st.columns(2)
                current_simple_model_vendor = st.session_state.config["vendor"][
                    "simple_model"
                ]
                current_simple_model = st.session_state.config[
                    current_simple_model_vendor
                ]["simple_model"]
                current_complex_model_vendor = st.session_state.config["vendor"][
                    "complex_model"
                ]
                current_complex_model = st.session_state.config[
                    current_complex_model_vendor
                ]["complex_model"]
                current_embeddings_model_vendor = st.session_state.config["vendor"][
                    "embeddings_model"
                ]
                current_embeddings_model = st.session_state.config[
                    current_embeddings_model_vendor
                ]["embeddings_model"]
                with models_col:
                    simple_model = st.text_input(
                        "Simple model", value=current_simple_model
                    )
                    complex_model = st.text_input(
                        "Complex model", value=current_complex_model
                    )
                    embeddings_model = st.text_input(
                        "Embeddings model", value=current_embeddings_model
                    )
                with vendor_col:
                    simple_model_vendor = st.text_input(
                        "Simple model vendor", value=current_simple_model_vendor
                    )
                    complex_model_vendor = st.text_input(
                        "Complex model vendor", value=current_complex_model_vendor
                    )
                    embeddings_model_vendor = st.text_input(
                        "Embeddings model vendor", value=current_embeddings_model_vendor
                    )
        if use_advanced_config:
            st.session_state.config["vendor"] = {
                "simple_model": simple_model_vendor,
                "complex_model": complex_model_vendor,
                "embeddings_model": embeddings_model_vendor,
            }
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

elif st.session_state.current_step == 3:
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

    # Langfuse configuration
    st.subheader("Langfuse Configuration")
    langfuse_enabled = st.checkbox(
        "Enable Langfuse",
        value=st.session_state.config.get("tracing", {})
        .get("langfuse", {})
        .get("use_langfuse", False),
    )

    if langfuse_enabled:
        st.info(
            """
        Please ensure you have the following environment variables set:
        - `LANGFUSE_SECRET_KEY="sk-lf-..."`
        - `LANGFUSE_PUBLIC_KEY="pk-lf-..."`

        Find setup instructions [here](https://langfuse.com/docs/deployment/self-host)
        """
        )

        langfuse_host = st.text_input(
            "Langfuse Host",
            value=st.session_state.config.get("tracing", {})
            .get("langfuse", {})
            .get("host", "https://cloud.langfuse.com"),
        )
        # Store in config
        if "tracing" not in st.session_state.config:
            st.session_state.config["tracing"] = {}
        st.session_state.config["tracing"]["langfuse"] = {
            "use_langfuse": langfuse_enabled,
            "host": langfuse_host,
        }

    # Langsmith configuration
    st.subheader("LangSmith Configuration")
    langsmith_enabled = st.checkbox(
        "Enable LangSmith",
        value=st.session_state.config.get("tracing", {})
        .get("langsmith", {})
        .get("use_langsmith", False),
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

elif st.session_state.current_step == 4:
    st.title("Speech Recognition Configuration")
    st.info(
        """
    Speech recognition (ASR - Automatic Speech Recognition) converts spoken words into text. This allows your assistant to understand voice input:
    - Local ASR uses Whisper and runs on your computer (recommended with GPU)
    - Device selection determines which microphone is used for voice input
    """
    )

    def get_recording_devices(reinitialize: bool = False) -> List[Dict[str, str | int]]:
        if reinitialize:
            sd._terminate()
            sd._initialize()
        devices: List[Dict[str, str | int]] = sd.query_devices()
        recording_devices = [
            device for device in devices if device.get("max_input_channels", 0) > 0
        ]
        return recording_devices

    recording_devices = get_recording_devices()
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
    )
    refresh_devices = st.button("Refresh devices")
    if refresh_devices:
        recording_devices = get_recording_devices(reinitialize=True)

    st.session_state.config["asr"]["recording_device_name"] = recording_device_name
    asr_vendor = st.selectbox(
        "Choose your ASR vendor",
        ["Local Whisper (Free)", "OpenAI (Cloud)"],
        placeholder="Select vendor",
    )
    if asr_vendor == "Local Whisper (Free)":
        st.info(
            """
        Recommended to use when Nvidia GPU is available.
        """
        )
        st.session_state.config["asr"]["vendor"] = "whisper"
    elif asr_vendor == "OpenAI (Cloud)":
        st.info(
            """
        OpenAI ASR uses the OpenAI API. Make sure to set `OPENAI_API_KEY` environment variable.
        """
        )
        st.session_state.config["asr"]["vendor"] = "openai"

    # Add ASR parameters
    st.subheader("ASR Parameters")

    language = st.text_input(
        "Language code",
        value=st.session_state.config.get("asr", {}).get("language", "en"),
        help="Language code for the ASR model (e.g., 'en' for English)",
    )

    silence_grace_period = st.number_input(
        "Silence grace period (seconds)",
        value=float(
            st.session_state.config.get("asr", {}).get("silence_grace_period", 1.0)
        ),
        min_value=0.1,
        help="Grace period in seconds after silence to stop recording",
    )

    vad_threshold = st.slider(
        "VAD threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.config.get("asr", {}).get("vad_threshold", 0.5)),
        help="Threshold for voice activity detection",
    )

    use_wake_word = st.checkbox(
        "Use wake word detection",
        value=st.session_state.config.get("asr", {}).get("use_wake_word", False),
    )
    if use_wake_word:
        wake_word_model = st.text_input(
            "Wake word model",
            value=st.session_state.config.get("asr", {}).get("wake_word_model", ""),
            help="Wake word model to use",
        )
        wake_word_threshold = st.slider(
            "Wake word threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(
                st.session_state.config.get("asr", {}).get("wake_word_threshold", 0.5)
            ),
            help="Threshold for wake word detection",
        )

    # Update config
    if "asr" not in st.session_state.config:
        st.session_state.config["asr"] = {}

    st.session_state.config["asr"].update(
        {
            "language": language,
            "silence_grace_period": silence_grace_period,
            "use_wake_word": use_wake_word,
            "vad_threshold": vad_threshold,
        }
    )

    if use_wake_word:
        st.session_state.config["asr"].update(
            {
                "wake_word_model": wake_word_model,
                "wake_word_threshold": wake_word_threshold,
            }
        )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        st.button("Next →", on_click=next_step)

elif st.session_state.current_step == 5:
    st.title("Text to Speech Configuration")
    st.info(
        """
    Text to Speech (TTS) converts your assistant's text responses into spoken words:
    - ElevenLabs provides high-quality, natural-sounding voices (requires API key)
    - OpenTTS runs locally on your computer with no API costs (requires Docker)
    """
    )

    tts_vendor = st.selectbox(
        "Choose your TTS vendor",
        ["ElevenLabs (Cloud)", "OpenTTS (Local)"],
        placeholder="Select vendor",
    )

    if tts_vendor == "ElevenLabs (Cloud)":
        st.info(
            """
        Please ensure you have the following environment variable set:
        ```sh
        export ELEVENLABS_API_KEY="..."
        ```

        To get your API key, follow the instructions [here](https://elevenlabs.io/docs/api-reference/getting-started)
        """
        )
        st.session_state.config["tts"]["vendor"] = "elevenlabs"

    elif tts_vendor == "OpenTTS (Local)":
        st.info(
            """
        Please ensure you have the Docker container running:
        ```sh
        docker run -it -p 5500:5500 synesthesiam/opentts:en
        ```

        To learn more about OpenTTS, visit [here](https://github.com/synesthesiam/opentts)
        """
        )
        st.session_state.config["tts"]["vendor"] = "opentts"
    keep_speaker_busy = st.checkbox("Keep speaker busy", value=False)
    st.info(
        """
    Some speakers enter power-saving mode when inactive.
    Enabling this option will keep the speaker active, reducing audio playback latency.
    """
    )
    st.session_state.config["tts"]["keep_speaker_busy"] = keep_speaker_busy
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        st.button("Next →", on_click=next_step)

elif st.session_state.current_step == 6:
    st.title("Additional Features Configuration")
    st.info(
        """
    These optional features enhance your assistant's capabilities but require additional dependencies.
    Follow the installation instructions below for the features you want to use.
    """
    )

    # OpenSET Instructions
    st.subheader("OpenSET (Visual Understanding)")
    st.markdown(
        """
    OpenSET provides visual understanding through Grounding DINO and Grounded SAM models.

    To install OpenSET dependencies, run:
    ```bash
    poetry install --with openset
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

elif st.session_state.current_step == 7:
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
        success = True
        progress = st.progress(0.0)

        vendor = st.session_state.config["vendor"]
        simple_model_vendor_name = st.session_state.config["vendor"]["simple_model"]
        complex_model_vendor_name = st.session_state.config["vendor"]["complex_model"]
        embeddings_model_vendor_name = st.session_state.config["vendor"][
            "embeddings_model"
        ]

        # create simple model
        progress.progress(0.1)
        try:
            if simple_model_vendor_name == "openai":
                simple_model = ChatOpenAI(
                    model=st.session_state.config["openai"]["simple_model"]
                )
            elif simple_model_vendor_name == "aws":
                simple_model = ChatBedrock(
                    model_id=st.session_state.config["aws"]["simple_model"]
                )
            elif simple_model_vendor_name == "ollama":
                simple_model = ChatOllama(
                    model=st.session_state.config["ollama"]["simple_model"],
                    base_url=st.session_state.config["ollama"]["base_url"],
                )
        except Exception as e:
            success = False
            st.error(f"Failed to initialize simple model: {e}")

        # create complex model
        progress.progress(0.2)
        try:
            if complex_model_vendor_name == "openai":
                complex_model = ChatOpenAI(
                    model=st.session_state.config["openai"]["complex_model"]
                )
            elif complex_model_vendor_name == "aws":
                complex_model = ChatBedrock(
                    model_id=st.session_state.config["aws"]["complex_model"]
                )
            elif complex_model_vendor_name == "ollama":
                complex_model = ChatOllama(
                    model=st.session_state.config["ollama"]["complex_model"],
                    base_url=st.session_state.config["ollama"]["base_url"],
                )
        except Exception as e:
            success = False
            st.error(f"Failed to initialize complex model: {e}")

        # create embeddings model
        progress.progress(0.3)
        try:
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
        except Exception as e:
            success = False
            st.error(f"Failed to initialize embeddings model: {e}")

        progress.progress(0.4)
        use_langfuse = st.session_state.config["tracing"]["langfuse"]["use_langfuse"]
        if use_langfuse:
            if not os.getenv("LANGFUSE_SECRET_KEY", "") or not os.getenv(
                "LANGFUSE_PUBLIC_KEY", ""
            ):
                success = False
                st.error(
                    "Langfuse is enabled but LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY is not set"
                )

        progress.progress(0.5)
        use_langsmith = st.session_state.config["tracing"]["langsmith"]["use_langsmith"]
        if use_langsmith:
            if not os.getenv("LANGCHAIN_API_KEY", ""):
                success = False
                st.error("Langsmith is enabled but LANGCHAIN_API_KEY is not set")

        progress.progress(0.6, text="Testing simple model")
        simple_answer = simple_model.invoke("Say hello!")
        if simple_answer.content is None:
            success = False
            st.error("Simple model is not working")

        progress.progress(0.7, text="Testing complex model")
        complex_answer = complex_model.invoke("Say hello!")
        if complex_answer.content is None:
            success = False
            st.error("Complex model is not working")

        progress.progress(0.8, text="Testing embeddings model")
        embeddings_answer = embeddings_model.embed_query("Say hello!")
        if embeddings_answer is None:
            success = False
            st.error("Embeddings model is not working")

        progress.progress(1.0, text="Done!")
        if success:
            st.success("Configuration is correct. You can save it now.")
        else:
            st.error("Configuration is incorrect")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        if st.button("Save Configuration"):
            # Save configuration to TOML file
            with open("config.toml", "wb") as f:
                tomli_w.dump(st.session_state.config, f)
            st.success("Configuration saved successfully!")
