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
    2. Set up speech recognition
    3. Configure text-to-speech
    4. Review and save your configuration

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

        advanced_config = st.container()
        advanced_config.subheader("Multivendor configuration (Advanced)")

        use_advanced_config = False
        with advanced_config:
            st.write(
                "If you have access to multiple vendors, you can configure the models to use different vendors."
            )
            use_advanced_config = st.checkbox("Use advanced configuration", value=False)
            models_col, vendor_col = st.columns(2)
            current_simple_model_vendor = st.session_state.config["vendor"][
                "simple_model"
            ]
            current_simple_model = st.session_state.config[current_simple_model_vendor][
                "simple_model"
            ]
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
                simple_model = st.text_input("Simple model", value=current_simple_model)
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

    # ... ASR configuration ...
    with st.expander("View available recording devices"):
        st.markdown(f"```python\n{sd.query_devices()}\n```")

    default_recording_device = st.number_input("Default recording device", value=0)
    local_asr = st.checkbox(
        "Enable local ASR (Whisper). Recommended when Nvidia GPU is available."
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
        ["ElevenLabs", "OpenTTS (Local)"],
        placeholder="Select vendor",
    )

    if tts_vendor == "ElevenLabs":
        st.info(
            """
        Please ensure you have the following environment variable set:
        - `ELEVENLABS_API_KEY`
        """
        )
    elif tts_vendor == "OpenTTS (Local)":
        st.info(
            """
        Please ensure you have the Docker container running:
        ```
        docker run -it -p 5500:5500 synesthesiam/opentts:en
        ```
        """
        )

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
    st.info(
        """
    This is the final step where you can:
    - Review all your configuration settings
    - Test the configuration to ensure everything works
    - Save the settings to a file that your assistant will use
    """
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
        # create complex model
        progress.progress(0.2)
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

        # create embeddings model
        progress.progress(0.3)
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
