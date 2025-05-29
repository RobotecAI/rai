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

import argparse
import os

import tomli
import tomli_w

CONFIG_CONTENT = """
[vendor]
simple_model = "openai"
complex_model = "openai"
embeddings_model = "openai"

[aws]
simple_model = "anthropic.claude-3-haiku-20240307-v1:0"
complex_model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
embeddings_model = "amazon.titan-embed-text-v1"
region_name = "us-east-1"

[openai]
simple_model = "gpt-4o-mini"
complex_model = "gpt-4o"
embeddings_model = "text-embedding-ada-002"
base_url = "https://api.openai.com/v1/"

[ollama]
simple_model = "llama3.2"
complex_model = "llama3.1:70b"
embeddings_model = "llama3.2"
base_url = "http://localhost:11434"

[tracing]
project = "rai"

[tracing.langfuse]
use_langfuse = false
host = "http://localhost:3000"

[tracing.langsmith]
use_langsmith = false
host = "https://api.smith.langchain.com"

[asr]
recording_device_name = "default"
transcription_model = "LocalWhisper"
language = "en"
vad_model = "SileroVAD"
silence_grace_period = 0.3
vad_threshold = 0.3
use_wake_word = false
wake_word_model = ""
wake_word_threshold = 0.5
wake_word_model_name = ""
transcription_model_name = "tiny"

[tts]
vendor = "ElevenLabs"
voice = ""
speaker_device_name = "default"
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        if args.force and os.path.exists("config.toml"):
            os.remove("config.toml")
        if os.path.exists("config.toml") and not args.force:
            print(
                "config.toml already exists. Use --force to overwrite with default values."
            )
            return

        config_dict = tomli.loads(CONFIG_CONTENT)

        with open("config.toml", "wb") as f:
            tomli_w.dump(config_dict, f)

    except (OSError, IOError) as e:
        print(f"Error writing config file: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    print("config.toml created successfully.")


if __name__ == "__main__":
    main()
