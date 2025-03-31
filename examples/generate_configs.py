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

from pathlib import Path

from rai_whoami.docs_processors.config_generator import ConfigGenerator


def main():
    """Generate configuration files from documentation."""
    # Get the path to the documentation directory
    docs_dir = Path("docs")

    # Create the config generator
    generator = ConfigGenerator(
        docs_path=docs_dir,
        model_name="gpt-4-turbo-preview",  # You can change this to use a different model
        recursive=True,
    )

    # Generate and save the configurations
    output_dir = Path("configs")
    config_paths = generator.generate_configs(output_dir=output_dir)

    print("Generated configuration files:")
    for config_type, path in config_paths.items():
        print(f"- {config_type}: {path}")


if __name__ == "__main__":
    main()
