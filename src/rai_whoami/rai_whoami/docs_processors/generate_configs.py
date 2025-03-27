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

import argparse
from pathlib import Path

from rai.utils.model_initialization import get_llm_model

from rai_whoami.docs_processors.config_generator import ConfigGenerator


def parse_args():
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate configuration files from documentation using RAI Whoami."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("docs"),
        help="Directory containing documentation files (default: docs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs"),
        help="Directory to save generated configuration files (default: configs)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search in input directory",
    )
    return parser.parse_args()


def generate_configs_from_docs(
    input_dir: Path = Path("docs"),
    output_dir: Path = Path("configs"),
    recursive: bool = True,
) -> dict[str, Path]:
    """Generate configuration files from documentation.

    Parameters
    ----------
    input_dir : Path, optional
        Directory containing documentation files, by default Path("docs")
    output_dir : Path, optional
        Directory to save generated configuration files, by default Path("configs")
    recursive : bool, optional
        Whether to search recursively in input directory, by default True

    Returns
    -------
    dict[str, Path]
        Dictionary mapping config types to their generated file paths
    """
    # Create the config generator
    generator = ConfigGenerator(
        docs_path=input_dir, model=get_llm_model("simple_model"), recursive=recursive
    )

    # Generate and save the configurations
    return generator.generate_configs(output_dir=output_dir)


def main():
    """CLI entry point for generating configuration files from documentation."""
    args = parse_args()

    # Generate and save the configurations
    config_paths = generate_configs_from_docs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
    )

    print("Generated configuration files:")
    for config_type, path in config_paths.items():
        print(f"- {config_type}: {path}")


if __name__ == "__main__":
    main()
