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


"""
Command-line interface for RAI fine-tuning utilities
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _add_common_extract_args(parser):
    """Add common arguments for data extraction"""
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--start-time", help="Start time for data extraction (ISO format)"
    )
    parser.add_argument(
        "--stop-time", help="Stop time for data extraction (ISO format)"
    )
    parser.add_argument(
        "--max-data-limit",
        type=int,
        default=5000,
        help="Maximum number of records to extract",
    )
    parser.add_argument(
        "--models",
        action="append",
        help="Filter data samples by model name substring(s)",
    )


def _add_langfuse_args(parser):
    """Add Langfuse-specific arguments"""
    _add_common_extract_args(parser)
    parser.add_argument(
        "--page-size", type=int, default=100, help="Page size for pagination"
    )
    parser.add_argument(
        "--type-filter",
        dest="type_filter",
        help="Observation type filter (e.g., GENERATION, EVENT, SPAN), langfuse only",
    )
    parser.add_argument(
        "--trace-id",
        help="Restrict to a specific trace ID, langfuse only",
    )


def _add_langsmith_args(parser):
    """Add LangSmith-specific arguments"""
    _add_common_extract_args(parser)
    parser.add_argument(
        "--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY env var)"
    )
    parser.add_argument(
        "--api-url", default="https://api.smith.langchain.com", help="LangSmith API URL"
    )
    parser.add_argument(
        "--project-name",
        help="LangSmith project name to filter runs, langsmith only",
    )
    parser.add_argument(
        "--run-id",
        help="Restrict to a specific run ID, langsmith only",
    )
    # Override max-data-limit default for LangSmith
    for action in parser._actions:
        if action.dest == "max_data_limit":
            action.default = 1000
            break


def _add_format_args(parser):
    """Add format command arguments"""
    parser.add_argument("--input", required=True, help="Input data file")
    parser.add_argument("--output", required=True, help="Output training data file")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant that can use tools to help users.",
        help="System prompt to use",
    )


def extract_langfuse_command(args):
    """Handle extract langfuse command"""
    try:
        from .data.extractors.langfuse_data import main as extract_main

        # Convert our args to the format expected by the extractor
        sys.argv = [sys.argv[0]] + args
        extract_main()
    except Exception as e:
        logger.error(f"Langfuse extraction failed: {e}")
        sys.exit(1)


def extract_langsmith_command(args):
    """Handle extract langsmith command"""
    try:
        from .data.extractors.langsmith_data import main as extract_main

        # Convert our args to the format expected by the extractor
        sys.argv = [sys.argv[0]] + args
        extract_main()
    except Exception as e:
        logger.error(f"LangSmith extraction failed: {e}")
        sys.exit(1)


def format_command(args):
    """Handle format command"""
    try:
        from .data.formatters.tool_calling import main as format_main

        # Convert our args to the format expected by the formatter
        sys.argv = [sys.argv[0]] + args
        format_main()
    except Exception as e:
        logger.error(f"Formatting failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RAI Fine-tuning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract data from Langfuse
  python -m rai_finetune.data_cli extract langfuse --models "gpt-4o" --models "gpt-4o-mini" --output observations.jsonl --page-size 100

  # Extract data from LangSmith
  python -m rai_finetune.data_cli extract langsmith --api-key YOUR_KEY --project-name "rai" --output observations.jsonl

  # Format data for training
  python -m rai_finetune.data_cli format --input observations.jsonl --output training_data.jsonl

  # Get help for specific commands
  python -m rai_finetune.data_cli extract langfuse --help
  python -m rai_finetune.data_cli extract langsmith --help
  python -m rai_finetune.data_cli format --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command with provider subcommands
    extract_parser = subparsers.add_parser(
        "extract", help="Extract data from various sources"
    )
    extract_subparsers = extract_parser.add_subparsers(
        dest="provider", help="Data source provider"
    )

    # Langfuse extract subcommand
    langfuse_parser = extract_subparsers.add_parser(
        "langfuse", help="Extract data from Langfuse"
    )
    _add_langfuse_args(langfuse_parser)

    # LangSmith extract subcommand
    langsmith_parser = extract_subparsers.add_parser(
        "langsmith", help="Extract data from LangSmith"
    )
    _add_langsmith_args(langsmith_parser)

    # Format command
    format_parser = subparsers.add_parser("format", help="Format data for training")
    _add_format_args(format_parser)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract" and not args.provider:
        extract_parser.print_help()
        sys.exit(1)

    # Convert args to list for passing to subcommands
    arg_list = []
    for key, value in vars(args).items():
        if key in ["command", "provider"]:
            continue
        if value is not None:
            # Convert underscores to hyphens for argument names
            arg_name = key.replace("_", "-")
            if isinstance(value, list):
                for v in value:
                    arg_list.extend([f"--{arg_name}", str(v)])
            else:
                arg_list.extend([f"--{arg_name}", str(value)])

    if args.command == "extract":
        if args.provider == "langfuse":
            extract_langfuse_command(arg_list)
        elif args.provider == "langsmith":
            extract_langsmith_command(arg_list)
        else:
            logger.error(f"Unknown provider: {args.provider}")
            sys.exit(1)
    elif args.command == "format":
        format_command(arg_list)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
