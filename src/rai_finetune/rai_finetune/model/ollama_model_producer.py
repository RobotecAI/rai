# Copyright (C) 2025 RAI Development Team
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

# Author: Julia Jia


import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# Import the LoRA merger
try:
    from .merge_with_base_model import LoRAMerger

    MERGER_AVAILABLE = True
except ImportError:
    try:
        from merge_with_base_model import LoRAMerger

        MERGER_AVAILABLE = True
    except ImportError:
        MERGER_AVAILABLE = False

# Import Unsloth for GGUF export - this is the primary requirement
try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Fail fast if Unsloth is not available since this is an Unsloth-focused tool
if not UNSLOTH_AVAILABLE:
    raise ImportError(
        "Unsloth is required for Ollama conversion. Please install unsloth."
    )

if not MERGER_AVAILABLE:
    raise ImportError("LoRAMerger is required for Ollama conversion.")


class OllamaConverter:
    """Convert fine-tuned models to Ollama format using Unsloth's GGUF export.
    Uses existing LoRAMerger for merging functionality."""

    def __init__(
        self,
        input_dir: str = None,
        base_model_path: str = None,
        output_dir: str = "qwen_finetuned_ollama",
        merged_model_path: str = None,
    ):
        """
        Initialize the converter

        Args:
            input_dir: Path to the fine-tuned model directory (LoRA adapter) - required if merged_model_path not provided
            base_model_path: Path to the local base model directory or HuggingFace model ID - required if merged_model_path not provided
            output_dir: Path to save the Ollama format model
            merged_model_path: Path to pre-merged model directory (skips merging step if provided)
        """
        self.input_dir = Path(input_dir) if input_dir else None
        self.base_model_path = base_model_path  # Keep as string for LoRAMerger
        self.output_dir = Path(output_dir)
        self.merged_model_path = Path(merged_model_path) if merged_model_path else None

        # Validate inputs
        if self.merged_model_path:
            # If merged model path is provided, validate it exists
            if not self.merged_model_path.exists():
                raise ValueError(
                    f"Merged model directory does not exist: {merged_model_path}"
                )
            logger.info("Using pre-merged model - skipping LoRA merging step")
        else:
            # If no merged model path, require input_dir and base_model_path
            if not input_dir or not base_model_path:
                raise ValueError(
                    "Either merged_model_path OR both input_dir and base_model_path must be provided"
                )
            if not self.input_dir.exists():
                raise ValueError(f"Input directory does not exist: {input_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized Ollama converter")
        if self.merged_model_path:
            logger.info(f"Pre-merged model path: {self.merged_model_path}")
        else:
            logger.info(f"Input directory: {self.input_dir}")
            logger.info(f"Base model path: {self.base_model_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def convert(self) -> bool:
        """
        Convert the model to Ollama format with GGUF export

        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            logger.info("Starting conversion to Ollama format...")

            # Use pre-merged model or merge LoRA weights
            if self.merged_model_path:
                logger.info("Using pre-merged model, skipping LoRA merging...")
                merged_model_path = self.merged_model_path
            else:
                # Use existing LoRAMerger to merge LoRA weights
                merged_model_path = self._merge_lora_weights()
                if not merged_model_path:
                    logger.error("Failed to merge LoRA weights")
                    return False

            # Convert merged model to GGUF format using Unsloth
            gguf_model_path = self._convert_to_gguf(merged_model_path)
            if not gguf_model_path:
                logger.error("Failed to convert to GGUF format")
                return False

            # Create Ollama model structure and files
            self._create_ollama_files(gguf_model_path)

            # Clean up temporary files (only if we created them)
            if not self.merged_model_path:
                self._cleanup_temp_files(merged_model_path)

            logger.info("Conversion completed successfully!")
            logger.info(f"Ollama model saved to: {self.output_dir}")
            self._print_usage_instructions()

            return True

        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            return False

    def _merge_lora_weights(self) -> Optional[Path]:
        """Use existing LoRAMerger to merge LoRA weights with base model"""
        try:
            logger.info("Merging LoRA weights using LoRAMerger...")

            # Create temporary directory for merged model
            temp_dir = Path(tempfile.mkdtemp())
            merged_model_path = temp_dir / "merged_model"

            # Use LoRAMerger with Unsloth's merged_16bit method for best compatibility
            merger = LoRAMerger(
                adapter_dir=str(self.input_dir),
                base_model_path=self.base_model_path,
                output_dir=str(merged_model_path),
            )
            success = merger.merge(save_method="merged_16bit")

            if success:
                logger.info("LoRA weights successfully merged")
                return merged_model_path
            else:
                logger.error("LoRA merging failed")
                return None

        except Exception as e:
            logger.error(f"Failed to merge LoRA weights: {e}")
            return None

    def _convert_to_gguf(self, merged_model_path: Path) -> Optional[Path]:
        """Convert merged model to GGUF format using Unsloth with memory optimizations"""

        model, tokenizer = None, None
        try:
            logger.info("Converting merged model to GGUF format using Unsloth...")

            # Clear GPU cache before loading
            import gc

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            # Try different loading strategies based on available memory
            loading_strategies = [
                # Strategy 1: Try full GPU with aggressive settings
                {
                    "load_in_4bit": True,
                    "device_map": "auto",
                    "max_memory": {0: "10GB", "cpu": "16GB"},
                    "low_cpu_mem_usage": True,
                },
                # Strategy 2: More conservative GPU memory
                {
                    "load_in_4bit": True,
                    "device_map": "auto",
                    "max_memory": {0: "8GB", "cpu": "20GB"},
                    "low_cpu_mem_usage": True,
                },
                # Strategy 3: CPU-only as fallback
                {"load_in_4bit": False, "device_map": "cpu", "low_cpu_mem_usage": True},
            ]

            for i, strategy in enumerate(loading_strategies):
                try:
                    logger.info(f"Attempting loading strategy {i + 1}/3...")

                    # For CPU strategy, use float16, otherwise let Unsloth auto-detect
                    dtype_param = (
                        torch.float16 if strategy.get("device_map") == "cpu" else None
                    )

                    # Load the merged model using Unsloth with current strategy
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        str(merged_model_path),
                        max_seq_length=1024,  # Reduce sequence length to save memory
                        dtype=dtype_param,
                        trust_remote_code=True,
                        **strategy,
                    )

                    logger.info(f"Successfully loaded model with strategy {i + 1}")
                    break

                except Exception as e:
                    logger.warning(f"Loading strategy {i + 1} failed: {e}")
                    if model is not None:
                        del model
                    if tokenizer is not None:
                        del tokenizer
                    model, tokenizer = None, None

                    # Clear memory before trying next strategy
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()

                    if i == len(loading_strategies) - 1:
                        raise e

            if model is None or tokenizer is None:
                raise ValueError("Failed to load model with any strategy")

            # Export directly to GGUF with q4_k_m quantization using Unsloth
            gguf_path = self.output_dir / "model.gguf"
            logger.info(f"Exporting directly to GGUF q4_k_m: {gguf_path}")

            model.save_pretrained_gguf(
                str(gguf_path.parent),
                tokenizer=tokenizer,
                quantization_method="q4_k_m",  # Direct quantization to q4_k_m
            )

            # Clear memory after conversion
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            logger.info(f"Successfully exported to GGUF q4_k_m: {gguf_path}")
            return gguf_path

        except Exception as e:
            logger.error(f"GGUF conversion failed: {e}")
            # Clean up any remaining objects
            if model is not None:  # noqa: F821
                del model  # noqa: F821
            if tokenizer is not None:  # noqa: F821
                del tokenizer  # noqa: F821
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            return None

    def _create_ollama_files(self, gguf_model_path: Path):
        """Create Ollama Modelfile and metadata with tool support"""
        try:
            # Find the actual GGUF file that Unsloth created
            gguf_files = list(self.output_dir.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError("No GGUF files found in output directory")

            # Use the first GGUF file found (should be the one we just created)
            actual_gguf_file = gguf_files[0]
            logger.info(f"Using GGUF file: {actual_gguf_file.name}")

            # Create tool-aware Modelfile with function definitions
            modelfile_content = f"""# Ollama Modelfile for fine-tuned model with tool support
FROM {actual_gguf_file.name}

# Model parameters optimized for Qwen
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Tool-aware system prompt
SYSTEM \"\"\"You are a helpful AI assistant that has been fine-tuned for robotic manipulation tasks. You have access to tools for object detection, manipulation, and camera operations.

When you need to use a tool, respond with a JSON-formatted function call. Available tools:
- get_object_positions: Retrieve positions of objects in the scene
- move_object_from_to: Move objects from one position to another
- reset_arm: Reset the robotic arm to its default position
- get_ros2_camera_image: Capture an image from the camera

Always use the proper JSON format for tool calls and wait for tool responses before proceeding.\"\"\"

# Stop tokens for Qwen format
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

# Template for tool calling support
TEMPLATE \"\"\"{{- if or .System .Tools }}
<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You have access to the following tools:

{{- range .Tools }}
- {{ .function.name }}: {{ .function.description }}
{{- end }}

Use tools by responding with JSON in this format:
{{"tool_calls": [{{"id": "call_1", "type": "function", "function": {{"name": "function_name", "arguments": "{{json_arguments}}"}}}}]}}
{{- end }}
<|im_end|>
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}
<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}
<|im_start|>assistant
{{- if .ToolCalls }}
{{ range .ToolCalls }}
{{"tool_calls": [{{"id": "{{ .ID }}", "type": "function", "function": {{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}}}}]}}
{{- end }}
{{- else }}
{{ .Content }}
{{- end }}<|im_end|>
{{- else if eq .Role "tool" }}
<|im_start|>tool
{{ .Content }}<|im_end|>
{{- end }}
{{- end }}
<|im_start|>assistant
\"\"\"
"""

            with open(self.output_dir / "Modelfile", "w") as f:
                f.write(modelfile_content)

            # Create tool definitions file for easy reference
            tool_definitions = {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_object_positions",
                            "description": "Retrieve the positions of all objects of a specified type in the target frame. This tool provides accurate positional data but does not distinguish between different colors of the same object type.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "object_name": {
                                        "type": "string",
                                        "description": "The name of the object to get the positions of",
                                    }
                                },
                                "required": ["object_name"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "move_object_from_to",
                            "description": "Move an object from one point to another. The tool will grab the object from the first point and then release it at the second point.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "x": {
                                        "type": "number",
                                        "description": "The x coordinate of the source point",
                                    },
                                    "y": {
                                        "type": "number",
                                        "description": "The y coordinate of the source point",
                                    },
                                    "z": {
                                        "type": "number",
                                        "description": "The z coordinate of the source point",
                                    },
                                    "x1": {
                                        "type": "number",
                                        "description": "The x coordinate of the target point",
                                    },
                                    "y1": {
                                        "type": "number",
                                        "description": "The y coordinate of the target point",
                                    },
                                    "z1": {
                                        "type": "number",
                                        "description": "The z coordinate of the target point",
                                    },
                                },
                                "required": ["x", "y", "z", "x1", "y1", "z1"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "reset_arm",
                            "description": "Reset the robotic arm to its default position.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_ros2_camera_image",
                            "description": "Capture an image from the camera to see the current state of the workspace.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        },
                    },
                ]
            }

            with open(self.output_dir / "tools.json", "w") as f:
                json.dump(tool_definitions, f, indent=2)

            # Enhanced metadata with tool support information
            metadata = {
                "name": "qwen-finetuned",
                "description": "Fine-tuned model converted to Ollama GGUF format with tool calling support",
                "format": "gguf",
                "created_with": "unsloth",
                "supports_tools": True,
                "available_tools": [
                    "get_object_positions",
                    "move_object_from_to",
                    "reset_arm",
                    "get_ros2_camera_image",
                ],
            }

            with open(self.output_dir / "model_info.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info("Created Ollama files with tool support")

        except Exception as e:
            logger.error(f"Failed to create Ollama files: {e}")
            raise

    def _print_usage_instructions(self):
        """Print usage instructions"""
        print("\nTo use this model with Ollama:")
        print(f"1. cd {self.output_dir}")
        print("2. ollama create qwen-finetuned -f Modelfile")
        print("3. ollama run qwen-finetuned")

    def _cleanup_temp_files(self, *temp_paths):
        """Clean up temporary files and directories"""
        try:
            for temp_path in temp_paths:
                if temp_path and temp_path.exists():
                    if temp_path.is_dir():
                        shutil.rmtree(temp_path.parent)  # Remove the temp directory
                    logger.info(f"Cleaned up temporary files: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert fine-tuned model to Ollama format using Unsloth"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory containing the LoRA adapter (adapter_config.json, adapter_model.safetensors)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        help="Path to the local base model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--merged-model-path",
        type=str,
        help="Path to pre-merged model directory (skips LoRA merging step)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="qwen_finetuned_ollama",
        help="Output directory for Ollama format model (default: qwen_finetuned_ollama)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Validate arguments
        if args.merged_model_path:
            if args.input_dir or args.base_model_path:
                print(
                    "Warning: --merged-model-path provided, ignoring --input-dir and --base-model-path"
                )
        else:
            if not args.input_dir or not args.base_model_path:
                print(
                    "Error: Either --merged-model-path OR both --input-dir and --base-model-path must be provided"
                )
                exit(1)

        # Create converter and run conversion
        converter = OllamaConverter(
            input_dir=args.input_dir,
            base_model_path=args.base_model_path,
            output_dir=args.output_dir,
            merged_model_path=args.merged_model_path,
        )
        success = converter.convert()

        if success:
            print("\nConversion completed successfully!")
        else:
            print("\n❌ Conversion failed. Check the logs for details.")
            exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
