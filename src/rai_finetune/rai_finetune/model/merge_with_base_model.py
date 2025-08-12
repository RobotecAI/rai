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

"""
The script merges the LoRA adapter back into the base model to create a complete, standalone fine-tuned model.
It uses Unsloth's native merging capabilities for optimal performance and compatibility.
It saves the merged model in standard HuggingFace format (safetensors/PyTorch) with proper tokenizer and configuration files.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Import Unsloth for LoRA merging (preferred approach)
try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# Import transformers as fallback for LoRA merging
try:
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log warnings
if not UNSLOTH_AVAILABLE:
    logger.warning(
        "Unsloth not available - will fallback to transformers for LoRA merging"
    )
    if not TRANSFORMERS_AVAILABLE:
        logger.error(
            "Neither Unsloth nor transformers available - LoRA merging will not work"
        )


class LoRAMerger:
    """Merge LoRA adapters with base model to create a complete fine-tuned model.

    This class uses Unsloth's native merging capabilities when available for optimal
    performance with Unsloth-trained models, with transformers as a fallback.
    """

    def __init__(
        self,
        adapter_dir: str,
        base_model_path: Optional[str] = None,
        output_dir: str = "merged_model",
    ):
        """
        Initialize the LoRA merger

        Args:
            adapter_dir: Path to the LoRA adapter directory (contains adapter_config.json, adapter_model.safetensors)
            base_model_path: Path to the local base model directory or HuggingFace model ID.
                           If None, will attempt to auto-detect from adapter config
            output_dir: Path to save the merged model
        """
        self.adapter_dir = Path(adapter_dir)
        self.output_dir = Path(output_dir)
        self._resolved_base_model_path = None
        self._base_model_id = None

        # Validate adapter directory
        if not self.adapter_dir.exists():
            raise ValueError(f"Adapter directory does not exist: {adapter_dir}")

        # Resolve base model path
        self._resolved_base_model_path, self._base_model_id = (
            self._resolve_base_model_path(base_model_path)
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized LoRA merger")
        logger.info(f"Adapter directory: {self.adapter_dir}")
        logger.info(f"Resolved base model path: {self._resolved_base_model_path}")
        if self._base_model_id:
            logger.info(f"Base model ID: {self._base_model_id}")
        logger.info(f"Output directory: {self.output_dir}")

        # Log which merging approach will be used
        if UNSLOTH_AVAILABLE:
            logger.info("Using Unsloth's native LoRA merging.")
        elif TRANSFORMERS_AVAILABLE:
            logger.info("Using transformers/PEFT for LoRA merging as a fallback.")
        else:
            raise ImportError(
                "Neither Unsloth nor transformers available for LoRA merging"
            )

    @property
    def base_model_path(self) -> Path:
        """Get the resolved base model path"""
        return self._resolved_base_model_path

    def _resolve_base_model_path(
        self, base_model_path: Optional[str]
    ) -> Tuple[Path, Optional[str]]:
        """
        Resolve the base model path from various sources

        Args:
            base_model_path: User-provided path or None for auto-detection

        Returns:
            Tuple of (resolved_path, model_id)
        """
        # If user provided a path, validate and use it
        if base_model_path:
            path = Path(base_model_path)

            # Handle HuggingFace cache paths with "latest" symlink
            if "snapshots" in str(path) and path.name == "latest":
                # Try to resolve the latest symlink or find the actual snapshot
                parent_dir = path.parent
                if parent_dir.exists():
                    snapshots = [
                        d
                        for d in parent_dir.iterdir()
                        if d.is_dir() and d.name != "latest"
                    ]
                    if snapshots:
                        # Use the latest snapshot by name (usually commit hash)
                        latest_snapshot = sorted(snapshots)[-1]
                        logger.info(
                            f"Resolved 'latest' to actual snapshot: {latest_snapshot}"
                        )
                        return latest_snapshot, None

            if path.exists():
                logger.info(f"Using provided local base model path: {path}")
                return path, None
            else:
                # Check if it's a HuggingFace model ID (not a file path)
                if not str(base_model_path).startswith("/") and not str(
                    base_model_path
                ).startswith("~"):
                    logger.info(
                        f"Provided path doesn't exist locally, treating as HuggingFace model ID: {base_model_path}"
                    )
                    if UNSLOTH_AVAILABLE or TRANSFORMERS_AVAILABLE:
                        return self._download_huggingface_model(
                            base_model_path
                        ), base_model_path
                    else:
                        raise ValueError(
                            f"Model ID provided but can't download: {base_model_path}"
                        )
                else:
                    # It's a local path that doesn't exist
                    raise ValueError(f"Local path does not exist: {base_model_path}")

        # Auto-detect from adapter config
        logger.info("Auto-detecting base model from adapter configuration...")
        base_model_name = self._extract_base_model_from_adapter()

        if not base_model_name:
            raise ValueError(
                "Could not determine base model path. Please provide --base-model-path"
            )

        # Check if it's a local path
        if Path(base_model_name).exists():
            logger.info(f"Found local base model path: {base_model_name}")
            return Path(base_model_name), None

        # Check HuggingFace cache
        cached_path = self._find_in_huggingface_cache(base_model_name)
        if cached_path:
            logger.info(f"Found base model in HuggingFace cache: {cached_path}")
            return cached_path, base_model_name

        # Download from HuggingFace
        logger.info(f"Downloading base model from HuggingFace: {base_model_name}")
        return self._download_huggingface_model(base_model_name), base_model_name

    def _extract_base_model_from_adapter(self) -> Optional[str]:
        """Extract base model name from adapter configuration"""
        try:
            # Check adapter_config.json
            adapter_config_path = self.adapter_dir / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path")
                if base_model:
                    return base_model

            # Check training_config.json
            training_config_path = self.adapter_dir / "training_config.json"
            if training_config_path.exists():
                with open(training_config_path, "r") as f:
                    training_config = json.load(f)
                base_model = training_config.get("model_name_or_path")
                if base_model and base_model != "unknown":
                    return base_model

            return None

        except Exception as e:
            logger.warning(f"Failed to extract base model from adapter config: {e}")
            return None

    def _find_in_huggingface_cache(self, model_name: str) -> Optional[Path]:
        """Find model in HuggingFace cache directory"""
        try:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            if not cache_dir.exists():
                return None

            model_name_clean = model_name.replace("/", "--")
            cache_path = cache_dir / f"models--{model_name_clean}"

            if cache_path.exists():
                snapshots_dir = cache_path / "snapshots"
                if snapshots_dir.exists():
                    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    if snapshots:
                        # Use the latest snapshot
                        latest_snapshot = sorted(snapshots)[-1]
                        return latest_snapshot

            return None

        except Exception as e:
            logger.warning(f"Failed to search HuggingFace cache: {e}")
            return None

    def _download_huggingface_model(self, model_id: str) -> Path:
        """Download model from HuggingFace Hub"""
        try:
            if not (UNSLOTH_AVAILABLE or TRANSFORMERS_AVAILABLE):
                raise ImportError(
                    "Neither Unsloth nor transformers available for downloading models"
                )

            logger.info(f"Downloading {model_id} from HuggingFace Hub...")
            cache_dir = snapshot_download(repo_id=model_id, cache_dir=None)
            logger.info(f"Downloaded to: {cache_dir}")
            return Path(cache_dir)

        except Exception as e:
            raise ValueError(f"Failed to download model {model_id}: {e}")

    def merge(self, save_method: str = "merged_16bit") -> bool:
        """
        Merge LoRA weights with base model and save in HuggingFace format

        Args:
            save_method: Merge method for Unsloth ('merged_16bit', 'merged_4bit_forced', 'lora')
                        Ignored when using transformers fallback

        Returns:
            bool: True if merging successful, False otherwise
        """
        try:
            logger.info("Starting LoRA weight merging...")

            # Load model configuration for validation
            config = self._load_adapter_config()
            if not config:
                logger.error("Failed to load adapter configuration")
                return False

            # Validate base model path contents (for local paths)
            if (
                self._resolved_base_model_path.exists()
                and not self._validate_base_model_path()
            ):
                logger.warning(
                    f"Base model path validation failed: {self.base_model_path}"
                )
                logger.info("Proceeding with merge attempt using model ID")

            # Attempt Unsloth merging first (recommended)
            if UNSLOTH_AVAILABLE:
                success = self._merge_with_unsloth(save_method)
                if success:
                    logger.info("LoRA merging completed successfully using Unsloth!")
                    logger.info(f"Merged model saved to: {self.output_dir}")
                    return True
                else:
                    logger.warning(
                        "Unsloth merging failed, falling back to transformers..."
                    )

            # Fallback to transformers/PEFT merging
            if TRANSFORMERS_AVAILABLE:
                success = self._merge_with_transformers()
                if success:
                    logger.info(
                        "LoRA merging completed successfully using transformers!"
                    )
                    logger.info(f"Merged model saved to: {self.output_dir}")
                    return True

            logger.error("All merging methods failed")
            return False

        except Exception as e:
            logger.error(f"❌ LoRA merging failed: {e}")
            return False

    def _merge_with_unsloth(self, save_method: str) -> bool:
        """Merge LoRA weights using Unsloth's native capabilities"""
        try:
            logger.info("Attempting LoRA merge using Unsloth...")

            # Determine model source (prefer model ID over local path for Unsloth)
            model_source = (
                self._base_model_id
                if self._base_model_id
                else str(self.base_model_path)
            )

            logger.info(
                f"Loading model and LoRA adapter with Unsloth from: {model_source}"
            )

            # Load model with LoRA adapter using Unsloth
            # First try to load from the adapter directory, but if that fails, load from base model
            try:
                logger.info(
                    "Attempting to load model and tokenizer from adapter directory..."
                )
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(self.adapter_dir),  # Path to LoRA adapter
                    max_seq_length=2048,  # Reasonable default
                    dtype=None,  # Auto-detect
                    load_in_4bit=True,  # Use 4-bit to save memory
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"Failed to load from adapter directory: {e}")
                logger.info(
                    "Falling back to loading from base model and applying adapter..."
                )

                # Load base model first with memory optimizations
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(self.base_model_path),  # Path to base model
                    max_seq_length=2048,  # Reasonable default
                    dtype=None,  # Auto-detect
                    load_in_4bit=True,  # Use 4-bit to save memory
                    trust_remote_code=True,
                )

                # Apply LoRA adapter manually
                logger.info("Applying LoRA adapter manually...")
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=16,  # Default LoRA rank
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    lora_alpha=32,
                    lora_dropout=0.1,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=42,
                    use_rslora=False,
                    loftq_config=None,
                )

                # Load the LoRA weights
                model.load_adapter(str(self.adapter_dir))

            # Enable inference mode for merging
            FastLanguageModel.for_inference(model)

            # Save merged model using Unsloth's native method
            logger.info(f"Saving merged model with method: {save_method}")
            model.save_pretrained_merged(
                str(self.output_dir),
                tokenizer,
                save_method=save_method,
            )

            # Copy additional files from adapter directory
            self._copy_additional_files()

            logger.info("Unsloth LoRA merging completed successfully")
            return True

        except Exception as e:
            logger.error(f"Unsloth merging failed: {e}")
            return False

    def _merge_with_transformers(self) -> bool:
        """Fallback: Merge LoRA weights using transformers/PEFT"""
        try:
            logger.info("Attempting LoRA merge using transformers/PEFT...")

            # Create temporary directory for merged model
            temp_dir = Path(tempfile.mkdtemp())
            merged_model_path = temp_dir / "merged_model"
            merged_model_path.mkdir(exist_ok=True)

            # Determine what to use for model loading
            model_source = (
                self._base_model_id
                if self._base_model_id
                else str(self.base_model_path)
            )
            tokenizer_source = model_source  # Try the same source first

            # Load the base model with aggressive memory optimizations
            logger.info(f"Loading base model from: {model_source}")
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float16,  # Force FP16 to save memory
                device_map="cpu",  # Load on CPU first to save GPU memory
                low_cpu_mem_usage=True,
                max_memory={0: "8GB", "cpu": "16GB"},  # More conservative GPU limit
                offload_folder="./tmp_offload",
                trust_remote_code=True,
                use_safetensors=True,
            )

            # Load tokenizer with multiple fallbacks
            tokenizer = None
            tokenizer_sources = [
                tokenizer_source,  # Base model source
                str(self.adapter_dir),  # Adapter directory (likely has tokenizer files)
                self._base_model_id
                if self._base_model_id
                else None,  # Original model ID
                str(self.base_model_path)
                if self._base_model_id
                and str(self.base_model_path) != self._base_model_id
                else None,
            ]

            for source in tokenizer_sources:
                if source is None:
                    continue
                try:
                    logger.info(f"Trying to load tokenizer from: {source}")
                    tokenizer = AutoTokenizer.from_pretrained(source)
                    logger.info(f"Successfully loaded tokenizer from: {source}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer from {source}: {e}")
                    continue

            if tokenizer is None:
                raise ValueError("Could not load tokenizer from any source")

            # Load LoRA adapter
            logger.info("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(model, str(self.adapter_dir))

            # Merge LoRA weights with base model
            logger.info("Merging LoRA weights...")
            model = model.merge_and_unload()

            # Save merged model
            logger.info("Saving merged model...")
            model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)

            # Copy merged model to output directory
            self._copy_merged_model(merged_model_path)

            # Copy additional files from adapter directory
            self._copy_additional_files()

            # Clean up temporary files and memory
            self._cleanup_temp_files(temp_dir)
            self._cleanup_memory()

            logger.info("Transformers LoRA merging completed successfully")
            return True

        except Exception as e:
            logger.error(f"Transformers merging failed: {e}")
            return False

    def _load_adapter_config(self) -> Optional[Dict[str, Any]]:
        """Load adapter configuration from adapter directory"""
        try:
            # Load adapter config
            adapter_config_path = self.adapter_dir / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
            else:
                logger.error("No adapter_config.json found in adapter directory")
                return None

            # Load training config if available
            training_config_path = self.adapter_dir / "training_config.json"
            if training_config_path.exists():
                with open(training_config_path, "r") as f:
                    training_config = json.load(f)
            else:
                training_config = {}

            # Combine configurations
            config = {
                "adapter_config": adapter_config,
                "training_config": training_config,
                "base_model": str(self.base_model_path),
                "base_model_id": self._base_model_id,
                "model_type": adapter_config.get("model_type", "unknown"),
                "peft_type": adapter_config.get("peft_type", "LORA"),
                "lora_config": {
                    "r": adapter_config.get("lora_r", 16),
                    "alpha": adapter_config.get("lora_alpha", 32),
                    "dropout": adapter_config.get("lora_dropout", 0.1),
                    "target_modules": adapter_config.get("target_modules", []),
                }
                if adapter_config.get("peft_type") == "LORA"
                else None,
            }

            logger.info(f"Loaded configuration for base model: {config['base_model']}")
            return config

        except Exception as e:
            logger.error(f"Failed to load adapter configuration: {e}")
            return None

    def _validate_base_model_path(self) -> bool:
        """Validate that the base model path contains required files"""
        try:
            # Skip validation if path doesn't exist (likely a model ID)
            if not self.base_model_path.exists():
                return True

            # Check for config.json
            if not (self.base_model_path / "config.json").exists():
                logger.error(
                    f"Missing config.json in base model path: {self.base_model_path}"
                )
                return False

            # Check for model weights - support various formats
            model_weight_patterns = [
                "pytorch_model.bin",  # Single file format
                "model.safetensors",  # Single safetensor format
                "pytorch_model-*.bin",  # Sharded pytorch format
                "model-*.safetensors",  # Sharded safetensor format
            ]

            has_model_weights = False
            for pattern in model_weight_patterns:
                matches = list(self.base_model_path.glob(pattern))
                if matches:
                    logger.info(f"Found model weights: {[f.name for f in matches]}")
                    has_model_weights = True
                    break

            if not has_model_weights:
                logger.error(
                    f"No model weights found in base model path: {self.base_model_path}"
                )
                logger.error(
                    f"Expected files matching patterns: {model_weight_patterns}"
                )
                return False

            # Check for tokenizer files - these can be missing if we have a HuggingFace model ID
            tokenizer_files = [
                "tokenizer.json",
                "vocab.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ]
            has_tokenizer = any(
                (self.base_model_path / f).exists() for f in tokenizer_files
            )

            if not has_tokenizer:
                if self._base_model_id:
                    logger.info(
                        f"Local tokenizer files not found, but can fallback to HuggingFace model ID: {self._base_model_id}"
                    )
                else:
                    logger.warning(
                        f"Limited tokenizer files found in base model path: {self.base_model_path}"
                    )
                    logger.warning(f"Expected some of: {tokenizer_files}")

            logger.info(f"Base model path validated: {self.base_model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to validate base model path: {e}")
            return False

    def _copy_merged_model(self, merged_model_path: Path):
        """Copy merged model files to output directory (for transformers fallback)"""
        try:
            logger.info("Copying merged model to output directory...")

            # Copy all files from merged model
            for item in merged_model_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.output_dir / item.name)
                    logger.info(f"Copied {item.name}")
                elif item.is_dir():
                    shutil.copytree(
                        item, self.output_dir / item.name, dirs_exist_ok=True
                    )
                    logger.info(f"Copied {item.name} directory")

            logger.info("Merged model files copied successfully")

        except Exception as e:
            logger.error(f"Failed to copy merged model files: {e}")
            raise

    def _copy_additional_files(self):
        """Copy additional files from adapter directory if they exist"""
        try:
            # Copy additional files from adapter directory if they exist
            additional_files = [
                "training_config.json",
                "README.md",
                "chat_template.jinja",
            ]
            for file_name in additional_files:
                src_path = self.adapter_dir / file_name
                if src_path.exists():
                    shutil.copy2(src_path, self.output_dir / file_name)
                    logger.info(f"Copied {file_name}")
        except Exception as e:
            logger.warning(f"Failed to copy additional files: {e}")

    def _cleanup_temp_files(self, *temp_paths):
        """Clean up temporary files and directories"""
        try:
            for temp_path in temp_paths:
                if temp_path and temp_path.exists():
                    if temp_path.is_dir():
                        shutil.rmtree(temp_path)
                    else:
                        temp_path.unlink()
                    logger.info(f"Cleaned up temporary files: {temp_path}")

            # Clean up offload folder if it exists
            offload_folder = Path("./tmp_offload")
            if offload_folder.exists():
                shutil.rmtree(offload_folder)
                logger.info("Cleaned up offload folder")

        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

    def _cleanup_memory(self):
        """Clean up GPU and system memory"""
        try:
            import gc

            import torch

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared GPU memory cache")

        except Exception as e:
            logger.warning(f"Failed to clean up memory: {e}")


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with base model using Unsloth"
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Directory containing the LoRA adapter (adapter_config.json, adapter_model.safetensors)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        help="Path to the local base model directory or HuggingFace model ID. If not provided, will auto-detect from adapter config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="merged_model",
        help="Output directory for the merged model (default: merged_model)",
    )
    parser.add_argument(
        "--save-method",
        type=str,
        default="merged_16bit",
        choices=["merged_16bit", "merged_4bit_forced", "lora"],
        help="Merge method for Unsloth (default: merged_16bit). Choices: merged_16bit, merged_4bit_forced, lora",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Create merger and run merging
        merger = LoRAMerger(args.adapter_dir, args.base_model_path, args.output_dir)
        success = merger.merge(save_method=args.save_method)

        if success:
            print("\nLoRA merging completed successfully!")
            print(f"Merged model saved to: {args.output_dir}")
            print("\nThe model is now ready for use in HuggingFace format")
        else:
            print("\n❌ LoRA merging failed. Check the logs for details.")
            exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
