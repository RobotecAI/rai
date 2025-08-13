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


import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import unsloth first to ensure optimizations are applied
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported

    UNSLOTH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unsloth not available: {e}")
    print("Falling back to standard transformers training (slower)")
    UNSLOTH_AVAILABLE = False

    # Mock the functions for compatibility
    def is_bfloat16_supported():
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


import torch
from transformers import Trainer, TrainingArguments

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Tool-aware ChatML template for Unsloth
DEFAULT_CHATML_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{% if message.get('tool_calls') %}
{% for tool_call in message['tool_calls'] %}
{"tool_calls": [{"id": "{{ tool_call['id'] }}", "type": "function", "function": {"name": "{{ tool_call['function']['name'] }}", "arguments": {{ tool_call['function']['arguments'] }}}}]}
{% endfor %}
{% else %}
{{ message['content'] }}
{% endif %}<|im_end|>
{% elif message['role'] == 'tool' %}
<|im_start|>tool
{% if message.get('tool_call_id') %}name={{ message.get('name', 'unknown') }}
{% endif %}{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


@dataclass
class FineTuningConfig:
    """Configuration for model fine-tuning"""

    # Model configuration
    model_name: str = "unsloth/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    dtype: str = "bfloat16" if is_bfloat16_supported() else "float16"
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Training configuration
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # Data configuration
    training_data_path: str = "training_data.jsonl"
    validation_split: float = 0.1
    max_samples: Optional[int] = None
    data_format: str = "chatml"  # Options: "sharegpt", "chatml", "alpaca", "raw_text"
    chat_template: Optional[str] = (
        "default"  # "default" for built-in ChatML, or path to custom template file
    )

    # Output configuration
    output_dir: str = "fine_tuned_model"
    save_model: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    # Advanced options
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Hardware optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention_2: bool = (
        True  # Note: Handled automatically by Unsloth when supported
    )
    use_rslora: bool = False  # Note: Handled automatically by Unsloth when supported
    use_unsloth_v2: bool = True  # Note: Handled automatically by Unsloth when supported


class TrainingDataLoader:
    """Load and prepare training data for fine-tuning with Unsloth format support"""

    def __init__(self, config: FineTuningConfig):
        self.config = config

    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from JSONL file"""
        training_data = []

        if not os.path.exists(self.config.training_data_path):
            raise FileNotFoundError(
                f"Training data file not found: {self.config.training_data_path}"
            )

        with open(self.config.training_data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        training_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue

        logger.info(f"Loaded {len(training_data)} training examples")
        return training_data

    def load_custom_chat_template(self, template_path: str) -> str:
        """Load a custom chat template from a file"""
        if not template_path or not os.path.exists(template_path):
            return None

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read().strip()
            logger.info(f"Loaded custom chat template from {template_path}")
            return template
        except Exception as e:
            logger.error(f"Failed to load chat template from {template_path}: {e}")
            return None

    def apply_chat_template(
        self, data: List[Dict[str, Any]], tokenizer
    ) -> List[Dict[str, Any]]:
        """Apply chat template to format data for training"""
        if not self.config.chat_template:
            logger.info("No custom chat template specified, using model's default")
            return data

        # Load custom template if it's a file path
        template_content = self.load_custom_chat_template(self.config.chat_template)
        if template_content:
            # Set the custom template on the tokenizer
            tokenizer.chat_template = template_content
        elif self.config.chat_template == "default":
            # Use the default ChatML template if "default" is specified
            tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE

        logger.info("Applying chat template to format data")

        formatted_data = []
        for item in data:
            try:
                # Apply the chat template to each conversation
                if "conversations" in item:
                    # ShareGPT format
                    formatted_text = tokenizer.apply_chat_template(
                        item["conversations"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    formatted_data.append({"text": formatted_text})
                elif "messages" in item:
                    # Direct messages format
                    formatted_text = tokenizer.apply_chat_template(
                        item["messages"], tokenize=False, add_generation_prompt=False
                    )
                    formatted_data.append({"text": formatted_text})
                else:
                    # Keep original format if not recognized
                    formatted_data.append(item)

            except Exception as e:
                logger.warning(f"Failed to apply chat template to item: {e}")
                formatted_data.append(item)

        logger.info(f"Applied chat template to {len(formatted_data)} examples")
        return formatted_data

    def convert_to_sharegpt_format(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert data to ShareGPT format for multi-turn conversations"""
        if self.config.data_format == "sharegpt":
            return data

        logger.info(
            f"Converting data from {self.config.data_format} to ShareGPT format"
        )

        if self.config.data_format == "alpaca":
            # Convert Alpaca format to ShareGPT
            converted_data = []
            for item in data:
                conversation = [
                    {
                        "from": "human",
                        "value": item.get("instruction", "")
                        + "\n"
                        + item.get("input", ""),
                    },
                    {"from": "gpt", "value": item.get("output", "")},
                ]
                converted_data.append({"conversations": conversation})
            return converted_data

        elif self.config.data_format == "raw_text":
            # Convert raw text to single-turn conversation
            converted_data = []
            for item in data:
                conversation = [
                    {"from": "human", "value": "Please continue the following text:"},
                    {"from": "gpt", "value": item.get("text", "")},
                ]
                converted_data.append({"conversations": conversation})
            return converted_data

        return data

    def split_train_validation(
        self, data: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into training and validation sets"""
        if self.config.validation_split <= 0:
            return data, []

        split_idx = int(len(data) * (1 - self.config.validation_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        logger.info(
            f"Split data: {len(train_data)} training, {len(val_data)} validation"
        )
        return train_data, val_data

    def limit_samples(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Limit the number of samples if specified"""
        if self.config.max_samples and len(data) > self.config.max_samples:
            data = data[: self.config.max_samples]
            logger.info(f"Limited to {len(data)} samples")

        return data

    def prepare_dataset(
        self, tokenizer
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Complete dataset preparation pipeline"""
        # Load raw data
        raw_data = self.load_training_data()

        # Convert to ShareGPT format if needed
        formatted_data = self.convert_to_sharegpt_format(raw_data)

        # Apply chat template if specified
        if self.config.chat_template:
            formatted_data = self.apply_chat_template(formatted_data, tokenizer)

        # Limit samples if specified
        formatted_data = self.limit_samples(formatted_data)

        # Split into train/validation
        train_data, val_data = self.split_train_validation(formatted_data)

        return train_data, val_data


class ModelManager:
    """Manage model loading, configuration, and saving"""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Determine dtype
        dtype = getattr(torch, self.config.dtype)

        if UNSLOTH_AVAILABLE:
            # Load model with Unsloth optimizations
            # Note: All optimizations (flash attention, RSLoRA, Unsloth v2) are handled automatically
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=dtype,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            )

            # Add LoRA adapters if using PEFT
            if self.config.use_peft:
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=self.config.lora_r,
                    target_modules=self.config.target_modules,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                    random_state=42,
                    use_rslora=self.config.use_rslora,
                    loftq_config=None,
                )
        else:
            # Fallback to standard transformers + PEFT
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.warning("Using standard transformers (slower than Unsloth)")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                device_map="auto",
                use_cache=False,
            )

            # Add LoRA adapters if using PEFT
            if self.config.use_peft:
                peft_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=self.config.target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, peft_config)

        logger.info("Model loaded successfully")
        return self.model, self.tokenizer

    def save_model(self, output_dir: str):
        """Save the fine-tuned model"""
        if not self.config.save_model:
            logger.info("Model saving disabled")
            return

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model to {output_dir}")

        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training configuration
        config_dict = {
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            "training_config": {
                "num_train_epochs": self.config.num_train_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            },
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
                "target_modules": self.config.target_modules,
            }
            if self.config.use_peft
            else None,
        }

        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Model saved to {output_dir}")

    def push_to_hub(self, hub_model_id: str):
        """Push the model to Hugging Face Hub"""
        if not self.config.push_to_hub:
            return

        logger.info(f"Pushing model to hub: {hub_model_id}")

        try:
            self.model.push_to_hub(hub_model_id)
            self.tokenizer.push_to_hub(hub_model_id)
            logger.info(f"Successfully pushed to hub: {hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")


class FineTuningTrainer:
    """Main fine-tuning trainer class"""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.data_loader = TrainingDataLoader(config)
        self.model_manager = ModelManager(config)

    def prepare_training_data(self):
        """Prepare training and validation data"""
        # Load model first to get tokenizer for data preparation
        model, tokenizer = self.model_manager.load_model()

        # Use the new comprehensive dataset preparation method
        train_data, val_data = self.data_loader.prepare_dataset(tokenizer)

        return train_data, val_data, model, tokenizer

    def create_training_arguments(
        self, val_data: List[Dict[str, Any]]
    ) -> TrainingArguments:
        """Create training arguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps" if val_data else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_data else False,
            metric_for_best_model="eval_loss" if val_data else None,
            greater_is_better=False if val_data else None,
            remove_unused_columns=False,
            push_to_hub=self.config.push_to_hub,
            hub_model_id=self.config.hub_model_id,
            report_to=None,  # Disable wandb/tensorboard for simplicity
        )

    def train(self):
        """Run the complete fine-tuning process"""
        try:
            logger.info("Starting fine-tuning process...")

            # Prepare data
            train_data, val_data, model, tokenizer = self.prepare_training_data()
            if not train_data:
                raise ValueError("No training data available")

            # Create training arguments
            training_args = self.create_training_arguments(val_data)

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data if val_data else None,
                tokenizer=tokenizer,
                data_collator=self._create_data_collator(tokenizer),
            )

            # Train the model
            logger.info("Starting training...")
            trainer.train()

            # Save the model
            if self.config.save_model:
                self.model_manager.save_model(self.config.output_dir)

            # Push to hub if requested
            if self.config.push_to_hub and self.config.hub_model_id:
                self.model_manager.push_to_hub(self.config.hub_model_id)

            logger.info("Fine-tuning completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _create_data_collator(self, tokenizer):
        """Create a data collator for the training data"""

        def collate_fn(batch):
            # Extract text from the batch
            texts = []
            for item in batch:
                if "text" in item:
                    texts.append(item["text"])
                elif "conversations" in item:
                    # Apply chat template if not already done
                    formatted_text = tokenizer.apply_chat_template(
                        item["conversations"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    texts.append(formatted_text)
                else:
                    # Fallback to string representation
                    texts.append(str(item))

            # Tokenize the texts
            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )

            # For language modeling, we need to create labels
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        return collate_fn


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description="Fine-tune a model using Unsloth")

    # Model configuration
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-4B-Instruct-2507",
        help="Base model to fine-tune (default: unsloth/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    # Training configuration
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Training batch size (default: 2)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )

    # Data configuration
    parser.add_argument(
        "--training-data",
        default="training_data.jsonl",
        help="Path to training data file (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (default: all)",
    )
    parser.add_argument(
        "--data-format",
        choices=["sharegpt", "chatml", "alpaca", "raw_text"],
        default="chatml",
        help="Data format for training data (default: chatml)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="default",
        help="Chat template: 'default' for built-in ChatML, or path to custom template file (default: default)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        default="fine_tuned_model",
        help="Output directory for the fine-tuned model (default: fine_tuned_model)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save the fine-tuned model"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push the model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub-model-id", default=None, help="Hugging Face Hub model ID for pushing"
    )

    # LoRA configuration
    parser.add_argument(
        "--lora-r", type=int, default=16, help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)"
    )

    args = parser.parse_args()

    # Create configuration
    config = FineTuningConfig(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        training_data_path=args.training_data,
        validation_split=args.validation_split,
        max_samples=args.max_samples,
        data_format=args.data_format,
        chat_template=args.chat_template,
        output_dir=args.output_dir,
        save_model=not args.no_save,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Run fine-tuning
    trainer = FineTuningTrainer(config)
    success = trainer.train()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
