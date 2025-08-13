# RAI Fine-tuning Module

## Module Overview

This module provides tools for extracting observations from tracing data and fine-tuning language models using RAI (Robotic AI) training data with Unsloth for efficient training. It includes:

**Data Preparation:**

-   **Observation Extractor for Langfuse**: Extract observations from Langfuse traces based on user input
-   **Training Data Formatter**: Converts RAI observations to training data format

**Fine-tune Helpers:**

-   **Model Fine-tuning**: Uses Unsloth for optimized training with 4-bit quantization and LoRA support
-   **LoRA Merger**: Merges LoRA adapter weights back into base models for standalone deployment
-   **Ollama Converter**: Converts fine-tuned models to Ollama format using GGUF export

The module is designed as a standalone package to avoid dependency conflicts between different versions of triton required by openai-whisper and unsloth-zoo.

## Environment Setup

This module requires Python 3.11.9 (Python 3.13 does not support Unsloth). Use pyenv to manage Python versions:

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libgdbm-dev \
    libnss3-dev \
    libtinfo6 \
    build-essential
```

### 2. Install Python 3.11.9 with pyenv

```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Add to shell profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload shell or source profile
source ~/.bashrc

# Install Python 3.11.9
pyenv install 3.11.9
```

### 3. Set up Poetry Environment

```bash
cd src/rai_finetune

# Set local Python version
pyenv local 3.11.9

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Create and activate Poetry environment
poetry env use python
poetry install
poetry run pip install flash-attn --no-build-isolation

# Activate the environment
. ./setup_finetune_shall.sh
```

### 4. Install llama.cpp Tools (Optional)

The Ollama conversion process requires the `llama-quantize` tool from llama.cpp. You have two options:

**Option A: Copy from existing installation (Recommended)**

```bash
# If you have llama.cpp built elsewhere, just copy the tool
cp ~/dev/llama.cpp/build/bin/llama-quantize ~/.local/bin/
# or copy to a location in your PATH
```

**Option B: Build from source**

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
# The llama-quantize tool will be in the build/bin directory
```

## Script Execution Flow

### 1. Observation Extraction

Extract observations from Langfuse for specific models:

```bash
python src/rai_finetune/rai_finetune/data/langfuse_obs_extractor.py \
  --models "gpt-4o" "gpt-4o-mini" \
  --output observations.jsonl
```

**Options:**

-   `--models`: List of model names to extract observations from
-   `--output`: Output file for extracted observations

### 2. Training Data Preparation

Convert RAI observations to training data:

For tool calling,

```bash
python src/rai_finetune/rai_finetune/data/tool_calling_data_formatter.py \
    --input observations.jsonl \
    --output tool_calling_training_data.jsonl
```

Process conversations and tool calling:

```bash
python src/rai_finetune/rai_finetune/data/training_data_formatter.py \
    --input observations.jsonl \
    --output training_data.jsonl
```

**Options:**

-   `--input`: Input observations file
-   `--output`: Output training data file
-   `--format`: Output format (default: unsloth)

### 3. Model Fine-tuning

Fine-tune a model with LoRA:

```bash
python src/rai_finetune/rai_finetune/model/trainer.py \
    --training-data tool_calling_training_data.jsonl \
    --output-dir fine_tuned_model
```

**Key Options:**

-   `--model`: Base model (default: unsloth/llama-3-8b-bnb-4bit)
-   `--epochs`: Training epochs (default: 3)
-   `--batch-size`: Batch size (default: 2)
-   `--learning-rate`: Learning rate (default: 2e-4)
-   `--lora-r`: LoRA rank (default: 16)
-   `--lora-alpha`: LoRA alpha (default: 32)
-   `--max-seq-length`: Sequence length (default: 2048)
-   `--validation-split`: Validation ratio (default: 0.1)

### 4. LoRA Weight Merging

Merge LoRA weights into base model:

```bash
python src/rai_finetune/rai_finetune/model/merge_with_base_model.py \
    --adapter-dir ./qwen_finetuned \
    --save-method merged_4bit_forced
```

**Options:**

-   `--adapter-dir`: LoRA adapter directory
-   `--output-dir`: Output directory (default: merged_model)
-   `--save-method`: Merge method (choices: merged_16bit, merged_4bit_forced, lora)

### 5. Ollama Conversion

Convert to Ollama format:

```bash
## make sure llama.cpp is git cloned, built, llama.cpp/llama-quantize exists in pwd before running the script
## ref: https://github.com/unslothai/unsloth/issues/1781
python src/rai_finetune/rai_finetune/model/ollama_model_producer.py \
       --merged-model-path merged_model \
       --output-dir qwen_ollama_test
```

**Options:**

-   `--merged-model-path`: Directory for LoRA and base model merged
-   `--output-dir`: Output directory (default: qwen_finetuned_ollama)

### 6. Ollama Deployment

```bash
cd /path/to/ollama/model
ollama create model-name -f Modelfile
ollama run model-name
```
