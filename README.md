# cllama

CLI wrapper for llama.cpp with Hugging Face integration.

## Features

- **Easy backend selection**: Switch between Vulkan, HIP, and ROCWMMA builds
- **Hugging Face integration**: Automatically download models from HF
- **Quantization support**: Download specific quantizations from multi-quant repos
- **Auto-download**: Models are downloaded automatically when running if not present
- **Multi-file support**: Automatically handles sharded models

## Installation

```bash
cd cllama
pip install -e .
```

This installs two commands:
- `cllama` - Main CLI for update, pull, and run commands
- `cllama-cli` - Direct wrapper for llama-cli

## Usage

### Update llama.cpp builds

```bash
cllama update
```

Runs the `update-llama.cpp.sh` script to update all llama.cpp builds.

### Pull a model

```bash
# Pull all .gguf files from a repo
cllama pull lefromage/Qwen3-Next-80B-A3B-Instruct-GGUF

# Pull specific quantization
cllama pull lefromage/Qwen3-Next-80B-A3B-Instruct-GGUF:Q4_K_M

# Or use --quant flag
cllama pull unsloth/GLM-4.7-Flash-GGUF --quant Q4_K_M
```

### Run llama-server

```bash
# Basic usage with auto-download
cllama run lefromage/Qwen3-Next-80B-A3B-Instruct-GGUF:Q4_K_M

# Specify backend
cllama run --backend vulkan lefromage/Qwen3-Next-80B:Q4_K_M --port 8084

# Pass additional arguments to llama-server
cllama run --backend hip lefromage/Qwen3-Next-80B --quant Q4_K_M -ngl 999 --jinja
```

### Run llama-cli

```bash
# Basic usage
cllama-cli lefromage/Qwen3-Next-80B-A3B-Instruct-GGUF:Q4_K_M --prompt "Hello"

# With backend and additional args
cllama-cli --backend vulkan lefromage/Qwen3-Next-80B:Q4_K_M -ngl 3 --prompt "What is AI?"
```

## Model Reference Format

Models can be specified in two ways:

1. `user/repo` - Uses the full repository
2. `user/repo:quant` - Uses specific quantization (e.g., `Q4_K_M`)

The `--quant` flag can also be used to override the quantization.

## Backends

Available backends:
- `vulkan` - Uses `./llama.cpp-vulkan/build/bin/`
- `hip` - Uses `./llama.cpp-hip/build/bin/`
- `rocwmma` - Uses `./llama.cpp-rocwmma/build/bin/`

Default backend: `vulkan`

## Directory Structure

- `./models/` - Downloaded models are stored here
- `./update-llama.cpp.sh` - Update script for llama.cpp builds

## Examples

```bash
# Download and run Q4_K_M quantization of a large model
cllama pull lefromage/Qwen3-Next-80B:Q4_K_M
cllama run --backend vulkan lefromage/Qwen3-Next-80B:Q4_K_M --port 8084 -ngl 3

# Auto-download on run
cllama run unsloth/GLM-4.7-Flash-GGUF --quant Q4_K_M -ngl 999 --jinja

# Use llama-cli for inference
cllama-cli --backend hip lefromage/Qwen3-Next-80B:Q4_K_M -ngl 0 --no-mmap -st --prompt "Hello"
```

## Requirements

- Python 3.8+
- `hf` CLI tool (Hugging Face CLI) installed and configured
- llama.cpp builds in the expected directories

## Configuration

Edit `cllama/config.py` to change:
- Default backend
- Models directory path
- Backend path mappings
