"""Configuration constants for cllama."""

from pathlib import Path


# my cwd is ~/strix-halo-testing/llm-bench/cllama
# Directory paths (relative to current working directory)
MODELS_DIR = Path("./models")
UPDATE_SCRIPT = Path("./update-llama.cpp.sh")

# Backend name to path mapping
BACKENDS = {
    "vulkan": "./llama.cpp-vulkan/build/bin",
    "hip": "./llama.cpp-hip/build/bin",
    "rocwmma": "./llama.cpp-rocwmma/build/bin",
}

# Default backend
DEFAULT_BACKEND = "vulkan"

# Llama.cpp binaries
LLAMA_SERVER = "llama-server"
LLAMA_CLI = "llama-cli"
