"""Configuration constants for cllama."""

import os
from pathlib import Path


# Base path for all relative paths (can be overridden via CLLAMA_BASE_PATH env var)
# Defaults to ~/strix-halo-testing/llm-bench/cllama
BASE_PATH = Path(os.getenv("CLLAMA_BASE_PATH", Path.cwd()))


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a relative path to an absolute path using BASE_PATH.

    Args:
        relative_path: Relative path string or Path object

    Returns:
        Absolute Path object
    """
    return (BASE_PATH / relative_path).resolve()


# Directory paths (relative to BASE_PATH)
MODELS_DIR_REL = "models"
UPDATE_SCRIPT_REL = "update-llama.cpp.sh"

# Resolved absolute paths
MODELS_DIR = resolve_path(MODELS_DIR_REL)
UPDATE_SCRIPT = resolve_path(UPDATE_SCRIPT_REL)

# Backend name to relative path mapping
BACKENDS_REL = {
    "vulkan": "llama.cpp-vulkan/build/bin",
    "hip": "llama.cpp-hip/build/bin",
    "rocwmma": "llama.cpp-rocwmma/build/bin",
}

# Backend name to absolute path mapping
BACKENDS = {name: str(resolve_path(path)) for name, path in BACKENDS_REL.items()}

# Default backend
DEFAULT_BACKEND = "vulkan"

# Llama.cpp binaries
LLAMA_SERVER = "llama-server"
LLAMA_CLI = "llama-cli"
