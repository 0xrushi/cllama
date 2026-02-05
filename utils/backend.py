"""Backend detection and binary path resolution."""

from pathlib import Path
from ..config import BACKENDS, LLAMA_SERVER, LLAMA_CLI


def get_backend_path(backend: str) -> Path:
    """Get the binary path for a given backend.

    Args:
        backend: Backend name (vulkan, hip, rocwmma)

    Returns:
        Path to the backend's bin directory

    Raises:
        ValueError: If backend is not supported
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: {', '.join(BACKENDS.keys())}"
        )
    return Path(BACKENDS[backend])


def get_llama_server_path(backend: str) -> Path:
    """Get the path to llama-server binary for a backend.

    Args:
        backend: Backend name

    Returns:
        Full path to llama-server binary
    """
    return get_backend_path(backend) / LLAMA_SERVER


def get_llama_cli_path(backend: str) -> Path:
    """Get the path to llama-cli binary for a backend.

    Args:
        backend: Backend name

    Returns:
        Full path to llama-cli binary
    """
    return get_backend_path(backend) / LLAMA_CLI


def validate_backend(backend: str) -> bool:
    """Check if a backend binary exists.

    Args:
        backend: Backend name

    Returns:
        True if binary exists, False otherwise
    """
    try:
        path = get_llama_server_path(backend)
        return path.exists() and path.is_file()
    except ValueError:
        return False
