"""Pytest fixtures and configuration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest


@pytest.fixture
def temp_base_path():
    """Create a temporary directory to use as BASE_PATH."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_base_path(temp_base_path, monkeypatch):
    """Set CLLAMA_BASE_PATH environment variable to temp directory."""
    monkeypatch.setenv("CLLAMA_BASE_PATH", str(temp_base_path))
    # Reload config module to pick up new env var
    import cllama.config
    import importlib
    importlib.reload(cllama.config)
    yield temp_base_path
    # Reload again to restore original
    importlib.reload(cllama.config)


@pytest.fixture
def temp_models_dir(temp_base_path):
    """Create a models directory in temp_base_path."""
    models_dir = temp_base_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


@pytest.fixture
def temp_backend_dirs(temp_base_path):
    """Create backend directories in temp_base_path."""
    backends = {}
    for backend in ["vulkan", "hip", "rocwmma"]:
        backend_path = temp_base_path / f"llama.cpp-{backend}" / "build" / "bin"
        backend_path.mkdir(parents=True, exist_ok=True)
        # Create dummy llama-server and llama-cli binaries
        (backend_path / "llama-server").touch()
        (backend_path / "llama-cli").touch()
        backends[backend] = backend_path
    return backends
