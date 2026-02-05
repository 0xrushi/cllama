"""Tests for backend module."""

from pathlib import Path
import pytest
from cllama.utils.backend import (
    get_backend_path,
    get_llama_server_path,
    get_llama_cli_path,
    validate_backend,
)


class TestGetBackendPath:
    """Test get_backend_path function."""

    def test_get_backend_path_vulkan(self):
        """Test getting vulkan backend path."""
        result = get_backend_path("vulkan")
        assert isinstance(result, Path)
        assert "vulkan" in str(result).lower()

    def test_get_backend_path_hip(self):
        """Test getting hip backend path."""
        result = get_backend_path("hip")
        assert isinstance(result, Path)
        assert "hip" in str(result).lower()

    def test_get_backend_path_rocwmma(self):
        """Test getting rocwmma backend path."""
        result = get_backend_path("rocwmma")
        assert isinstance(result, Path)
        assert "rocwmma" in str(result).lower()

    def test_get_backend_path_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_backend_path("invalid_backend")
        assert "Unsupported backend" in str(exc_info.value)
        assert "invalid_backend" in str(exc_info.value)

    def test_get_backend_path_case_sensitive(self):
        """Test that backend names are case sensitive."""
        with pytest.raises(ValueError):
            get_backend_path("VULKAN")  # Should be lowercase

    def test_get_backend_path_returns_absolute(self):
        """Test that returned paths are absolute."""
        for backend in ["vulkan", "hip", "rocwmma"]:
            result = get_backend_path(backend)
            assert result.is_absolute()


class TestGetLlamaServerPath:
    """Test get_llama_server_path function."""

    def test_get_llama_server_path_structure(self):
        """Test that llama-server path has correct structure."""
        result = get_llama_server_path("vulkan")
        assert isinstance(result, Path)
        assert str(result).endswith("llama-server")
        assert "vulkan" in str(result).lower()

    def test_get_llama_server_path_all_backends(self):
        """Test getting llama-server path for all backends."""
        for backend in ["vulkan", "hip", "rocwmma"]:
            result = get_llama_server_path(backend)
            assert isinstance(result, Path)
            assert "llama-server" in str(result)
            assert backend in str(result)

    def test_get_llama_server_path_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError):
            get_llama_server_path("nonexistent")


class TestGetLlamaCliPath:
    """Test get_llama_cli_path function."""

    def test_get_llama_cli_path_structure(self):
        """Test that llama-cli path has correct structure."""
        result = get_llama_cli_path("vulkan")
        assert isinstance(result, Path)
        assert str(result).endswith("llama-cli")
        assert "vulkan" in str(result).lower()

    def test_get_llama_cli_path_all_backends(self):
        """Test getting llama-cli path for all backends."""
        for backend in ["vulkan", "hip", "rocwmma"]:
            result = get_llama_cli_path(backend)
            assert isinstance(result, Path)
            assert "llama-cli" in str(result)
            assert backend in str(result)

    def test_get_llama_cli_path_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError):
            get_llama_cli_path("nonexistent")


class TestValidateBackend:
    """Test validate_backend function."""

    def test_validate_backend_invalid_backend_name(self):
        """Test validation of invalid backend name."""
        result = validate_backend("nonexistent")
        assert result is False

    def test_validate_backend_missing_binary(self):
        """Test validation when binary doesn't exist."""
        result = validate_backend("vulkan")
        # Will be False since we don't have the binary installed
        assert isinstance(result, bool)

    def test_validate_backend_returns_bool(self):
        """Test that validate_backend always returns a boolean."""
        for backend in ["vulkan", "hip", "rocwmma", "invalid"]:
            result = validate_backend(backend)
            assert isinstance(result, bool)

    def test_validate_backend_with_existing_binary(self, temp_backend_dirs, monkeypatch):
        """Test validation when binary exists."""
        from unittest.mock import patch
        import cllama.utils.backend

        # Mock BACKENDS to point to our temp directory
        mock_backends = {}
        for backend, path in temp_backend_dirs.items():
            mock_backends[backend] = str(path)

        with patch.object(cllama.utils.backend, "BACKENDS", mock_backends):
            result = validate_backend("vulkan")
            assert result is True

    def test_validate_backend_consistency(self):
        """Test that validation results are consistent across calls."""
        result1 = validate_backend("vulkan")
        result2 = validate_backend("vulkan")
        assert result1 == result2
