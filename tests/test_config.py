"""Tests for config module."""

import os
from pathlib import Path
import pytest
import importlib


class TestConfigPathResolution:
    """Test path resolution with BASE_PATH."""

    def test_resolve_path_with_string(self):
        """Test resolve_path with string argument."""
        from cllama.config import resolve_path, BASE_PATH
        result = resolve_path("models")
        assert result == (BASE_PATH / "models").resolve()

    def test_resolve_path_with_path_object(self):
        """Test resolve_path with Path object."""
        from cllama.config import resolve_path, BASE_PATH
        result = resolve_path(Path("models"))
        assert result == (BASE_PATH / "models").resolve()

    def test_resolve_path_returns_absolute(self):
        """Test that resolve_path returns absolute paths."""
        from cllama.config import resolve_path
        result = resolve_path("models")
        assert result.is_absolute()

    def test_resolve_path_nested(self):
        """Test resolve_path with nested paths."""
        from cllama.config import resolve_path, BASE_PATH
        result = resolve_path("llama.cpp-vulkan/build/bin")
        expected = (BASE_PATH / "llama.cpp-vulkan/build/bin").resolve()
        assert result == expected

    def test_models_dir_is_absolute(self):
        """Test that MODELS_DIR is an absolute path."""
        from cllama.config import MODELS_DIR
        assert MODELS_DIR.is_absolute()

    def test_update_script_is_absolute(self):
        """Test that UPDATE_SCRIPT is an absolute path."""
        from cllama.config import UPDATE_SCRIPT
        assert UPDATE_SCRIPT.is_absolute()

    def test_backends_are_absolute_strings(self):
        """Test that BACKENDS values are absolute path strings."""
        from cllama.config import BACKENDS
        for backend_name, backend_path in BACKENDS.items():
            path_obj = Path(backend_path)
            assert path_obj.is_absolute(), f"{backend_name} path is not absolute: {backend_path}"

    def test_base_path_defaults_to_cwd(self, monkeypatch):
        """Test that BASE_PATH defaults to current working directory when env var not set."""
        # Make sure CLLAMA_BASE_PATH is not set
        monkeypatch.delenv("CLLAMA_BASE_PATH", raising=False)

        # Reimport to pick up the new env state
        import cllama.config
        import importlib
        importlib.reload(cllama.config)

        from cllama.config import BASE_PATH
        # BASE_PATH should be current cwd when CLLAMA_BASE_PATH is not set
        assert BASE_PATH == Path.cwd()

    def test_base_path_from_env_variable(self, monkeypatch, tmp_path):
        """Test that BASE_PATH can be set via environment variable."""
        monkeypatch.setenv("CLLAMA_BASE_PATH", str(tmp_path))
        # Reimport to pick up the new env var
        import cllama.config
        importlib.reload(cllama.config)

        from cllama.config import BASE_PATH
        assert BASE_PATH == tmp_path

    def test_all_paths_relative_to_base_path(self, monkeypatch, tmp_path):
        """Test that all resolved paths are relative to BASE_PATH."""
        monkeypatch.setenv("CLLAMA_BASE_PATH", str(tmp_path))
        import cllama.config
        importlib.reload(cllama.config)

        from cllama.config import MODELS_DIR, BACKENDS, BASE_PATH

        # MODELS_DIR should start with BASE_PATH
        assert str(MODELS_DIR).startswith(str(BASE_PATH))

        # All backends should start with BASE_PATH
        for backend_path in BACKENDS.values():
            assert backend_path.startswith(str(BASE_PATH))

    def test_relative_path_constants_exist(self):
        """Test that relative path constants are defined."""
        from cllama.config import MODELS_DIR_REL, UPDATE_SCRIPT_REL, BACKENDS_REL
        assert isinstance(MODELS_DIR_REL, str)
        assert isinstance(UPDATE_SCRIPT_REL, str)
        assert isinstance(BACKENDS_REL, dict)

    def test_backends_rel_structure(self):
        """Test BACKENDS_REL has expected structure."""
        from cllama.config import BACKENDS_REL
        expected_backends = {"vulkan", "hip", "rocwmma"}
        assert set(BACKENDS_REL.keys()) == expected_backends

        for backend_name, backend_path in BACKENDS_REL.items():
            assert isinstance(backend_path, str)
            assert f"llama.cpp-{backend_name}" in backend_path


class TestConfigConstants:
    """Test configuration constants."""

    def test_default_backend_is_set(self):
        """Test that DEFAULT_BACKEND is set."""
        from cllama.config import DEFAULT_BACKEND
        assert DEFAULT_BACKEND == "vulkan"

    def test_llama_binaries_are_defined(self):
        """Test that LLAMA_SERVER and LLAMA_CLI are defined."""
        from cllama.config import LLAMA_SERVER, LLAMA_CLI
        assert LLAMA_SERVER == "llama-server"
        assert LLAMA_CLI == "llama-cli"

    def test_backends_dict_has_all_keys(self):
        """Test that BACKENDS dict has all backend types."""
        from cllama.config import BACKENDS
        expected = {"vulkan", "hip", "rocwmma"}
        assert set(BACKENDS.keys()) == expected
