"""Tests for Hugging Face integration."""

from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pytest
import cllama.utils.hf
from cllama.utils.hf import (
    get_repo_files,
    find_files_by_quant,
    get_local_model_path,
    is_model_downloaded,
)


class TestGetRepoFiles:
    """Test get_repo_files function."""

    @patch("cllama.utils.hf.list_repo_files")
    def test_get_repo_files_success(self, mock_list):
        """Test successful file listing."""
        mock_list.return_value = [
            "model.gguf",
            ".gitignore",
            "README.md",
            "config.json",
        ]

        result = get_repo_files("user/model")
        assert len(result) == 3
        assert ".gitignore" not in result
        assert "model.gguf" in result

    @patch("cllama.utils.hf.list_repo_files")
    def test_get_repo_files_filters_hidden(self, mock_list):
        """Test that files starting with dot are filtered."""
        mock_list.return_value = [
            ".git/config",
            ".gitignore",
            "model.gguf",
            ".env",
        ]

        result = get_repo_files("user/model")
        assert ".gitignore" not in result
        assert ".git/config" not in result
        assert ".env" not in result
        assert "model.gguf" in result

    @patch("cllama.utils.hf.list_repo_files")
    def test_get_repo_files_with_subdirs(self, mock_list):
        """Test file listing with subdirectories."""
        mock_list.return_value = [
            "models/part1.gguf",
            "models/part2.gguf",
            "README.md",
        ]

        result = get_repo_files("user/model")
        assert "models/part1.gguf" in result
        assert "models/part2.gguf" in result

    @patch("cllama.utils.hf.list_repo_files")
    def test_get_repo_files_api_error(self, mock_list):
        """Test error handling when API fails."""
        mock_list.side_effect = Exception("API Error")

        with pytest.raises(SystemExit):
            get_repo_files("user/invalid")

    @patch("cllama.utils.hf.list_repo_files")
    def test_get_repo_files_empty_repo(self, mock_list):
        """Test handling of empty repository."""
        mock_list.return_value = []

        result = get_repo_files("user/model")
        assert result == []


class TestFindFilesByQuant:
    """Test find_files_by_quant function."""

    @patch("cllama.utils.hf.get_repo_files")
    def test_find_files_by_quant_match(self, mock_get_files):
        """Test finding files matching quantization."""
        mock_get_files.return_value = [
            "model-Q4_K_M.gguf",
            "model-Q4_K_M.safetensors",
            "model-Q8_0.gguf",
        ]

        result = find_files_by_quant("user/model", "Q4_K_M")
        assert len(result) == 1
        assert "model-Q4_K_M.gguf" in result

    @patch("cllama.utils.hf.get_repo_files")
    def test_find_files_by_quant_case_insensitive(self, mock_get_files):
        """Test that matching is case insensitive."""
        mock_get_files.return_value = [
            "model-q4_k_m.gguf",
            "model-Q8_0.gguf",
        ]

        result = find_files_by_quant("user/model", "Q4_K_M")
        assert len(result) == 1
        assert "model-q4_k_m.gguf" in result

    @patch("cllama.utils.hf.get_repo_files")
    def test_find_files_by_quant_multiple_matches(self, mock_get_files):
        """Test multiple files matching quantization."""
        mock_get_files.return_value = [
            "model-Q4_K_M-00001.gguf",
            "model-Q4_K_M-00002.gguf",
            "model-Q4_K_M-00003.gguf",
            "model-Q8_0.gguf",
        ]

        result = find_files_by_quant("user/model", "Q4_K_M")
        assert len(result) == 3

    @patch("cllama.utils.hf.get_repo_files")
    def test_find_files_by_quant_no_match(self, mock_get_files):
        """Test when no files match quantization."""
        mock_get_files.return_value = [
            "model-Q8_0.gguf",
            "model-F16.gguf",
        ]

        result = find_files_by_quant("user/model", "Q4_K_M")
        assert result == []

    @patch("cllama.utils.hf.get_repo_files")
    def test_find_files_by_quant_only_gguf(self, mock_get_files):
        """Test that only .gguf files are returned."""
        mock_get_files.return_value = [
            "model-Q4_K_M.gguf",
            "model-Q4_K_M.safetensors",
            "README.md",
        ]

        result = find_files_by_quant("user/model", "Q4_K_M")
        assert len(result) == 1
        assert all(f.endswith(".gguf") for f in result)


class TestGetLocalModelPath:
    """Test get_local_model_path function from hf module."""

    def test_get_local_model_path_none_when_missing(self, temp_models_dir):
        """Test that None is returned for missing models."""
        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path("user/missing")
            assert result is None

    def test_get_local_model_path_flat_structure(self, temp_models_dir):
        """Test with flat file structure."""
        (temp_models_dir / "mymodel-Q4_K_M.gguf").touch()

        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path("user/mymodel")
            assert result == temp_models_dir

    def test_get_local_model_path_subdirectory(self, temp_models_dir):
        """Test with subdirectory structure."""
        model_subdir = temp_models_dir / "mymodel"
        model_subdir.mkdir()
        (model_subdir / "model.gguf").touch()

        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path("user/mymodel")
            assert result == model_subdir

    def test_get_local_model_path_with_quant_flat(self, temp_models_dir):
        """Test quantization filtering with flat structure."""
        (temp_models_dir / "mymodel-Q4_K_M.gguf").touch()
        (temp_models_dir / "mymodel-Q8_0.gguf").touch()

        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path("user/mymodel", quant="Q4_K_M")
            assert result == temp_models_dir

    def test_get_local_model_path_with_quant_no_match(self, temp_models_dir):
        """Test quantization filtering when no match."""
        (temp_models_dir / "mymodel-Q4_K_M.gguf").touch()

        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path("user/mymodel", quant="Q8_0")
            assert result is None

    def test_get_local_model_path_ignores_non_gguf_with_quant(self, temp_models_dir):
        """Test that non-GGUF files are ignored even with quant match."""
        (temp_models_dir / "mymodel-Q4_K_M.safetensors").touch()
        # No actual GGUF file

        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path("user/mymodel", quant="Q4_K_M")
            assert result is None


class TestIsModelDownloaded:
    """Test is_model_downloaded function."""

    def test_is_model_downloaded_true(self, temp_models_dir):
        """Test when model is downloaded."""
        (temp_models_dir / "mymodel-Q4_K_M.gguf").touch()

        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = is_model_downloaded("user/mymodel")
            assert result is True

    def test_is_model_downloaded_false(self, temp_models_dir):
        """Test when model is not downloaded."""
        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = is_model_downloaded("user/missing")
            assert result is False

    def test_is_model_downloaded_with_quant(self, temp_models_dir):
        """Test with specific quantization."""
        (temp_models_dir / "mymodel-Q4_K_M.gguf").touch()

        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            assert is_model_downloaded("user/mymodel", quant="Q4_K_M") is True
            assert is_model_downloaded("user/mymodel", quant="Q8_0") is False

    def test_is_model_downloaded_subdirectory(self, temp_models_dir):
        """Test with subdirectory structure."""
        model_subdir = temp_models_dir / "mymodel"
        model_subdir.mkdir()
        (model_subdir / "model.gguf").touch()

        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = is_model_downloaded("user/mymodel")
            assert result is True

    def test_is_model_downloaded_case_insensitive(self, temp_models_dir):
        """Test that quantization matching is case insensitive."""
        (temp_models_dir / "mymodel-q4_k_m.gguf").touch()

        with patch.object(cllama.utils.hf, "MODELS_DIR", temp_models_dir):
            result = is_model_downloaded("user/mymodel", quant="Q4_K_M")
            assert result is True
