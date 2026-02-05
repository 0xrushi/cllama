"""Tests for model reference and path resolution."""

from pathlib import Path
from unittest.mock import patch
import pytest
import cllama.utils.models
from cllama.utils.models import (
    ModelReference,
    parse_model_reference,
    get_local_model_path,
    get_model_files,
)


class TestModelReference:
    """Test ModelReference class."""

    def test_model_reference_repo_id_only(self):
        """Test ModelReference with repo_id only."""
        ref = ModelReference("user/model")
        assert ref.repo_id == "user/model"
        assert ref.quant is None

    def test_model_reference_with_quant(self):
        """Test ModelReference with quantization."""
        ref = ModelReference("user/model", "Q4_K_M")
        assert ref.repo_id == "user/model"
        assert ref.quant == "Q4_K_M"

    def test_model_reference_repr_without_quant(self):
        """Test string representation without quantization."""
        ref = ModelReference("user/model")
        assert str(ref) == "user/model"

    def test_model_reference_repr_with_quant(self):
        """Test string representation with quantization."""
        ref = ModelReference("user/model", "Q4_K_M")
        assert str(ref) == "user/model:Q4_K_M"


class TestParseModelReference:
    """Test parse_model_reference function."""

    def test_parse_simple_repo_id(self):
        """Test parsing simple repo ID."""
        result = parse_model_reference("user/model")
        assert result.repo_id == "user/model"
        assert result.quant is None

    def test_parse_repo_id_with_quant(self):
        """Test parsing repo ID with quantization."""
        result = parse_model_reference("user/model:Q4_K_M")
        assert result.repo_id == "user/model"
        assert result.quant == "Q4_K_M"

    def test_parse_repo_id_with_multiple_colons(self):
        """Test parsing when repo ID contains colons (uses rsplit)."""
        result = parse_model_reference("user/model:something:Q4_K_M")
        # rsplit(":", 1) splits from the right, so last part is quant
        assert result.repo_id == "user/model:something"
        assert result.quant == "Q4_K_M"

    def test_parse_quant_override(self):
        """Test that quant_override parameter works."""
        result = parse_model_reference("user/model:Q4_K_M", quant_override="Q8_0")
        assert result.repo_id == "user/model"
        assert result.quant == "Q8_0"

    def test_parse_quant_override_without_original_quant(self):
        """Test quant_override when no quant in original string."""
        result = parse_model_reference("user/model", quant_override="Q4_K_M")
        assert result.repo_id == "user/model"
        assert result.quant == "Q4_K_M"

    def test_parse_repo_with_numbers(self):
        """Test parsing repo ID with numbers."""
        result = parse_model_reference("user/model123:Q5_K")
        assert result.repo_id == "user/model123"
        assert result.quant == "Q5_K"

    def test_parse_various_quant_formats(self):
        """Test parsing various quantization formats."""
        quants = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K", "Q6_K", "Q8_0", "F16"]
        for quant in quants:
            result = parse_model_reference(f"user/model:{quant}")
            assert result.quant == quant

    def test_parse_returns_model_reference(self):
        """Test that parse_model_reference returns ModelReference instance."""
        result = parse_model_reference("user/model")
        assert isinstance(result, ModelReference)


class TestGetLocalModelPath:
    """Test get_local_model_path function."""

    def test_get_local_model_path_missing_model(self, temp_models_dir):
        """Test getting path for non-existent model raises FileNotFoundError."""
        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            with pytest.raises(FileNotFoundError):
                get_local_model_path(ModelReference("user/missing"))

    def test_get_local_model_path_flat_file_structure(self, temp_models_dir):
        """Test getting path for model stored as flat files."""
        # Create a model file in models directory
        (temp_models_dir / "mymodel-Q4_K_M.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path(ModelReference("user/mymodel"))
            assert result == temp_models_dir

    def test_get_local_model_path_subdirectory_structure(self, temp_models_dir):
        """Test getting path for model in subdirectory."""
        model_subdir = temp_models_dir / "mymodel"
        model_subdir.mkdir()
        (model_subdir / "model.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path(ModelReference("user/mymodel"))
            assert result == model_subdir

    def test_get_local_model_path_with_quant_match(self, temp_models_dir):
        """Test getting path when quantization matches."""
        model_subdir = temp_models_dir / "mymodel"
        model_subdir.mkdir()
        (model_subdir / "model-Q4_K_M.gguf").touch()
        (model_subdir / "model-Q8_0.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path(ModelReference("user/mymodel", "Q4_K_M"))
            assert result == model_subdir

    def test_get_local_model_path_with_quant_no_match(self, temp_models_dir):
        """Test getting path when quantization doesn't match raises FileNotFoundError."""
        model_subdir = temp_models_dir / "mymodel"
        model_subdir.mkdir()
        (model_subdir / "model-Q4_K_M.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            with pytest.raises(FileNotFoundError):
                get_local_model_path(ModelReference("user/mymodel", "Q8_0"))

    def test_get_local_model_path_flat_with_quant_match(self, temp_models_dir):
        """Test flat file structure with quantization match."""
        (temp_models_dir / "mymodel-Q4_K_M.gguf").touch()
        (temp_models_dir / "other-Q8_0.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path(ModelReference("user/mymodel", "Q4_K_M"))
            assert result == temp_models_dir

    def test_get_local_model_path_case_insensitive_quant(self, temp_models_dir):
        """Test that quantization matching is case insensitive."""
        model_subdir = temp_models_dir / "mymodel"
        model_subdir.mkdir()
        (model_subdir / "model-q4_k_m.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            result = get_local_model_path(ModelReference("user/mymodel", "Q4_K_M"))
            assert result == model_subdir

    def test_get_local_model_path_extracts_repo_name(self, temp_models_dir):
        """Test that repo name is correctly extracted."""
        # Create subdirectory with repo name
        model_subdir = temp_models_dir / "mymodel"
        model_subdir.mkdir()
        (model_subdir / "model.gguf").touch()

        # Test that user/mymodel and org/mymodel both find it
        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            result1 = get_local_model_path(ModelReference("user/mymodel"))
            result2 = get_local_model_path(ModelReference("org/mymodel"))
            assert result1 == result2 == model_subdir


class TestGetModelFiles:
    """Test get_model_files function."""

    def test_get_model_files_single_file(self, temp_models_dir):
        """Test getting single model file."""
        (temp_models_dir / "mymodel-Q4_K_M.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            files = get_model_files(parse_model_reference("user/mymodel"))
            assert len(files) == 1
            assert files[0].name == "mymodel-Q4_K_M.gguf"

    def test_get_model_files_multiple_files(self, temp_models_dir):
        """Test getting multiple model files."""
        (temp_models_dir / "mymodel-Q4_K_M-00001-of-00004.gguf").touch()
        (temp_models_dir / "mymodel-Q4_K_M-00002-of-00004.gguf").touch()
        (temp_models_dir / "mymodel-Q4_K_M-00003-of-00004.gguf").touch()
        (temp_models_dir / "mymodel-Q4_K_M-00004-of-00004.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            files = get_model_files(parse_model_reference("user/mymodel"))
            assert len(files) == 4

    def test_get_model_files_with_quant_filter(self, temp_models_dir):
        """Test that quantization filtering works."""
        (temp_models_dir / "mymodel-Q4_K_M.gguf").touch()
        (temp_models_dir / "mymodel-Q8_0.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            files = get_model_files(parse_model_reference("user/mymodel:Q4_K_M"))
            assert len(files) == 1
            assert "Q4_K_M" in files[0].name

    def test_get_model_files_missing_model(self, temp_models_dir):
        """Test that FileNotFoundError is raised for missing model."""
        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            with pytest.raises(FileNotFoundError):
                get_model_files(parse_model_reference("user/missing"))

    def test_get_model_files_sorted(self, temp_models_dir):
        """Test that model files are sorted."""
        # Create files in non-alphabetical order
        (temp_models_dir / "mymodel-00003-of-00004.gguf").touch()
        (temp_models_dir / "mymodel-00001-of-00004.gguf").touch()
        (temp_models_dir / "mymodel-00004-of-00004.gguf").touch()
        (temp_models_dir / "mymodel-00002-of-00004.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            files = get_model_files(parse_model_reference("user/mymodel"))
            assert len(files) == 4
            # Check that they are sorted
            names = [f.name for f in files]
            assert names == sorted(names)

    def test_get_model_files_subdirectory(self, temp_models_dir):
        """Test getting files from subdirectory."""
        model_subdir = temp_models_dir / "mymodel"
        model_subdir.mkdir()
        (model_subdir / "model-00001-of-00002.gguf").touch()
        (model_subdir / "model-00002-of-00002.gguf").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            files = get_model_files(parse_model_reference("user/mymodel"))
            assert len(files) == 2
            assert all(f.parent == model_subdir for f in files)

    def test_get_model_files_ignores_non_gguf(self, temp_models_dir):
        """Test that non-GGUF files are ignored."""
        (temp_models_dir / "mymodel.gguf").touch()
        (temp_models_dir / "mymodel.txt").touch()
        (temp_models_dir / "mymodel.json").touch()

        with patch.object(cllama.utils.models, "MODELS_DIR", temp_models_dir):
            files = get_model_files(parse_model_reference("user/mymodel"))
            assert len(files) == 1
            assert files[0].suffix == ".gguf"
