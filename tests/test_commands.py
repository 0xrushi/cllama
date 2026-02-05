"""Tests for CLI commands."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest
from click.testing import CliRunner
from cllama.commands.run import run
from cllama.commands.cli import cli_cmd
from cllama.commands.pull import pull
from cllama.commands.update import update


class TestPullCommand:
    """Test cllama pull command."""

    def test_pull_simple_repo(self):
        """Test pulling a simple repository without quantization."""
        runner = CliRunner()
        with patch("cllama.commands.pull.download_model") as mock_download:
            with patch("cllama.commands.pull.parse_model_reference") as mock_parse:
                mock_model_ref = MagicMock()
                mock_model_ref.repo_id = "user/model"
                mock_model_ref.quant = None
                mock_parse.return_value = mock_model_ref

                result = runner.invoke(pull, ["user/model"])

                assert result.exit_code == 0
                mock_parse.assert_called_once_with("user/model", None)
                mock_download.assert_called_once_with("user/model", None)

    def test_pull_with_quantization_in_model_string(self):
        """Test pulling with quantization specified in model string."""
        runner = CliRunner()
        with patch("cllama.commands.pull.download_model") as mock_download:
            with patch("cllama.commands.pull.parse_model_reference") as mock_parse:
                mock_model_ref = MagicMock()
                mock_model_ref.repo_id = "user/model"
                mock_model_ref.quant = "Q4_K_M"
                mock_parse.return_value = mock_model_ref

                result = runner.invoke(pull, ["user/model:Q4_K_M"])

                assert result.exit_code == 0
                mock_parse.assert_called_once_with("user/model:Q4_K_M", None)
                mock_download.assert_called_once_with("user/model", "Q4_K_M")

    def test_pull_with_quant_flag(self):
        """Test pulling with --quant flag."""
        runner = CliRunner()
        with patch("cllama.commands.pull.download_model") as mock_download:
            with patch("cllama.commands.pull.parse_model_reference") as mock_parse:
                mock_model_ref = MagicMock()
                mock_model_ref.repo_id = "user/model"
                mock_model_ref.quant = "Q8_0"
                mock_parse.return_value = mock_model_ref

                result = runner.invoke(pull, ["user/model", "--quant", "Q8_0"])

                assert result.exit_code == 0
                mock_parse.assert_called_once_with("user/model", "Q8_0")

    def test_pull_download_failure(self):
        """Test pull command when download fails."""
        runner = CliRunner()
        with patch("cllama.commands.pull.download_model") as mock_download:
            with patch("cllama.commands.pull.parse_model_reference") as mock_parse:
                mock_model_ref = MagicMock()
                mock_parse.return_value = mock_model_ref
                mock_download.side_effect = Exception("Download failed")

                result = runner.invoke(pull, ["user/model"])

                assert result.exit_code == 1


class TestRunCommand:
    """Test cllama run command."""

    def test_run_with_backend_and_model(self):
        """Test run command with backend and model."""
        runner = CliRunner()
        with patch("cllama.commands.run.validate_backend") as mock_validate:
            with patch("cllama.commands.run.parse_model_reference") as mock_parse:
                with patch("cllama.commands.run.resolve_model_path") as mock_resolve:
                    with patch("cllama.commands.run.get_model_files") as mock_get_files:
                        with patch("cllama.commands.run.get_llama_server_path") as mock_server_path:
                            with patch("cllama.commands.run.subprocess.run") as mock_subprocess:
                                mock_validate.return_value = True
                                mock_model_ref = MagicMock()
                                mock_parse.return_value = mock_model_ref
                                mock_resolve.return_value = Path("/tmp/model")
                                mock_file = MagicMock()
                                mock_file.is_file.return_value = True
                                mock_get_files.return_value = [mock_file]
                                mock_server_path.return_value = Path("/bin/llama-server")

                                result = runner.invoke(run, ["--backend", "vulkan", "user/model"])

                                assert result.exit_code == 0
                                mock_validate.assert_called_once_with("vulkan")
                                mock_subprocess.assert_called_once()

    def test_run_invalid_backend(self):
        """Test run command with invalid backend."""
        runner = CliRunner()
        with patch("cllama.commands.run.validate_backend") as mock_validate:
            mock_validate.return_value = False

            result = runner.invoke(run, ["--backend", "invalid", "user/model"])

            assert result.exit_code == 1

    def test_run_default_backend(self):
        """Test run command uses default backend."""
        runner = CliRunner()
        with patch("cllama.commands.run.validate_backend") as mock_validate:
            with patch("cllama.commands.run.parse_model_reference") as mock_parse:
                with patch("cllama.commands.run.resolve_model_path") as mock_resolve:
                    with patch("cllama.commands.run.get_model_files") as mock_get_files:
                        with patch("cllama.commands.run.get_llama_server_path") as mock_server_path:
                            with patch("cllama.commands.run.subprocess.run") as mock_subprocess:
                                mock_validate.return_value = True
                                mock_model_ref = MagicMock()
                                mock_parse.return_value = mock_model_ref
                                mock_resolve.return_value = Path("/tmp/model")
                                mock_file = MagicMock()
                                mock_file.is_file.return_value = True
                                mock_get_files.return_value = [mock_file]
                                mock_server_path.return_value = Path("/bin/llama-server")

                                result = runner.invoke(run, ["user/model"])

                                assert result.exit_code == 0
                                # Should use default backend (vulkan)
                                mock_validate.assert_called_once_with("vulkan")

    def test_run_passes_additional_args(self):
        """Test that run command passes additional arguments to llama-server."""
        runner = CliRunner()
        with patch("cllama.commands.run.validate_backend") as mock_validate:
            with patch("cllama.commands.run.parse_model_reference") as mock_parse:
                with patch("cllama.commands.run.resolve_model_path") as mock_resolve:
                    with patch("cllama.commands.run.get_model_files") as mock_get_files:
                        with patch("cllama.commands.run.get_llama_server_path") as mock_server_path:
                            with patch("cllama.commands.run.subprocess.run") as mock_subprocess:
                                mock_validate.return_value = True
                                mock_model_ref = MagicMock()
                                mock_parse.return_value = mock_model_ref
                                mock_resolve.return_value = Path("/tmp/model")
                                mock_file = MagicMock()
                                mock_file.is_file.return_value = True
                                mock_get_files.return_value = [mock_file]
                                mock_server_path.return_value = Path("/bin/llama-server")

                                result = runner.invoke(run, ["user/model", "--port", "8084", "-ngl", "999"])

                                assert result.exit_code == 0
                                # Check that additional args were included
                                call_args = mock_subprocess.call_args[0][0]
                                assert "--port" in call_args
                                assert "8084" in call_args

    def test_run_model_resolution_failure(self):
        """Test run command when model resolution fails."""
        runner = CliRunner()
        with patch("cllama.commands.run.validate_backend") as mock_validate:
            with patch("cllama.commands.run.parse_model_reference") as mock_parse:
                with patch("cllama.commands.run.resolve_model_path") as mock_resolve:
                    mock_validate.return_value = True
                    mock_parse.return_value = MagicMock()
                    mock_resolve.side_effect = FileNotFoundError("Model not found")

                    result = runner.invoke(run, ["user/missing"])

                    assert result.exit_code == 1


class TestCliCommand:
    """Test cllama-cli command."""

    def test_cli_with_backend_and_model(self):
        """Test cli command with backend and model."""
        runner = CliRunner()
        with patch("cllama.commands.cli.validate_backend") as mock_validate:
            with patch("cllama.commands.cli.parse_model_reference") as mock_parse:
                with patch("cllama.commands.cli.resolve_model_path") as mock_resolve:
                    with patch("cllama.commands.cli.get_model_files") as mock_get_files:
                        with patch("cllama.commands.cli.get_llama_cli_path") as mock_cli_path:
                            with patch("cllama.commands.cli.subprocess.run") as mock_subprocess:
                                mock_validate.return_value = True
                                mock_model_ref = MagicMock()
                                mock_parse.return_value = mock_model_ref
                                mock_resolve.return_value = Path("/tmp/model")
                                mock_file = MagicMock()
                                mock_file.is_file.return_value = True
                                mock_get_files.return_value = [mock_file]
                                mock_cli_path.return_value = Path("/bin/llama-cli")

                                result = runner.invoke(cli_cmd, ["--backend", "hip", "user/model"])

                                assert result.exit_code == 0
                                mock_validate.assert_called_once_with("hip")
                                mock_subprocess.assert_called_once()

    def test_cli_invalid_backend(self):
        """Test cli command with invalid backend."""
        runner = CliRunner()
        with patch("cllama.commands.cli.validate_backend") as mock_validate:
            mock_validate.return_value = False

            result = runner.invoke(cli_cmd, ["--backend", "invalid", "user/model"])

            assert result.exit_code == 1

    def test_cli_with_quant_flag(self):
        """Test cli command with quantization flag."""
        runner = CliRunner()
        with patch("cllama.commands.cli.validate_backend") as mock_validate:
            with patch("cllama.commands.cli.parse_model_reference") as mock_parse:
                with patch("cllama.commands.cli.resolve_model_path") as mock_resolve:
                    with patch("cllama.commands.cli.get_model_files") as mock_get_files:
                        with patch("cllama.commands.cli.get_llama_cli_path") as mock_cli_path:
                            with patch("cllama.commands.cli.subprocess.run") as mock_subprocess:
                                mock_validate.return_value = True
                                mock_model_ref = MagicMock()
                                mock_parse.return_value = mock_model_ref
                                mock_resolve.return_value = Path("/tmp/model")
                                mock_file = MagicMock()
                                mock_file.is_file.return_value = True
                                mock_get_files.return_value = [mock_file]
                                mock_cli_path.return_value = Path("/bin/llama-cli")

                                result = runner.invoke(cli_cmd, ["user/model", "--quant", "Q4_K_M"])

                                assert result.exit_code == 0
                                mock_parse.assert_called_once_with("user/model", "Q4_K_M")

    def test_cli_multi_file_model(self):
        """Test cli command with multi-file sharded model."""
        runner = CliRunner()
        with patch("cllama.commands.cli.validate_backend") as mock_validate:
            with patch("cllama.commands.cli.parse_model_reference") as mock_parse:
                with patch("cllama.commands.cli.resolve_model_path") as mock_resolve:
                    with patch("cllama.commands.cli.get_model_files") as mock_get_files:
                        with patch("cllama.commands.cli.get_llama_cli_path") as mock_cli_path:
                            with patch("cllama.commands.cli.subprocess.run") as mock_subprocess:
                                mock_validate.return_value = True
                                mock_model_ref = MagicMock()
                                mock_parse.return_value = mock_model_ref
                                model_dir = Path("/tmp/model")
                                mock_resolve.return_value = model_dir

                                # Multiple files
                                mock_files = [MagicMock() for _ in range(4)]
                                for mf in mock_files:
                                    mf.is_file.return_value = True
                                    mf.parent = model_dir
                                    mf.name = "model.gguf"

                                mock_get_files.return_value = mock_files
                                mock_cli_path.return_value = Path("/bin/llama-cli")

                                result = runner.invoke(cli_cmd, ["user/model"])

                                assert result.exit_code == 0
                                # Should pass model directory for sharded models
                                call_args = mock_subprocess.call_args[0][0]
                                assert str(model_dir) in call_args


class TestUpdateCommand:
    """Test cllama update command."""

    def test_update_success(self):
        """Test update command executes script successfully."""
        runner = CliRunner()
        with patch("cllama.commands.update.subprocess.run") as mock_subprocess:
            with patch("cllama.commands.update.Path") as mock_path_class:
                mock_script = MagicMock()
                mock_script.exists.return_value = True
                mock_script.is_file.return_value = True
                mock_path_class.return_value = mock_script

                result = runner.invoke(update)

                assert result.exit_code == 0
                mock_subprocess.assert_called_once()

    def test_update_script_not_found(self):
        """Test update command when script doesn't exist."""
        runner = CliRunner()
        with patch("cllama.commands.update.Path") as mock_path_class:
            mock_script = MagicMock()
            mock_script.exists.return_value = False
            mock_path_class.return_value = mock_script

            result = runner.invoke(update)

            assert result.exit_code == 1
