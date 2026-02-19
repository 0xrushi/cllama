"""Tests for llama-swap config update utility."""

import pytest
from pathlib import Path
from unittest.mock import patch, call
import cllama.utils.llama_swap as ls_module
from cllama.utils.llama_swap import update_llama_swap_config, _do_update


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = """\
models:
  "existing-model":
    cmd: |
      ${latest-llama}
      --port ${PORT}
      --model ${model-root}/existing/model.gguf
      -c 32768
      -ngl 999
      --jinja
    ttl: 600

groups:
  "main-group":
    members:
      - "existing-model"
      "embedding-group":
        members:
          - "embed-model"
"""


def make_config(tmp_path: Path, content: str = MINIMAL_CONFIG) -> Path:
    cfg = tmp_path / "llama-swap-config.yaml"
    cfg.write_text(content)
    return cfg


# ---------------------------------------------------------------------------
# update_llama_swap_config — top-level guard
# ---------------------------------------------------------------------------

class TestUpdateLlamaSwapConfigGuard:
    """Tests for the LLAMA_SWAP_CONFIG existence check."""

    def test_missing_config_exits(self, tmp_path):
        """When the config file does not exist, sys.exit(1) is called."""
        missing = tmp_path / "nonexistent.yaml"
        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", missing):
            with pytest.raises(SystemExit) as exc_info:
                update_llama_swap_config("user/MyModel-GGUF", "Q4_K_M")
        assert exc_info.value.code == 1

    def test_existing_config_does_not_exit(self, tmp_path):
        """When the config exists, no SystemExit is raised (even if _do_update is no-op)."""
        cfg = make_config(tmp_path)
        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "_do_update") as mock_do:
                update_llama_swap_config("user/MyModel-GGUF", "Q4_K_M")
        mock_do.assert_called_once_with("user/MyModel-GGUF", "Q4_K_M")

    def test_do_update_exception_is_non_fatal(self, tmp_path):
        """Unexpected errors inside _do_update are caught and logged as warnings."""
        cfg = make_config(tmp_path)
        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "_do_update", side_effect=RuntimeError("boom")):
                # Should NOT raise
                update_llama_swap_config("user/MyModel-GGUF", "Q4_K_M")


# ---------------------------------------------------------------------------
# _do_update — model path resolution
# ---------------------------------------------------------------------------

class TestDoUpdateModelPath:
    """Tests for local model path resolution inside _do_update."""

    def test_no_local_model_path_skips(self, tmp_path):
        """If get_local_model_path returns None, the config is left unchanged."""
        cfg = make_config(tmp_path)
        original = cfg.read_text()
        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "get_local_model_path", return_value=None):
                _do_update("user/MyModel-GGUF", "Q4_K_M")
        assert cfg.read_text() == original

    def test_no_gguf_files_in_dir_skips(self, tmp_path):
        """If the model dir contains no .gguf files, the config is left unchanged."""
        cfg = make_config(tmp_path)
        model_dir = tmp_path / "models" / "MyModel-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "README.md").touch()   # not a gguf

        original = cfg.read_text()
        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "get_local_model_path", return_value=model_dir):
                _do_update("user/MyModel-GGUF", "Q4_K_M")
        assert cfg.read_text() == original

    def test_quant_filter_skips_non_matching_files(self, tmp_path):
        """Files that don't match the requested quant are excluded."""
        cfg = make_config(tmp_path)
        model_dir = tmp_path / "models" / "MyModel-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "MyModel-Q8_0.gguf").touch()

        original = cfg.read_text()
        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "get_local_model_path", return_value=model_dir):
                _do_update("user/MyModel-GGUF", "Q4_K_M")
        assert cfg.read_text() == original


# ---------------------------------------------------------------------------
# _do_update — model key derivation
# ---------------------------------------------------------------------------

class TestDoUpdateKeyDerivation:
    """Tests for model key name generation."""

    def _run(self, tmp_path, repo_id, quant, filename):
        cfg = make_config(tmp_path)
        models_dir = tmp_path / "models"
        model_dir = models_dir / repo_id.split("/")[-1]
        model_dir.mkdir(parents=True)
        (model_dir / filename).touch()

        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "MODELS_DIR", models_dir):
                with patch.object(ls_module, "get_local_model_path", return_value=model_dir):
                    _do_update(repo_id, quant)
        return cfg.read_text()

    def test_underscores_replaced(self, tmp_path):
        content = self._run(tmp_path, "user/Qwen3-8B-GGUF", "Q4_K_M", "Qwen3-8B-Q4_K_M.gguf")
        assert '"qwen3-8b-gguf-q4-k-m"' in content

    def test_dots_replaced(self, tmp_path):
        content = self._run(tmp_path, "user/My.Model-GGUF", "Q4_K_M", "My.Model-Q4_K_M.gguf")
        assert '"my-model-gguf-q4-k-m"' in content

    def test_no_quant_uses_full(self, tmp_path):
        cfg = make_config(tmp_path)
        models_dir = tmp_path / "models"
        model_dir = models_dir / "MyModel-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "MyModel-F16.gguf").touch()

        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "MODELS_DIR", models_dir):
                with patch.object(ls_module, "get_local_model_path", return_value=model_dir):
                    _do_update("user/MyModel-GGUF", None)
        assert '"mymodel-gguf-full"' in cfg.read_text()

    def test_key_is_lowercase(self, tmp_path):
        content = self._run(tmp_path, "user/UPPER-GGUF", "Q4_K_M", "UPPER-Q4_K_M.gguf")
        assert '"upper-gguf-q4-k-m"' in content


# ---------------------------------------------------------------------------
# _do_update — split model shard selection
# ---------------------------------------------------------------------------

class TestDoUpdateSplitModel:
    """Tests for split-model (multi-shard) file selection."""

    def _setup(self, tmp_path, filenames):
        cfg = make_config(tmp_path)
        models_dir = tmp_path / "models"
        model_dir = models_dir / "MyModel-GGUF"
        model_dir.mkdir(parents=True)
        for name in filenames:
            (model_dir / name).touch()
        return cfg, models_dir, model_dir

    def test_split_model_uses_00001_shard(self, tmp_path):
        cfg, models_dir, model_dir = self._setup(tmp_path, [
            "MyModel-Q4_K_M-00001-of-00003.gguf",
            "MyModel-Q4_K_M-00002-of-00003.gguf",
            "MyModel-Q4_K_M-00003-of-00003.gguf",
        ])
        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "MODELS_DIR", models_dir):
                with patch.object(ls_module, "get_local_model_path", return_value=model_dir):
                    _do_update("user/MyModel-GGUF", "Q4_K_M")
        content = cfg.read_text()
        assert "00001-of-00003" in content
        assert "00002-of-00003" not in content

    def test_non_split_uses_first_sorted(self, tmp_path):
        cfg, models_dir, model_dir = self._setup(tmp_path, [
            "MyModel-Q4_K_M.gguf",
        ])
        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "MODELS_DIR", models_dir):
                with patch.object(ls_module, "get_local_model_path", return_value=model_dir):
                    _do_update("user/MyModel-GGUF", "Q4_K_M")
        assert "MyModel-Q4_K_M.gguf" in cfg.read_text()


# ---------------------------------------------------------------------------
# _do_update — yaml content insertion
# ---------------------------------------------------------------------------

class TestDoUpdateYamlInsertion:
    """Tests for correct text insertion into the config file."""

    def _run(self, tmp_path, config_content=MINIMAL_CONFIG, quant="Q4_K_M", filename=None):
        cfg = tmp_path / "llama-swap-config.yaml"
        cfg.write_text(config_content)
        models_dir = tmp_path / "models"
        model_dir = models_dir / "MyModel-GGUF"
        model_dir.mkdir(parents=True)
        fname = filename or f"MyModel-{quant}.gguf"
        (model_dir / fname).touch()

        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "MODELS_DIR", models_dir):
                with patch.object(ls_module, "get_local_model_path", return_value=model_dir):
                    _do_update("user/MyModel-GGUF", quant)
        return cfg.read_text()

    def test_model_entry_inserted_before_groups(self, tmp_path):
        content = self._run(tmp_path)
        groups_idx = content.index("\ngroups:")
        key_idx = content.index('"mymodel-gguf-q4-k-m"')
        assert key_idx < groups_idx

    def test_model_entry_contains_required_fields(self, tmp_path):
        content = self._run(tmp_path)
        assert "${latest-llama}" in content
        assert "--port ${PORT}" in content
        assert "--model ${model-root}/" in content
        assert "-c 32768" in content
        assert "-ngl 999" in content
        assert "--jinja" in content
        assert "ttl: 600" in content

    def test_member_line_inserted_before_embedding_group(self, tmp_path):
        content = self._run(tmp_path)
        member_idx = content.index('      - "mymodel-gguf-q4-k-m"')
        embedding_idx = content.index('      "embedding-group":')
        assert member_idx < embedding_idx

    def test_existing_content_preserved(self, tmp_path):
        content = self._run(tmp_path)
        assert '"existing-model"' in content
        assert "embed-model" in content

    def test_model_path_uses_model_root_variable(self, tmp_path):
        content = self._run(tmp_path)
        assert "${model-root}/MyModel-GGUF/MyModel-Q4_K_M.gguf" in content

    def test_missing_groups_marker_skips(self, tmp_path):
        bad_config = "models:\n  existing: {}\n"  # no \ngroups: line
        content = self._run(tmp_path, config_content=bad_config)
        # file unchanged — no new key inserted
        assert "mymodel-gguf" not in content

    def test_missing_embedding_group_still_inserts_model(self, tmp_path):
        config_without_embedding = """\
models:
  "existing-model":
    cmd: echo
    ttl: 600

groups:
  "main-group":
    members:
      - "existing-model"
"""
        content = self._run(tmp_path, config_content=config_without_embedding)
        # Model entry should still be present
        assert '"mymodel-gguf-q4-k-m"' in content
        # Member line not inserted (embedding-group missing), but no crash

    def test_idempotent_key_format(self, tmp_path):
        """Running twice should insert two entries (caller's responsibility to dedup)."""
        cfg = make_config(tmp_path)
        models_dir = tmp_path / "models"
        model_dir = models_dir / "MyModel-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "MyModel-Q4_K_M.gguf").touch()

        with patch.object(ls_module, "LLAMA_SWAP_CONFIG", cfg):
            with patch.object(ls_module, "MODELS_DIR", models_dir):
                with patch.object(ls_module, "get_local_model_path", return_value=model_dir):
                    _do_update("user/MyModel-GGUF", "Q4_K_M")
                    _do_update("user/MyModel-GGUF", "Q4_K_M")
        content = cfg.read_text()
        assert content.count('"mymodel-gguf-q4-k-m"') == 4  # 2 model entries + 2 member lines
