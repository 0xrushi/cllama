"""Utilities for managing the llama-swap config yaml."""

import sys
from loguru import logger

from ..config import LLAMA_SWAP_CONFIG, MODELS_DIR
from .hf import get_local_model_path


def update_llama_swap_config(repo_id: str, quant: str | None) -> None:
    """Update the llama-swap config yaml with a newly downloaded model.

    Args:
        repo_id: Repository ID (e.g., "user/model")
        quant: Optional quantization pattern (e.g., "Q4_K_M")
    """
    if not LLAMA_SWAP_CONFIG.exists():
        logger.error(
            f"llama-swap config not found at '{LLAMA_SWAP_CONFIG}'. "
            "Set the LLAMA_SWAP_CONFIG env var to the correct path."
        )
        sys.exit(1)

    try:
        _do_update(repo_id, quant)
    except Exception as e:
        logger.warning(f"Failed to update llama-swap config: {e}")


def _do_update(repo_id: str, quant: str | None) -> None:
    repo_name = repo_id.split("/")[-1]

    # Find the local model directory
    model_dir = get_local_model_path(repo_id, quant)
    if model_dir is None:
        logger.warning(f"Could not find local model path for '{repo_id}' — skipping config update.")
        return

    # List gguf files, filtered by quant if provided
    gguf_files = sorted(model_dir.glob("*.gguf"))
    if quant:
        quant_lower = quant.lower()
        gguf_files = [f for f in gguf_files if quant_lower in f.name.lower()]

    if not gguf_files:
        logger.warning("No local .gguf files found — skipping config update.")
        return

    # For split models, use the -00001-of- shard as the entry point
    split_files = [f for f in gguf_files if "-00001-of-" in f.name]
    chosen_file = split_files[0] if split_files else gguf_files[0]

    # Derive model key: lowercase repo_name-quant, dots/underscores → hyphens
    quant_part = quant if quant else "full"
    raw_key = f"{repo_name}-{quant_part}"
    model_key = raw_key.lower().replace(".", "-").replace("_", "-")

    # Compute path relative to MODELS_DIR for use with ${model-root}
    try:
        rel_path = chosen_file.relative_to(MODELS_DIR)
    except ValueError:
        rel_path = chosen_file  # fallback: use absolute

    model_path_str = f"${{model-root}}/{rel_path}"

    # Build yaml entry text (indented under models:)
    entry_text = (
        f'\n  "{model_key}":\n'
        f"    cmd: |\n"
        f"      ${{latest-llama}}\n"
        f"      --port ${{PORT}}\n"
        f"      --model {model_path_str}\n"
        f"      -c 32768\n"
        f"      -ngl 999\n"
        f"      --jinja\n"
        f"    ttl: 600\n"
    )

    # Member line for main-group
    member_line = f'      - "{model_key}"\n'

    content = LLAMA_SWAP_CONFIG.read_text()

    # Insert model entry just before the \ngroups: line
    groups_marker = "\ngroups:"
    if groups_marker not in content:
        logger.warning("Could not find 'groups:' section in llama-swap config — skipping config update.")
        return

    content = content.replace(groups_marker, entry_text + groups_marker, 1)

    # Append to main-group members list, just before the "embedding-group": line
    embedding_marker = '      "embedding-group":'
    if embedding_marker not in content:
        logger.warning("Could not find 'embedding-group' in llama-swap config — skipping member insertion.")
    else:
        content = content.replace(embedding_marker, member_line + embedding_marker, 1)

    LLAMA_SWAP_CONFIG.write_text(content)

    logger.success(f"llama-swap config updated: added '{model_key}'.")
    print("Remember to restart llama-swap for the changes to take effect.")
