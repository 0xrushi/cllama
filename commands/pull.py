"""Pull/download models from Hugging Face."""

import sys
from loguru import logger
import click

from ..utils.models import parse_model_reference, resolve_model_path
from ..utils.hf import download_model
from ..utils.llama_swap import update_llama_swap_config


@click.command()
@click.argument("model")
@click.option("--quant", "-q", help="Quantization pattern (e.g., Q4_K_M)")
def pull(model: str, quant: str | None):
    """Pull a model from Hugging Face.

    MODEL: Repository ID (e.g., user/repo or user/repo:quant)

    Examples:
        cllama pull lefromage/Qwen3-Next-80B-A3B-Instruct-GGUF
        cllama pull lefromage/Qwen3-Next-80B:Q4_K_M
        cllama pull lefromage/Qwen3-Next-80B --quant Q4_K_M
    """
    # Parse model reference
    model_ref = parse_model_reference(model, quant)

    logger.info(f"Pulling model: {model_ref}")

    # Download model
    try:
        download_model(model_ref.repo_id, model_ref.quant)
        logger.success("Model downloaded successfully!")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        sys.exit(1)

    update_llama_swap_config(model_ref.repo_id, model_ref.quant)
