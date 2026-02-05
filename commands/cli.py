"""Run llama-cli with Hugging Face model integration."""

import sys
import subprocess
from pathlib import Path
from loguru import logger
import click

from ..config import DEFAULT_BACKEND
from ..utils.models import parse_model_reference, resolve_model_path, get_model_files
from ..utils.backend import get_llama_cli_path, validate_backend


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--backend", "-b", default=DEFAULT_BACKEND,
              help="Backend to use (vulkan, hip, rocwmma)")
@click.option("--quant", "-q", help="Quantization pattern (e.g., Q4_K_M)")
@click.argument("model")
@click.argument("llama_args", nargs=-1, type=click.UNPROCESSED)
def cli_cmd(backend: str, quant: str | None, model: str, llama_args: tuple):
    """Run llama-cli with a Hugging Face model.

    MODEL: Repository ID (e.g., user/repo or user/repo:quant)

    LLAMA_ARGS: Additional arguments passed to llama-cli

    Examples:
        cllama-cli --backend vulkan lefromage/Qwen3-Next-80B:Q4_K_M --prompt "Hello"
        cllama-cli --backend hip lefromage/Qwen3-Next-80B --quant Q4_K_M -ngl 3
    """
    # Validate backend
    if not validate_backend(backend):
        logger.error(f"Backend '{backend}' not found or binary does not exist")
        logger.info(f"Available backends: vulkan, hip, rocwmma")
        sys.exit(1)

    # Parse model reference
    model_ref = parse_model_reference(model, quant)
    logger.info(f"Using model: {model_ref}")

    # Resolve model path (auto-download if needed)
    try:
        model_dir = resolve_model_path(model_ref, auto_download=True)
    except Exception as e:
        logger.error(f"Failed to resolve model: {e}")
        sys.exit(1)

    # Get model files
    try:
        model_files = get_model_files(model_ref)
        if len(model_files) == 1:
            model_path = model_files[0]
            logger.info(f"Using model file: {model_path.name}")
        else:
            # Multiple files - use directory for sharded models
            model_path = model_dir
            logger.info(f"Using model directory with {len(model_files)} file(s):")
            for f in model_files:
                logger.info(f"  - {f.name}")
    except Exception as e:
        logger.error(f"Failed to get model files: {e}")
        sys.exit(1)

    # Get llama-cli path
    cli_path = get_llama_cli_path(backend)
    logger.info(f"Using llama-cli: {cli_path}")

    # Build command
    cmd = [str(cli_path)]

    # Add model path
    if model_path.is_file():
        cmd.extend(["-m", str(model_path)])
    else:
        # For multi-file models, specify directory
        cmd.extend(["-m", str(model_path)])

    # Add additional arguments
    if llama_args:
        cmd.extend(list(llama_args))

    # Run llama-cli
    logger.info("Running llama-cli...")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running llama-cli: {e}")
        sys.exit(1)
