"""Hugging Face integration for model downloads."""

import subprocess
import sys
from pathlib import Path
from huggingface_hub import list_repo_files, whoami
from loguru import logger

from ..config import MODELS_DIR


def get_repo_files(repo_id: str) -> list[str]:
    """List all files in a Hugging Face repository.

    Args:
        repo_id: Repository ID (e.g., "user/model")

    Returns:
        List of file paths in the repository
    """
    try:
        files = list_repo_files(repo_id, repo_type="model")
        return [f for f in files if not f.startswith(".")]
    except Exception as e:
        logger.error(f"Failed to list files from {repo_id}: {e}")
        sys.exit(1)


def find_files_by_quant(repo_id: str, quant: str) -> list[str]:
    """Find all files matching a quantization pattern.

    Args:
        repo_id: Repository ID
        quant: Quantization pattern (e.g., "Q4_K_M")

    Returns:
        List of file paths matching the quantization
    """
    files = get_repo_files(repo_id)
    quant_lower = quant.lower()

    # Match files containing the quantization string
    matching_files = [
        f for f in files
        if quant_lower in f.lower() and f.endswith(".gguf")
    ]

    return matching_files


def download_model(repo_id: str, quant: str | None = None) -> Path:
    """Download a model from Hugging Face.

    If quant is specified, downloads all files matching that quantization.
    Otherwise, downloads all .gguf files.

    Args:
        repo_id: Repository ID (e.g., "user/model")
        quant: Optional quantization pattern

    Returns:
        Path to the download directory
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if quant:
        # Find all files matching the quantization
        matching_files = find_files_by_quant(repo_id, quant)

        if not matching_files:
            logger.error(f"No files found matching quantization '{quant}' in {repo_id}")
            logger.info("Available .gguf files:")
            for f in get_repo_files(repo_id):
                if f.endswith(".gguf"):
                    logger.info(f"  - {f}")
            sys.exit(1)

        logger.info(f"Found {len(matching_files)} file(s) matching '{quant}':")
        for f in matching_files:
            logger.info(f"  - {f}")

        # Download all matching files
        for file_path in matching_files:
            logger.info(f"Downloading {file_path}...")
            download_file(repo_id, file_path, MODELS_DIR)
    else:
        # Download all .gguf files
        logger.info(f"Downloading all .gguf files from {repo_id}...")
        try:
            subprocess.run(
                [
                    "hf", "download", repo_id,
                    "--local-dir", str(MODELS_DIR),
                    "--include", "*.gguf"
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model: {e}")
            sys.exit(1)

    return MODELS_DIR


def download_file(repo_id: str, file_path: str, local_dir: Path) -> None:
    """Download a specific file from a Hugging Face repository.

    Args:
        repo_id: Repository ID
        file_path: Path to the file in the repository
        local_dir: Local directory to save to
    """
    try:
        subprocess.run(
            ["hf", "download", repo_id, file_path, "--local-dir", str(local_dir)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {file_path}: {e}")
        sys.exit(1)


def get_local_model_path(repo_id: str, quant: str | None = None) -> Path | None:
    """Get the local path to a model file.

    Args:
        repo_id: Repository ID
        quant: Optional quantization pattern

    Returns:
        Path to model file or directory, None if not found
    """
    repo_name = repo_id.split("/")[-1]
    local_path = MODELS_DIR / repo_name

    # Check if model is in a subdirectory
    if local_path.exists() and local_path.is_dir():
        if quant:
            quant_lower = quant.lower()
            all_gguf = list(local_path.glob("*.gguf"))
            matching_files = [f for f in all_gguf if quant_lower in f.name.lower()]
            if matching_files:
                return local_path
            return None
        else:
            gguf_files = list(local_path.glob("*.gguf"))
            if gguf_files:
                return local_path
            return None

    # Check if model is stored as flat files in models directory
    all_gguf = list(MODELS_DIR.glob("*.gguf"))
    repo_files = [f for f in all_gguf if repo_name in f.name]

    if repo_files:
        if quant:
            quant_lower = quant.lower()
            matching_files = [f for f in repo_files if quant_lower in f.name.lower()]
            if matching_files:
                return MODELS_DIR
            return None
        return MODELS_DIR

    return None


def is_model_downloaded(repo_id: str, quant: str | None = None) -> bool:
    """Check if a model is already downloaded.

    Args:
        repo_id: Repository ID
        quant: Optional quantization pattern

    Returns:
        True if model files exist locally
    """
    return get_local_model_path(repo_id, quant) is not None
