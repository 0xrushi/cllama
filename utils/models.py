"""Model reference parsing and resolution."""

from pathlib import Path
from loguru import logger

from .hf import download_model, is_model_downloaded, find_files_by_quant
from ..config import MODELS_DIR


class ModelReference:
    """Parsed model reference."""

    def __init__(self, repo_id: str, quant: str | None = None):
        self.repo_id = repo_id
        self.quant = quant

    def __repr__(self) -> str:
        if self.quant:
            return f"{self.repo_id}:{self.quant}"
        return self.repo_id


def parse_model_reference(model_ref: str, quant_override: str | None = None) -> ModelReference:
    """Parse a model reference string.

    Supports formats:
    - user/repo
    - user/repo:quant

    Args:
        model_ref: Model reference string
        quant_override: Optional quantization to override parsed value

    Returns:
        ModelReference object
    """
    # Parse :quant suffix
    if ":" in model_ref:
        parts = model_ref.rsplit(":", 1)
        repo_id = parts[0]
        quant = parts[1]
    else:
        repo_id = model_ref
        quant = None

    # Override with explicit --quant flag if provided
    if quant_override:
        quant = quant_override

    return ModelReference(repo_id, quant)


def resolve_model_path(model_ref: ModelReference, auto_download: bool = True) -> Path:
    """Resolve a model reference to a local file path.

    If model is not found locally and auto_download is True,
    downloads the model from Hugging Face.

    Args:
        model_ref: Parsed model reference
        auto_download: Whether to auto-download if not found

    Returns:
        Path to model directory (for multi-file models)
    """
    # Check if model exists locally
    if is_model_downloaded(model_ref.repo_id, model_ref.quant):
        logger.info(f"Using local model: {model_ref.repo_id}")
        return get_local_model_path(model_ref)

    # Auto-download if enabled
    if auto_download:
        logger.info(f"Model not found locally. Downloading from Hugging Face...")
        download_model(model_ref.repo_id, model_ref.quant)
        return get_local_model_path(model_ref)

    raise FileNotFoundError(
        f"Model not found locally and auto-download disabled: {model_ref.repo_id}"
    )


def get_local_model_path(model_ref: ModelReference) -> Path:
    """Get local path for a model reference.

    Args:
        model_ref: Parsed model reference

    Returns:
        Path to local model directory or file

    Raises:
        FileNotFoundError: If model not found locally
    """
    repo_name = model_ref.repo_id.split("/")[-1]
    local_path = MODELS_DIR / repo_name

    # Check if model is stored as a directory
    if local_path.exists() and local_path.is_dir():
        # If quantization specified, verify matching files exist
        if model_ref.quant:
            quant_lower = model_ref.quant.lower()
            all_gguf = list(local_path.glob("*.gguf"))
            matching_files = [f for f in all_gguf if quant_lower in f.name.lower()]
            if not matching_files:
                raise FileNotFoundError(
                    f"No files found matching quantization '{model_ref.quant}' in {local_path}"
                )
        return local_path

    # Check if model is stored as a flat file in models directory
    # Look for files containing the repo name
    all_gguf = list(MODELS_DIR.glob("*.gguf"))
    gguf_files = [f for f in all_gguf if repo_name in f.name]

    if gguf_files:
        # If quantization specified, filter by it
        if model_ref.quant:
            quant_lower = model_ref.quant.lower()
            matching_files = [f for f in gguf_files if quant_lower in f.name.lower()]
            if matching_files:
                return MODELS_DIR  # Return models directory for flat files
            raise FileNotFoundError(
                f"No files found matching quantization '{model_ref.quant}' in {MODELS_DIR}"
            )
        return MODELS_DIR  # Return models directory for flat files

    raise FileNotFoundError(f"Model not found locally: {model_ref.repo_id}")


def get_model_files(model_ref: ModelReference) -> list[Path]:
    """Get all .gguf files for a model reference.

    Args:
        model_ref: Parsed model reference

    Returns:
        List of paths to .gguf files

    Raises:
        FileNotFoundError: If model or files not found
    """
    local_path = get_local_model_path(model_ref)
    repo_name = model_ref.repo_id.split("/")[-1]

    # Check if local_path is a directory or the models directory (flat files)
    if local_path.is_dir() and local_path != MODELS_DIR:
        # Model is in a subdirectory
        all_files = sorted(local_path.glob("*.gguf"))
        if model_ref.quant:
            quant_lower = model_ref.quant.lower()
            files = [f for f in all_files if quant_lower in f.name.lower()]
        else:
            files = all_files
    else:
        # Model is stored as flat files in models directory
        all_gguf = sorted(MODELS_DIR.glob("*.gguf"))
        # Filter files containing the repo name
        repo_files = [f for f in all_gguf if repo_name in f.name]

        if model_ref.quant:
            quant_lower = model_ref.quant.lower()
            files = [f for f in repo_files if quant_lower in f.name.lower()]
        else:
            files = repo_files

    if not files:
        raise FileNotFoundError(f"No .gguf files found for {model_ref.repo_id}")

    return files
