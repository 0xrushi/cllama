"""Update llama.cpp builds."""

import subprocess
import sys
from pathlib import Path
from loguru import logger
import click

from ..config import UPDATE_SCRIPT


@click.command()
def update():
    """Update llama.cpp builds by running update script."""
    script_path = Path(UPDATE_SCRIPT)

    if not script_path.exists():
        logger.error(f"Update script not found: {script_path}")
        logger.info("Make sure you're in the correct directory with update-llama.cpp.sh")
        sys.exit(1)

    if not script_path.is_file():
        logger.error(f"Update script is not a file: {script_path}")
        sys.exit(1)

    logger.info("Updating llama.cpp builds...")
    logger.info(f"Running: {script_path}")

    try:
        result = subprocess.run(
            [str(script_path)],
            check=True,
            shell=True
        )
        logger.success("Update completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Update failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running update: {e}")
        sys.exit(1)
