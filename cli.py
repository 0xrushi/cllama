"""Main CLI entry point for cllama."""

import sys
import click
from loguru import logger

from .commands.update import update
from .commands.pull import pull
from .commands.run import run
from .commands.cli import cli_cmd

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """cllama - CLI wrapper for llama.cpp with Hugging Face integration."""
    pass


# Add commands
main.add_command(update)
main.add_command(pull)
main.add_command(run)


def cli_entry():
    """Entry point for cllama-cli (separate command for llama-cli)."""
    # Standalone entry that directly calls cli_cmd
    # Parse sys.argv to extract model and args
    from .commands.cli import cli_cmd

    # Remove script name
    args = sys.argv[1:]

    # Check for --help
    if "--help" in args or "-h" in args or not args:
        # Call the Click command with --help
        import click
        ctx = click.Context(cli_cmd)
        click.echo(cli_cmd.get_help(ctx))
        return

    # Try to extract --backend and --quant flags
    backend = None
    quant = None
    model_idx = -1

    i = 0
    while i < len(args):
        if args[i] == "--backend" or args[i] == "-b":
            if i + 1 < len(args):
                backend = args[i + 1]
                i += 2
        elif args[i] == "--quant" or args[i] == "-q":
            if i + 1 < len(args):
                quant = args[i + 1]
                i += 2
        elif args[i].startswith("-"):
            i += 1
        else:
            # First non-flag argument is the model
            if model_idx == -1:
                model_idx = i
            i += 1

    if model_idx >= 0:
        model = args[model_idx]
        llama_args = args[model_idx + 1:]

        # Call cli_cmd directly with parsed arguments
        cli_cmd(backend=backend, quant=quant, model=model, llama_args=tuple(llama_args))
    else:
        logger.error("No model specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
