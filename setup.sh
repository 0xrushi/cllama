#!/bin/bash
# Setup script to activate cllama environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv/cllama"

echo "To use cllama, add the venv bin directory to your PATH:"
echo ""
echo "  export PATH=\"$VENV_DIR/bin:\$PATH\""
echo ""
echo "Or run commands directly:"
echo "  $VENV_DIR/bin/cllama --help"
echo "  $VENV_DIR/bin/cllama-cli --help"
echo ""
echo "Example commands:"
echo "  cllama update"
echo "  cllama pull lefromage/Qwen3-Next-80B:Q4_K_M"
echo "  cllama run --backend vulkan lefromage/Qwen3-Next-80B:Q4_K_M --port 8084"
echo "  cllama-cli --backend vulkan lefromage/Qwen3-Next-80B:Q4_K_M --prompt \"Hello\""
