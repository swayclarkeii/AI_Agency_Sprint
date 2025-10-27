#!/bin/bash
# Universal activation script for AI_Agency_Sprint

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Run: python3 -m venv .venv"
    return 1
fi

source "$VENV_PATH/bin/activate"

# Show clear confirmation
echo ""
echo "✅ Virtual environment activated"
echo "📁 Project: $PROJECT_ROOT"
echo "🐍 Python: $(which python)"
echo ""
echo "Type 'deactivate' to exit"
echo ""
