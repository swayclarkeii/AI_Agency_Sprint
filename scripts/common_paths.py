"""
File path utilities for data processing pipeline.
Defines the project directory structure and helper functions for file operations.
"""
from pathlib import Path
from typing import Optional

# Project directory structure
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
RAW = DATA / "raw"
RAW_HOOKS = RAW / "hooks"
KB = DATA / "kb"
CHUNKS = DATA / "chunks"
FRAMES = BASE / "frameworks"

def latest_file(folder: Path, ext: str) -> Optional[Path]:
    """
    Find the most recently modified file with the specified extension in the folder.
    
    Args:
        folder: Directory path to search
        ext: File extension to match (without the leading dot)
        
    Returns:
        Path to the most recently modified matching file, or None if no files found
    """
    if not folder.exists():
        return None
        
    files = sorted(folder.glob(f"*.{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

# Optional: Initialize directories if they don't exist
def ensure_directories_exist():
    """Create all defined directories if they don't already exist."""
    for directory in [DATA, RAW, RAW_HOOKS, KB, CHUNKS, FRAMES]:
        directory.mkdir(parents=True, exist_ok=True)

# Optional: Call this at import time if needed
# ensure_directories_exist()