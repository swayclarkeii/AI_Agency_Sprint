# paths.py
import os
from pathlib import Path

def resolve_project_drive() -> Path:
    """
    Return the Google Drive project root:
      $SPRINT_DRIVE if set,
      otherwise auto-detect ~/Library/CloudStorage/GoogleDrive-*/My Drive/AI_Agency_Sprint
    """
    env = os.getenv("SPRINT_DRIVE")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"SPRINT_DRIVE points to missing path: {p}")

    base = Path.home() / "Library" / "CloudStorage"
    candidates = list(base.glob("GoogleDrive-*/My Drive/AI_Agency_Sprint"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find AI_Agency_Sprint under {base}. "
            "Set $SPRINT_DRIVE to override."
        )
    return candidates[0]
