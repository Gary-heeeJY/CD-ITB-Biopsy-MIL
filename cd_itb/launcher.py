"""Run the preserved research implementation from stable public entry points."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = PROJECT_ROOT / "framework"


def run_legacy(script: str, argv: Iterable[str] | None = None) -> None:
    """Execute a framework script with the repository root as its working directory.

    The original implementation uses imports such as ``from models import ...``.
    Keeping ``framework`` on ``sys.path`` preserves those imports while making all
    relative data and output paths resolve from the public repository root.
    """

    script_path = FRAMEWORK_ROOT / script
    if not script_path.is_file():
        raise FileNotFoundError(f"Framework entry point not found: {script_path}")

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    inserted_paths = [str(PROJECT_ROOT), str(FRAMEWORK_ROOT)]
    for path in reversed(inserted_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.chdir(PROJECT_ROOT)
    sys.argv = [str(script_path), *(list(argv) if argv is not None else old_argv[1:])]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
