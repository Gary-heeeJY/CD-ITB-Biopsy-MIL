"""Shared helpers for the public command-line entry points."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def has_option(args: list[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in args)


def with_default(args: list[str], option: str, value: str) -> list[str]:
    if has_option(args, option):
        return args
    return [*args, option, value]


def current_args() -> list[str]:
    return sys.argv[1:]
