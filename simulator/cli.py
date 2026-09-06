from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .runner import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal satellite orbit simulator",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="complete standalone JSON config file",
    )
    return parser.parse_args()


def main() -> int:
    return run(load_config(parse_args().config))
