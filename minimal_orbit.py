#!/usr/bin/env python3
"""Compatibility wrapper for the modular Satmulator CLI."""

from simulator.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
