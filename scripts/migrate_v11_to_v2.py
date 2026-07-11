#!/usr/bin/env python3
"""Compatibility wrapper for ``lunavla-v2 migrate-config``."""

from __future__ import annotations

import sys

from lunavla.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["migrate-config", *sys.argv[1:]]))
