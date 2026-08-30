"""Suppress noisy stderr from third-party libs (yfinance curl warnings)."""
from __future__ import annotations

import contextlib
import os
import sys


@contextlib.contextmanager
def SuppressedStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(old, 2)
        os.close(devnull)
        os.close(old)
