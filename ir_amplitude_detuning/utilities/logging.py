"""
Logging
-------

Provides a basic logging setup for the package.
"""
from __future__ import annotations

import logging
import sys


def log_setup(**kwargs):
    """Set up a basic logger."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(levelname)7s | %(message)s | %(name)s",
        **kwargs,
    )
