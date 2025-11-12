"""
Fixtures defined in here are discovered by all tests automatically.
https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
See also https://stackoverflow.com/a/34520971.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest



def assert_exists_and_not_empty(file_path: Path):
    """Assert that a file exists and is not empty."""
    assert file_path.exists(), f"File {file_path} does not exist."
    assert file_path.stat().st_size > 0, f"File {file_path} is empty."