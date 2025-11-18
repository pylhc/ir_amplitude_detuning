"""
Fixtures defined in here are discovered by all tests automatically.
https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
See also https://stackoverflow.com/a/34520971.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import git

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

GITLAB_REPO_ACC_MODELS: str = "https://gitlab.cern.ch/acc-models/acc-models-{}.git"


def assert_exists_and_not_empty(file_path: Path):
    """Assert that a file exists and is not empty."""
    assert file_path.exists(), f"File {file_path} does not exist."
    assert file_path.stat().st_size > 0, f"File {file_path} is empty."


def clone_acc_models(tmp_path_factory: pytest.TempPathFactory, accel: str, year: int) -> Path:
    """Clone the acc-models directory for the specified accelerator from github into a temporary directory."""
    tmp_path_repo = tmp_path_factory.mktemp(f"acc-models-{accel}")
    tmp_path_year = tmp_path_repo / str(year)
    tmp_path_year.mkdir(exist_ok=True)
    git.Repo.clone_from(GITLAB_REPO_ACC_MODELS.format(accel), tmp_path_year, branch=str(year))
    return tmp_path_year
