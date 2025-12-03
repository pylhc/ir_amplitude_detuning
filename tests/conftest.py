"""
Fixtures defined in here are discovered by all tests automatically.
https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
See also https://stackoverflow.com/a/34520971.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import git
import matplotlib as mpl
import pytest

if sys.platform.startswith("win"):
    # Disable TKinter on Windows (not supported, at least in the github workflows).
    # This needs to be done BEFORE importing matplotlib anywhere else, so doing it
    # here should take care of that automatically.
    mpl.use("Agg")


# ============================================================================
# Important Paths
# ============================================================================

TEST_DIR: Path = Path(__file__).parent

TEST_EXAMPLES_DIR = TEST_DIR / "examples"
TEST_UNIT_DIR = TEST_DIR / "unit"

INPUTS_DIR: Path = TEST_DIR / "inputs"
SIMULATION_DATA_DIR: Path = INPUTS_DIR / "simulation2018"

PACKAGE_ROOT: Path = TEST_DIR.parent
EXAMPLES_DIR: Path = PACKAGE_ROOT / "examples"


# ============================================================================
# Make Accelerator Models avaliable
# ============================================================================

GITLAB_REPO_ACC_MODELS: str = "https://gitlab.cern.ch/acc-models/acc-models-{}.git"


@pytest.fixture(scope="module")
def prepare_models(tmp_path_factory: pytest.TempPathFactory):
    tmp_path_base = tmp_path_factory.mktemp("acc-models-lhc")

    # 2018
    tmp_path_2018 = tmp_path_base / "2018"
    tmp_path_2018.mkdir(exist_ok=True)

    for seq_file in SIMULATION_DATA_DIR.glob("*.seq"):
        shutil.copy(src=seq_file, dst=tmp_path_2018/seq_file.name)

    macro = SIMULATION_DATA_DIR / "macro.madx"
    toolkit_dir = tmp_path_2018 / "toolkit"
    toolkit_dir.mkdir(exist_ok=True)
    shutil.copy(src=macro, dst=toolkit_dir / macro.name)

    optics = SIMULATION_DATA_DIR / "opticsfile.22_ctpps2"
    optics_dir = tmp_path_2018 / "PROTON"
    optics_dir.mkdir(exist_ok=True)
    shutil.copy(src=optics, dst=optics_dir / optics.name)

    # 2022
    clone_acc_models(tmp_path_base, accel="lhc", year=2022)

    # Patch paths
    from ir_amplitude_detuning.simulation.lhc_simulation import PATHS

    mp = pytest.MonkeyPatch()
    mp.setitem(PATHS, "optics_runII", tmp_path_base)  # overwrite
    mp.setitem(PATHS, "acc_models_lhc", tmp_path_base)  # overwrite

    yield

    # teardown
    mp.undo()


def clone_acc_models(output_base: Path, accel: str, year: int) -> Path:
    """Clone the acc-models directory for the specified accelerator from github into the specified output base directory."""
    tmp_path_year = output_base / str(year)
    tmp_path_year.mkdir(exist_ok=True)
    git.Repo.clone_from(GITLAB_REPO_ACC_MODELS.format(accel), tmp_path_year, branch=str(year))
    return tmp_path_year


# ============================================================================
# Assertions
# ============================================================================

def assert_exists_and_not_empty(file_path: Path):
    """Assert that a file exists and is not empty."""
    assert file_path.exists(), f"File {file_path} does not exist."
    assert file_path.stat().st_size > 0, f"File {file_path} is empty."
