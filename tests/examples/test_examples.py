"""
Example Tests
-------------

Test that all the examples run, produce the expected output files and
the expected corretor settings.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tfs
from pandas.testing import assert_frame_equal

from ir_amplitude_detuning.simulation.lhc_simulation import PATHS
from ir_amplitude_detuning.utilities.constants import NAME
from tests.conftest import assert_exists_and_not_empty, clone_acc_models

if TYPE_CHECKING:
    from collections.abc import Iterable

EXAMPLES_TEST_DIR = Path(__file__).parent
INPUTS_DIR = EXAMPLES_TEST_DIR / "inputs"
SIMULATION_DATA_DIR = EXAMPLES_TEST_DIR / "simulation2018"
EXAMPLES_DIR = EXAMPLES_TEST_DIR.parents[1] / "examples"

if str(EXAMPLES_DIR.parent) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR.parent))

from examples import md3311, md6863, commissioning_2022  # noqa


# Fixtures to prepare Models ---------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def prepare_models(tmp_path_factory: pytest.TempPathFactory):
    # 2018
    temp_path_2018 = tmp_path_factory.mktemp("acc-models-lhc-2018")
    for seq_file in SIMULATION_DATA_DIR.glob("*.seq"):
        shutil.copy(src=seq_file, dst=temp_path_2018/seq_file.name)

    macro = SIMULATION_DATA_DIR / "macro.madx"
    toolkit_dir = temp_path_2018 / "toolkit"
    toolkit_dir.mkdir(exist_ok=True)
    shutil.copy(src=macro, dst=toolkit_dir / macro.name)

    optics = SIMULATION_DATA_DIR / "opticsfile.22_ctpps2"
    optics_dir = temp_path_2018 / "PROTON"
    optics_dir.mkdir(exist_ok=True)
    shutil.copy(src=optics, dst=optics_dir / optics.name)

    # 2022
    temp_path_2022 = clone_acc_models(tmp_path_factory, accel="lhc", year=2022)

    # UGLY :(
    PATHS.update({
        "optics2018": temp_path_2018,
        "optics_repo": temp_path_2022.parent,
    })



# Fixtures to prepare Outputdir ------------------------------------------------

@pytest.fixture(scope="class")
def example_md3311(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temp_path = tmp_path_factory.mktemp("md3311")
    md3311.LHCSimParams.outputdir = temp_path
    return temp_path


@pytest.fixture(scope="class")
def example_commissioning_2022(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temp_path = tmp_path_factory.mktemp("2022_commissioning")
    commissioning_2022.LHCSimParams.outputdir = temp_path
    return temp_path


@pytest.fixture(scope="class")
def example_md6863(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temp_path = tmp_path_factory.mktemp("2022_md6863")
    md6863.LHCSimParams.outputdir = temp_path
    return temp_path


# Tests ------------------------------------------------------------------------

@pytest.mark.example  # --------------------------------------------------------
class TestExamplesMD3311:
    """Tests for the 2018 MD3311 example."""
    INPUT_DIR = INPUTS_DIR / "md3311"

    @pytest.mark.dependency()
    def test_simulation(self, example_md3311: Path):
        """Test the simulation function and do-correction with passed beams."""
        output_dir = example_md3311

        lhc_beams = md3311.simulation()
        _check_simulation_output(output_dir)

        md3311.do_correction(lhc_beams=lhc_beams)
        _check_correction_output(output_dir, self.INPUT_DIR)

        md3311.check_correction(lhc_beams=lhc_beams)
        _check_correction_ptc_check_output(output_dir, self.INPUT_DIR)

        for beam in lhc_beams.values():
            beam.madx.exit()

    @pytest.mark.dependency(depends=["TestExamplesMD3311::test_simulation"])
    def test_do_correction(self, example_md3311: Path):
        """Test the correction calculation loading from files."""
        output_dir = example_md3311

        _clear_correction_output(output_dir)
        md3311.do_correction()
        _check_correction_output(output_dir, self.INPUT_DIR)

    @pytest.mark.dependency(depends=["TestExamplesMD3311::test_simulation"])
    def test_check_correction(self, example_md3311: Path):
        """Test the check correction ptc simulation when loading from files."""
        output_dir = example_md3311

        _clear_correction_output(output_dir)
        md3311.check_correction()
        _check_correction_ptc_check_output(output_dir, self.INPUT_DIR)

    @pytest.mark.dependency(depends=["TestExamplesMD3311::test_do_correction"])
    def test_plot_corrector_strengths(self, example_md3311: Path):
        """Test corrector strength plotting."""
        output_dir = example_md3311
        md3311.plot_corrector_strengths()

        assert_exists_and_not_empty(output_dir / "plot.b6_correctors.ip15.pdf")

    @pytest.mark.dependency(depends=["TestExamplesMD3311::test_do_correction"])
    def test_plot_detuning_compensation(self, example_md3311: Path):
        """Test detuning compensation plotting."""
        output_dir = example_md3311
        md3311.plot_detunig_compensation()

        for target_id in _get_target_ids(self.INPUT_DIR):
            for beam in (1, 2):
                assert_exists_and_not_empty(
                    output_dir / f"plot.ampdet_compensation.{target_id}.b{beam}.pdf"
                )


@pytest.mark.example  # --------------------------------------------------------
class TestExamplesCommissioning2022:
    """Tests for the 2022 commissioning example."""
    INPUT_DIR = INPUTS_DIR / "commissioning_2022"

    @pytest.mark.dependency()
    def test_simulation(self, example_commissioning_2022: Path):
        """Test the simulation function and do-correction with passed beams."""
        output_dir = example_commissioning_2022

        lhc_beams = commissioning_2022.simulation()
        _check_simulation_output(output_dir)

        commissioning_2022.do_correction(lhc_beams=lhc_beams)
        _check_correction_output(output_dir, self.INPUT_DIR)

        commissioning_2022.check_correction(lhc_beams=lhc_beams)
        _check_correction_ptc_check_output(output_dir, self.INPUT_DIR)

        for beam in lhc_beams.values():
            beam.madx.exit()

    @pytest.mark.dependency(depends=["TestExamplesCommissioning2022::test_simulation"])
    def test_do_correction(self, example_commissioning_2022: Path):
        """Test the correction calculation loading from files."""
        output_dir = example_commissioning_2022

        _clear_correction_output(output_dir)
        commissioning_2022.do_correction()
        _check_correction_output(output_dir, self.INPUT_DIR)

    @pytest.mark.dependency(depends=["TestExamplesCommissioning2022::test_simulation"])
    def test_check_correction(self, example_commissioning_2022: Path):
        """Test the check correction ptc simulation when loading from files."""
        output_dir = example_commissioning_2022

        _clear_correction_output(output_dir)
        commissioning_2022.check_correction()
        _check_correction_ptc_check_output(output_dir, self.INPUT_DIR, from_files=True)

    @pytest.mark.dependency(depends=["TestExamplesCommissioning2022::test_do_correction"])
    def test_plot_corrector_strengths(self, example_commissioning_2022: Path):
        """Test corrector strength plotting."""
        output_dir = example_commissioning_2022
        commissioning_2022.plot_corrector_strengths()

        assert_exists_and_not_empty(output_dir / "plot.b6_correctors.ip15.pdf")

    @pytest.mark.dependency(depends=["TestExamplesCommissioning2022::test_do_correction"])
    def test_plot_detuning_comparison(self, example_commissioning_2022: Path):
        """Test detuning compensation plotting."""
        output_dir = example_commissioning_2022
        commissioning_2022.plot_detuning_comparison()

        for beam in (1, 2):
            assert_exists_and_not_empty(
                output_dir / f"plot.ampdet_comparison.b{beam}.pdf"
            )

    @pytest.mark.dependency(depends=["TestExamplesCommissioning2022::test_do_correction"])
    def test_plot_simulation_comparison(self, example_commissioning_2022: Path):
        """Test detuning compensation plotting."""
        output_dir = example_commissioning_2022
        commissioning_2022.plot_detuning_comparison()

        for beam in (1, 2):
            assert_exists_and_not_empty(
                output_dir / f"plot.ampdet_sim_comparison.b{beam}.pdf"
            )


@pytest.mark.example  # --------------------------------------------------------
class TestExamplesMD6863:
    """Tests for the 2022 MD6863 example."""
    INPUT_DIR = INPUTS_DIR / "md6863"

    @pytest.mark.dependency()
    def test_simulation(self, example_md6863: Path):
        """Test the simulation function and do-correction with passed beams."""
        output_dir = example_md6863

        xing_configs = list(md6863.XingSchemes.keys())
        assert len(xing_configs) == 4

        lhc_beams = md6863.simulation()
        _check_simulation_output(output_dir, suffixes=xing_configs)

        md6863.do_correction(lhc_beams=lhc_beams)
        _check_correction_output(output_dir, self.INPUT_DIR)

        md6863.check_correction(lhc_beams=lhc_beams)
        _check_correction_ptc_check_output(output_dir, self.INPUT_DIR)

        for beam in lhc_beams.values():
            beam.madx.exit()

    @pytest.mark.dependency(depends=["TestExamplesMD6863::test_simulation"])
    def test_do_correction(self, example_md6863: Path):
        """Test the correction calculation loading from files."""
        output_dir = example_md6863

        _clear_correction_output(output_dir)
        md6863.do_correction()
        _check_correction_output(output_dir, self.INPUT_DIR)

    @pytest.mark.dependency(depends=["TestExamplesMD6863::test_simulation"])
    def test_check_correction(self, example_md6863: Path):
        """Test the check correction ptc simulation when loading from files."""
        output_dir = example_md6863

        _clear_correction_output(output_dir)
        md6863.check_correction()
        _check_correction_ptc_check_output(output_dir, self.INPUT_DIR)

    @pytest.mark.dependency(depends=["TestExamplesMD6863::test_do_correction"])
    def test_plot_corrector_strengths(self, example_md6863: Path):
        """Test corrector strength plotting."""
        output_dir = example_md6863
        md6863.plot_corrector_strengths()

        assert_exists_and_not_empty(output_dir / "plot.b6_correctors.ip15.pdf")

    @pytest.mark.dependency(depends=["TestExamplesMD6863::test_do_correction"])
    def test_plot_measurement_comparison(self, example_md6863: Path):
        """Test measurement comparison plotting."""
        output_dir = example_md6863
        md6863.plot_measurement_comparison()

        for beam in (1, 2):
            assert_exists_and_not_empty(
                output_dir / f"plot.ampdet_measured.all.b{beam}.pdf"
            )

    @pytest.mark.dependency(depends=["TestExamplesMD6863::test_do_correction"])
    def test_plot_target_comparison(self, example_md6863: Path):
        """Test target comparison plotting."""
        output_dir = example_md6863
        md6863.plot_target_comparison()

        for beam in (1, 2):
            for target_id in _get_target_ids(self.INPUT_DIR):
                assert_exists_and_not_empty(
                    output_dir / f"plot.ampdet_compensation.{target_id}.b{beam}.pdf"
                )

            assert_exists_and_not_empty(
                output_dir / f"plot.ampdet_compensation_and_measured_corrected.global.b{beam}.pdf"
            )


# Helper functions for the tests -------------------------------------------------------------------

def _get_target_ids(output_dir: Path) -> set[str]:
    target_ids = set()
    for settings_file in output_dir.glob("settings.lhc.b1.*.tfs"):
        parts = settings_file.stem.split(".")
        target_ids.add(parts[3])
    return target_ids


def _clear_correction_output(output_dir: Path):
    """Clear correction/correction check output files."""
    for glob in ("settings.*", "ampdet_calc*", "ampdet.*"):
        for found_file in output_dir.glob(glob):
            found_file.unlink()


def _check_correction_output(output_dir: Path, compare_dir: Path):
    """Check that correction output files are created."""
    target_ids = _get_target_ids(compare_dir)
    assert len(target_ids) == 3, "No settings found to compare test run to."  # test sanity check

    for target_id in target_ids:
        # Compare Expected Settings ---
        filename = f"settings.lhc.b1.{target_id}.tfs"
        df_new = tfs.read(output_dir / filename, index=NAME)
        df_compare = tfs.read(compare_dir / filename, index=NAME)
        assert_frame_equal(df_new, df_compare, rtol=1e-2, atol=1e-2)

        # Check files present ---
        for prefix in ("settings", "ampdet_calc", "ampdet_calc_err"):
            for beam in (1, 4):
                assert_exists_and_not_empty(output_dir / "{prefix}.lhc.b{beam}.{target_id}.tfs")

    # Compare Beams Settings ---
    b1_files = list(output_dir.glob("settings.lhc.b1*"))
    b4_files = list(output_dir.glob("settings.lhc.b4*"))
    assert len(b1_files) > 0, "No b1 settings found."
    assert len(b4_files) == len(b1_files), "Mismatch in number of b4 and b1 settings files."
    for b1_file, b4_file in zip(sorted(b1_files), sorted(b4_files)):
        assert b1_file.read_text() == b4_file.read_text(), f"Settings files {b1_file} and {b4_file} are different."


def _check_simulation_output(output_dir: Path, suffixes: Iterable[str] = ('',)):
    """Check that simulation output files are created."""
    for beam in (1, 4):
        for suffix in suffixes:
            simulation_dir = output_dir / f"{suffix}{'_' if suffix else ''}b{beam}"
            assert simulation_dir.is_dir()

            for file_name in (
                f"ampdet.lhc.b{beam}.nominal.tfs",
                f"twiss.lhc.b{beam}.nominal.tfs",
                "full_output.log",
                "madx_commands.log"
                ):
                assert_exists_and_not_empty(simulation_dir / file_name)
                if file_name.endswith(".tfs"):
                    tfs.read(simulation_dir / file_name, index=NAME)


def _check_correction_ptc_check_output(output_dir: Path, compare_dir: Path, from_files: bool = False):
    """Check that correction output files are created."""
    target_ids = _get_target_ids(compare_dir)
    assert target_ids > 0, "No settings found to compare test run to."  # test sanity check

    # Check files present ---
    for target_id in target_ids:
        for beam in (1, 4):
            assert_exists_and_not_empty(output_dir / f"ampdet.lhc.b{beam}.{target_id}.tfs")

    if from_files:
        for beam in (1, 4):
            tmp_ptc_dir = output_dir / f"tmp_ptc_b{beam}"
            assert tmp_ptc_dir.is_dir()
            assert_exists_and_not_empty(tmp_ptc_dir / "full_output.log")
            assert_exists_and_not_empty(tmp_ptc_dir / "madx_commands.log")
