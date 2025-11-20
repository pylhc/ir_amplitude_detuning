
from __future__ import annotations

import logging
import re
from functools import partial
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import tfs
from cpymad_lhc.general import match_tune
from cpymad_lhc.logging import LOG_CMD_LVL
from pandas.testing import assert_frame_equal, assert_series_equal

from ir_amplitude_detuning.lhc_detuning_corrections import get_optics
from ir_amplitude_detuning.simulation import lhc_simulation
from ir_amplitude_detuning.simulation.lhc_simulation import (
    ACC_MODELS,
    drop_allzero_columns,
    pathstr,
)
from tests.conftest import assert_exists_and_not_empty

if TYPE_CHECKING:
    from ir_amplitude_detuning.utilities.correctors import Corrector

# ==============================================================================
# LHC Beam tests and fixtures
# ==============================================================================

# Trigger fixture to prepare Models --------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def prepare_models_local(prepare_models):
    return prepare_models


# Parameterized fixtures -------------------------------------------------------

@pytest.fixture(params=(1, 4))
def beam_no(request):
    return request.param


@pytest.fixture(params=(2018, 2022))
def model_year(request):
    return request.param


# Beam Setup fixture -----------------------------------------------------------

@pytest.fixture()
def lhc_beam(beam_no, model_year, tmp_path_factory, caplog):
    """Initialize and set up the LHC-Beam.
    This is actually the first test, but as it runs quickly and both other
    beam related tests need this setup, it's put into a fixture."""

    caplog.set_level(LOG_CMD_LVL)

    # The MAD-X setup removes all log-handlers. But caplog already added some. Store them!
    root_logger = logging.getLogger("")
    handlers = root_logger.handlers[-2:]  # save the handlers from caplog

    # ==========================================================================
    # Init Simulation Class
    # ==========================================================================

    lhc_beam = lhc_simulation.LHCBeam(
        beam=beam_no,
        outputdir=tmp_path_factory.mktemp(f"b{beam_no}_{model_year}"),
        xing={'scheme': 'top'},
        optics=get_optics(model_year),  # NOTE: optics after loading have collision tune
        year=model_year,
        tune_x=62.28,
        tune_y=60.31,
        chroma=3,
    )
    root_logger.handlers.extend(handlers)   # add caplog handlers back

    mvars = lhc_beam.madx.globals

    # checks ---
    assert mvars["mylhcbeam"] == lhc_beam.beam
    assert (lhc_beam.outputdir / lhc_simulation.ACC_MODELS).is_symlink()

    # ==========================================================================
    # Setup LHC
    # ==========================================================================
    lhc_beam.setup_machine()

    # check if correct files were called ---
    seq_name = f"lhcb{1 if lhc_beam.beam == 1 else 2}"
    seq_file = lhc_simulation.ACC_MODELS / f"lhc{'b4' if lhc_beam.beam == 4 else ''}{'_as-built' if lhc_beam.year == 2018 else ''}.seq"
    bv = 1

    assert f"call, file=\"{seq_file}\";" in caplog.text
    assert f"call, file=\"{lhc_beam.optics}\";" in caplog.text
    assert f"beam, sequence={seq_name}, bv={bv}" in caplog.text
    assert f"use, sequence={seq_name};" in caplog.text

    # quick test for orbit setup ---
    mvars = lhc_beam.madx.globals
    if lhc_beam.year == 2018:
        assert mvars['on_x1'] == 160
        assert mvars['on_x8'] == -250
    else:
        assert mvars['on_x1_v'] == -160
        assert mvars['on_x8h'] == -200

    caplog.clear()
    return lhc_beam


# Test Class -------------------------------------------------------------------

class TestLHCBeam:
    """Test the LHCBeam class."""

    def test_always_use_beam4(self, tmp_path):
        """Test that Beam 4 is used, even if Beam 2 is given."""
        lhc_beam = lhc_simulation.LHCBeam(
            beam=2,
            outputdir=tmp_path,
            xing={'scheme': 'top'},
            optics=get_optics(2022),
            year=2022,
            tune_x=62.28,
            tune_y=60.31,
            chroma=3,
        )
        assert lhc_beam.beam == 4
        assert lhc_beam.madx.globals["mylhcbeam"] == 4
        assert lhc_beam.seq_name == "lhcb2"
        assert lhc_beam.seq_file == "lhcb4.seq"
        assert lhc_beam.bv_flag == 1

    def test_lhc_simulation_nominal(self, lhc_beam: lhc_simulation.LHCBeam, monkeypatch):
        """Test the nominal setup, i.e. matching to the correct tunes and get a twiss."""
        # make matching a bit faster
        monkeypatch.setattr(lhc_simulation, "match_tune", partial(match_tune, step=1e-5, tolerance=1e-7, calls=20))

        lhc_beam.get_ampdet = Mock()  # takes too long; tested in examples
        lhc_beam.get_ampdet.return_value = "I have returned"

        # before state ----
        df_before = lhc_beam.get_twiss()
        assert df_before.headers["Q1"] == pytest.approx(62.31)
        assert df_before.headers["Q2"] == pytest.approx(60.32)
        assert df_before.headers["DQ1"] < 2.5  # not a specific value, just what the optics
        assert df_before.headers["DQ2"] < 2.5  # files spit out. Important is that it's not 3!

        # run ---
        lhc_beam.save_nominal()

        # after state ---
        assert lhc_beam.get_ampdet.call_count == 1
        assert lhc_beam.df_ampdet_nominal == lhc_beam.get_ampdet.return_value

        assert_exists_and_not_empty(lhc_beam.output_path("twiss", "nominal"))
        assert_exists_and_not_empty(lhc_beam.output_path("twiss", "optics_ir"))

        assert isinstance(lhc_beam.df_twiss_nominal, tfs.TfsDataFrame)
        assert isinstance(lhc_beam.df_twiss_nominal_ir, tfs.TfsDataFrame)

        df = lhc_beam.df_twiss_nominal
        assert df.headers["Q1"] == pytest.approx(62.28)
        assert df.headers["Q2"] == pytest.approx(60.31)
        assert df.headers["DQ1"] == pytest.approx(3, rel=1e-4)
        assert df.headers["DQ2"] == pytest.approx(3, rel=1e-4)

        lhc_beam.madx.exit()


    def test_lhc_simulation_corrector_install(self, lhc_beam: lhc_simulation.LHCBeam, caplog):
        """Checks the installation of the corrector circuits/magnets. Does not need
        the tunes to be matched, so can be run independently from the nominal test."""
        mvars = lhc_beam.madx.globals
        log_pattern = r"\s*{name}\s+([0-9.e+\-]+)\s+([0-9.e+\-]+)\%\s+(\d+)\%"
        rng = np.random.default_rng(3897329)

        def get_n(corrector: Corrector) -> int:
            return int(corrector.field[-1]) - 1


        def loop_correctors(masks = (lhc_simulation.LHCCorrectors.b5, lhc_simulation.LHCCorrectors.b6)):
            assert len(masks)

            for ip in (1, 5):
                for side in "LR":
                    for corrector_mask in masks:
                        yield corrector_mask.get_corrector(side=side, ip=ip)

        # before ---
        values = {}
        df = lhc_beam.get_twiss()

        for corrector in loop_correctors():
            assert df.loc[corrector.magnet, f"K{get_n(corrector)}L"] == 0
            values[corrector.circuit] =  (rng.random() + 1) * 1000
            mvars[corrector.circuit] = values[corrector.circuit]
            assert mvars[f"l.{corrector.madx_type}"] == corrector.length

        # run ---
        lhc_beam.install_circuits_into_mctx()

        # after ---
        for corrector in loop_correctors():
            assert mvars[corrector.circuit] == 0  # should have been reset by the function
            mvars[corrector.circuit] = values[corrector.circuit]

        df = lhc_beam.get_twiss()

        for corrector in loop_correctors():
            twiss_knl = df.loc[corrector.magnet, f"K{get_n(corrector)}L"]
            assert twiss_knl != 0

            test_knl = values[corrector.circuit] * corrector.length
            if (corrector.field == lhc_simulation.FieldComponent.b6 and lhc_beam.beam != 1):
                test_knl = -test_knl
            assert twiss_knl == pytest.approx(test_knl)

        # run checks ---
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            lhc_beam.check_kctx_limits()

        for corrector in loop_correctors((lhc_simulation.LHCCorrectors.b6,)):
            line = re.search(log_pattern.format(name=corrector.circuit.upper()), caplog.text)
            assert line is not None
            assert float(line.group(1)) == pytest.approx(values[corrector.circuit], rel=1e-2)
            assert 0 < abs(float(line.group(2))) < 1
            assert 2 <= float(line.group(3)) <= 7

        # reset ---
        lhc_beam.reset_detuning_circuits()

        df = lhc_beam.get_twiss()
        for corrector in loop_correctors():
            assert mvars[corrector.circuit] == 0
            assert df.loc[corrector.magnet, f"K{get_n(corrector)}L"] == 0

        # run checks ---
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            lhc_beam.check_kctx_limits()

        for corrector in loop_correctors((lhc_simulation.LHCCorrectors.b6,)):
            line = re.search(log_pattern.format(name=corrector.circuit.upper()), caplog.text)
            assert line is not None
            assert float(line.group(1)) == 0
            assert float(line.group(2)) == 0
            assert float(line.group(3)) == 0

        lhc_beam.madx.exit()


# ==============================================================================
# Function Unit test
# ==============================================================================


class TestPathstr:
    """Tests for the `pathstr` helper."""

    def test_returns_string(self):
        """The function should always return a string."""
        result = pathstr("some", "subdir")
        assert isinstance(result, str), "pathstr should return a string"

    def test_joins_correctly(self):
        """Returned path must equal ACC_MODELS joined with the arguments."""
        expected = str(ACC_MODELS.joinpath("some", "subdir"))
        assert pathstr("some", "subdir") == expected

    def test_single_argument(self):
        """Single‑argument case should still join to the base directory."""
        expected = str(ACC_MODELS.joinpath("single"))
        assert pathstr("single") == expected

    def test_no_arguments(self):
        """Calling with no arguments should return the base path as a string."""
        expected = str(ACC_MODELS)
        assert pathstr() == expected


class TestDropAllZeroColumns:
    def test_drop_zero_columns(self):
        """Columns that contain only zeros should be removed."""
        df = pd.DataFrame({
            "a": [0, 0, 0],
            "b": [1, 2, 3],
            "c": [0, 5, 0]
        })
        result = drop_allzero_columns(df)
        assert set(result.columns) == {"b", "c"}

    def test_no_zero_columns(self):
        """If no column is all zeros, the original DataFrame should be unchanged."""
        df = pd.DataFrame({
            "x": [1, 0, 3],
            "y": [4, 5, 6]
        })
        result = drop_allzero_columns(df)
        assert_frame_equal(result, df)

    def test_keep_parameter_respected(self):
        """Columns listed in `keep` stay even if they are all zeros."""
        df = pd.DataFrame({
            "zero1": [0, 0, 0],
            "keep_me": [0, 0, 0],   # all zeros but should be kept
            "nonzero": [1, 2, 3]
        })
        # Request to keep the column named "keep_me"
        result = drop_allzero_columns(df, keep=["keep_me"])
        assert set(result.columns) == {"keep_me", "nonzero"}
        # Ensure the kept column data is unchanged
        assert_series_equal(result["keep_me"], df["keep_me"])
