from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ir_amplitude_detuning.detuning.equation_system import (
    BETA,
    ROW_ID,
    DetuningCorrectionEquationSystem,
    beam_symmetry_sign,
    build_detuning_correction_matrix,
    build_detuning_correction_matrix_per_entry,
    calculate_matrix_row,
    get_detuning_coeff,
)
from ir_amplitude_detuning.detuning.measurements import (
    Constraints,
    Detuning,
    FirstOrderTerm,
    MeasureValue,
    SecondOrderTerm,
)
from ir_amplitude_detuning.detuning.targets import TargetData
from ir_amplitude_detuning.utilities.correctors import Corrector, FieldComponent


# ============================================================================
# Tests for DetuningCorrectionEquationSystem
# ============================================================================

class TestDetuningCorrectionEquationSystem:
    """Test cases for the DetuningCorrectionEquationSystem dataclass."""

    def test_create_empty_no_columns(self):
        """Test creating an empty equation system without columns."""
        eqsys = DetuningCorrectionEquationSystem.create_empty()

        assert isinstance(eqsys.m, pd.DataFrame)
        assert isinstance(eqsys.v, pd.Series)
        assert isinstance(eqsys.m_constr, pd.DataFrame)
        assert isinstance(eqsys.v_constr, pd.Series)
        assert isinstance(eqsys.v_meas, pd.Series)

        assert len(eqsys.m) == 0
        assert len(eqsys.v) == 0
        assert len(eqsys.m_constr) == 0
        assert len(eqsys.v_constr) == 0
        assert len(eqsys.v_meas) == 0

    def test_create_empty_with_columns(self):
        """Test creating an empty equation system with columns."""
        columns = ["c1", "c2", "c3"]
        eqsys = DetuningCorrectionEquationSystem.create_empty(columns=columns)

        assert list(eqsys.m.columns) == columns
        assert list(eqsys.m_constr.columns) == columns
        assert len(eqsys.m) == 0
        assert len(eqsys.m_constr) == 0

    def test_append_series_to_matrix(self):
        """Test appending a series as a row to the matrix."""
        eqsys = DetuningCorrectionEquationSystem.create_empty(columns=["a", "b", "c"])
        series = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])

        eqsys.append_series_to_matrix(series)

        assert len(eqsys.m) == 1
        assert_series_equal(eqsys.m.iloc[0], series, check_names=False)

    def test_append_series_to_matrix_multiple(self):
        """Test appending multiple series to the matrix."""
        eqsys = DetuningCorrectionEquationSystem.create_empty(columns=["a", "b"])
        series1 = pd.Series([1.0, 2.0], index=["a", "b"])
        series2 = pd.Series([3.0, 4.0], index=["a", "b"])

        eqsys.append_series_to_matrix(series1)
        eqsys.append_series_to_matrix(series2)

        assert len(eqsys.m) == 2
        assert_series_equal(eqsys.m.iloc[0], series1, check_names=False)
        assert_series_equal(eqsys.m.iloc[1], series2, check_names=False)

    def test_append_series_to_constraints_matrix(self):
        """Test appending a series as a row to the constraints matrix."""
        eqsys = DetuningCorrectionEquationSystem.create_empty(columns=["a", "b"])
        series = pd.Series([1.0, 2.0], index=["a", "b"])

        eqsys.append_series_to_constraints_matrix(series)

        assert len(eqsys.m_constr) == 1
        assert_series_equal(eqsys.m_constr.iloc[0], series, check_names=False)

    def test_set_value_float(self):
        """Test setting a value in v and v_meas with a float."""
        eqsys = DetuningCorrectionEquationSystem.create_empty()
        eqsys.set_value("test", 5.0)

        assert eqsys.v["test"] == 5.0
        assert isinstance(eqsys.v_meas["test"], MeasureValue)
        assert eqsys.v_meas["test"].value == 5.0
        assert eqsys.v_meas["test"].error == 0.0

    def test_set_value_measure_value(self):
        """Test setting a value in v and v_meas with a MeasureValue."""
        eqsys = DetuningCorrectionEquationSystem.create_empty()
        measure = MeasureValue(3.5, 0.2)

        eqsys.set_value("test", measure)

        assert eqsys.v["test"] == 3.5
        assert eqsys.v_meas["test"] == measure

    def test_set_constraint(self):
        """Test setting a constraint value."""
        eqsys = DetuningCorrectionEquationSystem.create_empty()
        eqsys.set_constraint("constr1", 10.0)

        assert eqsys.v_constr["constr1"] == 10.0

    def test_append_all(self):
        """Test appending all matrices and vectors from another equation system."""
        eqsys = DetuningCorrectionEquationSystem.create_empty(columns=["a", "b"])

        eqsys1 = DetuningCorrectionEquationSystem.create_empty(columns=["a", "b"])
        eqsys1.append_series_to_matrix(pd.Series([1.0, 2.0], index=["a", "b"]))
        eqsys1.append_series_to_constraints_matrix(pd.Series([3.0, 4.0], index=["a", "b"]))
        eqsys1.set_value("v1", 5.0)
        eqsys1.set_constraint("v1", 6.0)

        eqsys2 = DetuningCorrectionEquationSystem.create_empty(columns=["a", "b"])
        eqsys2.append_series_to_matrix(pd.Series([3.0, 4.0], index=["a", "b"]))
        eqsys2.set_value("v2", 7.0)


        for i in range(2):
            eqsys.append_all(eqsys1)
            assert len(eqsys.m) == i+1
            assert len(eqsys.v) == i+1
            assert len(eqsys.v_meas) == i+1
            assert len(eqsys.m_constr) == i+1
            assert len(eqsys.v_constr) == i+1

            assert all(eqsys.m.iloc[i] == eqsys1.m)
            assert all(eqsys.v.iloc[i] == eqsys1.v)
            assert all(eqsys.v_meas.iloc[i] == eqsys1.v_meas)
            assert all(eqsys.m_constr.iloc[i] == eqsys1.m_constr)
            assert all(eqsys.v_constr.iloc[i] == eqsys1.v_constr)

        eqsys.append_all(eqsys2)
        assert len(eqsys.m) == 3
        assert len(eqsys.v) == 3
        assert len(eqsys.v_meas) == 3
        assert len(eqsys.m_constr) == 2
        assert len(eqsys.v_constr) == 2

        assert all(eqsys.m.iloc[-1] == eqsys2.m)
        assert all(eqsys.v.iloc[-1] == eqsys2.v)
        assert all(eqsys.v_meas.iloc[-1] == eqsys2.v_meas)
        assert all(eqsys.m_constr.iloc[-1] == eqsys1.m_constr)
        assert all(eqsys.v_constr.iloc[-1] == eqsys1.v_constr)

    def test_fillna(self):
        """Test filling NaN values with zeros."""
        eqsys = DetuningCorrectionEquationSystem.create_empty(columns=["a", "b", "c"])
        eqsys.m = pd.DataFrame([[1.0, np.nan, 3.0], [np.nan, 2.0, np.nan]], columns=["a", "b", "c"])
        eqsys.m_constr = pd.DataFrame([[np.nan, 1.0, 2.0]], columns=["a", "b", "c"])

        eqsys.fillna()

        assert not eqsys.m.isna().any().any()
        assert not eqsys.m_constr.isna().any().any()
        assert eqsys.m.loc[0, "b"] == 0.0
        assert eqsys.m.loc[1, "a"] == 0.0


# ============================================================================
# Tests for beam_symmetry_sign
# ============================================================================

class TestBeamSymmetrySign:
    """Test cases for the beam_symmetry_sign function."""

    def test_beam_symmetry_sign_beam_1(self):
        """Test symmetry sign for odd beam numbers."""
        assert beam_symmetry_sign(1) == 1
        assert beam_symmetry_sign(2) == -1
        assert beam_symmetry_sign(4) == -1


# ============================================================================
# Tests for get_detuning_coeff
# ============================================================================

class TestGetDetuningCoeff:
    """Test cases for the get_detuning_coeff function."""

    def test_first_order_direct_terms(self):
        """Test first order direct detuning coefficients."""
        beta = {"X": 100.0, "Y": 200.0}

        # X10 term
        coeff_x10 = get_detuning_coeff(FirstOrderTerm.X10, beta)
        assert coeff_x10 == pytest.approx(100.0**2 / (32 * np.pi))

        # Y01 term
        coeff_y01 = get_detuning_coeff(FirstOrderTerm.Y01, beta)
        assert coeff_y01 == pytest.approx(200.0**2 / (32 * np.pi))

    def test_first_order_cross_terms(self):
        """Test first order cross detuning coefficients."""
        beta = {"X": 100.0, "Y": 200.0}

        coeff_x01 = get_detuning_coeff(FirstOrderTerm.X01, beta)
        coeff_y10 = get_detuning_coeff(FirstOrderTerm.Y10, beta)

        assert coeff_x01 == coeff_y10 == pytest.approx(-100.0 * 200.0 / (16 * np.pi))

    def test_second_order_direct_terms(self):
        """Test second order direct detuning coefficients."""
        beta = {"X": 100.0, "Y": 200.0}

        # X20 term
        coeff_x20 = get_detuning_coeff(SecondOrderTerm.X20, beta)
        assert coeff_x20 == pytest.approx(100.0**3 / (384 * np.pi))

        # Y02 term
        coeff_y02 = get_detuning_coeff(SecondOrderTerm.Y02, beta)
        assert coeff_y02 == pytest.approx(-200.0**3 / (384 * np.pi))

    def test_second_order_cross_terms(self):
        """Test second order cross detuning coefficients."""
        beta = {"X": 100.0, "Y": 200.0}

        coeff_x11 = get_detuning_coeff(SecondOrderTerm.X11, beta)
        coeff_y20 = get_detuning_coeff(SecondOrderTerm.Y20, beta)
        assert coeff_y20 == coeff_x11 == pytest.approx(-100.0**2 * 200.0 / (128 * np.pi))

        coeff_y11 = get_detuning_coeff(SecondOrderTerm.Y11, beta)
        coeff_x02 = get_detuning_coeff(SecondOrderTerm.X02, beta)
        assert coeff_x02 == coeff_y11 == pytest.approx(100.0 * 200.0**2 / (128 * np.pi))


    def test_get_detuning_coeff_case_insensitive(self):
        """Test that get_detuning_coeff handles both upper and lower case terms."""
        beta = {"X": 100.0, "Y": 200.0}

        coeff1 = get_detuning_coeff("x10", beta)
        coeff2 = get_detuning_coeff("X10", beta)
        assert coeff1 == coeff2

    def test_get_detuning_coeff_unknown_term(self):
        """Test that unknown term raises KeyError."""
        beta = {"X": 100.0, "Y": 200.0}

        with pytest.raises(KeyError):
            get_detuning_coeff("UNKNOWN", beta)


# ============================================================================
# Tests for calculate_matrix_row
# ============================================================================

class TestCalculateMatrixRow:
    """Test cases for the calculate_matrix_row function."""

    def _create_mock_twiss(self, magnets):
        """Helper to create a mock twiss dataframe."""
        data = {
            "X": np.zeros(len(magnets)),
            "Y": np.zeros(len(magnets)),
            "BETX": 100.0 * np.ones(len(magnets)),
            "BETY": 200.0 * np.ones(len(magnets)),
        }
        return pd.DataFrame(data, index=magnets)

    def test_calculate_matrix_row_first_order_b4(self):
        """Test matrix row calculation for first order b4 term."""
        corrector = Corrector(
            field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1
        )
        twiss = self._create_mock_twiss([corrector.magnet])

        row = calculate_matrix_row(1, twiss, [corrector], FirstOrderTerm.X10)

        assert len(row) == 1
        assert corrector in row
        expected = get_detuning_coeff(FirstOrderTerm.X10, {"X": 100.0, "Y": 200.0})
        assert row[corrector] == pytest.approx(expected)


    @pytest.mark.parametrize("term, field", (
        (FirstOrderTerm.X10, FieldComponent.b4),
        (FirstOrderTerm.X01, FieldComponent.b6),
        (SecondOrderTerm.X11, FieldComponent.b6))
    )
    def test_calculate_matrix_row_b4_beam_symmetry(self, term, field):
        """Test that matrix row respects beam symmetry sign."""
        corrector = Corrector(
            field=field, length=0.5, magnet="c1b4", circuit="k1b4", ip=1
        )
        twiss = self._create_mock_twiss([corrector.magnet])

        row_beam1 = calculate_matrix_row(1, twiss, [corrector], term)
        row_beam2 = calculate_matrix_row(2, twiss, [corrector], term)

        assert row_beam1[corrector] == pytest.approx(-row_beam2[corrector])

    def test_calculate_matrix_row_b5_feeddown(self):
        """Test matrix row calculation for b5 feeddown to first order."""
        corrector = Corrector(
            field=FieldComponent.b5, length=0.5, magnet="c1b5", circuit="k1b5", ip=1
        )
        twiss = self._create_mock_twiss([corrector.magnet])
        twiss.loc[corrector.magnet, "X"] = 2.0  # offset in x

        row = calculate_matrix_row(1, twiss, [corrector], FirstOrderTerm.X10)

        coeff = get_detuning_coeff(FirstOrderTerm.X10, {"X": 100.0, "Y": 200.0})
        expected = 2.0 * coeff  # x * coeff
        assert row[corrector] == pytest.approx(expected)

    def test_calculate_matrix_row_b6_feeddown(self):
        """Test matrix row calculation for b6 feeddown to first order."""
        corrector = Corrector(
            field=FieldComponent.b6, length=0.5, magnet="c1b6", circuit="k1b6", ip=1
        )
        twiss = self._create_mock_twiss(["c1b6"])
        twiss.loc[corrector.magnet, "X"] = 3.0
        twiss.loc[corrector.magnet, "Y"] = 1.0

        row = calculate_matrix_row(1, twiss, [corrector], FirstOrderTerm.X10)

        coeff = get_detuning_coeff(FirstOrderTerm.X10, {"X": 100.0, "Y": 200.0})
        expected = 1 * 0.5 * (3.0**2 - 1.0**2) * coeff
        assert row[corrector] == pytest.approx(expected)

    def test_calculate_matrix_row_second_order_b6(self):
        """Test matrix row calculation for second order b6 term."""
        corrector = Corrector(
            field=FieldComponent.b6, length=0.5, magnet="c1b6", circuit="k1b6", ip=1
        )
        twiss = self._create_mock_twiss([corrector.magnet])

        row = calculate_matrix_row(1, twiss, [corrector], SecondOrderTerm.X20)

        expected = 1 * get_detuning_coeff(SecondOrderTerm.X20, {"X": 100.0, "Y": 200.0})
        assert row[corrector] == pytest.approx(expected)

    def test_calculate_matrix_row_multiple_correctors(self):
        """Test matrix row with multiple correctors."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c2b4", circuit="k2b4", ip=1),
            Corrector(field=FieldComponent.b5, length=0.5, magnet="c1b5", circuit="k1b5", ip=1),
            Corrector(field=FieldComponent.b6, length=0.5, magnet="c1b6", circuit="k1b6", ip=2),
        ]
        twiss = self._create_mock_twiss([c.magnet for c in correctors])
        twiss.loc["c1b5", "X"] = 1.0
        twiss.loc["c1b6", "X"] = np.sqrt(3)  # to get a total coefficient of 1
        twiss.loc["c1b6", "Y"] = 1.0

        row = calculate_matrix_row(1, twiss, correctors, FirstOrderTerm.X10)

        assert len(row) == len(correctors)
        coeff = get_detuning_coeff(FirstOrderTerm.X10, {"X": 100.0, "Y": 200.0})
        for i in range(len(correctors)):
            assert row[correctors[i]] == pytest.approx(coeff)

    def test_calculate_matrix_row_magnet_not_in_twiss(self, caplog):
        """Test that missing magnet in twiss is skipped."""
        corrector1 = Corrector(
            field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1
        )
        corrector2 = Corrector(
            field=FieldComponent.b4, length=0.5, magnet="c2b4", circuit="k2b4", ip=1
        )
        twiss = self._create_mock_twiss([corrector1.magnet])  # c2b4 not present

        with caplog.at_level(logging.DEBUG):
            row = calculate_matrix_row(1, twiss, [corrector1, corrector2], FirstOrderTerm.X10)

        assert "magnet c2b4 not in twiss" in caplog.text
        assert row[corrector1] != 0
        assert row[corrector2] == 0

    def test_calculate_matrix_row_no_correctors(self):
        """Test with empty correctors list raises ValueError."""
        twiss = self._create_mock_twiss([])

        with pytest.raises(ValueError, match="[Nn]o detuning correctors"):
            calculate_matrix_row(1, twiss, [], FirstOrderTerm.X10)

    def test_calculate_matrix_row_no_b6_for_second_order(self):
        """Test that second order term requires b6 corrector."""
        corrector = Corrector(
            field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1
        )
        twiss = self._create_mock_twiss([corrector.magnet])

        with pytest.raises(ValueError, match="no b6 correctors"):
            calculate_matrix_row(1, twiss, [corrector], SecondOrderTerm.X20)

    def test_calculate_matrix_row_wrong_order(self):
        """Test that second order term requires b6 corrector."""
        corrector = Corrector(
            field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1
        )
        twiss = self._create_mock_twiss([corrector.magnet])

        with pytest.raises(NotImplementedError, match="[Oo]rder must be 1 or 2"):
            calculate_matrix_row(1, twiss, [corrector], "X22")

    def test_calculate_matrix_row_invalid_field(self):
        """Test that invalid field raises ValueError."""
        # Create a corrector with an invalid field component.
        # The real Correctors check for this already in their init,
        # so this should never happen.
        corrector = Mock()
        corrector.field = "invalid_field"
        corrector.magnet = "test"

        twiss = self._create_mock_twiss(["test"])

        with pytest.raises(ValueError, match="[Ff]ield must be one of .*invalid"):
            calculate_matrix_row(1, twiss, [corrector], FirstOrderTerm.X10)


# ============================================================================
# Tests for build_detuning_correction_matrix_per_entry
# ============================================================================

class TestBuildDetuningCorrectionMatrixPerEntry:
    """Test cases for the build_detuning_correction_matrix_per_entry function."""

    def _create_twiss_dataframe(self, magnets):
        """Helper to create a twiss dataframe."""
        data = {
            "X": np.zeros(len(magnets)),
            "Y": np.zeros(len(magnets)),
            "BETX": 100.0 * np.ones(len(magnets)),
            "BETY": 200.0 * np.ones(len(magnets)),
        }
        return pd.DataFrame(data, index=magnets)

    def test_build_matrix_per_entry_single_beam_single_term(self):
        """Test building equation system for a single beam and term."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
        ]
        twiss_b1 = self._create_twiss_dataframe(["c1b4"])

        # Create actual Detuning and Constraints objects
        detuning_b1 = Detuning()
        detuning_b1[FirstOrderTerm.X10] = MeasureValue(5.0, 0.1)

        constraints_b1 = Constraints()

        # Create TargetData
        target_data = TargetData(
            label="test_label",
            correctors=correctors,
            optics={1: twiss_b1},
            detuning={1: detuning_b1},
            constraints={1: constraints_b1},
        )

        # Run
        eqsys = build_detuning_correction_matrix_per_entry(target_data)

        # Check
        assert len(eqsys.m) == 1
        assert len(eqsys.v) == 1
        assert len(eqsys.v_meas) == 1
        assert len(eqsys.m_constr) == 0
        assert len(eqsys.v_constr) == 0
        assert eqsys.v["b1.test_label.X10"] == pytest.approx(5.0)
        assert eqsys.v_meas["b1.test_label.X10"].value == pytest.approx(5.0)
        assert eqsys.v_meas["b1.test_label.X10"].error == pytest.approx(0.1)

    def test_build_matrix_per_entry_multiple_beams(self):
        """Test building equation system with multiple beams."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
        ]
        twiss_b1 = self._create_twiss_dataframe(["c1b4"])
        twiss_b2 = self._create_twiss_dataframe(["c1b4"])

        # Create Detuning and Constraints for both beams
        detuning_b1 = Detuning()
        detuning_b1[FirstOrderTerm.X10] = MeasureValue(5.0, 0.1)

        detuning_b2 = Detuning()
        detuning_b2[FirstOrderTerm.X10] = MeasureValue(4.5, 0.15)

        constraints_b1 = Constraints()
        constraints_b2 = Constraints()

        target_data = TargetData(
            label="test_label",
            correctors=correctors,
            optics={1: twiss_b1, 2: twiss_b2},
            detuning={1: detuning_b1, 2: detuning_b2},
            constraints={1: constraints_b1, 2: constraints_b2},
        )

        # Run
        eqsys = build_detuning_correction_matrix_per_entry(target_data)

        # Check: 2 beams * 1 term = 2 rows
        assert len(eqsys.m) == 2
        assert len(eqsys.v) == 2
        assert len(eqsys.v_meas) == 2
        assert len(eqsys.m_constr) == 0
        assert len(eqsys.v_constr) == 0
        assert eqsys.v["b1.test_label.X10"] == pytest.approx(5.0)
        assert eqsys.v["b2.test_label.X10"] == pytest.approx(4.5)

    def test_build_matrix_per_entry_multiple_terms(self):
        """Test building equation system with multiple detuning terms."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
        ]
        twiss_b1 = self._create_twiss_dataframe(["c1b4"])

        # Create Detuning with multiple terms
        detuning_b1 = Detuning()
        detuning_b1[FirstOrderTerm.X10] = MeasureValue(5.0, 0.1)
        detuning_b1[FirstOrderTerm.Y01] = MeasureValue(3.0, 0.05)

        constraints_b1 = Constraints()

        target_data = TargetData(
            label="test_label",
            correctors=correctors,
            optics={1: twiss_b1},
            detuning={1: detuning_b1},
            constraints={1: constraints_b1},
        )

        # Run
        eqsys = build_detuning_correction_matrix_per_entry(target_data)

        # Check: 1 beam * 2 terms = 2 rows
        assert len(eqsys.m) == 2
        assert len(eqsys.v) == 2
        assert len(eqsys.v_meas) == 2
        assert len(eqsys.m_constr) == 0
        assert len(eqsys.v_constr) == 0
        assert eqsys.v["b1.test_label.X10"] == pytest.approx(5.0)
        assert eqsys.v["b1.test_label.Y01"] == pytest.approx(3.0)

    def test_build_matrix_per_entry_with_constraints(self):
        """Test building equation system with constraints."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
        ]
        twiss_b1 = self._create_twiss_dataframe(["c1b4"])

        detuning_b1 = Detuning()
        detuning_b1[FirstOrderTerm.X10] = MeasureValue(5.0, 0.1)

        # Create Constraints with a term
        constraints_b1 = Constraints()
        constraints_b1[FirstOrderTerm.Y01] = "<=10"

        target_data = TargetData(
            label="test_label",
            correctors=correctors,
            optics={1: twiss_b1},
            detuning={1: detuning_b1},
            constraints={1: constraints_b1},
        )

        # Run
        eqsys = build_detuning_correction_matrix_per_entry(target_data)

        # Check: 1 detuning row + 1 constraint row
        assert len(eqsys.m) == 1
        assert len(eqsys.m_constr) == 1
        assert len(eqsys.v) == 1
        assert len(eqsys.v_meas) == 1
        assert len(eqsys.v_constr) == 1
        assert eqsys.v["b1.test_label.X10"] == pytest.approx(5.0)
        assert eqsys.v_constr["b1.test_label.Y01"] == pytest.approx(10.0)

    def test_build_matrix_per_entry_constraints_with_negative_sign(self):
        """Test building equation system with constraints that have negative sign."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
        ]
        twiss_b1 = self._create_twiss_dataframe(["c1b4"])

        detuning_b1 = Detuning()
        detuning_b1[FirstOrderTerm.X10] = MeasureValue(5.0, 0.1)

        # Create Constraints with negative sign
        constraints_b1 = Constraints()
        constraints_b1[FirstOrderTerm.Y01] = ">=-15"  # sign=-1, value=15.0

        target_data = TargetData(
            label="test_label",
            correctors=correctors,
            optics={1: twiss_b1},
            detuning={1: detuning_b1},
            constraints={1: constraints_b1},
        )

        # Run
        eqsys = build_detuning_correction_matrix_per_entry(target_data)

        # Check: constraint row should be multiplied by -1
        assert len(eqsys.m_constr) == 1
        assert eqsys.v_constr["b1.test_label.Y01"] == pytest.approx(15.0)
        # The matrix row should be negated
        base_row = calculate_matrix_row(1, twiss_b1, correctors, FirstOrderTerm.Y01)
        negated_row = -1 * base_row
        assert_series_equal(
            eqsys.m_constr.iloc[0],
            negated_row,
            check_names=False,
        )


# ============================================================================
# Tests for build_detuning_correction_matrix
# ============================================================================

class TestBuildDetuningCorrectionMatrix:
    """Test cases for the build_detuning_correction_matrix function."""

    def _create_twiss_dataframe(self, magnets):
        """Helper to create a twiss dataframe."""
        data = {
            "X": np.zeros(len(magnets)),
            "Y": np.zeros(len(magnets)),
            "BETX": 100.0 * np.ones(len(magnets)),
            "BETY": 200.0 * np.ones(len(magnets)),
        }
        return pd.DataFrame(data, index=magnets)

    @patch("ir_amplitude_detuning.detuning.equation_system.DetuningCorrectionEquationSystem")
    @patch("ir_amplitude_detuning.detuning.equation_system.build_detuning_correction_matrix_per_entry")
    def test_build_matrix_fully_mocked(self, mock_build_per_entry, mock_det_corr_eqs):
        """Tests if the function calls the expected downstream functions.

        This should be the only test necessary, as the downstream functions have
        been fully tested above, but on the other hand this test assumes the logic
        of this main function will never change.
        So maybe it's good to also test the outputs (as done below).
        """
        # Prepare ---
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
        ]

        target_data1 = Mock()
        target_data1.correctors = correctors
        target_data2 = Mock()
        target_data2.correctors = correctors

        target = Mock()
        target.correctors = correctors
        target.data = [target_data1, target_data2]

        eqsys_full = Mock()
        eqsys_full.append_all = Mock()
        eqsys_full.fillna = Mock()

        eqsys_partial = Mock()
        mock_build_per_entry.return_value = eqsys_partial

        mock_det_corr_eqs.create_empty.return_value = eqsys_full

        # Run ---
        eqsys = build_detuning_correction_matrix(target)

        # Assert ---
        assert mock_det_corr_eqs.create_empty.call_count == 1
        assert mock_det_corr_eqs.create_empty.call_args.kwargs == {"columns": correctors}
        assert eqsys_full.fillna.call_count == 1
        assert eqsys_full.append_all.call_count == 2
        assert mock_build_per_entry.call_count == 2
        assert mock_build_per_entry.call_args_list[0][0][0] == target_data1
        assert mock_build_per_entry.call_args_list[1][0][0] == target_data2
        assert eqsys is eqsys_full

    def test_build_matrix_single_target_data(self):
        """Test building full equation system with single target data."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
        ]
        twiss_b1 = self._create_twiss_dataframe(["c1b4"])

        detuning_b1 = Detuning()
        detuning_b1[FirstOrderTerm.X10] = MeasureValue(5.0, 0.1)

        constraints_b1 = Constraints()

        target_data = TargetData(
            label="test_label",
            correctors=correctors,
            optics={1: twiss_b1},
            detuning={1: detuning_b1},
            constraints={1: constraints_b1},
        )

        target = Mock()
        target.correctors = correctors
        target.data = [target_data]

        # Run
        eqsys = build_detuning_correction_matrix(target)

        # Check
        assert isinstance(eqsys, DetuningCorrectionEquationSystem)
        assert len(eqsys.m) == 1
        assert eqsys.v["b1.test_label.X10"] == pytest.approx(5.0)

    def test_build_matrix_multiple_target_data(self):
        """Test building full equation system with multiple target data entries."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
            Corrector(field=FieldComponent.b6, length=0.5, magnet="c1b6", circuit="k1b6", ip=1),
        ]
        twiss_b1 = self._create_twiss_dataframe(["c1b4", "c1b6"])

        # First target data: first order
        detuning_b1_td1 = Detuning()
        detuning_b1_td1[FirstOrderTerm.X10] = MeasureValue(5.0, 0.1)
        constraints_b1_td1 = Constraints()

        target_data_1 = TargetData(
            label="first_order",
            correctors=correctors,
            optics={1: twiss_b1},
            detuning={1: detuning_b1_td1},
            constraints={1: constraints_b1_td1},
        )

        # Second target data: second order
        detuning_b1_td2 = Detuning()
        detuning_b1_td2[SecondOrderTerm.X20] = MeasureValue(3.0, 0.05)
        constraints_b1_td2 = Constraints()

        target_data_2 = TargetData(
            label="second_order",
            correctors=correctors,
            optics={1: twiss_b1},
            detuning={1: detuning_b1_td2},
            constraints={1: constraints_b1_td2},
        )

        target = Mock()
        target.correctors = correctors
        target.data = [target_data_1, target_data_2]

        # Run
        eqsys = build_detuning_correction_matrix(target)

        # Check: 1 row from target_data_1 + 1 row from target_data_2
        assert isinstance(eqsys, DetuningCorrectionEquationSystem)
        assert len(eqsys.m) == 2
        assert eqsys.v["b1.first_order.X10"] == pytest.approx(5.0)
        assert eqsys.v["b1.second_order.X20"] == pytest.approx(3.0)

    def test_build_matrix_fillna_called(self):
        """Test that fillna is called on the final equation system."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="c1b4", circuit="k1b4", ip=1),
            Corrector(field=FieldComponent.b6, length=0.5, magnet="c1b6", circuit="k1b6", ip=2),
        ]
        twiss_b1 = self._create_twiss_dataframe([correctors[0].magnet])  # c1b6 not present

        detuning_b1 = Detuning()
        detuning_b1[FirstOrderTerm.X10] = MeasureValue(5.0, 0.1)

        constraints_b1 = Constraints()

        target_data = TargetData(
            label="test_label",
            correctors=correctors,
            optics={1: twiss_b1},
            detuning={1: detuning_b1},
            constraints={1: constraints_b1},
        )

        target = Mock()
        target.correctors = correctors
        target.data = [target_data]

        # Run
        eqsys = build_detuning_correction_matrix(target)

        # Check that NaN values were filled with zeros (no NaN should remain)
        assert not eqsys.m.isna().any().any()
        assert not eqsys.m_constr.isna().any().any()
        # c1b6 column should be 0 since magnet not in twiss
        assert (eqsys.m.loc[:, correctors[1]] == 0.0).all()
