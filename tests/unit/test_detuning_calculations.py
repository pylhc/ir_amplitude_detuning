from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from cvxpy.settings import INFEASIBLE, SOLVER_ERROR, UNBOUNDED
from pandas.testing import assert_frame_equal

from ir_amplitude_detuning.detuning.calculations import (
    FIELDS,
    IP,
    Method,
    calc_effective_detuning,
    calculate_correction,
)
from ir_amplitude_detuning.detuning.equation_system import DetuningCorrectionEquationSystem
from ir_amplitude_detuning.detuning.measurements import (
    FirstOrderTerm,
    MeasureValue,
    SecondOrderTerm,
)
from ir_amplitude_detuning.utilities.correctors import Corrector, FieldComponent

# ============================================================================
# Tests for Method Enum
# ============================================================================

class TestMethodEnum:
    """Test cases for the Method enum."""

    def test_method_enum_values(self):
        """Test that all method values are correct."""
        assert Method.auto == "auto"
        assert Method.cvxpy == "cvxpy"
        assert Method.numpy == "numpy"

    def test_method_in_enum(self):
        """Test that all methods are recognized by the enum."""
        # check if all values are in the enum (`list` only necessary until py3.12)
        assert "auto" in list(Method)
        assert "cvxpy" in list(Method)
        assert "numpy" in list(Method)

        # similarly, creating instances should also work:
        assert Method("auto") == Method.auto
        assert Method("cvxpy") == Method.cvxpy
        assert Method("numpy") == Method.numpy

    def test_method_not_in_enum(self):
        """Test that unknown methods raise ValueError."""
        with pytest.raises(ValueError):
            Method("invalid")


# ============================================================================
# Tests for calculate_correction
# ============================================================================

class TestCalculateCorrection:
    """Test cases for the calculate_correction function."""

    def test_calculate_correction_invalid_method(self):
        """Test that invalid method raises ValueError."""
        mock_target = Mock()

        with pytest.raises(ValueError):
            calculate_correction(mock_target, method="invalid")

    @pytest.mark.parametrize("method", [Method.auto, Method.numpy, Method.cvxpy])
    @patch('ir_amplitude_detuning.detuning.calculations.build_detuning_correction_matrix')
    def test_calculate_correction_exact_no_constraints(self, mock_build_matrix, method):
        """Test calculate_correction with auto method and no constraints."""
        # Very simple eqation system:
        matrix = [[1, 1], [1, -1]]  # inverse is matrix/2
        values = [MeasureValue(3, 0.2), MeasureValue(1, 0.1)]
        expected_values = {"a": 2, "b": 1}
        expected_error = np.sqrt(np.mean([v.error**2 for v in values])/2)

        # Mock equation system building ---
        mock_eqsys = DetuningCorrectionEquationSystem(
            m = pd.DataFrame(matrix, columns=expected_values.keys()),
            v = pd.Series([v.value for v in values]),
            m_constr = pd.DataFrame(),
            v_constr = pd.Series(dtype=float),
            v_meas = pd.Series(values),
        )

        mock_build_matrix.return_value = mock_eqsys
        mock_target = Mock()

        # Run the calculation ---
        result = calculate_correction(mock_target, method=method)

        # Check the results ---
        mock_build_matrix.assert_called_with(mock_target)

        assert isinstance(result, pd.Series)
        assert len(result) == 2

        if method in (Method.numpy, Method.auto):
            assert result["a"].value == pytest.approx(expected_values["a"])
            assert result["b"].value == pytest.approx(expected_values["b"])
            assert result["a"].error == pytest.approx(result["b"].error) == pytest.approx(expected_error)
        else:
            assert result["a"] == pytest.approx(expected_values["a"])
            assert result["b"] == pytest.approx(expected_values["b"])

            with pytest.raises(AttributeError):
                result["a"].value

            with pytest.raises(AttributeError):
                result["b"].value

    def _get_equation_system_to_optimize(self) -> tuple[list[MeasureValue], list[list[int]], float]:
        """Simple, not exact solvable Eqs for the next tests."""
        values = [MeasureValue(3, 0.2), MeasureValue(5, 0.1)]
        matrix = [[1, 1], [2, 2]]
        expected = 1.3  # optimal value without constraints
        return values, matrix, expected

    @patch('ir_amplitude_detuning.detuning.calculations.build_detuning_correction_matrix')
    def test_calculate_correction_optimize_no_constraints(self, mock_build_matrix):
        """Test calculate_correction with auto method and no constraints."""
        # Prepare ---
        values, matrix, expected = self._get_equation_system_to_optimize()
        mock_eqsys = DetuningCorrectionEquationSystem(
            m = pd.DataFrame(matrix, columns=["a", "b"]),
            v = pd.Series([v.value for v in values]),
            m_constr = pd.DataFrame(),
            v_constr = pd.Series(dtype=float),
            v_meas = pd.Series(values),
        )
        mock_build_matrix.return_value = mock_eqsys

        # Run ---
        result_numpy = calculate_correction(Mock(), method=Method.auto)

        # Check ---
        assert result_numpy["a"].value == pytest.approx(expected)
        assert result_numpy["b"].value == pytest.approx(expected)
        assert result_numpy["a"].error > 0
        assert result_numpy["b"].error > 0

    @patch('ir_amplitude_detuning.detuning.calculations.build_detuning_correction_matrix')
    def test_calculate_correction_optimize_with_wide_constraints(self, mock_build_matrix):
        """Test calculate_correction with auto method and constraints that don't really matter."""
        # Prepare ---
        values, matrix, expected = self._get_equation_system_to_optimize()
        mock_eqsys = DetuningCorrectionEquationSystem(
            m = pd.DataFrame(matrix, columns=["a", "b"]),
            v = pd.Series([v.value for v in values]),
            m_constr = pd.DataFrame([[1, 1]]),   # sum of variables
            v_constr = pd.Series([3]),           # to be smaller than 3
            v_meas = pd.Series(values),
        )
        mock_build_matrix.return_value = mock_eqsys

        # Run ---
        result_cvxpy = calculate_correction(Mock(), method=Method.auto)

        # Check ---
        assert result_cvxpy["a"] == pytest.approx(expected)
        assert result_cvxpy["b"] == pytest.approx(expected)

    @patch('ir_amplitude_detuning.detuning.calculations.build_detuning_correction_matrix')
    def test_calculate_correction_optimize_with_constraints(self, mock_build_matrix):
        """Test calculate_correction with auto method and constraints."""
        # Prepare ---
        values, matrix, expected = self._get_equation_system_to_optimize()
        mock_eqsys = DetuningCorrectionEquationSystem(
            m = pd.DataFrame(matrix, columns=["a", "b"]),
            v = pd.Series([v.value for v in values]),
            m_constr = pd.DataFrame([[-1, -1]]),  # sum of variables
            v_constr = pd.Series([-3]),           # to be larger than 3
            v_meas = pd.Series(values),
        )
        mock_build_matrix.return_value = mock_eqsys

        # Run ---
        result_cvxpy = calculate_correction(Mock(), method=Method.auto)

        # Check ---
        assert np.sum(result_cvxpy) == pytest.approx(3)  # should optimize to 1.5, 1.5


    @patch('ir_amplitude_detuning.detuning.calculations.build_detuning_correction_matrix')
    @patch('ir_amplitude_detuning.detuning.calculations.cvx.Problem')
    def test_cvxpy_fails(self, mock_problem_class, mock_build_matrix):
        """Test calculate_correction with cvxpy method."""
        # Setup mocks
        mock_eqsys = Mock()
        mock_eqsys.m = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
        mock_eqsys.v = pd.Series([5, 6])
        mock_eqsys.m_constr = pd.DataFrame()
        mock_eqsys.v_constr = pd.Series(dtype=float)
        mock_eqsys.v_meas = pd.Series([5, 6])

        mock_build_matrix.return_value = mock_eqsys

        # Mock cvxpy solver
        for error_status in (INFEASIBLE, UNBOUNDED, SOLVER_ERROR):
            mock_prob = Mock()
            mock_problem_class.return_value = mock_prob
            mock_prob.status = error_status
            mock_prob.solve.return_value = None

            with pytest.raises(ValueError) as e:
                calculate_correction(Mock(), method=Method.cvxpy)

            assert "failed" in str(e)
            assert error_status in str(e)

            # Check that cvxpy solver was used
            mock_prob.solve.assert_called_once()


# ============================================================================
# Tests for calc_effective_detuning
# ============================================================================

class TestCalcEffectiveDetuning:
    """Test cases for the calc_effective_detuning function."""

    def test_calc_effective_detuning_empty_optics(self):
        """Test with empty optics dictionary."""
        result = calc_effective_detuning({}, pd.Series(0, index=[Mock(ip=None)], dtype=float))

        # Returns empty dict with no beams
        assert isinstance(result, dict)
        assert len(result) == 0

    @patch("ir_amplitude_detuning.detuning.calculations.calculate_matrix_row")
    def test_calc_effective_detuning_no_ips(self, mock_calculate_matrix_row):
        """Test with correctors without IPs."""
        # Create mocks ---
        all_correctors = [
            Corrector(
                field=field,
                length=0.5,
                magnet=f"{type_}ip{ip or 0}{field}",
                circuit=f"k{type_}ip{ip or 0}{field}",
                ip=ip,
            )
            for type_, field, ip in (
                ("c1", FieldComponent.b4, None),
                ("c2", FieldComponent.b4, None),
            )
        ]
        values = pd.Series([1, 2], index=all_correctors, dtype=float)

        mock_optics = {1: Mock()}
        def mocked_calulation(beam, optics, correctors, term):
            assert correctors == all_correctors  # no filtering as all ips are None, and both have same field
            return np.ones([1, len(correctors)])

        mock_calculate_matrix_row.side_effect = mocked_calulation
        all_terms = list(FirstOrderTerm) + list(SecondOrderTerm)

        # Run ---
        result = calc_effective_detuning(mock_optics, values)

        # Check results ---
        assert isinstance(result, dict)
        assert len(result) == 1  # one beam
        assert len(result[1]) == 1  # one field, "one" ip (None)
        assert all(result[1].loc[:, all_terms] == values.sum())  # calculation returns [1, 1]
        assert mock_calculate_matrix_row.call_count == len(all_terms)

    @patch("ir_amplitude_detuning.detuning.calculations.calculate_matrix_row")
    def test_calc_effective_detuning(self, mock_calculate_matrix_row):
        """Test with single beam."""
        # Prepare fake data ---
        # Create correctors with different IPs and fields
        correctors = [
            Corrector(
                field=field,
                length=0.5,
                magnet=f"{type_}ip{ip or 0}{field}",
                circuit=f"k{type_}ip{ip or 0}{field}",
                ip=ip,
            )
            for type_, field, ip in (
                ("c1", FieldComponent.b4, 1),
                ("c2", FieldComponent.b4, 1),
                ("c1", FieldComponent.b5, 1),
                ("c1", FieldComponent.b6, 1),
                ("c2", FieldComponent.b6, 1),
                ("c1", FieldComponent.b6, 2),
                ("c2", FieldComponent.b6, 2),
                ("c2", FieldComponent.b4, None),
            )
        ]
        all_terms = list(FirstOrderTerm) + list(SecondOrderTerm)
        values = pd.Series(np.arange(len(correctors)), index=correctors)

        # Create mocks
        mock_optics = {1: Mock(), 2: Mock()}
        def mocked_calulation(beam, optics, correctors, term):
            assert optics == mock_optics[beam]  # already some checks
            assert term in all_terms
            return np.ones([1, len(correctors)]) * (all_terms.index(term) + 1) * beam

        mock_calculate_matrix_row.side_effect = mocked_calulation


        # Test the function ---
        result = calc_effective_detuning(mock_optics, values)

        # Check the result ---
        assert isinstance(result, dict)

        # n calls = n_terms * (n_ips + 1) * (n_fields + 1) * n_beams
        assert mock_calculate_matrix_row.call_count == len(all_terms) * 3 * 4 * 2

        # Check that result 2 is the same as 1 but multiplied by 2 (beam in mock calculation)
        df_mul = result[1].copy()
        df_mul.loc[:, all_terms] = df_mul.loc[:, all_terms] * 2
        assert_frame_equal(df_mul, result[2])

        # Test grouping by fields and ips
        def filter_correctors(field, ip):
            return list(filter(lambda c: (c.field in field) and (c.ip is None or str(c.ip) in ip), correctors))

        for field in ("b4", "b5", "b6", "b4b5b6"):
            df_field = result[1].loc[result[1][FIELDS] == field, :]
            for ip in ("1", "2", "12"):
                df_field_ip = df_field.loc[df_field[IP] == ip, :]
                value = df_field_ip[all_terms[0]].iloc[0]
                assert len(df_field_ip) == 1
                assert all(df_field_ip[all_terms] == value * np.arange(1, len(all_terms) + 1))  # because  of mock return
                contributing_correctors = filter_correctors(field=field, ip=ip)
                assert value == sum(values.loc[contributing_correctors])  # because mock return for first term is (1,1,1..)
