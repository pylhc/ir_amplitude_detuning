from __future__ import annotations

from unittest.mock import patch

import pytest

from ir_amplitude_detuning.detuning.measurements import MeasureValue
from ir_amplitude_detuning.utilities.latex import (
    dqd2j,
    exp_m,
    partial_dqd2j,
    print_correction_and_error_as_latex,
    term2dqdj,
    term2partial_dqdj,
    unit_exp_m,
    ylabel_from_detuning_term,
)

# ============================================================================
# Tests for print_correction_and_error_as_latex
# ============================================================================

class TestPrintCorrectionAndErrorAsLatex:
    """Test cases for the print_correction_and_error_as_latex function."""

    @patch("ir_amplitude_detuning.utilities.latex.LOG")
    def test_print_with_measure_values_and_errors(self, mock_log):
        """Test printing with MeasureValue objects that have errors."""
        values = [MeasureValue(value=1.23, error=0.05), MeasureValue(value=2.45, error=1.1)]
        correctors = ["MCTX.3L1", "MCTX.3L2"]

        print_correction_and_error_as_latex(values, correctors)

        mock_log.info.assert_called_once()
        latex_output = mock_log.info.call_args[0][0]
        assert "MCTX.3L1 & MCTX.3L2\\" in latex_output
        assert "1.230(050) & 2.450(1.100)\\" in latex_output

    @patch("ir_amplitude_detuning.utilities.latex.LOG")
    def test_print_with_scalar_values_no_error(self, mock_log):
        """Test printing with scalar values without error attribute."""
        values = [1.23, 2.45]
        correctors = ["MCTX.3L1", "MCTX.3L2"]

        print_correction_and_error_as_latex(values, correctors)

        mock_log.info.assert_called_once()
        latex_output = mock_log.info.call_args[0][0]
        assert "MCTX.3L1 & MCTX.3L2\\" in latex_output
        assert "1.230 & 2.450\\" in latex_output

    @patch("ir_amplitude_detuning.utilities.latex.LOG")
    def test_print_with_measure_values_and_errors_scaled(self, mock_log):
        """Test printing with MeasureValue objects that have errors."""
        values = [MeasureValue(value=123.4, error=50), MeasureValue(value=245.4, error=11)]
        correctors = ["MCTX.3L1", "MCTX.3L2"]

        print_correction_and_error_as_latex(values, correctors, exponent=3)

        mock_log.info.assert_called_once()
        latex_output = mock_log.info.call_args[0][0]
        assert "MCTX.3L1 & MCTX.3L2\\" in latex_output
        assert "0.123(050) & 0.245(011)\\" in latex_output


# ============================================================================
# Tests for ylabel_from_detuning_term
# ============================================================================

class TestYlabelFromDetuningTerm:
    """Test cases for the ylabel_from_detuning_term function."""

    def test_ylabel_without_exponent(self):
        """Test ylabel generation without exponent."""
        result = ylabel_from_detuning_term("X01")

        assert "Q_{x,y}" in result
        assert "$m$^{-1}" in result
        assert "10^" not in result

    def test_ylabel_with_exponent(self):
        """Test ylabel generation with exponent."""
        result = ylabel_from_detuning_term("Y02", exponent=3)

        assert "Q_{y,yy}" in result
        assert "10^{3}" in result
        assert "$m$^{-2}" in result



# ============================================================================
# Tests for dqd2j and partial_dqd2j
# ============================================================================

class TestDqdFormatting:
    """Test cases for detuning term latex formatting functions."""

    def test_dqd2j_x_y(self):
        """Test dqd2j with x tune and y action."""
        result = dqd2j("x", "y")
        assert result == "Q_{x,y}"

    def test_dqd2j_x_xy(self):
        """Test dqd2j with x tune and mixed action."""
        result = dqd2j("y", "xy")
        assert result == "Q_{y,xy}"


class TestPartialDqdj:
    """Test cases for the partial_dqd2j function."""

    def test_partial_dqdj_single_derivative(self):
        """Test partial_dqdj with single derivative."""
        result = partial_dqd2j("x", "y")
        assert result == r"\partial_{2J_y}Q_x"

    def test_partial_dqdj_double_same_derivative(self):
        """Test partial_dqdj with double same derivative."""
        result = partial_dqd2j("x", "yy")
        assert result == r"\partial^{2}_{2J_y}Q_x"

    def test_partial_dqdj_mixed_derivatives(self):
        """Test partial_dqdj with mixed derivatives."""
        result = partial_dqd2j("y", "xy")
        assert result == r"\partial_{2J_x}\partial_{2J_y}Q_y"

    def test_partial_dqdj_invalid_action(self):
        """Test partial_dqdj raises error for derivatives > 2."""
        with pytest.raises(NotImplementedError):
            partial_dqd2j("x", "yyy")


# ============================================================================
# Tests for term2dqdj and term2partial_dqdj wrappers
# ============================================================================

class TestTermFormatWrappers:
    """Test cases for term wrapper functions."""

    def test_term2dqdj(self):
        """Test term2dqdj wrapper."""
        result = term2dqdj("X02")
        assert result == "Q_{x,yy}"

    def test_term2partial_dqdj(self):
        """Test term2partial_dqdj wrapper."""
        result = term2partial_dqdj("Y11")
        assert result == r"\partial_{2J_x}\partial_{2J_y}Q_y"


# ============================================================================
# Tests for exp_m and unit_exp_m
# ============================================================================

class TestUnitFormatting:
    """Test cases for unit formatting functions."""

    def test_exp_m_with_zero_e_power(self):
        """Test exp_m with zero exponent."""
        result = exp_m(0, 1)
        assert r"\;$m$^{1}" in result
        assert "10^" not in result

    def test_exp_m_with_zero_m_power(self):
        """Test exp_m with zero exponent."""
        result = exp_m(2, 0)
        assert r"\;$m$^{0}" in result
        assert "10^{2}" in result

    def test_exp_m_with_mixed_exponents(self):
        """Test exp_m with positive exponent."""
        result = exp_m(3, -1)
        assert r"\cdot 10^{3}\;$m$^{-1}" in result

    def test_unit_exp_m_format(self):
        """Test unit_exp_m formatting."""
        result = unit_exp_m(3, -1)
        assert r"\; [10^{3}" in result
        assert "$m$^{-1}]" in result

    def test_unit_exp_m_format_zero_m_power(self):
        """Test unit_exp_m formatting."""
        result = unit_exp_m(0, 3)
        assert r"\; [$m$^{3}]" in result
