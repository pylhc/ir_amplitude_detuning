from __future__ import annotations

import numpy as np
import pytest

from ir_amplitude_detuning.detuning.measurements import (
    Constraints,
    Detuning,
    DetuningMeasurement,
    MeasureValue,
    scaled_contraints,
    scaled_detuning,
    scaled_detuningmeasurement,
)
from ir_amplitude_detuning.detuning.terms import FirstOrderTerm


class TestMeasureValue:
    """Tests for the MeasureValue class."""

    def test_init_default(self):
        """Test default initialization."""
        mv = MeasureValue()
        assert mv.value == 0.0
        assert mv.error == 0.0

    def test_init_with_values(self):
        """Test initialization with custom values."""
        mv = MeasureValue(value=5.0, error=0.5)
        assert mv.value == 5.0
        assert mv.error == 0.5

    def test_add_measure_values(self):
        """Test addition of two MeasureValue objects."""
        mv1 = MeasureValue(value=3.0, error=0.3)
        mv2 = MeasureValue(value=4.0, error=0.4)
        result = mv1 + mv2
        assert result.value == 7.0
        assert result.error == pytest.approx(np.sqrt(0.3**2 + 0.4**2))

    def test_add_with_scalar(self):
        """Test addition with zero scalar."""
        mv = MeasureValue(value=5.0, error=0.5)
        result = mv + 0
        assert result is not mv
        assert result.value == 5.0
        assert result.error == 0.5

        result = mv + 2
        assert result.value == 7.0
        assert result.error == 0.5

    def test_radd_measure_values(self):
        """Test right addition of MeasureValue objects."""
        mv1 = MeasureValue(value=3.0, error=0.3)
        mv2 = MeasureValue(value=4.0, error=0.4)
        result = mv2 + mv1
        assert result.value == 7.0
        assert result.error == pytest.approx(np.sqrt(0.3**2 + 0.4**2))

    def test_radd_with_scalar(self):
        """Test right addition with scalar (for sum())."""
        mv1 = MeasureValue(value=3.0, error=0.3)
        result = 3 + mv1
        assert result.value == 6.0
        assert result.error == 0.3

        mv2 = MeasureValue(value=4.0, error=0.8)
        addition = mv1 + mv2
        summation = sum((mv1, mv2))
        assert summation.value == pytest.approx(addition.value)
        assert summation.error == pytest.approx(addition.error)

    def test_sub_measure_values(self):
        """Test subtraction of two MeasureValue objects."""
        mv1 = MeasureValue(value=5.0, error=0.5)
        mv2 = MeasureValue(value=3.0, error=0.3)
        result = mv1 - mv2
        assert result.value == 2.0
        assert result.error == pytest.approx(np.sqrt(0.5**2 + 0.3**2))

    def test_sub_with_scalar(self):
        """Test subtraction with scalar."""
        mv = MeasureValue(value=5.0, error=0.5)
        result = mv - 2.0
        assert result.value == 3.0
        assert result.error == 0.5

    def test_neg(self):
        """Test negation operator."""
        mv = MeasureValue(value=5.0, error=0.5)
        result = -mv
        assert result.value == -5.0
        assert result.error == 0.5

    def test_mul_by_scalar(self):
        """Test multiplication by scalar."""
        mv = MeasureValue(value=4.0, error=0.4)
        result = mv * 2.5
        assert result.value == 10.0
        assert result.error == 1.0

    def test_rmul_by_scalar(self):
        """Test right multiplication by scalar."""
        mv = MeasureValue(value=4.0, error=0.4)
        result = 2.5 * mv
        assert result.value == 10.0
        assert result.error == 1.0

    def test_truediv_by_scalar(self):
        """Test division by scalar."""
        mv = MeasureValue(value=8.0, error=0.8)
        result = mv / 2.0
        assert result.value == 4.0
        assert result.error == 0.4

    def test_abs(self):
        """Test absolute value."""
        mv = MeasureValue(value=-5.0, error=0.5)
        result = abs(mv)
        assert result.value == 5.0
        assert result.error == 0.5

    def test_str(self):
        """Test string representation."""
        mv = MeasureValue(value=5.0, error=0.5)
        assert str(mv) == "5.0 +- 0.5"

    def test_format(self):
        """Test format method."""
        mv = MeasureValue(value=5.123, error=0.456)
        formatted = f"{mv:.2f}"
        assert formatted == "5.12 +- 0.46"

    def test_repr(self):
        """Test repr."""
        mv = MeasureValue(value=5.0, error=0.5)
        assert repr(mv) == "5.0 +- 0.5"

    def test_iter(self):
        """Test iteration over MeasureValue."""
        mv = MeasureValue(value=5.0, error=0.5)
        value, error = mv
        assert value == 5.0
        assert error == 0.5

    def test_rms_single_value(self):
        """Test rms with single value."""
        mv = MeasureValue(value=3.0, error=0.3)
        result = MeasureValue.rms([mv])
        assert result.value == pytest.approx(3.0)
        assert result.error == pytest.approx(0.3)

    def test_rms_multiple_values(self):
        """Test rms with multiple values."""
        measurements = [
            MeasureValue(value=3.0, error=0.3),
            MeasureValue(value=4.0, error=0.4),
            MeasureValue(value=8.0, error=0.1),
        ]
        result = MeasureValue.rms(measurements)

        # RMS error propagation step-by-step
        values = np.array([m.value for m in measurements])
        errors = np.array([m.error for m in measurements])

        squared_values = values**2
        squared_errors = 2*np.abs(values)*errors

        mean_squared_values = np.mean(squared_values)
        mean_squared_errors = np.sqrt(np.sum(squared_errors**2)) / len(measurements)

        expected_value = np.sqrt(mean_squared_values)
        expected_error = 0.5 * mean_squared_errors / expected_value

        assert result.value == pytest.approx(expected_value)
        assert result.error == pytest.approx(expected_error)

    def test_weighted_rms(self):
        """Test weighted_rms."""
        measurements = [
            MeasureValue(value=3.0, error=0.3),
            MeasureValue(value=4.0, error=0.4),
        ]

        result = MeasureValue.weighted_rms(measurements)

        # RMS error propagation step-by-step
        values = np.array([m.value for m in measurements])
        errors = np.array([m.error for m in measurements])
        weights = 1/errors**2
        sum_weights = np.sum(weights)

        squared_values = values**2
        squared_errors = 2*np.abs(values)*errors

        sum_squared_values = np.sum(squared_values * weights)
        sum_squared_errors = np.sqrt(np.sum((squared_errors * weights)**2))

        av_squared_values = sum_squared_values / sum_weights
        av_squared_errors = sum_squared_errors / sum_weights

        expected_value = np.sqrt(av_squared_values)
        expected_error = 0.5 * av_squared_errors / expected_value

        assert result.value == pytest.approx(expected_value)
        assert result.error == pytest.approx(expected_error)

    def test_mean(self):
        """Test mean calculation."""
        measurements = [
            MeasureValue(value=3.0, error=0.3),
            MeasureValue(value=5.0, error=0.5),
            MeasureValue(value=8.0, error=0.1),
        ]
        result = MeasureValue.mean(measurements)

        # calculate direct on MeasValue objects
        # (note: MeasureValue.mean() is more efficiently implemented)
        expected = np.mean(measurements)

        assert result.value == pytest.approx(expected.value)
        assert result.error == pytest.approx(expected.error)

    def test_weighted_mean(self):
        """Test weighted mean."""
        measurements = [
            MeasureValue(value=3.0, error=0.3),
            MeasureValue(value=5.0, error=0.5),
            MeasureValue(value=8.0, error=0.1),
        ]

        # calculate direct on MeasValue objects
        # (note: weighted_mean is more efficiently implemented)
        weights = np.array([1/m.error**2 for m in measurements])
        expected = np.average(measurements, weights=weights)

        result = MeasureValue.weighted_mean(measurements)
        assert result.value == pytest.approx(expected.value)
        assert result.error == pytest.approx(expected.error)

    def test_from_value_float(self):
        """Test from_value with float."""
        result = MeasureValue.from_value(5.0)
        assert result.value == 5.0
        assert result.error == 0.0

    def test_from_value_measure_value(self):
        """Test from_value with MeasureValue (creates copy)."""
        original = MeasureValue(value=5.0, error=0.5)
        result = MeasureValue.from_value(original)
        assert result.value == 5.0
        assert result.error == 0.5
        # Verify it's a copy, not the same object
        assert result is not original


class TestDetuning:
    """Tests for the Detuning class."""

    def test_init_empty(self):
        """Test initialization with no terms."""
        det = Detuning()
        assert list(det.terms()) == []

    def test_init_with_first_order_terms(self):
        """Test initialization with first order terms."""
        det = Detuning(X10=1.0, X01=2.0)
        terms = list(det.terms())
        assert "X10" in terms
        assert "X01" in terms
        assert len(terms) == 2

    def test_init_with_second_order_terms(self):
        """Test initialization with second order terms."""
        det = Detuning(X20=1.0, Y11=2.0, X02=3.0)
        terms = list(det.terms())
        assert "X20" in terms
        assert "Y11" in terms
        assert "X02" in terms

    def test_init_with_scale(self):
        """Test initialization with scaling."""
        det = Detuning(X10=1.0, X01=2.0, scale=1e3)
        assert det[FirstOrderTerm.X10] == pytest.approx(1e3)
        assert det[FirstOrderTerm.X01] == pytest.approx(2e3)

    def test_all_terms_no_filter(self):
        """Test all_terms returns all possible terms."""
        terms = Detuning.all_terms()
        assert len(terms) == 10  # 4 first order + 6 second order

    def test_all_terms_first_order(self):
        """Test all_terms with order=1."""
        terms = Detuning.all_terms(order=1)
        assert len(terms) == 4
        assert "X10" in terms
        assert "Y01" in terms

    def test_all_terms_second_order(self):
        """Test all_terms with order=2."""
        terms = Detuning.all_terms(order=2)
        assert len(terms) == 6
        assert "X20" in terms
        assert "Y02" in terms

    def test_getitem_set_term(self):
        """Test __getitem__ for set terms."""
        det = Detuning(X10=5.0)
        assert det["X10"] == 5.0

    def test_getitem_unset_term_raises(self):
        """Test __getitem__ for unset terms raises KeyError."""
        det = Detuning(X10=5.0)
        with pytest.raises(KeyError):
            det["X01"]

    def test_setitem_valid_term(self):
        """Test __setitem__ for valid terms."""
        det = Detuning()
        det["X10"] = 5.0
        assert det.X10 == 5.0

    def test_setitem_invalid_term_raises(self):
        """Test __setitem__ for invalid term raises KeyError."""
        det = Detuning()
        with pytest.raises(KeyError):
            det["INVALID"] = 5.0

    def test_items(self):
        """Test items iterator."""
        det = Detuning(X10=1.0, Y01=2.0)
        items = dict(det.items())
        assert items["X10"] == 1.0
        assert items["Y01"] == 2.0
        assert len(items) == 2

    def test_add_detunings(self):
        """Test addition of two Detuning objects."""
        det1 = Detuning(X10=1.0, Y01=2.0)
        det2 = Detuning(X10=3.0, Y01=4.0)
        result = det1 + det2
        assert result["X10"] == 4.0
        assert result["Y01"] == 6.0

    def test_add_different_terms_raises(self):
        """Test addition with different terms raises KeyError."""
        det1 = Detuning(X10=1.0)
        det2 = Detuning(X10=3.0, Y01=4.0)

        # this way around term is ignored/debug logged
        res1 = det1 + det2
        assert res1.Y01 is None

        # this way around KeyError is raised
        with pytest.raises(KeyError):
            det2 + det1

    def test_sub_detunings(self):
        """Test subtraction of two Detuning objects."""
        det1 = Detuning(X10=5.0, Y01=6.0)
        det2 = Detuning(X10=2.0, Y01=1.0)
        result = det1 - det2
        assert result["X10"] == 3.0
        assert result["Y01"] == 5.0

    def test_neg_detuning(self):
        """Test negation of Detuning."""
        det = Detuning(X10=5.0, Y01=-3.0)
        result = -det
        assert result["X10"] == -5.0
        assert result["Y01"] == 3.0

    def test_mul_by_scalar(self):
        """Test multiplication by scalar."""
        det = Detuning(X10=2.0, Y01=3.0)
        result = det * 2.5
        assert result["X10"] == 5.0
        assert result["Y01"] == 7.5

    def test_mul_detunings(self):
        """Test element-wise multiplication of two Detuning objects."""
        det1 = Detuning(X10=2.0, Y01=3.0)
        det2 = Detuning(X10=5.0, Y01=4.0)
        result = det1 * det2
        assert result["X10"] == 10.0
        assert result["Y01"] == 12.0

    def test_truediv_by_scalar(self):
        """Test division by scalar."""
        det = Detuning(X10=10.0, Y01=6.0)
        result = det / 2.0
        assert result["X10"] == 5.0
        assert result["Y01"] == 3.0

    def test_truediv_detunings(self):
        """Test element-wise division of two Detuning objects."""
        det1 = Detuning(X10=10.0, Y01=6.0)
        det2 = Detuning(X10=2.0, Y01=3.0)
        result = det1 / det2
        assert result["X10"] == 5.0
        assert result["Y01"] == 2.0

    def test_apply_acdipole_correction_first_order(self):
        """Test AC-Dipole correction on first order terms."""
        # X10, Y01, X11, Y11 should be divided by 2
        det = Detuning(X10=4.0, Y01=6.0)
        corrected = det.apply_acdipole_correction()
        assert corrected["X10"] == pytest.approx(2.0)
        assert corrected["Y01"] == pytest.approx(3.0)

    def test_apply_acdipole_correction_second_order(self):
        """Test AC-Dipole correction on second order terms."""
        # X20, Y02 should be divided by 3
        det = Detuning(X20=9.0, Y02=12.0)
        corrected = det.apply_acdipole_correction()
        assert corrected["X20"] == pytest.approx(3.0)
        assert corrected["Y02"] == pytest.approx(4.0)

    def test_apply_acdipole_correction_mixed(self):
        """Test AC-Dipole correction with mixed terms."""
        det = Detuning(X10=4.0, X20=9.0, X11=6.0, Y02=12.0)
        corrected = det.apply_acdipole_correction()
        assert corrected["X10"] == pytest.approx(2.0)
        assert corrected["X20"] == pytest.approx(3.0)
        assert corrected["X11"] == pytest.approx(3.0)
        assert corrected["Y02"] == pytest.approx(4.0)

    def test_apply_acdipole_correction_only_sets_terms(self):
        """Test AC-Dipole correction only affects set terms."""
        det = Detuning(X10=4.0)
        corrected = det.apply_acdipole_correction()
        # Only X10 should be set
        assert list(corrected.terms()) == ["X10"]

    def test_merge_first_order_crossterm_no_y10(self):
        """Test merge when Y10 is not set."""
        det = Detuning(X10=2.0)
        merged = det.merge_first_order_crossterm()
        assert merged["X10"] == 2.0
        assert merged.Y10 is None

    def test_merge_first_order_crossterm_only_y10(self):
        """Test merge when only Y10 is set."""
        det = Detuning(Y10=3.0)
        merged = det.merge_first_order_crossterm()
        assert merged["X01"] == 3.0
        assert merged.Y10 is None

    def test_merge_first_order_crossterm_both_set(self):
        """Test merge when both X01 and Y10 are set."""
        det = Detuning(X01=4.0, Y10=6.0)
        merged = det.merge_first_order_crossterm()
        assert merged["X01"] == 5.0  # (4.0 + 6.0) * 0.5
        assert merged.Y10 is None

    def test_check_terms_compatible(self):
        """Test _check_terms with compatible terms."""
        det1 = Detuning(X10=1.0, Y01=2.0)
        det2 = Detuning(X10=3.0, Y01=4.0)
        # Should not raise
        det1._check_terms(det2)

    def test_check_terms_incompatible(self):
        """Test _check_terms with incompatible terms."""
        det1 = Detuning(X10=1.0)
        det2 = Detuning(Y01=4.0)
        with pytest.raises(KeyError):
            det1._check_terms(det2)


class TestDetuningMeasurement:
    """Tests for the DetuningMeasurement class."""

    def test_init_with_measure_values(self):
        """Test initialization with MeasureValue objects."""
        meas = DetuningMeasurement(
            X10=MeasureValue(1.0, 0.1),
            Y01=MeasureValue(2.0, 0.2)
        )
        assert meas["X10"].value == 1.0
        assert meas["X10"].error == 0.1

    def test_init_with_tuples_converts_to_measure_value(self):
        """Test initialization with tuples converts to MeasureValue."""
        meas = DetuningMeasurement(X10=(1.0, 0.1))
        assert isinstance(meas["X10"], MeasureValue)
        assert meas["X10"].value == 1.0
        assert meas["X10"].error == 0.1

    def test_init_with_too_long_tuples_fails(self):
        """Test initialization with tuples converts to MeasureValue."""
        with pytest.raises(ValueError):
            DetuningMeasurement(X10=(1.0, 0.1, 0.1))

    def test_init_with_floats_converts_to_measure_value(self):
        """Test initialization with floats converts to MeasureValue."""
        meas = DetuningMeasurement(X10=1.0)
        assert isinstance(meas["X10"], MeasureValue)
        assert meas["X10"].value == 1.0

    def test_post_init_with_scale(self):
        """Test __post_init__ applies scaling."""
        meas = DetuningMeasurement(X10=MeasureValue(1.0, 0.1), scale=1e3)
        assert meas["X10"].value == pytest.approx(1e3)
        assert meas["X10"].error == pytest.approx(1e2)

    def test_get_detuning(self):
        """Test get_detuning returns Detuning with values only."""
        meas = DetuningMeasurement(
            X10=MeasureValue(1.5, 0.1),
            Y01=MeasureValue(2.5, 0.2)
        )
        det = meas.get_detuning()
        assert isinstance(det, Detuning)
        assert det["X10"] == 1.5
        assert det["Y01"] == 2.5

    def test_from_detuning(self):
        """Test from_detuning creates DetuningMeasurement with zero errors."""
        det = Detuning(X10=1.5, Y01=2.5)
        meas = DetuningMeasurement.from_detuning(det)
        assert isinstance(meas, DetuningMeasurement)
        assert meas["X10"].value == 1.5
        assert meas["X10"].error == 0.0
        assert meas["Y01"].value == 2.5
        assert meas["Y01"].error == 0.0

    def test_arithmetic_operations_with_measure_values(self):
        """Test arithmetic operations work with MeasureValues."""
        meas1 = DetuningMeasurement(X10=MeasureValue(3.0, 0.3))
        meas2 = DetuningMeasurement(X10=MeasureValue(2.0, 0.2))
        result = meas1 + meas2
        assert result["X10"].value == 5.0
        assert result["X10"].error == pytest.approx(np.sqrt(0.3**2 + 0.2**2))

    def test_add_measurement_with_scale(self):
        """Test addition of scaled measurements."""
        meas1 = DetuningMeasurement(X10=MeasureValue(1.0, 0.1), scale=1e3)
        meas2 = DetuningMeasurement(X10=MeasureValue(2.0, 0.2), scale=1e3)
        result = meas1 + meas2
        assert result["X10"].value == pytest.approx(3e3)


class TestConstraints:
    """Tests for the Constraints class."""

    def test_init_empty(self):
        """Test initialization with no constraints."""
        const = Constraints()
        assert list(const.terms()) == []

    def test_init_with_le_constraint(self):
        """Test initialization with <= constraint."""
        const = Constraints(X10="<=5.0")
        assert const["X10"] == "<=5.0"

    def test_init_with_ge_constraint(self):
        """Test initialization with >= constraint."""
        const = Constraints(X10=">=2.0")
        assert const["X10"] == ">=2.0"

    def test_init_with_whitespace(self):
        """Test initialization with whitespace is handled."""
        const = Constraints(X10="<= 5.0")
        assert const["X10"] == "<= 5.0"

    def test_init_with_invalid_comparison_raises(self):
        """Test initialization with invalid comparison raises ValueError."""
        with pytest.raises(ValueError, match="Unknown constraint"):
            Constraints(X10="<5.0")

    def test_init_with_invalid_value_raises(self):
        """Test initialization with non-numeric value raises ValueError."""
        with pytest.raises(ValueError, match="does not parse to float"):
            Constraints(X10="<=abc")

    def test_parse_value_le(self):
        """Test _parse_value with <= constraint."""
        const = Constraints()
        comparison, value = const._parse_value("<=5.0")
        assert comparison == "<="
        assert value == 5.0

    def test_parse_value_ge(self):
        """Test _parse_value with >= constraint."""
        const = Constraints()
        comparison, value = const._parse_value(">=2.5")
        assert comparison == ">="
        assert value == 2.5

    def test_parse_value_negative(self):
        """Test _parse_value with negative values."""
        const = Constraints()
        comparison, value = const._parse_value("<=-3.5")
        assert comparison == "<="
        assert value == -3.5

    def test_parse_value_with_spaces(self):
        """Test _parse_value removes spaces."""
        const = Constraints()
        comparison, value = const._parse_value(">= 2.5")
        assert comparison == ">="
        assert value == 2.5

    def test_getitem_set_constraint(self):
        """Test __getitem__ for set constraints."""
        const = Constraints(X10="<=5.0")
        assert const["X10"] == "<=5.0"

    def test_getitem_unset_constraint_raises(self):
        """Test __getitem__ for unset constraint raises KeyError."""
        const = Constraints(X10="<=5.0")
        with pytest.raises(KeyError):
            const["X01"]

    def test_setitem_valid_constraint(self):
        """Test __setitem__ for valid constraint."""
        const = Constraints()
        const["X10"] = "<=5.0"
        assert const.X10 == "<=5.0"

    def test_setitem_invalid_term_raises(self):
        """Test __setitem__ for invalid term raises KeyError."""
        const = Constraints()
        with pytest.raises(KeyError):
            const["INVALID"] = "<=5.0"

    def test_setitem_invalid_constraint_raises(self):
        """Test __setitem__ with invalid constraint raises ValueError."""
        const = Constraints()
        with pytest.raises(ValueError):
            const["X10"] = "<5.0"

    def test_terms(self):
        """Test terms method."""
        const = Constraints(X10="<=5.0", Y01=">=2.0")
        terms = list(const.terms())
        assert "X10" in terms
        assert "Y01" in terms
        assert len(terms) == 2

    def test_all_terms_no_filter(self):
        """Test all_terms returns all possible terms."""
        terms = Constraints.all_terms()
        assert len(terms) == 10

    def test_all_terms_first_order(self):
        """Test all_terms with order=1."""
        terms = Constraints.all_terms(order=1)
        assert len(terms) == 4

    def test_get_leq_le_positive(self):
        """Test get_leq with <= and positive value."""
        const = Constraints(X10="<=4")
        sign, value = const.get_leq("X10")
        assert sign == 1
        assert value == 4

    def test_get_leq_ge_positive(self):
        """Test get_leq with >= and positive value."""
        const = Constraints(X10=">=3")
        sign, value = const.get_leq("X10")
        assert sign == -1
        assert value == -3

    def test_get_leq_ge_negative(self):
        """Test get_leq with >= and negative value."""
        const = Constraints(X10=">=-2")
        sign, value = const.get_leq("X10")
        assert sign == -1
        assert value == 2

    def test_get_leq_le_negative(self):
        """Test get_leq with <= and negative value."""
        const = Constraints(X10="<=-2")
        sign, value = const.get_leq("X10")
        assert sign == 1
        assert value == -2

    def test_get_leq_with_scale(self):
        """Test get_leq applies scaling."""
        const = Constraints(X10="<=4", scale=1e3)
        sign, value = const.get_leq("X10")
        assert sign == 1
        assert value == pytest.approx(4e3)

    def test_get_leq_with_scale_ge(self):
        """Test get_leq with >= and scaling."""
        const = Constraints(X10=">=3", scale=1e3)
        sign, value = const.get_leq("X10")
        assert sign == -1
        assert value == pytest.approx(-3e3)

    def test_get_leq_unset_constraint_raises(self):
        """Test get_leq for unset constraint raises KeyError."""
        const = Constraints(X10="<=5.0")
        with pytest.raises(KeyError):
            const.get_leq("X01")


class TestScaledPartials:
    """Tests for the scaled partial functions."""

    def test_scaled_detuning(self):
        """Test scaled_detuning creates Detuning with 1e3 scale."""
        det = scaled_detuning(X10=1.0, Y01=2.0)
        assert det["X10"] == pytest.approx(1e3)
        assert det["Y01"] == pytest.approx(2e3)

    def test_scaled_contraints(self):
        """Test scaled_contraints creates Constraints with 1e3 scale."""
        const = scaled_contraints(X10="<=1.0")
        sign, value = const.get_leq("X10")
        assert value == pytest.approx(1e3)

    def test_scaled_detuningmeasurement(self):
        """Test scaled_detuningmeasurement creates DetuningMeasurement with 1e3 scale."""
        meas = scaled_detuningmeasurement(X10=MeasureValue(1.0, 0.1))
        assert meas["X10"].value == pytest.approx(1e3)
        assert meas["X10"].error == pytest.approx(100)


class TestIntegration:
    """Integration tests combining multiple classes."""

    def test_detuning_measurement_workflow(self):
        """Test typical workflow with DetuningMeasurement."""
        # Create measurements
        meas1 = DetuningMeasurement(
            X10=MeasureValue(1.0, 0.1),
            Y01=MeasureValue(2.0, 0.2)
        )
        meas2 = DetuningMeasurement(
            X10=MeasureValue(1.5, 0.15),
            Y01=MeasureValue(2.5, 0.25)
        )

        # Combine
        combined = meas1 + meas2
        assert combined["X10"].value == 2.5
        assert combined["Y01"].value == 4.5

        # Extract detuning
        det = combined.get_detuning()
        assert isinstance(det, Detuning)
        assert det["X10"] == 2.5
