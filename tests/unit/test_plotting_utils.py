from __future__ import annotations

import re
from unittest.mock import Mock

import pytest
from matplotlib.colors import to_rgb

from ir_amplitude_detuning.detuning.measurements import (
    Constraints,
    DetuningMeasurement,
)
from ir_amplitude_detuning.detuning.targets import Target, TargetData
from ir_amplitude_detuning.plotting.utils import (
    OtherColors,
    get_color_for_field,
    get_color_for_ip,
    get_default_scaling,
    get_full_target_labels,
)
from ir_amplitude_detuning.utilities.correctors import Corrector, FieldComponent

# ============================================================================
# Tests for get_default_scaling
# ============================================================================

class TestGetDefaultScaling:
    """Test cases for the get_default_scaling function."""

    @pytest.mark.parametrize(
        "term,expected_exponent,expected_scaling",
        [
            ("X02", 12, 1e-12),
            ("Y01", 3, 1e-3),
            ("Y11", 12, 1e-12),
            ("X10", 3, 1e-3),
        ],
    )
    def test_get_default_scaling(self, term: str, expected_exponent: int, expected_scaling: float):
        """Test default scaling factors for various detuning terms."""
        exponent, scaling = get_default_scaling(term)
        assert exponent == expected_exponent
        assert scaling == pytest.approx(expected_scaling)

    def test_get_default_scaling_invalid_sum(self):
        """Test that invalid term sums raise KeyError."""
        with pytest.raises(KeyError):
            get_default_scaling("X00")  # sum is 0, not in dict

        with pytest.raises(KeyError):
            get_default_scaling("Y31")  # sum is 4, not in dict


# ============================================================================
# Tests for get_color_for_field
# ============================================================================

class TestGetColorForField:
    """Test cases for the get_color_for_field function."""

    @pytest.mark.parametrize("field", list(FieldComponent))
    def test_get_color_for_field_valid(self, field: FieldComponent):
        """Test that valid fields return colors."""
        result = get_color_for_field(field)  # asserts all fields are valid
        _, _, _ = to_rgb(result)  # asserts that it is convertable to RGB

    def test_all_colors_different(self):
        """Test that all colors are different."""
        colors = [get_color_for_field(field) for field in list(FieldComponent)]
        assert len(set(colors)) == len(colors)

    def test_get_color_for_field_invalid(self):
        """Test that invalid fields raise NotImplementedError."""
        # Create a mock field that doesn't match any case
        mock_field = Mock(spec=FieldComponent)
        mock_field.__str__ = Mock(return_value="invalid_field")

        with pytest.raises(NotImplementedError, match="Field must be one of"):
            get_color_for_field(mock_field)


# ============================================================================
# Tests for get_color_for_ip
# ============================================================================

class TestGetColorForIp:
    """Test cases for the get_color_for_ip function."""

    @pytest.mark.parametrize("ip", ["15", "1", "5"])
    def test_get_color_for_ip_valid(self, ip: str):
        """Test that valid IPs return colors."""
        result = get_color_for_ip(ip)  # asserts all IPs are valid
        _, _, _ = to_rgb(result)  # asserts that it is convertable to RGB

    def test_all_colors_different(self):
        """Test that all colors are different."""
        colors = [get_color_for_ip(ip) for ip in ["15", "1", "5"]]
        assert len(set(colors)) == len(colors)

    @pytest.mark.parametrize("invalid_ip", ["2", "8", "invalid", "", "1 ", "15a", "0"])
    def test_get_color_for_ip_invalid(self, invalid_ip: str):
        """Test that invalid IPs raise NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match=f"IP must be one of \\['15', '1', '5'\\], got {invalid_ip}\\."
        ):
            get_color_for_ip(invalid_ip)


# ============================================================================
# Tests for OtherColors
# ============================================================================

class TestOtherColors:
    """Test cases for the OtherColors class."""

    def test_other_colors_estimated(self):
        """Test that OtherColors.estimated has correct value."""
        _, _, _ = to_rgb(OtherColors.estimated)
        assert OtherColors.flat != OtherColors.estimated

    def test_other_colors_flat(self):
        """Test that OtherColors.flat has correct value."""
        _, _, _ = to_rgb(OtherColors.estimated)
        assert OtherColors.flat != OtherColors.estimated

# ============================================================================
# Tests for get_full_target_labels
# ============================================================================

class TestGetFullTargetLabels:
    """Test cases for the get_full_target_labels function."""

    def test_get_full_target_labels_single_target_no_suffixes(self, target_data):
        """Test with single target and no suffixes."""
        result = get_full_target_labels([Target(name="target_name", data=[target_data])])

        assert isinstance(result, dict)
        assert "target_name" in result
        target_label = result["target_name"]
        assert isinstance(target_label, str)
        assert re.search(r"\$Q_\{x,yy\}\$\s+=\s+1\.5\s+\|\s+3\.5", target_label)  # values from the fixture
        assert re.search(r"\$Q_\{y,xy\}\$\s+=\s+--\s+\|\s+2\.5", target_label)


    def test_get_full_target_labels_multiple_targets_with_suffixes(self, target_data):
        """Test with multiple targets and suffixes."""
        result = get_full_target_labels(
            [Target(name="target1", data=[target_data]),
            Target(name="target2", data=[target_data])],
            suffixes=["suffix_1", "suffix_2"]
        )

        assert len(result) == 2
        assert "target1" in result
        assert "target2" in result
        assert "suffix_1" in result["target1"]
        assert "suffix_2" in result["target2"]
        assert result["target1"].replace("suffix_1","") == result["target2"].replace("suffix_2","")

    def test_get_full_target_labels_empty_targets(self):
        """Test with empty targets list."""
        result = get_full_target_labels([])

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_get_full_target_labels_mismatched_suffixes_error(self, target_data):
        """Test that mismatched suffixes count raises ValueError."""
        with pytest.raises(ValueError, match="Number of suffixes must match number of targets"):
            get_full_target_labels(
                [
                    Target(name="target1", data=[target_data]),
                    Target(name="target2", data=[target_data]),
                ],
                suffixes=["only_one_suffix"],
            )

    def test_get_full_target_labels_custom_scale_exponent(self, target_data):
        """Test with custom scale exponent."""
        result = get_full_target_labels([Target(name="target_name", data=[target_data])], rescale=4)

        assert isinstance(result, dict)
        assert "target_name" in result  # replace "target_data.name" with the actual name of the target
        target_label = result["target_name"]
        assert isinstance(target_label, str)
        assert re.search(r"\$Q_\{x,yy\}\$\s+=\s+0\.2\s+\|\s+0\.4", target_label)  # values /10 from the fixture
        assert re.search(r"\$Q_\{y,xy\}\$\s+=\s+--\s+\|\s+0\.3", target_label)


@pytest.fixture
def target_data():
    """Fixture for TargetData. Used in the TestGetFullTargetLabels class."""
    correctors = [
        Corrector(field=FieldComponent.b4, circuit="k4", magnet="K4", length=0.5),
        Corrector(field=FieldComponent.b5, circuit="k5", magnet="K5", length=0.5),
    ]
    optics = {1: Mock(), 2: Mock()}
    detuning = {
        1: DetuningMeasurement(X02=(1.51, 2.5), scale=1e3),
        2: DetuningMeasurement(Y11=(2.51, 2.5), X02=(3.52, 4.5), scale=1e3)
    }
    constraints = {1: Constraints(Y02="<=10"), 2: Constraints(Y02=">=11")}
    return TargetData(
        correctors=correctors, optics=optics, detuning=detuning, constraints=constraints
    )
