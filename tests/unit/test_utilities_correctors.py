from __future__ import annotations

import pytest

from ir_amplitude_detuning.utilities.correctors import (
    Corrector,
    CorrectorMask,
    FieldComponent,
    fill_corrector_masks,
    get_fields,
)

# ============================================================================
# Tests for FieldComponent Enum
# ============================================================================

class TestFieldComponent:
    """Test cases for the FieldComponent enum."""

    def test_field_component_values(self):
        """Test that all field component values are correct."""
        assert FieldComponent.b4 == "b4"
        assert FieldComponent.b5 == "b5"
        assert FieldComponent.b6 == "b6"


# ============================================================================
# Tests for Corrector Class
# ============================================================================

class TestCorrector:
    """Test cases for the Corrector class."""

    def test_corrector_initialization(self):
        """Test basic corrector initialization."""
        corrector = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="MCTX.3L1",
            circuit="kctx3.l1",
        )
        assert corrector.field == FieldComponent.b4
        assert corrector.length == 0.5
        assert corrector.magnet == "MCTX.3L1"
        assert corrector.circuit == "kctx3.l1"
        assert corrector.ip is None
        assert corrector.madx_type is None

    def test_corrector_initialization_with_optional_fields(self):
        """Test corrector initialization with optional fields."""
        corrector = Corrector(
            field=FieldComponent.b6,
            length=0.3,
            magnet="MCTX.2R2",
            circuit="kctx2.r2",
            ip=2,
            madx_type="MCTX",
        )
        assert corrector.ip == 2
        assert corrector.madx_type == "MCTX"

    def test_corrector_invalid_field(self):
        """Test that invalid field raises ValueError."""
        with pytest.raises(ValueError, match="[Ff]ield must be one of"):
            Corrector(
                field="b7",  # Invalid field
                length=0.5,
                magnet="MCTX.3L1",
                circuit="kctx3.l1",
            )

    def test_corrector_set_field(self):
        """Test that invalid field raises ValueError."""
        c = Corrector(
                field=FieldComponent.b4,
                length=0.5,
                magnet="MCTX.3L1",
                circuit="kctx3.l1",
            )

        # valid setting of field
        c.field = FieldComponent.b5
        assert c.field == "b5"

        # invalid setting of field
        with pytest.raises(ValueError, match="[Ff]ield must be one of"):
            c.field = "b7"

    def test_corrector_hash(self):
        """Test corrector hash is consistent and unique."""
        c1 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="MCTX.3L1",
            circuit="kctx3.l1",
        )
        c2 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="MCTX.3L1",
            circuit="kctx3.l1",
        )
        c3 = Corrector(
            field=FieldComponent.b5,
            length=0.5,
            magnet="MCTX.3L1",
            circuit="kctx3.l1",
        )
        c4 = Corrector(
            field=FieldComponent.b5,
            length=0.5,
            magnet="MCDX.3L1",
            circuit="kcdx3.l1",
        )

        # Same corrector should have same hash
        assert hash(c1) == hash(c2)

        # Beware, that changing only the field component will lead to the same corrector hash
        # (as field component is more of a meta-information,
        # the corrector is defined by magnet and circuit)
        assert hash(c1) == hash(c3)

        # Different magnet/circuit should have different hash
        assert hash(c1) != hash(c4)

    def test_corrector_less_than_by_field(self):
        """Test corrector comparison by field component."""
        c_b4 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="MCTX.3L1",
            circuit="kctx3.l1",
        )
        c_b5 = Corrector(
            field=FieldComponent.b5,
            length=0.5,
            magnet="MCTX.3L1",
            circuit="kctx3.l1",
        )
        c_b6 = Corrector(
            field=FieldComponent.b6,
            length=0.5,
            magnet="MCTX.3L1",
            circuit="kctx3.l1",
        )

        assert c_b4 < c_b5
        assert c_b5 < c_b6
        assert c_b4 < c_b6
        assert not (c_b5 < c_b4)

    def test_corrector_less_than_by_circuit(self):
        """Test corrector comparison falls back to circuit when magnet pattern doesn't match."""
        c1 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="NOMATCH1",
            circuit="kctx1.l1",
        )
        c2 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="NOMATCH2",
            circuit="kctx2.l1",
        )

        assert c1 < c2
        assert not (c2 < c1)

    def test_corrector_less_than_by_element(self):
        """Test corrector comparison falls back to circuit when magnet pattern doesn't match."""
        c_3l1 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="3l1",
            circuit="3.l1",
        )
        c_4l1 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="4l1",
            circuit="4.l1",
        )
        c_3l2 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="3l2",
            circuit="3.l2",
        )
        c_3r1 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="3r1",
            circuit="3.r1",
        )
        c_4r1 = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="4r1",
            circuit="4.r1",
        )

        # assert ip1 < ip2
        assert c_3l1 < c_3l2

        # assert left before right
        assert c_3l1 < c_3r1
        assert c_4l1 < c_3r1

        # assert going from left to right
        assert c_4l1 < c_3l1
        assert c_3l1 < c_3r1
        assert c_3r1 < c_4r1

    def test_corrector_sorting(self):
        """Test that correctors can be sorted."""
        correctors = [
            Corrector(
                field=FieldComponent.b6,
                length=0.5,
                magnet="MCTX.3L1",
                circuit="kctx3.l1",
            ),
            Corrector(
                field=FieldComponent.b4,
                length=0.5,
                magnet="MCTX.3L1",
                circuit="kctx3.l1",
            ),
            Corrector(
                field=FieldComponent.b5,
                length=0.5,
                magnet="MCTX.3L1",
                circuit="kctx3.l1",
            ),
        ]

        sorted_correctors = sorted(correctors)
        assert sorted_correctors[0].field == FieldComponent.b4
        assert sorted_correctors[1].field == FieldComponent.b5
        assert sorted_correctors[2].field == FieldComponent.b6


# ============================================================================
# Tests for CorrectorMask Class
# ============================================================================

class TestCorrectorMask:
    """Test cases for the CorrectorMask class."""

    def test_get_corrector_left_side(self):
        """Test generating corrector from mask on left side."""
        mask = CorrectorMask(
            field=FieldComponent.b4,
            length=0.5,
            magnet_pattern="MCTX.3{side}{ip}",
            circuit_pattern="kctx3.{side}{ip}",
        )
        corrector = mask.get_corrector("L", 1)

        assert corrector.field == FieldComponent.b4
        assert corrector.length == 0.5
        assert corrector.magnet == "MCTX.3L1"
        assert corrector.circuit == "kctx3.l1"
        assert corrector.ip == 1

    def test_get_corrector_right_side(self):
        """Test generating corrector from mask on right side."""
        mask = CorrectorMask(
            field=FieldComponent.b6,
            length=0.3,
            magnet_pattern="MCTX.2{side}{ip}",
            circuit_pattern="kctx2.{side}{ip}",
        )
        corrector = mask.get_corrector("R", 2)

        assert corrector.magnet == "MCTX.2R2"
        assert corrector.circuit == "kctx2.r2"
        assert corrector.ip == 2

    def test_get_corrector_lowercase_side(self):
        """Test generating corrector with lowercase side input."""
        mask = CorrectorMask(
            field=FieldComponent.b5,
            length=0.4,
            magnet_pattern="MCTX.4{side}{ip}",
            circuit_pattern="kctx4.{side}{ip}",
        )
        corrector = mask.get_corrector("r", 3)

        assert corrector.magnet == "MCTX.4R3"
        assert corrector.circuit == "kctx4.r3"


# ============================================================================
# Tests for Utility Functions
# ============================================================================

class TestGetFields:
    """Test cases for the get_fields function."""

    def test_get_fields_empty_list(self):
        """Test get_fields with empty corrector list."""
        result = get_fields([])
        assert result == []

    def test_get_fields_single_field(self):
        """Test get_fields with correctors of single field type."""
        correctors = [
            Corrector(
                field=FieldComponent.b4,
                length=0.5,
                magnet=f"MCTX.3L{i}",
                circuit=f"kctx3.l{i}",
            )
            for i in range(3)
        ]
        result = get_fields(correctors)
        assert result == [FieldComponent.b4]

    def test_get_fields_multiple_fields(self):
        """Test get_fields with correctors of multiple field types."""
        correctors = [
            Corrector(
                field=FieldComponent.b4,
                length=0.5,
                magnet="MCTX.3L1",
                circuit="kctx3.l1",
            ),
            Corrector(
                field=FieldComponent.b5,
                length=0.4,
                magnet="MCTX.4L1",
                circuit="kctx4.l1",
            ),
            Corrector(
                field=FieldComponent.b6,
                length=0.3,
                magnet="MCTX.5L1",
                circuit="kctx5.l1",
            ),
        ]
        result = get_fields(correctors)
        assert result == [FieldComponent.b4, FieldComponent.b5, FieldComponent.b6]

    def test_get_fields_duplicate_fields(self):
        """Test get_fields removes duplicates and sorts."""
        correctors = [
            Corrector(
                field=FieldComponent.b6,
                length=0.5,
                magnet="MCTX.1L1",
                circuit="kctx1.l1",
            ),
            Corrector(
                field=FieldComponent.b4,
                length=0.5,
                magnet="MCTX.2L1",
                circuit="kctx2.l1",
            ),
            Corrector(
                field=FieldComponent.b6,
                length=0.5,
                magnet="MCTX.3L1",
                circuit="kctx3.l1",
            ),
            Corrector(
                field=FieldComponent.b4,
                length=0.5,
                magnet="MCTX.4L1",
                circuit="kctx4.l1",
            ),
        ]
        result = get_fields(correctors)
        assert result == [FieldComponent.b4, FieldComponent.b6]


class TestFillCorrectorMasks:
    """Test cases for the fill_corrector_masks function."""

    def test_fill_corrector_masks_single_mask_single_ip_single_side(self):
        """Test fill_corrector_masks with single mask, single IP, single side."""
        mask = CorrectorMask(
            field=FieldComponent.b4,
            length=0.5,
            magnet_pattern="MCTX.3{side}{ip}",
            circuit_pattern="kctx3.{side}{ip}",
        )
        result = fill_corrector_masks([mask], ips=[1], sides="L")

        assert len(result) == 1
        assert result[0].magnet == "MCTX.3L1"
        assert result[0].circuit == "kctx3.l1"
        assert result[0].ip == 1
        assert result[0].length == 0.5

    def test_fill_corrector_masks_single_mask_multiple_ips(self):
        """Test fill_corrector_masks with single mask, multiple IPs."""
        mask = CorrectorMask(
            field=FieldComponent.b4,
            length=0.5,
            magnet_pattern="MCTX.3{side}{ip}",
            circuit_pattern="kctx3.{side}{ip}",
        )
        result = fill_corrector_masks([mask], ips=[1, 2], sides="L")

        assert len(result) == 2
        assert result[0].magnet == "MCTX.3L1"
        assert result[0].circuit =="kctx3.l1"
        assert result[1].magnet == "MCTX.3L2"
        assert result[1].circuit == "kctx3.l2"
        assert get_fields(result) == [FieldComponent.b4]
        assert all(c.length == 0.5 for c in result)

    def test_fill_corrector_masks_single_mask_multiple_sides(self):
        """Test fill_corrector_masks with single mask, multiple sides."""
        mask = CorrectorMask(
            field=FieldComponent.b4,
            length=0.5,
            magnet_pattern="MCTX.3{side}{ip}",
            circuit_pattern="kctx3.{side}{ip}",
        )
        result = fill_corrector_masks([mask], ips=[1], sides="LR")

        assert len(result) == 2
        assert result[0].magnet == "MCTX.3L1"
        assert result[0].circuit =="kctx3.l1"
        assert result[1].magnet == "MCTX.3R1"
        assert result[1].circuit == "kctx3.r1"
        assert get_fields(result) == [FieldComponent.b4]
        assert all(c.length == 0.5 for c in result)

    def test_fill_corrector_masks_single_mask_multiple_ips_and_sides(self):
        """Test fill_corrector_masks with single mask, multiple IPs and sides."""
        mask = CorrectorMask(
            field=FieldComponent.b4,
            length=0.5,
            magnet_pattern="MCTX.3{side}{ip}",
            circuit_pattern="kctx3.{side}{ip}",
        )
        result = fill_corrector_masks([mask], ips=[1, 2], sides="LR")

        assert len(result) == 4
        magnets = sorted([c.magnet for c in result])
        assert magnets == ["MCTX.3L1", "MCTX.3L2", "MCTX.3R1", "MCTX.3R2"]
        circuits = sorted([c.circuit for c in result])
        assert circuits == ["kctx3.l1", "kctx3.l2", "kctx3.r1", "kctx3.r2"]
        assert all(c.length == 0.5 for c in result)
        assert get_fields(result) == [FieldComponent.b4]

    def test_fill_corrector_masks_multiple_masks(self):
        """Test fill_corrector_masks with multiple masks."""
        masks = [
            CorrectorMask(
                field=FieldComponent.b4,
                length=0.5,
                magnet_pattern="MCTX.3{side}{ip}",
                circuit_pattern="kctx3.{side}{ip}",
            ),
            CorrectorMask(
                field=FieldComponent.b6,
                length=0.3,
                magnet_pattern="MCTX.5{side}{ip}",
                circuit_pattern="kctx5.{side}{ip}",
            ),
        ]
        result = fill_corrector_masks(masks, ips=[1], sides="L")

        assert len(result) == 2
        assert get_fields(result) == [FieldComponent.b4, FieldComponent.b6]
        assert result[0].magnet == "MCTX.3L1"
        assert result[0].circuit =="kctx3.l1"
        assert result[1].magnet == "MCTX.5L1"
        assert result[1].circuit == "kctx5.l1"

    def test_fill_corrector_masks_with_pre_filled_correctors(self):
        """Test fill_corrector_masks with mix of masks and correctors."""
        mask = CorrectorMask(
            field=FieldComponent.b4,
            length=0.5,
            magnet_pattern="MCTX.3{side}{ip}",
            circuit_pattern="kctx3.{side}{ip}",
        )
        corrector = Corrector(
            field=FieldComponent.b6,
            length=0.3,
            magnet="MCTX.5L1",
            circuit="kctx5.l1",
        )
        result = fill_corrector_masks([mask, corrector], ips=[1], sides="L")

        assert len(result) == 2
        # Corrector should pass through unchanged
        assert corrector in result

    def test_fill_corrector_masks_sorting(self):
        """Test that fill_corrector_masks returns sorted correctors."""
        mask_b6 = CorrectorMask(
            field=FieldComponent.b6,
            length=0.3,
            magnet_pattern="MCTX.5{side}{ip}",
            circuit_pattern="kctx5.{side}{ip}",
        )
        mask_b4 = CorrectorMask(
            field=FieldComponent.b4,
            length=0.5,
            magnet_pattern="MCTX.3{side}{ip}",
            circuit_pattern="kctx3.{side}{ip}",
        )
        result = fill_corrector_masks([mask_b6, mask_b4], ips=[1], sides="L")

        assert len(result) == 2
        assert result[0] < result[1]
