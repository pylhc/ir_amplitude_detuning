"""
Unit tests for detuning targets classes.
"""

from __future__ import annotations

import pytest

from ir_amplitude_detuning.detuning.measurements import Constraints, Detuning
from ir_amplitude_detuning.detuning.targets import Target, TargetData
from ir_amplitude_detuning.utilities.common import BeamDict


class TestTarget:
    """Tests for Target class."""

    def test_target_creation(self):
        """Test basic Target creation."""
        # Create mock TargetData
        mock_correctors = ["MQX1", "MQX2"]
        mock_optics = {1: "optics1", 2: "optics2"}
        mock_detuning = {1: Detuning(1.0, 0.0), 2: Detuning(2.0, 0.0)}
        mock_constraints = {1: Constraints("<= 4", ">= 1"), 2: Constraints(">= 1", "<= 0")}

        target_data = TargetData(
            correctors=mock_correctors,
            optics=mock_optics,
            detuning=mock_detuning,
            constraints=mock_constraints,
            label="test_label"
        )

        target = Target(name="test_target", data=[target_data])

        assert target.name == "test_target"
        assert len(target.data) == 1
        assert target.correctors == sorted(mock_correctors)

    def test_target_creation_with_period_in_name(self):
        """Test that Target raises NameError when name contains period."""
        mock_correctors = ["MQX1", "MQX2"]
        mock_optics = {1: "optics1", 2: "optics2"}
        mock_detuning = {1: Detuning(1.0, 0.0), 2: Detuning(2.0, 0.0)}

        target_data = TargetData(
            correctors=mock_correctors,
            optics=mock_optics,
            detuning=mock_detuning,
            label="test_label"
        )

        with pytest.raises(NameError, match="No periods allowed in target name!"):
            Target(name="test.target", data=[target_data])

    def test_target_creation_with_duplicate_labels(self):
        """Test that Target raises ValueError when TargetData has duplicate labels."""
        mock_correctors = ["MQX1", "MQX2"]
        mock_optics = {1: "optics1", 2: "optics2"}
        mock_detuning = {1: Detuning(1.0, 0.0), 2: Detuning(2.0, 0.0)}

        target_data1 = TargetData(
            correctors=mock_correctors,
            optics=mock_optics,
            detuning=mock_detuning,
            label="duplicate_label"
        )

        target_data2 = TargetData(
            correctors=mock_correctors,
            optics=mock_optics,
            detuning=mock_detuning,
            label="duplicate_label"
        )

        with pytest.raises(ValueError, match="All TargetData in Target 'test_target' must have unique labels"):
            Target(name="test_target", data=[target_data1, target_data2])

    def test_target_correctors_aggregation(self):
        """Test that Target correctly aggregates correctors from all TargetData."""
        mock_correctors1 = ["MQX1", "MQX2"]
        mock_correctors2 = ["MQX3", "MQX4"]
        mock_optics = {1: "optics1", 2: "optics2"}
        mock_detuning = {1: Detuning(1.0, 0.0), 2: Detuning(2.0, 0.0)}

        target_data1 = TargetData(
            correctors=mock_correctors1,
            optics=mock_optics,
            detuning=mock_detuning,
            label="test_label1"
        )

        target_data2 = TargetData(
            correctors=mock_correctors2,
            optics=mock_optics,
            detuning=mock_detuning,
            label="test_label2"
        )

        target = Target(name="test_target", data=[target_data1, target_data2])

        # Should contain all unique correctors from both TargetData
        expected_correctors = sorted(["MQX1", "MQX2", "MQX3", "MQX4"])
        assert target.correctors == expected_correctors


class TestTargetData:
    """Tests for TargetData class."""

    def test_target_data_creation(self):
        """Test basic TargetData creation."""
        mock_correctors = ["MQX1", "MQX2"]
        mock_optics = {1: "optics1", 2: "optics2"}
        mock_detuning = {1: Detuning(1.0, 0.0), 2: Detuning(2.0, 0.0)}
        mock_constraints = {1: Constraints("<= 4", ">= 1"), 2: Constraints(">= 1", "<= 0")}

        target_data = TargetData(
            correctors=mock_correctors,
            optics=mock_optics,
            detuning=mock_detuning,
            constraints=mock_constraints,
            label="test_label"
        )

        assert target_data.label == "test_label"
        assert target_data.correctors == sorted(mock_correctors)
        assert target_data.optics == mock_optics
        assert isinstance(target_data.detuning, BeamDict)
        assert isinstance(target_data.detuning[0], Detuning)  # defaults to empty Detuning
        assert isinstance(target_data.constraints, BeamDict)
        assert isinstance(target_data.constraints[0], Constraints)  # defaults to empty Constraints
        assert target_data.beams == (1, 2)

    def test_target_data_creation_without_label(self):
        """Test TargetData creation without explicit label."""
        mock_correctors = ["MQX1", "MQX2"]
        mock_optics = {1: "optics1", 2: "optics2"}
        mock_detuning = {1: Detuning(1.0, 0.0), 2: Detuning(2.0, 0.0)}

        target_data = TargetData(
            correctors=mock_correctors,
            optics=mock_optics,
            detuning=mock_detuning,
            label=None  # Should generate hash-based label
        )

        # Label should be a hash string
        assert isinstance(target_data.label, str)
        assert len(target_data.label) > 0

    def test_target_data_creation_with_none_constraints(self):
        """Test TargetData creation with None constraints."""
        mock_correctors = ["MQX1", "MQX2"]
        mock_optics = {1: "optics1", 2: "optics2"}
        mock_detuning = {1: Detuning(1.0, 0.0), 2: Detuning(2.0, 0.0)}

        target_data = TargetData(
            correctors=mock_correctors,
            optics=mock_optics,
            detuning=mock_detuning,
            constraints=None,
            label="test_label"
        )

        # Constraints should be an empty BeamDict
        assert isinstance(target_data.constraints, BeamDict)
        assert len(target_data.constraints) == 0
        assert isinstance(target_data.constraints[1], Constraints)

    def test_target_data_beams_extraction(self):
        """Test that TargetData correctly extracts beams from optics."""
        mock_correctors = ["MQX1", "MQX2"]
        mock_optics = {1: "optics1", 4: "optics4"}
        mock_detuning = {1: Detuning(1.0, 0.0), 2: Detuning(4.0, 0.0)}

        target_data = TargetData(
            correctors=mock_correctors,
            optics=mock_optics,
            detuning=mock_detuning,
            label="test_label"
        )

        assert target_data.beams == (1, 4)
        assert target_data.detuning[4] == mock_detuning[2]
