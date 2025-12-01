
from __future__ import annotations

from unittest.mock import patch

import pytest
from matplotlib import pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

from ir_amplitude_detuning.detuning.measurements import (
    Detuning,
    MeasureValue,
)
from ir_amplitude_detuning.plotting.detuning import (
    PlotSetup,
    get_average,
    get_handles_labels,
    get_measured_detuning_terms,
    get_ylabel,
    plot_measurements,
    plot_value_or_measurement,
)

# ============================================================================
# Tests for get_ylabel
# ============================================================================

class TestGetYlabel:
    """Test cases for the get_ylabel function."""

    def test_get_ylabel_no_rescale_no_delta(self):
        """Test label generation without rescaling or delta."""
        ylabel = get_ylabel(rescale=0, delta=False)

        assert "Q$_{a,b}$" in ylabel
        assert "[m$^{-1}$]" in ylabel
        assert "10$^" not in ylabel
        assert "$\\Delta$" not in ylabel

    def test_get_ylabel_with_rescale(self):
        """Test label generation with rescaling."""
        ylabel = get_ylabel(rescale=3, delta=False)

        assert "Q$_{a,b}$" in ylabel
        assert "10$^3$" in ylabel
        assert "[10$^3$ m$^{-1}$]" in ylabel

    def test_get_ylabel_with_delta(self):
        """Test label generation with delta flag."""
        ylabel = get_ylabel(rescale=0, delta=True)

        assert "$\\Delta$" in ylabel
        assert "Q$_{a,b}$" in ylabel

    def test_get_ylabel_with_delta_and_rescale(self):
        """Test label generation with both delta and rescaling."""
        ylabel = get_ylabel(rescale=2, delta=True)

        assert "$\\Delta$" in ylabel
        assert "10$^2$" in ylabel


# ============================================================================
# Tests for get_measured_detuning_terms
# ============================================================================

class TestGetMeasuredDetuningTerms:
    """Test cases for the get_measured_detuning_terms function."""

    def test_get_measured_detuning_terms_filters_none_values(self):
        """Test that terms with None values are filtered out."""
        setups = [
            PlotSetup(
                label="Setup1",
                measurement=Detuning(X10=MeasureValue(0.1, 0.01), Y01=MeasureValue(0.2, 0.02)),
                simulation=None,
            ),
            PlotSetup(
                label="Setup2",
                measurement=Detuning(X10=None, Y01=MeasureValue(0.15, 0.015)),
                simulation=None,
            ),
        ]

        terms = ["X10", "Y01", "X01"]
        result = get_measured_detuning_terms(setups, terms)

        assert "X10" in result
        assert "Y01" in result
        assert "X01" not in result

    def test_get_measured_detuning_terms_all_none(self):
        """Test when all measurements are None."""
        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(X10=None, Y01=None),
            simulation=None,
        )

        terms = ["X10", "Y01"]
        result = get_measured_detuning_terms([setup], terms)

        assert len(result) == 0

    def test_get_measured_detuning_terms_preserves_order(self):
        """Test that the order of terms is preserved."""
        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=MeasureValue(0.1, 0.01),
                Y01=MeasureValue(0.2, 0.02),
                X01=MeasureValue(0.3, 0.03),
            ),
            simulation=None,
        )

        terms = ["X01", "X10", "Y01"]
        result = get_measured_detuning_terms([setup], terms.copy())

        assert result == terms


# ============================================================================
# Tests for get_average
# ============================================================================


class TestGetAverage:
    """Test cases for the get_average function with mocked dependencies."""

    @patch.object(MeasureValue, 'weighted_rms')
    def test_get_average_auto_method_with_measure_values(self, mock_weighted_rms):
        """Test that auto method selects weighted_rms for MeasureValue objects."""
        mock_weighted_rms.return_value = MeasureValue(0.15, 0.015)

        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=MeasureValue(0.1, 0.01),
                Y01=MeasureValue(0.2, 0.02),
            ),
            simulation=None,
        )

        av_meas, av_label = get_average(setup, terms=["X10", "Y01"])

        assert av_label == "RMS"
        mock_weighted_rms.assert_called_once()
        assert av_meas == mock_weighted_rms.return_value

    @patch('numpy.sqrt')
    @patch('numpy.mean')
    def test_get_average_auto_method_with_floats(self, mock_mean, mock_sqrt):
        """Test that auto method selects rms for float values."""
        mock_mean.return_value = 0.025
        mock_sqrt.return_value = 0.158

        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(X10=0.1, Y01=0.2),
            simulation=None,
        )

        av_meas, av_label = get_average(setup, terms=["X10", "Y01"])

        assert av_meas == mock_sqrt.return_value
        assert av_label == "RMS"
        mock_mean.assert_called_once()
        mock_sqrt.assert_called_once()

    def test_get_average_filters_none_values(self):
        """Test that terms with None values are filtered out."""
        setup = PlotSetup(
                label="Setup2",
                measurement=Detuning(X10=None, Y01=MeasureValue(0.15, 0.015)),
                simulation=None,
            )

        terms = ["X10", "Y01", "X01"]
        result = get_average(setup, terms)
        assert result[0] == MeasureValue(0.15, 0.015)
        assert result[1] == "RMS"

    @patch.object(MeasureValue, 'weighted_rms')
    def test_get_average_only_requested_terms(self, mock_rms):
        """Test that rms method calls MeasureValue.rms for MeasureValue objects."""
        mock_rms.return_value = MeasureValue(0.16, 0.016)

        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=MeasureValue(0.1, 0.01),
                Y01=MeasureValue(0.2, 0.02),
                X01=MeasureValue(0.15, 0.015),
            ),
            simulation=None,
        )

        av_meas, av_label = get_average(setup, terms=["X10", "Y01"])

        assert av_meas == mock_rms.return_value
        assert av_label == "RMS"
        mock_rms.assert_called_once_with([setup.measurement.X10, setup.measurement.Y01])

    @patch.object(MeasureValue, 'rms')
    def test_get_average_rms_method_with_measure_values(self, mock_rms):
        """Test that rms method calls MeasureValue.rms for MeasureValue objects."""
        mock_rms.return_value = MeasureValue(0.16, 0.016)

        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=MeasureValue(0.1, 0.01),
                Y01=MeasureValue(0.2, 0.02),
                X01=MeasureValue(0.15, 0.015),
            ),
            simulation=None,
        )

        av_meas, av_label = get_average(setup, terms=["X10", "Y01", "X01"], method="rms")

        assert av_meas == mock_rms.return_value
        assert av_label == "RMS"
        mock_rms.assert_called_once()

    @patch('numpy.sqrt')
    @patch('numpy.mean')
    def test_get_average_rms_method_with_floats(self, mock_mean, mock_sqrt):
        """Test that rms method works with float values when not all are MeasureValue."""
        mock_mean.return_value = 0.0233
        mock_sqrt.return_value = 0.153

        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=0.1,
                Y01=0.2,
                X01=0.3,
            ),
            simulation=None,
        )

        av_meas, av_label = get_average(setup, terms=["X10", "Y01", "X01"], method="rms")

        assert av_meas == mock_sqrt.return_value
        assert av_label == "RMS"
        mock_mean.assert_called_once()
        mock_sqrt.assert_called_once()

    @patch.object(MeasureValue, 'weighted_rms')
    def test_get_average_explicit_weighted_rms(self, mock_weighted_rms):
        """Test that weighted_rms method calls MeasureValue.weighted_rms."""
        mock_weighted_rms.return_value = MeasureValue(0.18, 0.018)

        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=MeasureValue(0.1, 0.01),
                Y01=MeasureValue(0.25, 0.025),
                X01=MeasureValue(0.2, 0.02),
            ),
            simulation=None,
        )

        av_meas, av_label = get_average(setup, terms=["X10", "Y01", "X01"], method="weighted_rms")

        assert av_meas is mock_weighted_rms.return_value
        assert av_label == "RMS"
        mock_weighted_rms.assert_called_once()

    @patch.object(MeasureValue, 'mean')
    def test_get_average_mean_method_with_measure_values(self, mock_mean):
        """Test that mean method calls MeasureValue.mean for MeasureValue objects."""
        mock_mean.return_value = MeasureValue(0.15, 0.015)

        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=MeasureValue(0.1, 0.01),
                Y01=MeasureValue(0.2, 0.02),
            ),
            simulation=None,
        )

        av_meas, av_label = get_average(setup, terms=["X10", "Y01"],  method="mean")

        assert av_meas == mock_mean.return_value
        assert av_label == "Mean"
        mock_mean.assert_called_once()

    @patch('numpy.mean')
    def test_get_average_mean_method_with_floats(self, mock_mean):
        """Test that mean method works with float values."""
        mock_mean.return_value = 0.15

        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=0.1,
                Y01=0.2,
            ),
            simulation=None,
        )

        av_meas, av_label = get_average(setup, terms=["X10", "Y01"], method="mean")

        assert av_meas == mock_mean.return_value
        assert av_label == "Mean"
        mock_mean.assert_called_once()

    @patch.object(MeasureValue, 'weighted_mean')
    def test_get_average_weighted_mean_method(self, mock_weighted_mean):
        """Test that weighted_mean method calls MeasureValue.weighted_mean."""
        mock_weighted_mean.return_value = MeasureValue(0.17, 0.017)

        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=MeasureValue(0.1, 0.01),
                Y01=MeasureValue(0.25, 0.025),
            ),
            simulation=None,
        )

        av_meas, av_label = get_average(setup, terms=["X10", "Y01"], method="weighted_mean")

        assert av_meas == mock_weighted_mean.return_value
        assert av_label == "Mean"
        mock_weighted_mean.assert_called_once()

    def test_get_average_weighted_method_raises_error_with_non_measure_values(self):
        """Test that weighted methods raise error when mixed with non-MeasureValue objects."""
        setup = PlotSetup(
            label="Setup",
            measurement=Detuning(
                X10=0.1,  # float instead of MeasureValue
                Y01=MeasureValue(0.2, 0.02),
            ),
            simulation=None,
        )

        with pytest.raises(ValueError, match="requires all measurements to be of type MeasureValue"):
            get_average(setup, terms=["X10", "Y01"],  method="weighted_rms")

        with pytest.raises(ValueError, match="requires all measurements to be of type MeasureValue"):
            get_average(setup, terms=["X10", "Y01"],  method="weighted_mean")


# ============================================================================
# Tests for plot_value_or_measurement
# ============================================================================

class TestPlotValueOrMeasurement:
    """Test cases for the plot_value_or_measurement function."""

    def test_plot_value_or_measurement_with_error(self):
        """Test plotting a MeasureValue with error."""
        fig, ax = plt.subplots()
        measurement = MeasureValue(0.1, 0.01)

        result = plot_value_or_measurement(ax, measurement, x=1.0, label="test", color="red")

        assert result is not None
        assert isinstance(result, ErrorbarContainer)
        assert len(ax.containers) > 0
        plt.close(fig)

    def test_plot_value_or_measurement_without_error(self):
        """Test plotting a MeasureValue without error."""
        fig, ax = plt.subplots()
        measurement = MeasureValue(0.1, 0)

        result = plot_value_or_measurement(ax, measurement, x=1.0, label="test", color="blue")

        assert result is not None
        assert isinstance(result[0], Line2D)
        assert not len(ax.containers)
        assert len(ax.lines) > 0
        plt.close(fig)

    def test_plot_value_or_measurement_float_value(self):
        """Test plotting a float value."""
        fig, ax = plt.subplots()
        measurement = 0.15

        result = plot_value_or_measurement(ax, measurement, x=1.0, label="test", color="green")

        assert result is not None
        assert isinstance(result[0], Line2D)
        assert not len(ax.containers)
        assert len(ax.lines) > 0
        plt.close(fig)


# ============================================================================
# Tests for get_handles_labels
# ============================================================================

class TestGetHandlesLabels:
    """Test cases for the get_handles_labels function."""

    def test_get_handles_labels_basic(self):
        """Test handle and label generation from PlotSetup objects."""
        setups = [
            PlotSetup(label="Measurement 1", measurement=None, simulation=None),
            PlotSetup(label="Measurement 2", measurement=None, simulation=None),
        ]

        handles, labels = get_handles_labels(setups)

        assert len(handles) == 2
        assert len(labels) == 2
        assert labels == ["Measurement 1", "Measurement 2"]
        for handle in handles:
            assert handle.get_label() in labels

    def test_get_handles_labels_with_custom_colors(self):
        """Test that custom colors are preserved in handles."""
        setups = [
            PlotSetup(label="Setup 1", measurement=None, simulation=None, color="red"),
            PlotSetup(label="Setup 2", measurement=None, simulation=None, color="blue"),
        ]

        handles, labels = get_handles_labels(setups)

        assert len(handles) == 2
        assert handles[0].get_color() == "red"
        assert handles[1].get_color() == "blue"


# ============================================================================
# Tests for plot_measurements
# ============================================================================

class TestPlotMeasurements:
    """Test cases for the plot_measurements function."""

    @pytest.mark.parametrize("with_simulation", [True, False])
    def test_plot_measurements_basic(self, with_simulation):
        """Test basic plotting with measurements and simulations."""
        setups = [
            PlotSetup(
                label="Measurement A",
                measurement=Detuning(X10=MeasureValue(0.1, 0.01), Y01=MeasureValue(0.2, 0.02)),
                simulation=Detuning(X10=0.11, Y01=0.21) if with_simulation else None,
            ),
        ]

        fig = plot_measurements(setups, rescale=0, ylim=(-0.5, 0.5))

        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Q$_{a,b}$ [m$^{-1}$]"
        assert len(ax.lines) >= 2
        assert len(ax.patches) == (2 if with_simulation else 0)  # from the bars
        plt.close(fig)

    @pytest.mark.parametrize("with_average", [True, False])
    def test_plot_measurements_multiple_setups(self, with_average):
        """Test plotting with multiple measurement setups."""
        setups = [
            PlotSetup(
                label="Setup 1",
                measurement=Detuning(X10=MeasureValue(0.1, 0.01)),
                simulation=Detuning(X10=0.11),
            ),
            PlotSetup(
                label="Setup 2",
                measurement=Detuning(X10=MeasureValue(0.15, 0.015)),
                simulation=Detuning(X10=0.16),
            ),
        ]

        fig = plot_measurements(setups, rescale=0, ylim=(-0.3, 0.3), average=with_average, measured_only=True)

        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.lines) >= 2
        assert len(ax.patches) == 2
        assert len(ax.get_xticklabels()) == (2 if with_average else 1)
        plt.close(fig)

    @pytest.mark.parametrize("measured_only", [True, False])
    def test_plot_measurements_measured_only_filter(self, measured_only):
        """Test that measured_only filters out terms with no measurements."""
        setups = [
            PlotSetup(
                label="Partial Setup",
                measurement=Detuning(X10=MeasureValue(0.1, 0.01), Y01=None),
                simulation=None,
            ),
        ]

        terms = ["X10", "Y01"]
        fig = plot_measurements(
            setups,
            rescale=0,
            terms=terms,
            measured_only=measured_only,
            ylim=(-0.2, 0.2),
        )

        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.get_xticklabels()) == (1 if measured_only else len(terms))
        plt.close(fig)

    @pytest.mark.parametrize("rescale", [0, 3])
    def test_plot_measurements_with_rescale(self, rescale):
        """Test plotting with rescaling applied."""
        setups = [
            PlotSetup(
                label="Setup",
                measurement=Detuning(X10=MeasureValue(1000, 10)),
                simulation=Detuning(X10=1001),
            ),
        ]

        fig = plot_measurements(setups, rescale=rescale, ylim=None)

        assert fig is not None
        ax = fig.axes[0]
        if rescale:
            assert "10$^3$" in ax.get_ylabel()
        else:
            assert "10" not in ax.get_ylabel()

        ylim = ax.get_ylim()
        if rescale:   # 0 or 3 (== *10**-3)
            assert ylim[1] / 10 < 1  # y-max should be slightly larger than 1
        else:
            assert ylim[1] / 10 > 100  # y-max should be slightly larger than 1000
        plt.close(fig)


    def test_plot_measurements_with_delta(self):
        """Test plotting with delta flag."""
        setups = [
            PlotSetup(
                label="Delta Setup",
                measurement=Detuning(X10=MeasureValue(0.05, 0.005)),
                simulation=None,
            ),
        ]

        fig = plot_measurements(setups, rescale=0, is_shift=True, ylim=(-0.1, 0.1))

        assert fig is not None
        ax = fig.axes[0]
        assert "$\\Delta$" in ax.get_ylabel()
        plt.close(fig)
