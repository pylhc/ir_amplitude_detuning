"""
Detuning Plots
--------------

Plotting utilities to compare detuning measurements and simulation results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from omc3.plotting.utils import annotations as pannot
from omc3.plotting.utils import colors as pcolors
from omc3.plotting.utils import style as pstyle

from ir_amplitude_detuning.detuning.measurements import (
    Detuning,
    DetuningMeasurement,
    MeasureValue,
)
from ir_amplitude_detuning.utilities import latex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.container import ErrorbarContainer


LOG = logging.getLogger(__name__)


@dataclass
class PlotSetup:
    """Container to define different detuning measurements to plot
    with the plot_measurements function.

    Args:
        label (str): Label for the measurement.
        measurement (DetuningMeasurement | Detuning | None):
                    Measurement to plot. Is expected, but can be `None` if you only want to plot simulation results.
        simulation (Detuning | DetuningMeasurement | None):
                    Simulation results to plot, corresponding to the given measurement.
                    Is expected to be a `Detuning` object, so without errors, but can be given as `DetuningMeasurement` instead,
                    yet the errors will be ignored.
        color (str, optional): Color for the measurement.
    """
    label: str
    measurement: DetuningMeasurement | Detuning | None
    simulation: Detuning | DetuningMeasurement | None = None
    color: str = None

    def get_color(self, idx: int):
        if self.color is not None:
            return self.color
        return pcolors.get_mpl_color(idx)


def plot_measurements(setups: Sequence[PlotSetup], **kwargs):
    """Plot multiple measurements on the same plot.

    Args:
        measurements (Sequence[MeasurementSetup]): List of MeasurementSetup objects to plot.

    Keyword Args:
        style (str): The plot style to use.
        manual_style (dict): Dictionary of matplotlib style settings.
        is_shift (bool): Indicate if the given data is a "detuning shift" e.g. difference between two setups.
            This simply adds a "Delta" prefix to the y-axis label, if no label is given.
        ylim (Sequence[float, float]): y-axis limits.
        rescale (int): Exponent of the scaling factor.
            (e.g. 3 to give data in units of 10^3, which multiplies the data by 10^-3)
            Default: 3.
        ncol (int): Number of columns in the plot.
        transpose_legend (bool): Transpose the legend order.
        terms (Sequence[str]): Terms to plot.
        measured_only (bool): Only plot terms for which at least one measurement has a value.
        average (bool | str): Add an average values to the plot,
            Can be "rms", "weighted_rms", "mean", "weighted_mean".
            The default for `True` is "weighted_rms".
            Default: False.
    """
    # Set Style ---
    manual_style = {
        "figure.figsize": [6.50, 3.0],
        "figure.subplot.left": 0.12,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.right": 0.99,
        "figure.subplot.top": 0.77,
        "errorbar.capsize": 5,
        "lines.marker": "x",
        "lines.markersize": 4,
        "axes.grid": False,
        "ytick.minor.visible": True,
    }
    manual_style.update(kwargs.pop('manual_style', {}))
    pstyle.set_style(kwargs.pop("style", "standard"), manual_style)

    # Parse Keyword Args ---
    rescale: int = kwargs.pop('rescale', 3)
    is_shift: bool = kwargs.pop("is_shift", False)
    ylabel: str = kwargs.pop("ylabel", get_ylabel(rescale=rescale, delta=is_shift))
    ylim: Sequence[float, float] = kwargs.pop("ylim")
    ncol: int = kwargs.pop('ncol', 3)
    transpose_legend: bool = kwargs.pop('transpose_legend', False)
    terms: Sequence[str] = kwargs.pop('terms', Detuning.all_terms())
    measured_only: Sequence[str] = kwargs.pop('measured_only', False)
    average: str | bool = kwargs.pop('average', False)
    if average is True:
        average = "auto"

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}")

    # Prepare Constants ---
    detuning_terms = terms
    if measured_only:
        detuning_terms = get_measured_detuning_terms(setups, terms)

    n_components = len(detuning_terms) + bool(average)
    n_measurements = len(setups)
    measurement_width = 1 / (n_measurements + 1)
    bar_width = measurement_width * 0.15
    rescale_value = 10**-rescale

    # Generate Plot ------
    fig, ax = plt.subplots()

    # plot Lines ---
    ax.axhline(0, color="black", lw=1, ls="-", marker="", zorder=-10)  # y = 0
    for idx in range(1, n_components):
        ax.axvline(idx, color="grey", lw=1, ls="--", marker="", zorder=-10)  # split components

    for idx_measurement, measurement_setup in enumerate(setups):
        color = measurement_setup.get_color(idx_measurement)
        for idx_component, detuning_component in enumerate(detuning_terms):
            x_pos = idx_component + (idx_measurement + 1) * measurement_width
            line_label = f"_{measurement_setup.label}_{detuning_component}"

            # Plot Measurement ---
            if measurement_setup.measurement is not None:
                measurement: MeasureValue | float | None = getattr(measurement_setup.measurement, detuning_component)
                if measurement is not None:
                    plot_value_or_measurement(
                        ax,
                        measurement=measurement * rescale_value,
                        x=x_pos,
                        label=line_label,
                        color=color,
                    )

            # Plot Simulation ---
            if measurement_setup.simulation is not None:
                simulation : MeasureValue | float | None = getattr(measurement_setup.simulation, detuning_component)
                if simulation is not None:
                    simulation = getattr(simulation, "value", simulation)  # no errors on the bars. Just show the value
                    ax.bar(
                        x=x_pos,
                        height=simulation * rescale_value,
                        width=bar_width,
                        bottom=0,
                        label=f"{line_label}_sim",
                        color=color,
                        alpha=0.3,
                    )

        # Add Average ---
        extra_labels = []
        if average:
            av_meas, av_label = get_average(measurement_setup, terms=detuning_terms, method=average)
            x_pos = n_components - 1 + (idx_measurement + 1) * measurement_width
            plot_value_or_measurement(
                ax, measurement=av_meas * rescale_value,
                x=x_pos,
                label=f"_{measurement_setup.label}_{av_label}",
                color=measurement_setup.get_color(idx_measurement),
            )
            extra_labels = [av_label]

    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

    ax.set_xticks([x + 0.5 for x in range(n_components)])
    ax.set_xticklabels([f"${latex.term2dqdj(term)}$" for term in detuning_terms] + extra_labels)
    ax.set_xlim([0, n_components])

    # Add Legend ---
    handles, labels = get_handles_labels(setups)
    pannot.make_top_legend(ax, ncol=ncol, frame=False, handles=handles, labels=labels, transposed=transpose_legend)
    return fig


# Helper Functions -------------------------------------------------------------

def get_ylabel(rescale: int = 0, delta: bool = False) -> str:
    """Generate a y-axis label for the plot.

    Args:
        rescale (int, optional): The rescaling factor for the y-axis.
        delta (bool, optional): Indicate if the data is a "detuning shift" e.g. difference between two setups;
                                adds a "Delta" prefix.
    """
    rescale_str = f"10$^{rescale:d}$ " if rescale else ""
    delta_str = r"$\Delta$" if delta else ""
    return f"{delta_str}Q$_{{a,b}}$ [{rescale_str}m$^{{-1}}$]"


def get_measured_detuning_terms(measurements: Sequence[PlotSetup], terms: Sequence[str]) -> list[str]:
    """Get all terms for which at least one measurement has a value.

    Args:
        measurements (Sequence[MeasurementSetup]): The setups to check.
    """
    return [
        term
        for term in terms
        if any(getattr(m.measurement, term) is not None for m in measurements)
    ]


def get_average(measurement_setup: PlotSetup, terms: Sequence[str], method: str = "auto") -> tuple[MeasureValue | float, str]:
    """Calculate the average of the measurements.

    Args:
        measurement_setup (MeasurementSetup): The measurements to average.
        terms (Sequence[str]): The terms to average.
        method (str): The average method to use.
    """
    meas_values = [getattr(measurement_setup.measurement, detuning_component) for detuning_component in terms]
    meas_values = [mv for mv in meas_values if mv is not None]
    is_measurement = all(isinstance(mv, MeasureValue) for mv in meas_values)

    # determine average method
    weighted = "weighted_"
    if method == "auto":
        method = f"{weighted if is_measurement else ''}rms"

    if weighted in method and not all(isinstance(mv, MeasureValue) for mv in meas_values):
        raise ValueError(f"Average {method} requires all measurements to be of type MeasureValue.")

    # calculate average
    match method:
        case "rms":
            if is_measurement:
                av_meas: MeasureValue = MeasureValue.rms(meas_values)
            else:
                av_meas: float = np.sqrt(np.mean([getattr(mv, "value", mv)**2 for mv in meas_values]))
            av_label = "RMS"
        case "weighted_rms":
            av_meas: MeasureValue = MeasureValue.weighted_rms(meas_values)
            av_label = "RMS"
        case "mean":
            if is_measurement:
                av_meas: MeasureValue = MeasureValue.mean(meas_values)
            else:
                av_meas: float  = np.mean(meas_values)
            av_label = "Mean"
        case "weighted_mean":
            av_meas: MeasureValue = MeasureValue.weighted_mean(meas_values)
            av_label = "Mean"

    LOG.debug(f"{measurement_setup.label} RMS: {str(av_meas)}")
    return av_meas, av_label


def plot_value_or_measurement(
    ax: Axes,
    measurement: MeasureValue | float,
    x: float,
    label: str = None,
    color: str = None,
    ) -> Line2D | ErrorbarContainer:
    """Plots an errorbar if the given measurement has an error,
    otherwise a simple point.

    Args:
        ax (Axes): The axes to plot on.
        measurement (MeasureValue | float): The measurement to plot.
        x (float): The x-position of the measurement.
        label (str, optional): Label for the measurement.
        color (str, optional): Color for the measurement.
    """
    if hasattr(measurement, "error") and measurement.error:
        return ax.errorbar(
            x=x,
            y=measurement.value,
            yerr=measurement.error,
            label=label,
            color=color,
            elinewidth=1,  # looks offset otherwise
            ls="",
        )
    return ax.plot(x, getattr(measurement, "value", measurement), label=label, color=color, ls="")


def get_handles_labels(measurements: Sequence[PlotSetup]) -> tuple[list[Line2D], list[str]]:
    """Generate the handles and labels for the legend based on the given measurements.

    Args:
        measurements (Sequence[MeasurementSetup]): The measurements to plot.
    """
    return (
        [
            Line2D([], [], color=measurement_setup.get_color(idx_measurement), label=measurement_setup.label, ls="")
            for idx_measurement, measurement_setup in enumerate(measurements)
        ],
        [measurement_setup.label for measurement_setup in measurements],
    )
