"""
Detuning Plots
--------------

Plotting utilities to compare detuning measurements and simulation results.ßß
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from omc3.plotting.utils import annotations as pannot
from omc3.plotting.utils import colors as pcolors
from omc3.plotting.utils import style as pstyle

from ir_amplitude_detuning.detuning.measurements import Detuning, DetuningMeasurement, MeasureValue
from ir_amplitude_detuning.utilities import latex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.container import ErrorbarContainer
    from matplotlib.lines import Line2D


LOG = logging.getLogger(__name__)


@dataclass
class MeasurementSetup:
    """ Container to define different detuning measurements to plot
    with the plot_measurements function.

    Args:
        label (str): Label for the measurement.
        measurement (DetuningMeasurement | Detuning): Measurement to plot.
        simulation (Detuning, optional): Simulation results to plot, corresponding to the given measurement.
        color (str, optional): Color for the measurement.
    """
    label: str
    measurement: DetuningMeasurement | Detuning
    simulation: Detuning = None
    color: str = None

    def get_color(self, idx: int):
        if self.color is not None:
            return self.color
        return pcolors.get_mpl_color(idx)


def plot_measurements(measurements: Sequence[MeasurementSetup], **kwargs):
    """ Plot multiple measurements on the same plot.

    Args:
        measurements (Sequence[MeasurementSetup]): List of MeasurementSetup objects to plot.

    Keyword Args (optional):
        manual_style (dict): Dictionary of matplotlib style settings.
        is_shift (bool): Indicate if the given data is a "detuning shift" e.g. difference between two setups.
                         This simply adds a "Delta" prefix to the y-axis label, if no label is given.
        ylim (Sequence[float, float]): y-axis limits.
        rescale (int): Exponent of the scaling factor.
                       (e.g. 3 to give data in units of 10^3, which multiplies the data by 10^-3)
                       Default: 3.
        ncol (int): Number of columns in the plot.
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

    rescale: int = kwargs.pop('rescale', 3)
    is_shift: bool = kwargs.pop("is_shift", False)
    ylabel: str = kwargs.pop("ylabel", get_ylabel(rescale=rescale, delta=is_shift))
    ylim: Sequence[float, float] = kwargs.pop("ylim")
    ncol: int = kwargs.pop('ncol', 3)
    average: str | bool = kwargs.pop('average', False)

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}")

    # Prepare Constatns ---
    detuning_terms = get_defined_detuning_terms(measurements)
    n_components = len(detuning_terms) + bool(average)
    n_measurements = len(measurements)
    measurement_width = 1 / (n_measurements + 1)
    bar_width = measurement_width * 0.15
    rescale_value = 10**-rescale

    # Generate Plot ---
    fig, ax = plt.subplots()

    # plot lines
    ax.axhline(0, color="black", lw=1, ls="-", marker="", zorder=-10)  # y = 0
    for idx in range(1, n_components):
        ax.axvline(idx, color="grey", lw=1, ls="--", marker="", zorder=-10)  # split components

    for idx_measurement, measurement_setup in enumerate(measurements):
        for idx_component, detuning_component in enumerate(detuning_terms):
            x_pos = idx_component + (idx_measurement + 1) * measurement_width
            label_prefix = f"_{detuning_component}" if idx_component else ""

            measurement: MeasureValue | float = getattr(measurement_setup.measurement, detuning_component)
            if measurement is not None:
                measurement = measurement * rescale_value
                plot_value_or_measurement(ax,
                    measurement=measurement,
                    x=x_pos,
                    label=f"{label_prefix}{measurement_setup.label}",
                    color=measurement_setup.get_color(idx_measurement)
                )

            if measurement_setup.simulation is not None:
                simulation : Detuning = getattr(measurement_setup.simulation, detuning_component)
                if simulation is not None:
                    simulation = simulation * rescale_value
                    ax.bar(
                        x=x_pos, height=simulation,
                        width=bar_width, bottom=0,
                        label=f"_{detuning_component}{measurement_setup.label}_sim",
                        color=measurement_setup.get_color(idx_measurement),
                        alpha=0.3,
                    )

        av_label = []
        if average:
            meas_values = [getattr(measurement_setup.measurement, detuning_component) for detuning_component in DetuningMeasurement.all_terms()]
            meas_values = [mv for mv in meas_values if mv is not None]
            if average is True:
                average = "weighted_rms"

            match average:
                case "rms":
                    av_meas: MeasureValue = MeasureValue.rms(meas_values) * rescale_value
                    av_label = "RMS"
                case "weighted_rms":
                    av_meas: MeasureValue = MeasureValue.weighted_rms(meas_values) * rescale_value
                    av_label = "RMS"
                case "mean":
                    av_meas: MeasureValue = MeasureValue.mean(meas_values) * rescale_value
                    av_label = "Mean"
                case "weighted_mean":
                    av_meas: MeasureValue = MeasureValue.weighted_mean(meas_values) * rescale_value
                    av_label = "Mean"

            LOG.debug(f"{measurement_setup.label} RMS: {str(av_meas)}")

            x_pos = n_components - 1 + (idx_measurement + 1) * measurement_width
            ax.errorbar(x=x_pos, y=av_meas.value,
                        # yerr=rms.error,
                        label=f"_{measurement_setup.label}{av_label}",
                        color=measurement_setup.get_color(idx_measurement),
                        elinewidth=1,  # looks offset otherwise
                        ls="",  # for the legend only
                        )
            av_label = [av_label]

    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

    ax.set_xticks([x + 0.5 for x in range(n_components)])
    ax.set_xticklabels([f"${latex.term2dqdj(term)}$" for term in detuning_terms] + av_label)
    ax.set_xlim([0, n_components])

    pannot.make_top_legend(ax, ncol=ncol, frame=False)
    return fig


# Helper Functions -------------------------------------------------------------

def get_ylabel(rescale: int = 0, delta: bool = False) -> str:
    """ Generate a y-axis label for the plot.

    Args:
        rescale (int, optional): The rescaling factor for the y-axis.
        delta (bool, optional): Indicate if the data is a "detuning shift" e.g. difference between two setups;
                                adds a "Delta" prefix.
    """
    rescale_str = f"10$^{rescale:d}$ " if rescale else ""
    delta_str = r"$\Delta$" if delta else ""
    return f"{delta_str}Q$_{{a,b}}$ [{rescale_str}m$^{{-1}}$]"


def get_defined_detuning_terms(measurements: Sequence[MeasurementSetup]) -> list[str]:
    """ Get all terms for which at least one measurement has a value.

    Args:
        measurements (Sequence[MeasurementSetup]): The measurements to check.
    """
    terms = list(DetuningMeasurement.all_terms())
    for term in DetuningMeasurement.all_terms():
        if all(getattr(m.measurement, term) is None for m in measurements):
            terms.remove(term)
    return terms


def plot_value_or_measurement(
    ax: Axes,
    measurement: MeasureValue | float,
    x: float,
    label: str = None,
    color: str = None,
    ) -> Line2D | ErrorbarContainer:
    """ Plots an errorbar if the given measurement has an error,
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
    return ax.plot(x=x, y=measurement, label=label, color=color, ls="")
