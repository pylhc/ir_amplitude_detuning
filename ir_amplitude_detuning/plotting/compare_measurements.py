from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from omc3.plotting.utils import annotations as pannot
from omc3.plotting.utils import colors as pcolors
from omc3.plotting.utils import style as pstyle

from ir_amplitude_detuning.detuning.measurements import DetuningMeasurement, MeasureValue
from ir_amplitude_detuning.utilities import latex
from ir_amplitude_detuning.utilities.misc import detuning_short_to_planes

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG = logging.getLogger(__name__)


@dataclass
class MeasurementSetup:
    label: str
    measurement: DetuningMeasurement
    color = None

    def get_color(self, idx: int):
        if self.color is not None:
            return self.color
        return pcolors.get_mpl_color(idx)


def get_ylabel(rescale: int = 0, delta: bool = False) -> str:
    """ Generate a y-axis label for the plot. """
    rescale_str = f"10$^{rescale:d}$ " if rescale else ""
    delta_str = r"$\Delta$" if delta else ""
    return f"{delta_str}Q$_{{a,b}}$ [{rescale_str}m$^{{-1}}$]"


def get_components(measurements: Sequence[MeasurementSetup]):
    """ Get all terms for which at least one measurement has a value. """
    fields = list(DetuningMeasurement.fieldnames())
    for field in DetuningMeasurement.fieldnames():
        if all(getattr(m.measurement, field) is None for m in measurements):
            fields.remove(field)
    return fields


def plot_measurements(measurements: Sequence[MeasurementSetup], **kwargs):
    """ Plot multiple measurements on the same plot.

    Args:
        measurements (Sequence[MeasurementSetup]): List of MeasurementSetup objects to plot.

    Keyword Args (optional):
        manual_style (dict): Dictionary of matplotlib style settings.
        ylim (Sequence[float, float]): y-axis limits.
        rescale (int): Number of digits to rescale to.
        ncol (int): Number of columns in the plot.
        add_rms (bool): Add rms values to the plot.
    """

    # Set Style ---
    pstyle.set_style(kwargs.get("style", "standard"), kwargs.get("manual_style"))

    rescale: int = kwargs.get('rescale', 3)
    rescale_value = 10**-rescale
    ylabel: str = kwargs.get("ylabel", get_ylabel(rescale=rescale))
    ylim: Sequence[float, float] = kwargs.get("ylim")
    ncol: int = kwargs.get('ncol', 3)
    add_rms: bool = kwargs.get('add_rms', False)

    # Prepare Constatns ---
    field_components = get_components(measurements)
    n_components = len(field_components) + add_rms
    n_measurements = len(measurements)
    measurement_width = 1 / (n_measurements + 1)

    # Generate Plot ---
    fig, ax = plt.subplots()

    # plot lines
    ax.axhline(0, color="black", lw=1, ls="-", marker="", zorder=-10)  # y = 0
    for idx in range(1, n_components):
        ax.axvline(idx, color="grey", lw=1, ls="--", marker="", zorder=-10)  # split components

    for idx_measurement, measurement_setup in enumerate(measurements):
        for idx_component, detuning_component in enumerate(field_components):
            x_pos = idx_component + (idx_measurement + 1) * measurement_width
            pre_label = f"_{detuning_component}" if idx_component else ""
            measurement: MeasureValue = getattr(measurement_setup.measurement, detuning_component)
            if measurement is None:
                continue

            measurement = measurement * rescale_value
            ax.errorbar(x=x_pos, y=measurement.value,
                        yerr=measurement.error,
                        label=f"{pre_label}{measurement_setup.label}",
                        color=measurement_setup.get_color(idx_measurement),
                        elinewidth=1,  # looks offset otherwise
                        ls="",  # for the legend only
                        )

        if add_rms:
            meas_values = [getattr(measurement_setup.measurement, detuning_component) for detuning_component in DetuningMeasurement.fieldnames()]
            meas_values = [abs(mv) for mv in meas_values if mv is not None]
            # rms = MeasureValue.rms(meas_values) * rescale_value
            rms = MeasureValue.weighted_rms(meas_values) * rescale_value
            # mean = MeasureValue.weigthted_mean(meas_values) * rescale_value
            LOG.info(f"{measurement_setup.label} RMS: {str(rms)}")

            x_pos = n_components - 1 + (idx_measurement + 1) * measurement_width
            ax.errorbar(x=x_pos, y=rms.value,
                        # yerr=rms.error,
                        label=f"_{measurement_setup.label}rms",
                        color=measurement_setup.get_color(idx_measurement),
                        elinewidth=1,  # looks offset otherwise
                        ls="",  # for the legend only
                        )


    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

    ax.set_xticks([x + 0.5 for x in range(n_components)])
    ax.set_xticklabels([f"${latex.dqd2j(*detuning_short_to_planes(fc))}$" for fc in field_components] + (["RMS"] if add_rms else []))
    ax.set_xlim([0, n_components])

    pannot.make_top_legend(ax, ncol=ncol, frame=False)
    return fig
