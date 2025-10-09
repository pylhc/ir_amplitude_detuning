from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from omc3.plotting.utils import annotations as pannot
from omc3.plotting.utils import colors as pcolors
from omc3.plotting.utils import style as pstyle
from omc3.plotting.utils.lines import MarkerList

from ir_amplitude_detuning.detuning.measurements import FirstOrderTerm, SecondOrderTerm
from ir_amplitude_detuning.simulation.common import get_detuning_from_ptc_output
from ir_amplitude_detuning.utilities import latex
from ir_amplitude_detuning.utilities.classes_accelerator import FieldComponent
from ir_amplitude_detuning.utilities.constants import AMPDET_CALC_ID, AMPDET_ID, NOMINAL_ID
from ir_amplitude_detuning.utilities.latex import XLABEL_MAP, YLABEL_MAP
from ir_amplitude_detuning.utilities.misc import to_loop

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

LOG = logging.getLogger(__name__)  # setup in main()


SETTINGS_ID_MAP = {"corrected": "mcx_b1b4"}


BEAMS = 1, 4

ALL_IDS = list(XLABEL_MAP.keys())

FIELDS: str = "FIELDS"
IP: str = "IP"

COLOR_MARKER_MAP = dict(zip(
        XLABEL_MAP.keys(),
        [
            # b5b6
            (pcolors.change_color_brightness(pcolors.get_mpl_color(0), 1.0), "o"),
            (pcolors.change_color_brightness(pcolors.get_mpl_color(1), 1.0), "o"),
            (pcolors.change_color_brightness(pcolors.get_mpl_color(2), 1.0), "o"),
            # Nominal
            ("k", "o"),
        ],
    ))

def get_scaling(term: str) -> float:
    exponent = {1: 3, 2: 12}[int(term[1]) + int(term[2])]
    scaling = 10**-exponent
    return exponent, scaling


def load_calculated_detuning(folder: Path, beam: int, id_: str) -> tfs.TfsDataFrame:
    return load_detuning(folder, beam, id_, AMPDET_CALC_ID)


def load_simulated_detuning(folder: Path, beam: int, id_: str) -> pd.DataFrame:
    df = load_detuning(folder, beam, id_, AMPDET_ID)
    series = get_detuning_from_ptc_output(df, beam=None, log=False, terms=list(FirstOrderTerm) + list(SecondOrderTerm))
    return series.to_frame().T


def load_detuning(folder: Path, beam: int, id_: str, type_: str):
    glob = f"{type_}.*.b{beam}.{id_}.tfs"
    for filename in folder.glob(glob):
        return tfs.read(filename)
    raise FileNotFoundError(f"No file matching '{glob}' in {folder}.")



def get_calc_detuning_for_ip(folder: Path, beam: int, id_: str, ip: str) -> pd.DataFrame:
    df = load_calculated_detuning(folder, beam, id_)
    ip_mask = df[IP] == ip
    return df.loc[ip_mask, :].set_index(FIELDS, drop=True)


def get_calc_detuning_for_field(folder: Path, beam: int, id_: str, fields: Iterable[FieldComponent] | FieldComponent
    ) -> pd.DataFrame:
    df = load_calculated_detuning(folder, beam, id_)

    if not isinstance(fields, FieldComponent):
        fields = ''.join(sorted(fields))

    fields_mask = df[FIELDS] == fields
    return df.loc[fields_mask, :].set_index(IP, drop=True)


def subtract_detuning(detuning_a: dict[int, pd.DataFrame], detuning_b: dict[int, pd.DataFrame]):
    return {beam: detuning_a[beam] - detuning_b[beam] for beam in detuning_a}


def get_all_detuning_data_for_ip(folder: Path, ids: Iterable[str], beams: Iterable[int], ip: str
    ) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, dict[int, pd.DataFrame]]]:
    """ Load and sort the detuning data for a given IP."""
    ptc_data = {id_: {beam: load_simulated_detuning(folder, beam, id_) for beam in beams} for id_ in ids}
    nominal_data = {beam: load_simulated_detuning(folder, beam, NOMINAL_ID) for beam in beams}

    calculated_data = {id_: {beam: get_calc_detuning_for_ip(folder, beam, id_, ip) for beam in beams} for id_ in ids}

    for id_ in ids:
        ptc_data[id_] = subtract_detuning(ptc_data[id_], nominal_data)
        calculated_data[id_] = subtract_detuning(calculated_data[id_], nominal_data)

    return ptc_data, calculated_data


def get_all_detuning_data_for_field(
    folder: Path,
    ids: Iterable[str],
    beams: Iterable[int],
    fields: Iterable[FieldComponent] | FieldComponent
    ) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, dict[int, pd.DataFrame]]]:
    """ Load and sort the detuning data for a given set of fields."""
    ptc_data = {id_: {beam: load_simulated_detuning(folder, beam, id_) for beam in beams} for id_ in ids}
    nominal_data = {beam: load_simulated_detuning(folder, beam, NOMINAL_ID) for beam in beams}

    calculated_data = {id_: {beam: get_calc_detuning_for_field(folder, beam, id_, fields) for beam in beams} for id_ in ids}

    for id_ in ids:
        ptc_data[id_] = subtract_detuning(ptc_data[id_], nominal_data)
        calculated_data[id_] = subtract_detuning(calculated_data[id_], nominal_data)

    return ptc_data, calculated_data


def get_color_for_field(field: FieldComponent):
    match field:
        case FieldComponent.b5:
            return pcolors.get_mpl_color(4)
        case FieldComponent.b6:
            return pcolors.get_mpl_color(2)
        case FieldComponent.b4:
            return pcolors.get_mpl_color(1)



def plot_detuning_by_fields(
    folder: Path,
    ids: Iterable[str] | dict[str, str],
    fields: Iterable[FieldComponent],
    ips: str,
    measurement=None,
    **kwargs):

    # STYLE -------
    size = kwargs.pop('size', None)
    if not size:
        fig_width = 0.8 * len(ids)
        min_width = 4.8 + 2*bool(measurement)
        if fig_width < min_width:
            fig_width = min_width
        size = [fig_width, 4.80]

    manual = {
        "figure.figsize": size,
        "markers.fillstyle": "none",
        "grid.alpha": 0,
        "savefig.format": "pdf",
    }

    tickrotation: float = kwargs.pop('tickrotation', 45)
    ylims: tuple[float, float] = kwargs.pop('ylims', None)
    beams: tuple[int, ...] = kwargs.pop('beams', BEAMS)
    plot_styles: Iterable[Path | str] = kwargs.pop('plot_styles', 'standard')

    manual.update(kwargs)
    pstyle.set_style(plot_styles, manual)

    # Data
    data, calculated_data = get_all_detuning_data_for_ip(folder, ids, beams, ips)

    bar_width = 1/(len(beams) + 1)
    stack_width = 0.15 * bar_width
    xlim = [- bar_width / 2, (len(ids) - 1) + bar_width * (len(beams) + 0.5)]
    figs = {}

    all_terms = list(FirstOrderTerm) + list(SecondOrderTerm)
    for term in all_terms:
        scale_exponent, scaling = get_scaling(term)
        fig, ax = plt.subplots()
        figs[term] = fig

        # plot zero line
        ax.axhline(0, color="black", lw=1, ls="-", marker="", zorder=0)

        # plot separation lines
        for idx, id_ in enumerate(ids):
            if not idx:
                continue
            ax.axvline(idx-bar_width/2, ls="--", lw=1, color="black", alpha=0.2, marker="", zorder=-5)

        for idx_beam, beam in enumerate(beams):
            color = pcolors.get_mpl_color(idx_beam)

            # plot measurement
            meas_val, meas_err = None, None
            if measurement:
                try:
                    meas_val, meas_err = measurement[beam][term]
                except KeyError:
                    pass
                else:
                    ax.axhspan(ymin=meas_val-meas_err, ymax=meas_val+meas_err, color=color, alpha=0.3)

            for idx_id, id_name in enumerate(ids):
                x_pos = idx_id + bar_width * (idx_beam + 0.5)

                # Plot calculated Data ---
                loop_fields: list[str] = to_loop(sorted(fields))
                field_strings: list[str] = [''.join(map(str, fs)) for fs in loop_fields]

                # Individual contributions per field component
                for field in field_strings[1:]:
                    y_pos = calculated_data[id_name][beam].loc[field, term] * scaling
                    ax.bar(
                        x_pos, y_pos, stack_width, bottom=0,
                        label=f"_{beam}.{id_name}.{field}",
                        color=get_color_for_field(field), alpha=0.3
                    )

                # Total contribution
                y_pos = calculated_data[id_name][beam].loc[field_strings[0], term] * scaling
                ax.plot(x_pos, y_pos, ls="none", marker="x", color=color, alpha=0.5)

                # Plot PTC Data
                y_pos = data[id_name][beam][term] * scaling
                ax.plot(x_pos, y_pos, marker='o', color=color, label=f'_b{beam}.{id_name}')

        ax.set_ylabel(latex.ylabel_from_detuning_term(term, scale_exponent))
        ax.set_xticks(np.arange(len(ids)) + (bar_width * len(beams)) / 2)

        ax.set_xticklabels(ids.values(), rotation=tickrotation)
        ax.set_xlim(xlim)
        ax.set_ylim(ylims)

        fig.canvas.manager.set_window_title(f"ampdet.{term}")
    return figs


def plot_detuning_ips(folder, ids, fields='b6', labels=None, measurement=None, output_id='', **kwargs):
    # STYLE -------
    size = kwargs.pop('size', None)
    delta = kwargs.pop('delta', False)
    if not size:
        fig_width = 0.8 * len(ids)
        min_width = 4.8 + 2*bool(measurement)
        if fig_width < min_width:
            fig_width = min_width
        size = [fig_width, 4.80]

    manual = {
        "figure.figsize": size,
        "legend.columnspacing": 1,
        "legend.handlelength": 1.5,
        "markers.fillstyle": "none",
        "grid.alpha": 0,
        "savefig.format": "pdf",
        "ytick.minor.visible": True,
    }

    tickrotation: np.numeric = kwargs.pop('tickrotation', 45)
    ylims: np.numeric = kwargs.pop('ylims', None)
    beams: np.numeric = kwargs.pop('beams', BEAMS)
    alternative: str = kwargs.pop('alternative', 'normal')
    plot_styles: Iterable[Path | str] = kwargs.pop('plot_styles', 'standard')

    manual.update(kwargs)
    pstyle.set_style(plot_styles, manual)

    # Data
    data, calculated_data = get_all_detuning_data(folder, ids, beams, for_ips=True)

    bar_width = 1/(len(beams) + 1)
    stack_width = 0.15 * bar_width
    xlim = [- bar_width / 2, (len(ids) - 1) + bar_width * (len(beams) + 0.5)]
    figs = dict.fromkeys(YLABEL_MAP.keys())

    color_ip1 = pcolors.get_mpl_color(4)
    color_ip5 = pcolors.get_mpl_color(2)

    for term in YLABEL_MAP:
        order = sum(int(c) for c in term[1:])
        scale = SCALE[order]
        fig, ax = plt.subplots()
        figs[term] = fig

        # plot zero line
        ax.axhline(0, color="black", lw=1, ls="-", marker="", zorder=0)

        # plot separation lines
        for idx, id_ in enumerate(ids):
            if not idx:
                continue
            ax.axvline(idx-bar_width/2, ls="--", lw=1, color="black", alpha=0.2, marker="", zorder=-5)

        for idx_beam, beam in enumerate(beams):
            color = pcolors.get_mpl_color(idx_beam)

            # plot measurement
            meas_val, meas_err = None, None
            if measurement:
                try:
                    meas_val, meas_err = measurement[beam][term].to_list()
                except KeyError:
                    pass
                else:
                    ax.axhspan(ymin=(meas_val-meas_err)*scale, ymax=(meas_val+meas_err)*scale, color=color, alpha=0.3)

            for idx_id, id_name in enumerate(ids):
                x_pos = idx_id + bar_width * (idx_beam + 0.5)

                # Plot calculated Data:
                calc = {f: calculated_data[beam][id_name].loc[f, term]*scale for f in ("5", "1", "all")}
                if alternative == 'normal':
                    ax.bar(x_pos, calc["5"], stack_width, bottom=0, label=f"_{beam}.{id_name}.ip5", color=color_ip5, alpha=0.3)
                    ax.bar(x_pos, calc["1"], stack_width, bottom=calc["5"], label=f"_{beam}.{id_name}.ip1", color=color_ip1, alpha=0.3)
                    ax.plot(x_pos, calc["all"], ls="none", marker="_", color=color, alpha=0.3)
                elif alternative == 'separate':
                    ax.bar(x_pos, calc["5"], stack_width, bottom=0, label=f"_{beam}.{id_name}.ip5", color=color_ip5, alpha=0.3)
                    ax.bar(x_pos, calc["1"], stack_width, bottom=0, label=f"_{beam}.{id_name}.ip1", color=color_ip1, alpha=0.3)
                    ax.plot(x_pos, calc["all"], ls="none", marker="x", color=color, alpha=0.5)


                y_pos = data[beam][id_name][term] * scale
                ax.plot(x_pos, y_pos, marker='o', color=color, label=f'_b{beam}.{id_name}')


        if delta:
            ax.set_ylabel(fr"$\Delta {YLABEL_MAP[term][1:]}")
            ax.tick_params(axis="x", pad=12)  # to be able to remove cross-term ticklabels in latex
        else:
            ax.set_ylabel(YLABEL_MAP[term])

        ax.set_xticks(np.arange(len(ids)) + (bar_width * len(beams)) / 2)
        if labels is None:
            ax.set_xticklabels([XLABEL_MAP[c] for c in ids], rotation=tickrotation)
        else:
            ax.set_xticklabels(labels, rotation=tickrotation)
        ax.set_xlim(xlim)
        ax.set_ylim(ylims[order])

        empty = Line2D([0], [0], ls='none', marker='', label='')
        b1_color = pcolors.get_mpl_color(0)
        b2_color = pcolors.get_mpl_color(1)

        if alternative == 'normal':
            pannot.make_top_legend(ax, ncol=3 + bool(measurement), frame=False,
                                   handles=[
                                               Line2D([0], [0], marker='', ls='none', color=b1_color, label='Beam 1'),
                                               Line2D([0], [0], marker='', ls='none', color=b2_color, label='Beam 2'),
                                               Line2D([0], [0], marker='o', ls='none', color=b1_color, label='PTC'),
                                               Line2D([0], [0], marker='o', ls='none', color=b2_color, label='PTC'),
                                           ]
                                           + ([Patch(facecolor=b1_color, edgecolor=b1_color, alpha=0.3, label='Measured'), Patch(facecolor=b2_color, edgecolor=b2_color, alpha=0.3, label='Measured')] if measurement else [])
                                           + [Patch(facecolor=color_ip5, edgecolor=color_ip5, label='IP5', alpha=0.3), Patch(facecolor=color_ip1, edgecolor=color_ip1, alpha=0.3, label='IP1')],
                                   )
        elif alternative == 'separate':
            pannot.make_top_legend(ax, ncol=4 + bool(measurement), frame=False,
                                   handles=[
                                               Line2D([0], [0], marker='', ls='none', color=b1_color, label='Beam 1'),
                                               Line2D([0], [0], marker='', ls='none', color=b2_color, label='Beam 2'),
                                               Line2D([0], [0], marker='o', ls='none', color=b1_color, label='PTC'),
                                               Line2D([0], [0], marker='o', ls='none', color=b2_color, label='PTC'),
                                               Line2D([0], [0], marker='x', ls='none', color=b1_color, label='Eq.'),
                                               Line2D([0], [0], marker='x', ls='none', color=b2_color, label='Eq.'),
                                           ]
                                           + ([Patch(facecolor=b1_color, edgecolor=b1_color, alpha=0.3, label='Measured'), Patch(facecolor=b2_color, edgecolor=b2_color, alpha=0.3, label='Measured')] if measurement else [])
                                           + [Patch(facecolor=color_ip5, edgecolor=color_ip5, label='IP5', alpha=0.3), Patch(facecolor=color_ip1, edgecolor=color_ip1, alpha=0.3, label='IP1')],
                                   )

        fig.canvas.manager.set_window_title(f"ampdet.ips_{term}{output_id or ''}")
        if output_id is not None:
            fig.savefig(folder / f"plot.{fig.canvas.get_default_filename()}")
    # plt.show()
    return figs
