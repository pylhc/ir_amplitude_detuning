from __future__ import annotations

from collections.abc import Sequence
import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import tfs
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from omc3.plotting.utils import annotations as pannot
from omc3.plotting.utils import colors as pcolors
from omc3.plotting.utils import style as pstyle
from omc3.plotting.utils.lines import MarkerList

from ir_dodecapole_corrections.simulation.lhc_simulation import get_detuning_from_ptc_output
from ir_dodecapole_corrections.utilities.latex import XLABEL_MAP, YLABEL_MAP

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

LOG = logging.getLogger(__name__)  # setup in main()

AMPDET_FILE_PATTERN = "ampdet_calc.lhc.b{beam}.{id_}.tfs"
SETTINNGS_FILE_PATTERN = "settings.lhc.b{beam}.{id_}.madx"

SETTINGS_ID_MAP = {"corrected": "mcx_b1b4"}

SCALE = {1: 1e-3, 2: 1e-12}  # first order 10^3, second order 10^12

BEAMS = 1, 4

ALL_IDS = list(XLABEL_MAP.keys())

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


def get_left_right_keys(dict_):
    def get_key(side):
        for k in dict_:
            if re.match(fr".+\d+\.{side}\d+", k, re.IGNORECASE):
                return k
        raise KeyError(f"No key matching '*.{side}*' found in {list(dict_.keys())}.")
    return get_key('l'), get_key('r')


def get_detuning(folder, beam, id_):
    filename = f"ampdet.lhc.b{beam}.{id_}.tfs"
    try:
        file = list(folder.glob(f"**/{filename}"))[0]
    except IndexError:
        LOG.error(f"No file matching '{filename}' in {folder}")
        return dict.fromkeys(YLABEL_MAP.keys(), np.nan)
    df = tfs.read(file)
    return get_detuning_from_ptc_output(df, beam=None, log=False, terms=YLABEL_MAP.keys())


def get_calc_detuning(folder: Path, beam: int, id_: str, ip: str = "all"):
    ampdet_filename = AMPDET_FILE_PATTERN.format(beam=beam, id_=id_)
    try:
        file = list(folder.glob(f"**/{ampdet_filename}"))[0]
    except IndexError:
        raise FileNotFoundError(f"No file matching '{ampdet_filename}' in {folder}.")

    df = tfs.read(file)
    ip_mask = df["IP"] == ip
    return df.loc[ip_mask, :].set_index("FIELDS", drop=True)


def get_calc_detuning_ips(
    folder: Path,
    beam: int,
    id_: str,
    ips: Sequence[str] = ('5', '1', 'all'),
    fields: str ='b6'
):
    ampdet_filename = AMPDET_FILE_PATTERN.format(beam=beam, id_=id_)
    try:
        file = list(folder.glob(f"**/{ampdet_filename}"))[0]
    except IndexError:
        raise FileNotFoundError(f"No file matching '{ampdet_filename}' in {folder}.")

    df = tfs.read(file)
    fields_mask = df["FIELDS"] == fields
    return df.loc[fields_mask, :].set_index("IP", drop=True).loc[ips, :]


def get_corrector_strengths(folder: Path, beam: int, id_: str, corrector_mask: str):
    id_ = SETTINGS_ID_MAP.get(id_, id_)
    settings_filename = SETTINNGS_FILE_PATTERN.format(beam=beam, id_=id_)
    try:
        file = list(folder.glob(f"**/{settings_filename}"))[0]
    except IndexError:
        raise FileNotFoundError(f"No file matching '{settings_filename}' in {folder}.")

    txt = file.read_text()
    matches: list[str] = re.findall(fr"({corrector_mask})\s*:=\s*([0-9+-.e]+)\s*", txt, flags=re.IGNORECASE)
    if not matches:
        raise AttributeError(f"No matching corrector '{corrector_mask}' values found.")
    return {m[0].lower(): float(m[1]) for m in matches}


def get_all_detuning_data(folder, ids, beams=BEAMS, for_ips=False):
    data = {beam: {id_: get_detuning(folder, beam, id_) for id_ in ids} for beam in beams}
    nominal_data = {beam: get_detuning(folder, beam, 'nominal') for beam in beams}
    if for_ips:
        calculated_data = {beam: {id_: get_calc_detuning_ips(folder, beam, id_) for id_ in ids} for beam in beams}
    else:
        calculated_data = {beam: {id_: get_calc_detuning(folder, beam, id_) for id_ in ids} for beam in beams}

    # Subtract Nominal Value:
    for beam in data:
        for id_ in data[beam]:
            if id_ == "nominal":
                continue
            for term in data[beam][id_]:
                data[beam][id_][term] = data[beam][id_][term] - nominal_data[beam][term]
    return data, calculated_data


def plot_correctors(folder, ids, labels, output_id='', corrector_pattern=r'kctx3\.[lr]5', order="6", beam=1, **kwargs):
    # STYLE -------
    size = kwargs.pop('size', None)
    if not size:
        size = [7.68, 5.12]
    manual = {
        "figure.figsize": size,
        "markers.fillstyle": "none",
        "grid.alpha": 0,
        "savefig.format": "pdf",
    }

    lim: np.numeric = kwargs.pop('lim', None)
    ncol: int = kwargs.pop('ncol', 2)
    plot_styles: Iterable[Path | str] = kwargs.pop('plot_styles', 'standard')

    manual.update(kwargs)
    pstyle.set_style(plot_styles, manual)
    mlist = MarkerList()

    data = {id_: get_corrector_strengths(folder, beam, id_, corrector_pattern) for id_ in ids}

    fig, ax = plt.subplots()

    # plot zero lines
    ax.axhline(0, color="black", ls="--", marker="", zorder=-10, alpha=0.1)
    ax.axvline(0, color="black", ls="--", marker="", zorder=-10, alpha=0.1)

    for idx, ((id_, values), label) in enumerate(zip(data.items(), labels)):
        color, marker = pcolors.get_mpl_color(idx),  mlist.get_marker(idx)
        if len(values) == 2:
            left, right = get_left_right_keys(values)
            ax.plot(values[left], values[right], ls='none',  c=color, marker=marker, label=label)
        else:
            for ip_idx, ip_values in enumerate([dict(list(values.items())[2*i:2*i+2]) for i in range(len(values)//2)]):
                ip_name = list(ip_values.keys())[0][-1]
                left, right = get_left_right_keys(ip_values)
                # num_marker_size = mpl.rcParams['lines.markersize'] * 0.9
                # ax.plot(values[left], values[right], ls='none',  c=color, marker=marker, label=f"_{label}{ip_name}" if ip_idx else label)
                # ax.plot(values[left], values[right], ls='none',  c=color, marker=f"${ip_name}$", markersize=num_marker_size, label=f"_{label}{ip_name}num")
                ax.plot(values[left], values[right], ls='none',  c=color, marker=f"${ip_name}$", label=f"_{label}{ip_name}" if ip_idx else label)


    ax.set_xlabel(f"$K_{{{order}}}L$ Left")
    ax.set_ylabel(f"$K_{{{order}}}L$ Right")
    ax.set_aspect("equal")

    if not lim:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # lim = [min(xlim + ylim), max(xlim + ylim)]
        lim = max(np.abs(xlim + ylim))
        lim = [-lim, lim]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    pannot.make_top_legend(ax, ncol=ncol, frame=False)

    fig.canvas.manager.set_window_title(f"settings{output_id or ''}")
    if output_id is not None:
        fig.savefig(folder / f"plot.{fig.canvas.get_default_filename()}")
    # plt.show()
    return fig


def plot_detuning(folder, ids, labels=None, measurement=None, output_id='', fields="b5b6", **kwargs):
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

    tickrotation: np.numeric = kwargs.pop('tickrotation', 45)
    ylims: np.numeric = kwargs.pop('ylims', None)
    beams: np.numeric = kwargs.pop('beams', BEAMS)
    alternative: str = kwargs.pop('alternative', 'normal')
    plot_styles: Iterable[Path | str] = kwargs.pop('plot_styles', 'standard')

    manual.update(kwargs)
    pstyle.set_style(plot_styles, manual)

    # Data
    data, calculated_data = get_all_detuning_data(folder, ids, beams)

    bar_width = 1/(len(beams) + 1)
    stack_width = 0.15 * bar_width
    xlim = [- bar_width / 2, (len(ids) - 1) + bar_width * (len(beams) + 0.5)]
    figs = dict.fromkeys(YLABEL_MAP.keys())

    # colorb5 = pcolors.get_mpl_color(2)
    # colorb6 = pcolors.get_mpl_color(3)
    # hacked for IPAC plot:
    colorb5 = pcolors.get_mpl_color(4)
    colorb6 = pcolors.get_mpl_color(2)

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
                    meas_val, meas_err = measurement[beam][term]
                except KeyError:
                    pass
                else:
                    ax.axhspan(ymin=meas_val-meas_err, ymax=meas_val+meas_err, color=color, alpha=0.3)

            for idx_id, id_name in enumerate(ids):
                x_pos = idx_id + bar_width * (idx_beam + 0.5)

                # Plot calculated Data:
                calc = {f: calculated_data[beam][id_name].loc[f, term]*scale for f in ("b5", "b6", "b5b6")}
                if alternative == 'normal':
                    ax.bar(x_pos, calc["b5"], stack_width, bottom=0, label=f"_{beam}.{id_name}.b5", color=colorb5, alpha=0.3)
                    ax.bar(x_pos, calc["b6"], stack_width, bottom=calc["b5"], label=f"_{beam}.{id_name}.b6", color=colorb6, alpha=0.3)
                    ax.plot(x_pos, calc["b5b6"], ls="none", marker="_", color=color, alpha=0.3)
                elif alternative == 'separate':
                    ax.bar(x_pos, calc["b5"], stack_width, bottom=0, label=f"_{beam}.{id_name}.b5", color=colorb5, alpha=0.3)
                    ax.bar(x_pos, calc["b6"], stack_width, bottom=0, label=f"_{beam}.{id_name}.b6", color=colorb6, alpha=0.3)
                    ax.plot(x_pos, calc["b5b6"], ls="none", marker="x", color=color, alpha=0.5)

                y_pos = data[beam][id_name][term] * scale
                ax.plot(x_pos, y_pos, marker='o', color=color, label=f'_b{beam}.{id_name}')

        ax.set_ylabel(YLABEL_MAP[term])
        ax.set_xticks(np.arange(len(ids)) + (bar_width * len(beams)) / 2)
        if labels is None:
            ax.set_xticklabels([XLABEL_MAP[c] for c in ids], rotation=tickrotation)
        else:
            ax.set_xticklabels(labels, rotation=tickrotation)
        ax.set_xlim(xlim)
        ax.set_ylim(ylims[order])

        b1_color = pcolors.get_mpl_color(0)
        b2_color = pcolors.get_mpl_color(1)
        # hacked for iPAC plot
        # pannot.make_top_legend(ax, ncol=2 + bool(calculated_data), frame=False,
        #                        handles=[
        #                                 Line2D([0], [0], marker='o', ls='none', color=b1_color, label='Beam 1 PTC'),
        #                                 Line2D([0], [0], marker='o', ls='none', color=b2_color, label='Beam 2 PTC'),]
        #                                + ([Patch(facecolor=b1_color, edgecolor=b1_color, alpha=0.3, label='Measurement'), Patch(facecolor=b2_color, edgecolor=b2_color, alpha=0.3, label='Measurement')] if measurement else [])
        #                                + ([p for p in [Patch(facecolor=colorb5, edgecolor=colorb5, label='$K_5L$', alpha=0.3) if "b5" in fields else None,
        #                                                Patch(facecolor=colorb6, edgecolor=colorb6, alpha=0.3, label='$K_6L$') if "b6" in fields else None] if p is not None]),
        #                        # handles=[Patch(facecolor=colorb6, edgecolor=colorb6, alpha=0.3, label='$K_6L$')],
        # )
        fig.tight_layout()

        fig.canvas.manager.set_window_title(f"ampdet.{term}{output_id or ''}")
        if output_id is not None:
            fig.savefig(folder / f"plot.{fig.canvas.get_default_filename()}")
    # plt.show()
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


def plot_detuning_folders(folders, labels, id_, measurement=None, output_id='', **kwargs):
    # STYLE -------
    size = kwargs.pop('size', None)
    if not size:
        fig_width = 0.8 * len(folders)
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

    tickrotation: np.numeric = kwargs.pop('tickrotation', 45)
    ylims: np.numeric = kwargs.pop('ylims', None)
    beams: np.numeric = kwargs.pop('beams', BEAMS)
    plot_styles: Iterable[Path | str] = kwargs.pop('plot_styles', 'standard')

    manual.update(kwargs)
    pstyle.set_style(plot_styles, manual)

    # Data
    data = {beam: {folder: get_detuning(folder, beam, id_) for folder in folders} for beam in beams}
    nominal_data = {beam: get_detuning(folders[0], beam, 'nominal') for beam in beams}
    calculated_data = {beam: {folder: get_calc_detuning(folder, beam, id_) for folder in folders} for beam in beams}

    # Subtract Nominal Value:
    for beam in data:
        for folder in data[beam]:
            if id_ == "nominal":
                continue
            for term in data[beam][folder]:
                data[beam][folder][term] = data[beam][folder][term] - nominal_data[beam][term]

    bar_width = 1/(len(beams) + 1)
    stack_width = 0.15 * bar_width
    xlim = [- bar_width / 2, (len(folders) - 1) + bar_width * (len(beams) + 0.5)]
    figs = dict.fromkeys(YLABEL_MAP.keys())

    colorb5 = pcolors.get_mpl_color(2)
    colorb6 = pcolors.get_mpl_color(3)

    for term in YLABEL_MAP:
        order = sum(int(c) for c in term[1:])
        scale = SCALE[order]
        fig, ax = plt.subplots()
        figs[term] = fig

        # plot zero line
        ax.axhline(0, color="black", lw=1, ls="-", marker="", zorder=0)

        # plot separation lines
        for idx, folder in enumerate(folders):
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

            for idx_folder, (folder, label) in enumerate(zip(folders, labels)):
                x_pos = idx_folder + bar_width * (idx_beam + 0.5)

                # Plot calculated Data:
                calc = {f: calculated_data[beam][folder].loc[f, term]*scale for f in ("b5", "b6", "b5b6")}
                ax.bar(x_pos, calc["b5"], stack_width, bottom=0, label=f"_{beam}.{label}.b5", color=colorb5, alpha=0.3)
                ax.bar(x_pos, calc["b6"], stack_width, bottom=calc["b5"], label=f"_{beam}.{label}.b6", color=colorb6, alpha=0.3)
                ax.plot(x_pos, calc["b5b6"], ls="none", marker="_", color=color, alpha=0.3)

                y_pos = data[beam][folder][term] * scale
                ax.plot(x_pos, y_pos, marker='o', color=color, label=f'_b{beam}.{label}')

        ax.set_ylabel(YLABEL_MAP[term])
        ax.set_xticks(np.arange(len(labels)) + (bar_width * len(beams)) / 2)
        ax.set_xticklabels(list(labels), rotation=tickrotation)
        ax.set_xlim(xlim)
        ax.set_ylim(ylims[order])

        empty = Line2D([0], [0], ls='none', marker='', label='')
        pannot.make_top_legend(ax, ncol=2 + bool(calculated_data), frame=False,
                               handles=[Line2D([0], [0], marker='o', ls='none', color=pcolors.get_mpl_color(0), label='Beam 1'),
                                        Line2D([0], [0], marker='o', ls='none', color=pcolors.get_mpl_color(1), label='Beam 2'),]
                                       + ([Patch(facecolor='grey', edgecolor='grey', label='Measurement'), empty] if measurement else [])
                                       + [Patch(facecolor=colorb5, edgecolor=colorb5, label='$K_5L$', alpha=0.3), Patch(facecolor=colorb6, edgecolor=colorb6, alpha=0.3, label='$K_6L$')],
                               )
        fig.tight_layout()

        fig.canvas.manager.set_window_title(f"ampdet.{term}{f'.{output_id}' or ''}")
        if output_id is not None:
            fig.savefig(f"plots/plot.{fig.canvas.get_default_filename()}")
    # plt.show()
    return figs
