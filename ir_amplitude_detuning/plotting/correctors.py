"""
Plot Corrector Strengths
------------------------

Plots the calculated corrector strengths.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import tfs
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from omc3.plotting.utils import annotations as pannot
from omc3.plotting.utils import colors as pcolors
from omc3.plotting.utils import style as pstyle

from ir_amplitude_detuning.utilities.constants import CIRCUIT, ERR, KNL, SETTINGS_ID

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import pandas as pd

    from ir_amplitude_detuning.utilities.correctors import FieldComponent

LOG = logging.getLogger(__name__)


def plot_correctors(
    folder: Path,
    beam: int,
    ids: dict[str, str] | Iterable[str],
    field: FieldComponent,
    corrector_pattern: str = ".*",
    **kwargs
    ):
    """Plot the corrector strengths for a given beam and corrector pattern.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number (to select the right output files).
        ids (dict[str, str] | Iterable[str]): The ids to plot (from the targets).
                                              Use a dictionary to specify labels.
        field (str): The field of the used correctors, e.g. "a6" for K6SL.
        corrector_pattern (str, optional): The corrector pattern to match,
                                 in case you don't want all correctors to be plotted.

    Keyword Args:
        style (str): The plot style to use.
        manual_style (dict): Dictionary of matplotlib style settings.
        lim (float): The y-axis limit.
        rescale (int): Exponent of the scaling factor.
            (e.g. 3 to give data in units of 10^3, which multiplies the data by 10^-3)
            Default: 3.
        ncol (int): The number of columns in the figure.
        plot_styles (Iterable[Path | str]): The plot styles to use.
    """
    # STYLE -------
    manual = {
        "figure.figsize": [5.2, 4.8],
        "markers.fillstyle": "full",
        # "lines.markeredgecolor": "none",
        "xtick.minor.ndivs": 10,
        "ytick.minor.ndivs": 10,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "grid.alpha": 0,
    }
    manual.update(kwargs.pop('manual_style', {}))
    plot_styles: Iterable[Path | str] = kwargs.pop('plot_styles', 'standard')
    pstyle.set_style(plot_styles, manual)

    rescale: int = kwargs.pop('rescale', 3)
    rescale_value = 10**-rescale
    lim: tuple[float, float] | None = kwargs.pop('lim', None)
    ncol: int = kwargs.pop('ncol', 2)


    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {', '.join(kwargs.keys())}")

    if not isinstance(ids, dict):
        ids = {id_: id_ for id_ in ids}
    data = {label: get_corrector_strengths(folder, beam, id_, corrector_pattern) for id_, label in ids.items()}

    fig, ax = plt.subplots()

    # plot zero lines
    ax.axhline(0, color="gray", ls="-", marker="", zorder=-10, alpha=0.1, lw=0.2)
    ax.axvline(0, color="gray", ls="-", marker="", zorder=-10, alpha=0.1, lw=0.2)

    handles, labels = [], []

    for idx, (label, (values, errors)) in enumerate(data.items()):
        color = pcolors.get_mpl_color(idx)
        ip_correctors = pair_correctors(values.index)
        handles.append(Line2D([0], [0], marker=f"${''.join(ip_correctors.keys())}$", color=color, ls='none', label=label))
        labels.append(label)

        for ip, correctors in pair_correctors(values.index).items():
            left, right = correctors.get("l"), correctors.get("r")
            if not left or not right:
                raise ValueError(f"Could not find both correctors for {ip_correctors}")
            if errors is not None:
                x = (values[left] - errors[left]) * rescale_value
                y = (values[right] - errors[right]) * rescale_value
                width = errors[left] * 2 * rescale_value
                height = errors[right] * 2 * rescale_value
                ax.add_patch(
                    Rectangle(
                        (x, y), width, height,
                        alpha=0.3,
                        color=color,
                        label=f"_{label}{ip}err"
                    )
                )
            v_left = values[left] * rescale_value
            v_right = values[right] * rescale_value
            ax.plot(v_left, v_right, ls='none',  c=color, marker=f"${ip}$", label=f"_{label}{ip}")

    xlabel, ylabel = get_labels(field, rescale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

    if not lim:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        lim = max(np.abs(xlim + ylim))  # '+' here adds lists, not values
        lim = [-lim, lim]

    unit = np.floor(np.log10(lim[1] - lim[0]))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10**unit))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10**unit))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    pannot.make_top_legend(ax, ncol=ncol, frame=False, handles=handles, labels=labels)

    fig.canvas.manager.set_window_title("Corrector Strengths")
    return fig


def get_settings_file(folder: Path, beam: int, id_: str) -> Path:
    """Return the settings file for a given beam and id.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number (to select the right output files).
        id_ (str): The id of the data (target name).

    Returns:
        Path: The settings file.
    """
    glob = f"{SETTINGS_ID}.*.b{beam}.{id_}.tfs"
    for filename in folder.glob(glob):
        return filename
    raise FileNotFoundError(f"No file matching '{glob}' in {folder}.")


def get_corrector_strengths(folder: Path, beam: int, id_: str, corrector_pattern: str) -> pd.Series:
    """Get the corrector strengths for a given beam, id and corrector pattern.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number (to select the right output files).
        id_ (str): The id of the data (target name).
        corrector_pattern (str): The corrector pattern to match.

    Returns:
        pd.Series: The corrector strengths KNL values.
    """
    settings_file = get_settings_file(folder, beam, id_)
    df = tfs.read(settings_file, index=CIRCUIT)
    df = df.loc[df.index.str.match(corrector_pattern, flags=re.IGNORECASE), :]
    if df.empty:
        raise AttributeError(f"No matching corrector '{corrector_pattern}' values found.")

    errknl = f"{ERR}{KNL}"
    if errknl not in df.columns:
        return df[KNL], None
    return df[KNL], df[errknl]


def pair_correctors(correctors: Sequence[str]) -> dict[str, dict[str, str]]:
    """Returns a dictionary of ips with a dictionary left and right correctors.
    Assumes per IP and side there is only one corrector given and does
    not distinguish between corrector types.

    Args:
        correctors (Sequence[str]): The correctors to pair.

    Returns:
        dict[str, dict[str, str]]: The dictionary of ips with a dictionary left and right correctors.
    """
    pairs = defaultdict(dict)
    for k in correctors:
        pairs[k[-1]][k[-2].lower()] = k
    return {k: pairs[k] for k in sorted(pairs.keys())}


def get_labels(field: FieldComponent, rescale: int = 0) -> tuple[str, str]:
    """Generate the y-axis label for the plot.

    Args:
        field (FieldComponent): The field component.
        rescale (int, optional): The rescaling factor for the y-axis.
    """
    order = int(field[1])
    skew = "" if field[0].lower() == "b" else "S"
    knl = f"$K_{order}{skew}L$"

    rescale_str = f"10$^{rescale:d}$ " if rescale else ""
    unit = fr"[{rescale_str}m$^{{{-(order-1)}}}$]"
    return f"{knl} Left {unit}", f"{knl} Right {unit}"
