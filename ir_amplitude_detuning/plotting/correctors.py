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

from matplotlib.patches import Rectangle
import numpy as np
import tfs
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from omc3.plotting.utils import annotations as pannot
from omc3.plotting.utils import colors as pcolors
from omc3.plotting.utils import style as pstyle

from ir_amplitude_detuning.utilities.constants import CIRCUIT, ERR, KNL, SETTINGS_ID

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import pandas as pd

    from ir_amplitude_detuning.utilities.classes_accelerator import FieldComponent

LOG = logging.getLogger(__name__)


def plot_correctors(
    folder: Path,
    beam: int,
    ids: dict[str, str] | Iterable[str],
    field: FieldComponent,
    corrector_pattern: str = ".*",
    **kwargs
    ):
    """ Plot the corrector strengths for a given beam and corrector pattern.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number (to select the right output files).
        ids (dict[str, str] | Iterable[str]): The ids to plot (from the targets).
                                              Use a dictionary to specify labels.
        field (str): The field of the used correctors, e.g. "a6" for K6SL.
        corrector_pattern (str, optional): The corrector pattern to match,
                                 in case you don't want all correctors to be plotted.

    Keyword Args (optional):
        figsize (Iterable[float]): The figure size.
        lim (float): The y-axis limit.
        ncol (int): The number of columns in the figure.
        plot_styles (Iterable[Path | str]): The plot styles to use.
    """
    # STYLE -------
    manual = {
        "figure.figsize": [5.2, 4.8],
        "markers.fillstyle": "none",
        "grid.alpha": 0,
        "savefig.format": "pdf",
    }
    manual.update(kwargs.pop('manual_style', {}))
    plot_styles: Iterable[Path | str] = kwargs.pop('plot_styles', 'standard')
    pstyle.set_style(plot_styles, manual)

    lim: tuple[float, float] | None = kwargs.pop('lim', None)
    ncol: int = kwargs.pop('ncol', 2)

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {', '.join(kwargs.keys())}")

    if not isinstance(ids, dict):
        ids = {id_: id_ for id_ in ids}
    data = {label: get_corrector_strengths(folder, beam, id_, corrector_pattern) for id_, label in ids.items()}

    fig, ax = plt.subplots()

    # plot zero lines
    ax.axhline(0, color="black", ls="--", marker="", zorder=-10, alpha=0.1)
    ax.axvline(0, color="black", ls="--", marker="", zorder=-10, alpha=0.1)

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
                ax.add_patch(
                    Rectangle(
                        (values[left] - errors[left], values[right] - errors[right]),
                        errors[left] * 2,
                        errors[right] * 2,
                        alpha=0.3,
                        color=color,
                        label=f"_{label}{ip}err"
                    )
                )
            ax.plot(values[left], values[right], ls='none',  c=color, marker=f"${ip}$", label=f"_{label}{ip}")


    skew = "" if field[0].lower() == "b" else "S"
    ax.set_xlabel(f"$K_{{{field[1]}}}{skew}L$ Left")
    ax.set_ylabel(f"$K_{{{field[1]}}}{skew}L$ Right")
    ax.set_aspect("equal")

    if not lim:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        lim = max(np.abs(xlim + ylim))  # '+' here adds lists, not values
        lim = [-lim, lim]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    pannot.make_top_legend(ax, ncol=ncol, frame=False, handles=handles, labels=labels)

    fig.canvas.manager.set_window_title("Corrector Strengths")
    return fig


def get_settings_file(folder: Path, beam: int, id_: str) -> Path:
    """ Return the settings file for a given beam and id.

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
    """ Get the corrector strengths for a given beam, id and corrector pattern.

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
    """ Returns a dictionary of ips with a dictionary left and right correctors.

    Args:
        correctors (Sequence[str]): The correctors to pair.

    Returns:
        dict[str, dict[str, str]]: The dictionary of ips with a dictionary left and right correctors.
    """
    pairs = defaultdict(dict)
    for k in correctors:
        pairs[k[-1]][k[-2].lower()] = k
    return {k: pairs[k] for k in sorted(pairs.keys())}
