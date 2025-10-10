"""
General Plotting Utilities
--------------------------

This module contains general utilities to help with the plotting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ir_amplitude_detuning.utilities import latex
from ir_amplitude_detuning.utilities.classes_accelerator import FieldComponent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ir_amplitude_detuning.detuning.measurements import FirstOrderTerm, SecondOrderTerm
    from ir_amplitude_detuning.utilities.classes_targets import Target, TargetData


def get_default_scaling(term: FirstOrderTerm | SecondOrderTerm) -> tuple[int, float]:
    """ Get the default scaling factor for a detuning term.

    Args:
        term (str): Detuning term, e.g. "X02"

    Returns:
        tuple[int, float]: (exponent, scaling)
    """
    exponent = {1: 3, 2: 12}[int(term[1]) + int(term[2])]
    scaling = 10**-exponent
    return exponent, scaling


def get_color_for_field(field: FieldComponent):
    """ Get predefined colors for the fields. """
    match field:
        case FieldComponent.b5:
            return '#7f7f7f'  # middle gray
        case FieldComponent.b6:
            return '#d62728'  # brick red
        case FieldComponent.b4:
            return '#bcbd22'  # curry yellow-green
    raise NotImplementedError(f"Field must be one of {list(FieldComponent)}, got {field}.")


def get_color_for_ip(ip: str):
    """ Get predefined colors for the IPs. """
    match ip:
        case "15":
            return '#1f77b4'  # muted blue
        case "1":
            return '#9467bd'  # muted purple
        case "5":
            return '#2ca02c'  # cooked asparagus green
    raise NotImplementedError(f"IP must be one of ['15', '1', '5'], got {ip}.")


def get_full_target_labels(targets: Sequence[Target], suffixes: Sequence[str] | None = None, scale_exponent: float = 3) -> dict[str, str]:
    """ Get a label that includes all detuning terms so that they can be easily compared.
    To save space only the first target_data is used.

    Args:
        targets (Sequence[Target]): List of Target objects to get labels for.
        suffixes (Sequence[str] | None): List of suffixes to add to the labels.
        scale (float): Scaling factor for the detuning values.

    Returns:
        dict[str, str]: Dictionary of labels for each target identified by its name.
    """
    if suffixes is not None and len(suffixes) != len(targets):
        raise ValueError("Number of suffixes must match number of targets.")

    scaling = 10**-scale_exponent

    names = [target.name for target in targets]
    labels = [None for _ in targets]
    for idx_target, target in enumerate(targets):
        target_data: TargetData = target.data[0]
        scaled_values = {
            term: (target_data.detuning[1][term] * scaling, target_data.detuning[2][term] * scaling)
            for term in target_data.detuning[1].terms()
        }
        label = "\n".join(
            [
                f"${latex.term2dqdj(term)}$ = {f'{values[0].value: 5.1f} | {values[1].value: 5.1f}'.center(15)}"
                for term, values in scaled_values.items()
            ]
        )
        if suffixes is not None:
            label += f"\n{suffixes[idx_target]}"
        labels[idx_target] = label
    return dict(zip(names, labels))
