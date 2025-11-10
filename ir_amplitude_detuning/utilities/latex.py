"""
Latex Utilities
---------------

Utilities to convert data to latex, useful for plotting and
copy-pasting into reports.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ir_amplitude_detuning.detuning.terms import DetuningTerm, detuning_term_to_planes, get_order

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ir_amplitude_detuning.detuning.measurements import MeasureValue

LOG = logging.getLogger(__name__)


def print_correction_and_error_as_latex(values: Sequence[MeasureValue], correctors: Sequence[str], length: Sequence[float] | float = 0.615) -> None:
    """Print the correction values with errors as latex table snippet.

    Args:
        values: List of MeasureValue with the correction values
        correctors: List of corrector names, same length as values
        length: Length of the correctors in m, default 0.615 m for MCTs
    """
    try:
        len(length)
    except TypeError:
        length = [length] * len(values)

    assert len(values) == len(correctors) == len(length), "Values, correctors and length must have the same length."

    values_scaled = np.array([v * 1e-3 / l for v, l in zip(values, length)])  # convert to KNL [10^3] # noqa: E741

    def mv2s(data: MeasureValue) -> str:
        """Covert MeasureValue to string with error in paranthesis."""
        if not hasattr(data, "error"):
            return fr"{data:.3f}"

        uncert = (
            f"{int(data.error * 1000.0):03d}"   # only the digits after the comma if < 1
            if data.error < 1 else
            f"{data.error:.3f}"                 # full number if >= 1
        )

        return fr"{data.value:.3f}({uncert})"

    LOG.info(
        f"Latex table snippet for correctors (KNL values [10^-3]):\n\n"
        f" & {' & '.join(correctors)}\\\\\n"
        f" & {' & '.join(mv2s(x) for x in values_scaled)}\\\\\n"
    )


def ylabel_from_detuning_term(detuning_term: DetuningTerm, exponent: float = None) -> str:
    """Get the latex representation of a detuning term with partial derivatives to be used as y-label of a plot.

    Args:
        detuning_term (str): Detuning term, e.g. "X02"
        exponent (float, optional): Exponent of 10 to be included in the latex representation.
    """
    order = get_order(detuning_term)
    scale = fr" 10^{{{exponent}}} " if exponent else ""
    return fr"${term2dqdj(detuning_term)}\; [{scale}$m$^{{-{order}}}]$"


def term2dqdj(term: DetuningTerm) -> str:
    """Wrapper to get the latex representation of a detuning term as in the shorthand.

    Args:
        term (str): Detuning term, e.g. "X02"
    """
    tune, action = detuning_term_to_planes(term)
    return dqd2j(tune, action)


def term2partial_dqdj(term: DetuningTerm) -> str:
    """Wrapper to get the latex representation of a detuning term with partial derivatives.

    Args:
        term (str): Detuning term, e.g. "X02"
    """
    tune, action = detuning_term_to_planes(term)
    return partial_dqd2j(tune, action)


def partial_dqd2j(tune: str, action: str) -> str:
    r"""Latex representation of detuning term.
    Examples:
        partial_dqdj("x", "yy") -> "\partial^{2}_{y}Q_{x}".
        partial_dqdj("x", "xy") -> "\partial_{x}\partial_{y}Q_{x}".
        partial_dqdj("y", "x") -> "\partial_{x}Q_{y}".

    Args:
        tune: "x" or "y"
        action: "x" or "y"
        power: integer power, default 1
    """
    if len(action) > 2:
        raise NotImplementedError("Not implemented for derivative > 2.")

    def partial(plane: str, exponent: int = 1) -> str:
        if exponent == 1:
            return fr"\partial_{{2J_{plane}}}"
        return fr"\partial^{{{exponent}}}_{{2J_{plane}}}"

    if len(action) == 1:
        return fr"{partial(action)}Q_{tune}"

    if action[0] != action[1]:
        return fr"{partial(action[0])}{partial(action[1])}Q_{tune}"

    return fr"{partial(action[0], 2)}Q_{tune}"


def dqd2j(tune: str, action: str) -> str:
    """Latex representation of detuning term
    (in the shorthand version, used in my thesis/paper, jdilly).

    Examples:
        dqd2j("x", "y", 2) -> "Q_{x,yy}"
        dqd2j("x", "xy") -> "Q_{x,xy}"
        dqd2j("y", "x") -> "Q_{y,x}"

    Args:
        tune: "x" or "y"
        action: "x" or "y"
        power: integer power, default 1
    """
    return f"Q_{{{tune},{action}}}"


def exp_m(e_power: int, m_power: int) -> str:
    """Latex representation of unit 10^power m^inv.
    Example: em(3, -1) -> "\\cdot 10^{3}\\;$m$^{-1}".

    Args:
        power: integer power of 10
        inv: integer power of m
    """
    if not e_power:
        return fr"\;$m$^{{{m_power:d}}}"
    return fr"\cdot 10^{{{e_power:d}}}\;$m$^{{{m_power:d}}}"


def unit_exp_m(e_power: int, m_power: int) -> str:
    """Latex representation of unit 10^power m^inv.
    Example: unit(3, -1) -> "\\; [10^{3}\\;$m$^{-1}]".

    Args:
        power: integer power of 10
        inv: integer power of m
    """
    return fr"\; [10^{{{e_power:d}}} $m$^{{{m_power:+d}}}]"
