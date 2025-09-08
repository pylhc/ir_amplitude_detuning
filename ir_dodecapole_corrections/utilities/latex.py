from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ir_dodecapole_corrections.utilities.classes import MeasureValue

LOG = logging.getLogger(__name__)


YLABEL_MAP = {
    "X10": r"$\partial Q_x / \partial (2J_x) \; [10^3 $m$^{-1}]$",
    "Y01": r"$\partial Q_y / \partial (2J_y) \; [10^3 $m$^{-1}]$",
    "X01": r"$\partial Q_{x,y} / \partial (2J_{y,x}) \; [10^3 $m$^{-1}]$",
    "X20": r"$\partial^2 Q_x / \partial (2J_x)^2 \; [10^{12} $m$^{-2}]$",
    "Y02": r"$\partial^2 Q_y / \partial (2J_y)^2 \; [10^{12} $m$^{-2}]$",
    "X02": r"$\partial^2 Q_x / \partial (2J_y)^2 \; [10^{12} $m$^{-2}]$",
    "Y20": r"$\partial^2 Q_y / \partial (2J_x)^2 \; [10^{12} $m$^{-2}]$",
}

XLABEL_MAP = {
    "w_ampdet_b5b6": "$K_5L & K_6L$",
    "w_ampdet_b5": "$K_5L$",
    "w_ampdet_b6": "$K_6L$",
    "nominal": "Nominal",
}

def print_correction_and_error_as_latex(x_pseudo: float, correctors: list[str]):
    """ Print the correction values with errors as latex table snippet."""
    length = 0.615  # corrector length
    x_scaled = x_pseudo * 1e-3 / length  # convert to KNL [10^3]

    def mv2s(data: MeasureValue) -> str:
        """ Covert MeasureValue to string with error in paranthesis. """
        if not hasattr(data, "error"):
            return fr"{data:.3f} & (\%)"

        uncert = (
            f"{int(data.error * 1000.0):03d}"   # only the digits after the comma if < 1
            if data.error < 1 else
            f"{data.error:.3f}"                 # full number if >= 1
        )

        return fr"{data.value:.3f}({uncert})& (\%)"

    LOG.info(
        f"\n & {' & '.join(correctors)}\\\\\n"
        f" & {' & '.join(mv2s(x) for x in x_scaled)}\\\\\n"
    )


def partial_dqdj(tune: str, action: str, power: int = 1) -> str:
    r""" Latex representation of detuning term.
    Example: partial_dqdj("x", "y", 2) -> "\partial^{2}_{y}Q_{x}".

    Args:
        tune: "x" or "y"
        action: "x" or "y"
        power: integer power, default 1
    """
    if power == 1:
        return fr"\partial_{action}Q_{tune}"
    return fr"\partial^{{{power}}}_{action}Q_{tune}"


def dqd2j(tune: str, action: str, power: int = 1) -> str:
    """ Latex representation of detuning term
    (in the shorthand version, used in my thesis/paper, jdilly).
    Example: dqd2j("x", "y", 2) -> "Q_{x,y^{2}}".

    Args:
        tune: "x" or "y"
        action: "x" or "y"
        power: integer power, default 1
    """
    if power == 1:
        return f"Q_{{{tune},{action}}}"
    return f"Q_{{{tune},{action}^{{{power}}}}}"


def exp_m(e_power: int, m_power: int) -> str:
    """ Latex representation of unit 10^power m^inv.
    Example: em(3, -1) -> "\\cdot 10^{3}\\;$m$^{-1}".

    Args:
        power: integer power of 10
        inv: integer power of m
    """
    if not e_power:
        return fr"\;$m$^{{{m_power:d}}}"
    return fr"\cdot 10^{{{e_power:d}}}\;$m$^{{{m_power:d}}}"


def unit_exp_m(e_power: int, m_power: int) -> str:
    """ Latex representation of unit 10^power m^inv.
    Example: unit(3, -1) -> "\\; [10^{3}\\;$m$^{-1}]".

    Args:
        power: integer power of 10
        inv: integer power of m
    """
    return fr"\; [10^{{{e_power:d}}} $m$^{{{m_power:+d}}}]"
