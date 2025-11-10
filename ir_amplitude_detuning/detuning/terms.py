"""
Detuning Terms
--------------

This module contains the definitions of how the detuning terms are represented
in this code and in string outputs (TFS columns, logging, etc.).
This follows the PTC-output convention ``ANH[XY] ORDER1 ORDER2 ORDER3 ORDER4``,
reduced to the plane of the tune (X or Y) and the order of the first two derivatives (
2Jx and 2Jy respectively).

This representation is chosen over the LaTeX shorthand "Qxx" to have compact and
same-length strings that are still readable.
"""
from __future__ import annotations

from typing import TypeAlias

from ir_amplitude_detuning.utilities.common import StrEnum


class FirstOrderTerm(StrEnum):
    X10: str = "X10"  # d Qx / d 2Jx
    X01: str = "X01"  # d Qx / d 2Jy
    Y10: str = "Y10"  # d Qy / d 2Jx
    Y01: str = "Y01"  # d Qy / d 2Jy


class SecondOrderTerm(StrEnum):
    X20: str = "X20"  # d^2 Qx / (d 2Jx)^2
    X11: str = "X11"  # d^2 Qx / (d 2Jx)(d Jy)
    X02: str = "X02"  # d^2 Qx / (d 2Jy)^2
    Y20: str = "Y20"  # d^2 Qy / (d 2Jx)^2
    Y11: str = "Y11"  # d^2 Qy / (d 2Jx)(d Jy)
    Y02: str = "Y02"  # d^2 Qy / (d 2Jy)^2


DetuningTerm: TypeAlias = FirstOrderTerm | SecondOrderTerm


def get_order(term: DetuningTerm) -> int:
    """Get the order of the detuning, e.g. from X11 -> order 2, Y10 -> order 1.

    Args:
        term (str): Detuning Term, e.g. "X11"
    """
    return int(term[1]) + int(term[2])


def detuning_term_to_planes(term: DetuningTerm) -> tuple[str, str]:
    """Get the tune and action planes given detuning term.

    Args:
        term (str): Detuning term, e.g. "X02"

    Returns:
        tuple[str, str]: (tune, action), e.g. ("x", "yy")
    """
    tune = term[0].lower()
    action = "x" * int(term[1]) + "y" * int(term[2])
    return tune, action
