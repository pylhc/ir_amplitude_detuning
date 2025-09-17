"""
Classes for Accelerators
------------------------

Classes used to hold accelerator specific data to make the
code machine independent.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import LiteralString

from ir_dodecapole_corrections.utilities.misc import StrEnum

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class Corrector:
    """ Class to hold corrector information.

    Args:
        field: magnetic field component shorthand (e.g. 'b5' or 'b6')
        length: length of the corrector in m
        magnet: MAD-X magnet name pattern, e.g. "MCTX.3{side}{ip}"
        circuit: MAD-X circuit name pattern, e.g. "kctx3.{side}{ip}"
        pattern: regex pattern to identify the magnets, e.g. "MCTX.*[15]$"
    """
    field: str
    length: float
    magnet: str
    circuit: str
    pattern: str


Correctors = Sequence[Corrector]

class CorrectorFillAttributes(StrEnum):
    circuit: str = "circuit"
    magnet: str = "magnet"


def get_filled_corrector_attributes(ips: Sequence[str], correctors: Correctors, attribute: str) -> list[str]:
    """
    Returns a list of filled-in circuits for the given IPs, adding 'L' and 'R' for sides.
    Assures that the order is always the same.

    Args:
        ips (Sequence[str]): List of IPs as strings, e.g. ["1", "5"]
        correctors (Correctors): Sequence of correctors to use.
        attribute (str): Attribute of the corrector to return, e.g. "circuit" or "magnet"
    """
    if attribute not in list(CorrectorFillAttributes):
        raise ValueError(f"Attribute must be one of {list(CorrectorFillAttributes)}, got {attribute}.")

    sorted_correctors = sort_correctors(correctors)

    return [
        getattr(corrector, attribute).format(side=side, ip=ip)
        for ip in ips for side in "LR" for corrector in sorted_correctors
    ]


def get_fields(correctors: Correctors) -> list[str]:
    """ Get the field components available in the correctors. """
    return sorted({corrector.field for corrector in correctors})


def sort_correctors(correctors: Correctors) -> list[Corrector]:
    """ Get the correctors sorted by field and circuit. """
    return sorted(correctors, key=lambda c: (c.field, c.circuit))
