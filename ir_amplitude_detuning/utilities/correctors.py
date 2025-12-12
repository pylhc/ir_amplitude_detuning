"""
Correctors
----------

Utilities for working with Correctors,
implemented to make the code more machine-independent,
as you should be able to easily add new corrector definitions for a new machine.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from ir_amplitude_detuning.utilities.common import StrEnum

LOG = logging.getLogger(__name__)


class FieldComponent(StrEnum):
    """Fields for which detuning calculations are implemented."""
    b4: str = "b4"
    b5: str = "b5"
    b6: str = "b6"


@dataclass(slots=True)
class Corrector:
    """Class to hold corrector information.

    Args:
        field: magnetic field component shorthand (e.g. 'b5' or 'b6')
        length: length of the corrector in m
        magnet: MAD-X magnet name, e.g. "MCTX.3L1"
        circuit: MAD-X circuit name, e.g. "kctx3.l1"
        ip: IP the corrector is located at (for filtering if only certain IPs are corrected)
        madx_type: MAD-X magnet type, e.g. "MCTX"
    """
    field: FieldComponent
    length: float
    magnet: str
    circuit: str
    ip: int | None = None
    madx_type: str | None = None

    def __post_init__(self):
        if self.field not in list(FieldComponent):
            raise ValueError(f"Field must be one of {list(FieldComponent)}, got {self.field}.")

    def __setattr__(self, name: str, value) -> None:
        if name == "field" and value not in list(FieldComponent):
            raise ValueError(f"Field must be one of {list(FieldComponent)}, got {value}.")
        object.__setattr__(self, name, value)

    def __lt__(self, other: Corrector) -> bool:
        """Less-than operator for sorting correctors.
        Sorting is done by field, then by IP side (L before R), then by IP number (lower before higher),
        then by position along the beamline (upstream before downstream).
        """
        if self.field != other.field:
            return self.field < other.field

        pattern = re.compile(r".*(\d+)([lr]\d)$", re.IGNORECASE)
        self_match = pattern.match(self.magnet)
        other_match = pattern.match(other.magnet)
        if self_match is None or other_match is None:
            return self.circuit < other.circuit

        self_ipside = self_match.group(2)
        other_ipside = other_match.group(2)

        if self_ipside != other_ipside:
            # Sort by IP then left to right
            return (self_ipside[1:], self_ipside[0]) < (other_ipside[1:], other_ipside[0])

        # Correctors are at the same side of the same IP
        if self_ipside[0].lower() == "l":
            return self_match.group(1) > other_match.group(1)  # larger = further downstream
        return self_match.group(1) < other_match.group(1)  # larger = further upstream

    def __hash__(self):
        return hash(self.magnet + self.circuit)

    def __repr__(self):
        return f"{self.circuit}({self.magnet}>{self.field})"


@dataclass(slots=True)
class CorrectorMask:
    """Class to hold corrector templates to be filled in with IP and side.

    Args:
        field: magnetic field component shorthand (e.g. 'b5' or 'b6')
        length: length of the corrector in m
        magnet_pattern: MAD-X magnet name pattern, e.g. "MCTX.3{side}{ip}"
        circuit_pattern: MAD-X circuit name pattern, e.g. "kctx3.{side}{ip}"
        ip:
        madx_type: MAD-X magnet type, e.g. "MCTX"
    """
    field: FieldComponent
    length: float
    magnet_pattern: str
    circuit_pattern: str
    madx_type: str | None = None

    def get_corrector(self, side: str, ip: int) -> Corrector:
        return Corrector(
            field=self.field,
            length=self.length,
            magnet=self.magnet_pattern.format(side=side.upper(), ip=ip),
            circuit=self.circuit_pattern.format(side=side.lower(), ip=ip),
            ip=ip,
            madx_type=self.madx_type,
        )

Correctors: TypeAlias = Sequence[Corrector]


def get_fields(correctors: Correctors) -> list[FieldComponent]:
    """Get all field components available for correction by the correctors.

    Args:
        correctors (Correctors): list of correctors

    Returns:
        list[FieldComponent]: sorted list of uniqe field components
    """
    return sorted({corrector.field for corrector in correctors})


def fill_corrector_masks(corrector_masks: Sequence[CorrectorMask | Corrector], ips: Sequence[int], sides: Sequence[str] = "LR") -> Correctors:
    """Fill the corrector masks with the ips and sides.

    Args:
        corrector_masks (Sequence[CorrectorMask | Corrector]): list of corrector masks or correctors.
        ips (Sequence[int]): list of ips.
        sides (Sequence[str]): list of sides.

    Returns:
        list[Corrector]: sorted list of correctors
    """
    output = []
    for mask in corrector_masks:
        try:
            for ip in ips:
                for side in sides:
                    output.append(mask.get_corrector(side, ip))
        except AttributeError:
            output.append(mask)
    return sorted(output)
