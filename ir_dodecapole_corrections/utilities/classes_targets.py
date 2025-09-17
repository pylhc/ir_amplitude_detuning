"""
Classes for Targets
-------------------

Classes used to define correction targets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from ir_dodecapole_corrections.detuning.targets import Constraints, Detuning

if TYPE_CHECKING:
    from collections.abc import Sequence


LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class TargetData:
    """ Class to hold the Data of a Target.
        Single IPs are converted into a tuple automatically.
        The detuning values should be a dictionary defining the
        beams.

        In the getter methods, Beam 2 and Beam 4 are used interchangeably:
        if only one is defined its values will be returned when
        the other one is requested.
     """
    ips: Sequence[int] | int
    detuning: dict[int, Detuning]
    constraints: dict[int, Constraints] = None
    xing: str = None
    MAIN_XING: ClassVar[str] = 'main'

    def __post_init__(self):
        if self.xing is None:
            self.xing = self.MAIN_XING

        if isinstance(self.ips, int):
            self.ips = (self.ips,)

    def beams(self) -> list[int]:
        """ Get the list of beams defined in the detuning data. """
        return list(self.detuning.keys())

    def get_detuning(self, beam: int) -> Detuning:
        """ Get the detuning for a specific beam,
        same as self.detuning[beam], but returns an empty Detuning if the beam is not defined
        and does not differentiate between Beam 2 and Beam 4.
        """
        try:
            return self.detuning[beam]
        except KeyError:
            if beam == 2 and (4 in self.beams()):
                return self.detuning[4]
            if beam == 4 and (2 in self.beams()):
                return self.detuning[2]

            LOG.debug(f"Beam {beam} not defined. Returning empty detuning definition")
            return Detuning()

    def constraints_beams(self) -> list[int]:
        """ Get the list of beams defined in the TargetData. """
        if self.constraints is None:
            return []
        return list(self.constraints.keys())

    def get_contraints(self, beam: int) -> Constraints:
        """ Get the constraints for a specific beam,
        same as self.constraints[beam], but returns an empty Constraints if the beam is not defined
        and does not differentiate between Beam 2 and Beam 4.

        """
        if self.constraints is None:
            return Constraints()

        try:
            return self.constraints[beam]
        except (KeyError, TypeError):
            if beam == 2 and (4 in self.constraints_beams()):
                return self.constraints[4]
            if beam == 4 and (2 in self.constraints_beams()):
                return self.constraints[2]

            LOG.debug(f"Beam {beam} not defined. Returning empty Constraints")
            return Constraints()


@dataclass(slots=True)
class Target:
    """ Class to hold correction Target information.

        The `data` input can be a single TargetData or a list of TargetData
        each containing the detuning data for this correction Target.
        This is done so that the TargetData can be split up depending on which IPs are targeted.
        Single TargetData are converted into a List automatically, so that the
        data attribute is always a Sequence.
    """
    name: str
    data: Sequence[TargetData] | TargetData  # single only for __init__, Sequence when accessed as attribute
    ips: Sequence[int] = field(init=False)  # set in __post_init__

    def __post_init__(self):
        if isinstance(self.data, TargetData):
            self.data = [self.data]
        self.ips = [ip for data in self.data for ip in data.ips]

        if len(self.ips) != len(set(self.ips)):
            LOG.debug(f"Duplicate ips found for {self.name}")
            self.ips = set(self.ips)
