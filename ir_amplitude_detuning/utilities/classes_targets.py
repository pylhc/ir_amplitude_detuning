"""
Classes for Targets
-------------------

Classes used to define correction targets.
Each Target will be used to calcualate the correction,
i.e. transformed into a single detuning equation system.

To accurately correct for different detuning contributions from the errors and
crossing in each IP (or combination of IPs), each Target contains a list of
TargetData, which defines the measured detuning and constraints for each machine configuration.
These are used to build the rows of the equation system, with a single TargetData
defining multiple rows, depending on the amount of beams and detuning components given.

This allows for a combined "local", targeting the detunning stemming from a single IP,
and "global", correcting for the combined detuning in the machine, correction if needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ir_amplitude_detuning.detuning.measurements import Constraints, Detuning
from ir_amplitude_detuning.utilities.common import BeamDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    import tfs

    from ir_amplitude_detuning.utilities.classes_accelerator import Correctors


LOG = logging.getLogger(__name__)


class TargetData:
    """ Class to hold the detunig data of a Target.

    The TargetData is used to build multiple lines in an equation system,
    for each beam and each detuning/constraint component.

    Args:
        correctors (Correctors): List of correctors to be used for optimizing the detuning/constraints.
        detuning (dict[int, Detuning]): Dictionary defining the detuning for each beam.
        optics (dict[int, tfs.TfsDataFrame]): Dictionary defining the optics for each beam.
        constraints (dict[int, Constraints] | None): Dictionary defining the constraints for each beam.
    """
    def __init__(self,
        correctors: Correctors,
        optics: dict[int, tfs.TfsDataFrame],
        detuning: dict[int, Detuning],
        constraints: dict[int, Constraints] | None = None
        ):
        self.correctors = sorted(correctors)
        self.optics = optics
        self.detuning: BeamDict = BeamDict.from_dict(detuning, default=Detuning)
        self.constraints: BeamDict = BeamDict.from_dict(constraints or {}, default=Constraints)

        self.beams = tuple(self.optics.keys())  # needs to come form optics as beam 2 and beam 4 are important for these!
        self.ips = tuple({c.ip for c in self.correctors if c.ip is not None})


@dataclass(slots=True)
class Target:
    """ Class to hold correction Target information,
    which can be used to construct a single equation system,
    calculating a combined correction optimizing for all TargetData definitions.

    Args:
        name (str): Name of the Target
        data (Sequence[TargetData]): Data for the Target
    """
    name: str
    data: Sequence[TargetData]
    correctors: Correctors = field(init=False)  # All correctors for all TargetData
    ips: tuple[int, ...] = field(init=False)  # All ips for all TargetData

    def __post_init__(self):
        if "." in self.name:
            raise NameError("No periods allowed in target name!")

        self.correctors = sorted({c for data in self.data for c in data.correctors})
        self.ips = tuple({ip for data in self.data for ip in data.ips})
