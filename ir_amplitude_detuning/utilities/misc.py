"""
Misc Utilities
--------------
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ir_amplitude_detuning.detuning.measurements import Detuning, DetuningMeasurement


class StrEnum(str, Enum):
    """ Enum with string representation.

    Note: Can be removed in Python 3.11 as it is implemented there as `enum.StrEnum`.
    """
    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


# Convenience functions to loop over dicts ---

class Addable(Protocol):
    def __add__(self, other):
        ...


class Subtractable(Protocol):
    def __sub__(self, other):
        ...


def get_sum( a: dict[Any, Addable], b: dict[Any, Addable]) -> dict[Any, Addable]:
    """ Add the values of two dicts for each entry in a.
    Assumes all keys in a are present in b.
    """
    return {key: a[key] + b[key] for key in a}


def get_diff( a: dict[Any, Subtractable], b: dict[Any, Subtractable]) -> dict[Any, Subtractable]:
    """ Subtract the values of meas_b from meas_a for each entry in the dicts.
    Assumes all keys in a are present in b.
    """
    return {key: a[key] - b[key] for key in a}


def get_detuning(meas: dict[Any, DetuningMeasurement]) -> dict[Any, Detuning]:
    """ Get detuning values (i.e. without errors) from the measurement data."""
    return {key: meas[key].get_detuning() for key in meas}
