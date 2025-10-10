"""
Common Utilities
----------------

This module contains common utilities for use in other modules.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ir_amplitude_detuning.detuning.measurements import Detuning, DetuningMeasurement


class StrEnum(str, Enum):
    """ Enum with string representation.

    Note: Can be removed in Python 3.11 as it is implemented there as `enum.StrEnum`.
    """
    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class ContainerMeta(type):
    """ MetaClass to store data in class attributes.
    Minimal implementation to make this usable as a 'Mapping', i.e. dict-like.
    """
    def __getitem__(cls, key):
        return cls.__dict__[key]

    def __iter__(cls):
        # everything in the class, but ignore all private attributes and functions/attributes defined here
        return iter(key for key in cls.__dict__ if not key.startswith("_") and key not in ContainerMeta.__dict__)

    def __len__(cls):
        return len(tuple(cls.__iter__()))

    def keys(cls):
        return cls.__iter__()


class Container(metaclass=ContainerMeta):
    """ Convenience wrapper to inherit directly, instead of using a metaclass. """
    ...


class BeamDict(dict):
    __default_when_missing__: callable | None = None

    def __missing__(self, key):
        if key == 2 and 4 in self:
            return self[4]

        if key == 4 and 2 in self:
            return self[2]

        if self.__default_when_missing__ is not None:
            return self.__default_when_missing__()  # e.g. used to return an empty instance

        raise KeyError(f"Beam {key} not defined.")

    @classmethod
    def from_dict(cls, d: dict[int, Any], default: callable = None):
        obj = cls(d)
        obj.__default_when_missing__ = default
        return obj


# Looping Related Utilities -----------------------------------------------------

def to_loop(iterable: Iterable[Any]) -> list[Iterable[int]]:
    """ Get a list to loop over.

    If there is only one entry, the return list will only have this entry wrapped in a list.
    If there are multiple entry, the first element will be a list of all entries combined,
    and then single-element lists containing one entry each.

    Args:
        iterable (Iterable[Any]): List to loop over

    Returns:
        list[Iterable[int]]: List of lists of elements
    """
    combined = [iterable]

    if len(iterable) == 1:
        return combined

    return combined + [[entry] for entry in iterable]


# Loop over dicts ---

class Addable(Protocol):
    def __add__(self, other):
        ...


class Subtractable(Protocol):
    def __sub__(self, other):
        ...


def dict_sum(a: dict[Any, Addable], b: dict[Any, Addable]) -> dict[Any, Addable]:
    """ Add the values of two dicts for each entry in a.
    Assumes all keys in a are present in b.
    """
    return {key: a[key] + b[key] for key in a}


def dict_diff(a: dict[Any, Subtractable], b: dict[Any, Subtractable]) -> dict[Any, Subtractable]:
    """ Subtract the values of meas_b from meas_a for each entry in the dicts.
    Assumes all keys in a are present in b.
    """
    return {key: a[key] - b[key] for key in a}


def get_dict_detuning(meas: dict[Any, DetuningMeasurement]) -> dict[Any, Detuning]:
    """ Get detuning values (i.e. without errors) from the measurement data for each of the items in meas."""
    return {key: meas[key].get_detuning() for key in meas}
