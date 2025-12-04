"""
Common Utilities
----------------

This module contains common utilities for use in other modules.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


class StrEnum(str, Enum):
    """Enum with string representation.

    Note:
          Can possibly be removed in Python 3.11 as it is implemented there as `enum.StrEnum`.
          But beware, that `"value" in StrEnum` raises `TypeError` until Python 3.12,
          workaround is `"value" in list(StrEnum)` or to try `StrEnum(value)`.
    """
    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class ContainerMeta(type):
    """MetaClass to store data in class attributes.
    Minimal implementation to make this usable as a 'Mapping', i.e. dict-like.
    """
    def __getitem__(cls, key):
        return cls.__dict__[key]

    def __iter__(cls):
        # everything in the class, but ignore all private attributes and functions/attributes defined here
        return iter(key for key in cls.__dict__ if not key.startswith("_") and key not in ContainerMeta.__dict__)

    def __len__(cls) -> int:
        return len(tuple(cls.__iter__()))

    def keys(cls) -> list[str]:
        return list(cls.__iter__())


class Container(metaclass=ContainerMeta):
    """Convenience wrapper to inherit directly, instead of using a metaclass."""
    ...


class BeamDict(dict):
    """Dictionary with beam number as keys, where beam 2 and 4 are interchangeable.
    Also implements basic arithmetic operations, to be applied to all beams.
    """
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
        """Create a BeamDict from a regular dict, setting a default factory for missing keys."""
        obj = cls(d)
        obj.__default_when_missing__ = default
        return obj

    def __add__(self, other: BeamDict) -> BeamDict:
        """Add two BeamDicts together, adding the values for each beam."""
        result = BeamDict()
        for beam in self.keys():
            result[beam] = self[beam] + other[beam]
        return result

    def __sub__(self, other: BeamDict) -> BeamDict:
        """Subtract two BeamDicts, subtracting the values for each beam."""
        result = BeamDict()
        for beam in self.keys():
            result[beam] = self[beam] - other[beam]
        return result

    def __truediv__(self, scalar: float) -> BeamDict:
        """Divide all values in the BeamDict by a scalar."""
        result = BeamDict()
        for beam in self.keys():
            result[beam] = self[beam] / scalar
        return result

    def __mul__(self, scalar: float) -> BeamDict:
        """Multiply all values in the BeamDict by a scalar."""
        result = BeamDict()
        for beam in self.keys():
            result[beam] = self[beam] * scalar
        return result

    def __rmul__(self, scalar: float) -> BeamDict:
        """Multiply all values in the BeamDict by a scalar."""
        return self.__mul__(scalar)


# Looping Related Utilities -----------------------------------------------------

def to_loop(iterable: Iterable[Any]) -> list[Iterable[int]]:
    """Get a list to loop over.

    If there is only one entry, the return list will only have this entry wrapped in a list.
    If there are multiple entry, the first element will be a list of all entries combined,
    and then single-element lists containing one entry each.

    Args:
        iterable (Iterable[Any]): List to loop over

    Returns:
        list[Iterable[int]]: List of lists of elements
    """
    if not iterable:
        raise ValueError("Nothing to loop over.")

    combined = [iterable]

    if len(iterable) == 1:
        return combined

    return combined + [[entry] for entry in iterable]
