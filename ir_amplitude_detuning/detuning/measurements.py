
"""
Classes for Detuning
--------------------

Classes used to hold and manipulate individual detuning (measurement) data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ir_amplitude_detuning.detuning.terms import FirstOrderTerm, SecondOrderTerm

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    try:
        from typing import Self  # py 3.11+
    except ImportError:
        from typing_extensions import Self  # py 3.10

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class MeasureValue:
    """Class to hold a value with its error and do basic arithmetics.

    Args:
        value (float): value of the measurement
        error (float): error of the measurement, treated as standard deviation
    """
    value: float = 0.0
    error: float = 0.0

    def __add__(self, other: float | MeasureValue):
        if isinstance(other, MeasureValue):
            return MeasureValue(value=self.value + other.value, error=np.sqrt(self.error**2 + other.error**2))
        return MeasureValue(value=self.value + other, error=self.error)

    def __radd__(self, other: MeasureValue | float):
        return self + other  # __add__, note `0 + obj` is used by sum()

    def __sub__(self, other: MeasureValue | float):
        if isinstance(other, MeasureValue):
            return MeasureValue(value=self.value - other.value, error=np.sqrt(self.error**2 + other.error**2))
        return MeasureValue(value=self.value - other, error=self.error)

    def __neg__(self):
        return MeasureValue(value=-self.value, error=self.error)

    def __mul__(self, other: float):
        return MeasureValue(value=self.value * other, error=self.error * abs(other))

    def __rmul__(self, other: float):
        return self * other  # __mul__

    def __truediv__(self, other: float):
        return self * (1 / other)

    def __abs__(self):
        return MeasureValue(value=abs(self.value), error=self.error)

    def __str__(self):
        return f"{self.value} +- {self.error}"

    def __format__(self,fmt):
        return f"{self.value:{fmt}} +- {self.error:{fmt}}"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter((self.value, self.error))

    @staticmethod
    def rms(measurements: Sequence[MeasureValue]):
        """Returns rms of values and errors."""
        values = np.array([m.value for m in measurements])
        errors = np.array([m.error for m in measurements])

        def rms(x):
            return np.sqrt(np.mean(x**2))

        n = len(measurements)
        rms_values = rms(values)
        rms_err_times_value = rms(errors * values)
        return MeasureValue(
            value=rms_values,
            error=1/np.sqrt(n) * rms_err_times_value / rms_values,
        )

    @staticmethod
    def weighted_rms(measurements: Sequence[MeasureValue]):
        """Returns weighted rms of values and errors."""
        values = np.array([m.value for m in measurements])
        errors = np.array([m.error for m in measurements])

        if np.any(errors == 0):
            raise ValueError("Cannot compute weighted RMS with zero errors.")

        weights = 1 / errors**2
        sum_weights = np.sum(weights)
        return MeasureValue(
            value=np.sqrt(np.sum(values**2 * weights) / sum_weights),
            error=1/np.sqrt(sum_weights)
        )

    @staticmethod
    def mean(measurements: Sequence[MeasureValue]):
        """Returns mean of the measurements."""
        values = np.array([m.value for m in measurements])
        errors = np.array([m.error for m in measurements])

        return MeasureValue(
            value=np.mean(values),
            error=np.sqrt(np.sum(errors**2)) / len(measurements),
        )

    @staticmethod
    def weighted_mean(measurements: Sequence[MeasureValue]):
        """Returns a mean weighted proportionally to the errors."""
        values = np.array([m.value for m in measurements])
        errors = np.array([m.error for m in measurements])

        if np.any(errors == 0):
            raise ValueError("Cannot compute weighted RMS with zero errors.")

        weights = 1 / errors**2
        sum_weights = np.sum(weights)

        return MeasureValue(
            value=np.sum(values * weights) / sum_weights,
            error = 1 / np.sqrt(sum_weights)
        )

    @classmethod
    def from_value(cls, value: float | MeasureValue):
        if isinstance(value, float):
            return cls(value)

        # make a copy:
        return cls(value.value, value.error)


@dataclass(slots=True)
class Detuning:
    """Class holding first and second order detuning values.
    Only set values are returned via `__getitem__` or `terms()`.
    For convenience, the input values are scaled by the given `scale` parameter."""
    # first order
    X10: float | None = None
    X01: float | None = None
    Y10: float | None = None
    Y01: float | None = None
    # second order
    X20: float | None = None
    X11: float | None = None
    X02: float | None = None
    Y20: float | None = None
    Y11: float | None = None
    Y02: float | None = None
    scale: float | None = None

    def __post_init__(self):
        if self.scale:
            for term in self.terms():
                self[term] = self[term] * self.scale

    def terms(self) -> Iterator[str]:
        """Return names for all set terms."""
        return iter(name for name in self.all_terms() if getattr(self, name) is not None)

    def items(self) -> Iterator[tuple[str, float]]:
        return iter((name, getattr(self, name)) for name in self.terms())

    @staticmethod
    def all_terms(order: int | None = None) -> tuple[str, ...]:
        """Return all float-terms.

        Args:
            order (int): 1 or 2, for first and second order detuning terms respectively.
                         Or `None` for all terms (Default: `None`).
        """
        mapping = {
            1: tuple(FirstOrderTerm),
            2: tuple(SecondOrderTerm),
        }
        if order:
            return mapping[order]
        return tuple(e for m in mapping.values() for e in m)

    def __getitem__(self, item: str):
        """Convenience wrapper to access terms via `[]` .
        Not set terms will raise a KeyError.
        """
        if item not in self.terms():
            raise KeyError(f"'{item}' is not set in Detuning object.")
        return getattr(self, item)

    def __setitem__(self, item: str, value: float):
        """Convenience wrapper to set terms via `[]` ."""
        if item not in self.all_terms():
            raise KeyError(f"'{item}' is not in the available terms of a {self.__class__.__name__} object.")
        return setattr(self, item, value)

    def __add__(self, other: Self) -> Self:
        self._check_terms(other)
        return self.__class__(**{term: self[term] + other[term] for term in self.terms()})

    def __sub__(self, other: Self) -> Self:
        self._check_terms(other)
        return self.__class__(**{term: self[term] - other[term] for term in self.terms()})

    def __neg__(self) -> Self:
        return self.__class__(**{term: -self[term] for term in self.terms()})

    def __mul__(self, other: float | Self) -> Self:
        if isinstance(other, self.__class__):
            self._check_terms(other)
            return self.__class__(**{term: self[term] * other[term] for term in self.terms()})
        return self.__class__(**{term: self[term] * other for term in self.terms()})

    def __truediv__(self, other: float | Self) -> Self:
        if isinstance(other, self.__class__):
            self._check_terms(other)
            return self.__class__(**{term: self[term] / other[term] for term in self.terms()})
        return self.__class__(**{term: self[term] / other for term in self.terms()})

    def _check_terms(self, other: Self):
        not_in_other = [term for term in self.terms() if term not in other.terms()]
        if len(not_in_other):
            raise KeyError(
                f"Term '{not_in_other}' are not in the other {other.__class__.__name__} object. "
                f"Subtraction not possible."
            )

        not_in_self = [term for term in other.terms() if term not in self.terms()]
        if len(not_in_self):
            LOG.debug(
                f"Term '{not_in_self}' from the other object are not in this "
                f"{self.__class__.__name__} object. Terms ignored."
            )

    def apply_acdipole_correction(self) -> Self:
        """Correct for the influence of the AC-Dipole kick in measurement data.

        See Eqs. (78) - (81) and Eqs. (94) - (99) in [DillyAmplitudeDetuning2023]_
        and the derivations therein.

        Returns:
            Detuning: The corrected detuning
        """
        copy = self.__class__(**{term: self[term] for term in self.terms()})

        corrections = {
            2.: [FirstOrderTerm.X10, FirstOrderTerm.Y01, SecondOrderTerm.X11, SecondOrderTerm.Y11],  # Eqs. 78, 81, 95, 98
            3.: [SecondOrderTerm.X20, SecondOrderTerm.Y02],  #Eqs. 94, 99
        }
        for value, terms in corrections.items():
            for term in terms:
                if getattr(copy, term):
                    copy[term] = copy[term] / value
        return copy

    def merge_first_order_crossterm(self) -> Self:
        """Merge the cross-terms in the first order detuning into a single term X01,
        to avoid too much weight when fitting.

        Returns:
            Detuning: The merged detuning
        """
        the_copy = self.__class__(**dict(self.items()))

        if not self.Y10:
            return the_copy

        if not self.X01:
            the_copy.X01 = self.Y10  # put Y10 into X01
        else:
            the_copy.X01 = (self.X01 + self.Y10) * 0.5  # create an average

        the_copy.Y10 = None
        return the_copy


@dataclass(slots=True)
class DetuningMeasurement(Detuning):
    """Class holding first and second order detuning measurement values (i.e. with error)."""
    # first order
    X10: MeasureValue = None
    X01: MeasureValue = None
    Y10: MeasureValue = None
    Y01: MeasureValue = None
    # second order
    X20: MeasureValue = None
    X11: MeasureValue = None
    X02: MeasureValue = None
    Y20: MeasureValue = None
    Y11: MeasureValue = None
    Y02: MeasureValue = None

    def __post_init__(self):
        for term in self.terms():
            if not isinstance(self[term], MeasureValue):
                try:
                    if len(self[term]) > 2:
                        raise ValueError(
                            f"Found {len(self[term])} values to initialize term {term} "
                            f"of {self.__class__.__name__}, but a maximum of 2 values are allowed."
                        )
                except TypeError:  # from len(); assumes single number
                    self[term] = MeasureValue(self[term])
                else:
                    self[term] = MeasureValue(*self[term])

        Detuning.__post_init__(self)

    def get_detuning(self) -> Detuning:
        """Returns a Detuning object with the values (no errors) of this measurement."""
        return Detuning(**{term: self[term].value for term in self.terms()})

    @classmethod
    def from_detuning(cls, detuning) -> Self:
        """Create a DetuningMeasurement from a Detuning object, with zero errors."""
        return cls(**{term: MeasureValue(detuning[term]) for term in detuning.terms()})


@dataclass(slots=True)
class Constraints:
    """Class for holding detuning contraints.
    These are useful when trying to force a detuning term to have a specific sign,
    but not a specific value.
    Examples of this can be found in Fig. (7.1) of [DillyThesis2024]_.

    Only set definitions are returned via `__getitem__` or `terms()`,
    yet as they are used to build an equation system with minimization constraints,
    it is assumed that the values will only be used via the `get_leq()` method,
    which also applies the set scaling.

    Only ">=" and "<=" are implemented.
    E.g. ``X10 = "<=0"``.
    """
    X10: str | None = None
    X01: str | None = None
    Y10: str | None = None
    Y01: str | None = None
    #
    X20: str | None = None
    X11: str | None = None
    X02: str | None = None
    Y20: str | None = None
    Y11: str | None = None
    Y02: str | None = None
    #
    scale: float | None = None

    def __post_init__(self):
        for term in self.terms():
            self._parse_value(getattr(self, term))

    def _parse_value(self, given: str) -> tuple[str, float]:
        """Parse a single input value.
        Runs checks that the given value is valid and returns
        the parsed comparison and value."""

        try:
            val = given.replace(" ", "")
        except AttributeError as e:
            raise ValueError(f"Invalid input {given}, does not appear to be a string.") from e

        comparison = val[:2]
        if comparison not in ("<=", ">="):
            raise ValueError(f"Unknown constraint {val}, use either `<=` or `>=`.")

        try:
            value = float(val[2:])
        except ValueError as e:
            raise ValueError(f"Invalid value for constraint {val}, does not parse to float.") from e

        return comparison, value

    def terms(self) -> Iterator[str]:
        """Return names for all set terms as iterable."""
        return iter(name for name in self.all_terms() if getattr(self, name) is not None)

    @staticmethod
    def all_terms(order: int | None = None) -> tuple[str, ...]:
        """Return all float-terms."""
        return Detuning.all_terms(order)

    def __getitem__(self, item: str) -> str:
        if item not in self.terms():
            raise KeyError(f"'{item}' is not set in Constraints object.")
        return getattr(self, item)

    def __setitem__(self, item: str, value: str):
        if item not in self.all_terms():
            raise KeyError(f"'{item}' is not in the available terms of a Constraints object.")

        self._parse_value(value)
        setattr(self, item, value)

    def __setattr__(self, name, value):
        if name in self.all_terms() and value is not None:  # `None` is also fine and happens in __init__
            self._parse_value(value)
        object.__setattr__(self, name, value)

    def get_leq(self, item: str) -> tuple[int, float]:
        """Returns a tuple ``(sign, value)`` such that
        the given contraint is converted into a minimization constraint
        of the form ``sign * term <= value``.

        .. admonition:: Examples

            | ``"<=4"`` returns ``(1, 4)``
            | ``">=3"`` returns ``(-1, -3)``
            | ``">=-2"`` returns ``(-1, 2)``

        Values are rescaled if scale is set.

        Args:
            item (str): term name, e.g. ``"X10"``.
        """
        definition = self[item]
        comparison, value = self._parse_value(definition)
        sign = 1 if comparison == "<=" else -1

        if self.scale:
            value *= self.scale

        return sign, sign*value
