
"""
Classes for Detuning
--------------------

Classes used to hold and manipulate individual detuning (measurement) data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from omc3.utils.stats import weighted_mean

from ir_amplitude_detuning.utilities.common import StrEnum

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class MeasureValue:
    """ Class to hold a value with its error and do basic arithmetics.

    Args:
        value (float): value of the measurement
        error (float): error of the measurement, treated as standard deviation
    """
    value: float = 0
    error: float = 0

    def __add__(self, other: float | MeasureValue):
        if isinstance(other, float):
            if other:
                raise NotImplementedError(
                    "Addition of Measurements with scalar values other than 0 are not implemented.")
            return MeasureValue(value=self.value, error=self.error)

        return MeasureValue(value=self.value + other.value, error=np.sqrt(self.error**2 + other.error**2))

    def __radd__(self, other: MeasureValue | float):  # make sum work
        if isinstance(other, MeasureValue):
            return MeasureValue(value=self.value + other.value, error=np.sqrt(self.error**2 + other.error**2))
        return MeasureValue(value=self.value + other, error=self.error)

    def __sub__(self, other: MeasureValue | float):
        if isinstance(other, MeasureValue):
            return MeasureValue(value=self.value - other.value, error=np.sqrt(self.error**2 + other.error**2))
        return MeasureValue(value=self.value - other, error=self.error)

    def __neg__(self):
        return MeasureValue(value=-self.value, error=self.error)

    def __mul__(self, other: float):
        return MeasureValue(value=self.value * other, error=self.error * other)

    def __rmul__(self, other: float):
        return MeasureValue(value=self.value * other, error=self.error * other)

    def __truediv__(self, other: float):
        return MeasureValue(value=self.value / other, error=self.error / other)

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
        """ Returns rms of values and errors. """
        rms_values = np.sqrt(np.mean([m.value**2 for m in measurements]))
        rms_errors = np.sqrt(np.mean([m.error**2 for m in measurements]))
        return MeasureValue(
            value=rms_values,
            error=1/np.sqrt(len(measurements)) * rms_errors / rms_values,
        )

    @staticmethod
    def weighted_rms(measurements: Sequence[MeasureValue]):
        """ Returns rms of values and errors. """
        rms_values = np.sqrt(np.average([m.value**2 for m in measurements], weights=[1/m.error**2 for m in measurements]))
        rms_errors = np.sqrt(np.average([m.error**2 for m in measurements], weights=[1/m.error**2 for m in measurements]))
        return MeasureValue(
            value=rms_values,
            error=1/np.sqrt(len(measurements)) * rms_errors / rms_values,
        )

    @staticmethod
    def mean(measurements: Sequence[MeasureValue]):
        """ Returns mean of values and MAE. """
        return np.mean(measurements)

    @staticmethod
    def weighted_mean(measurements: Sequence[MeasureValue]):
        """ Returns a mean weighted by the errors and normal mean for errors."""
        values = np.array([m.value for m in measurements])
        errors = np.array([m.error for m in measurements])
        return MeasureValue(
            value=weighted_mean(data=values, errors=errors),
            # error=weighted_error(data=values, errors=errors),
            error=np.mean([m.error for m in measurements]),
        )

    @classmethod
    def from_value(cls, value: float | MeasureValue):
        if isinstance(value, float):
            return cls(value)

        # make a copy:
        return cls(value.value, value.error)


class FirstOrderTerm(StrEnum):
    X10: str = "X10"  # d Qx / d Jx
    X01: str = "X01"  # d Qx / d Jy
    Y10: str = "Y10"  # d Qy / d Jx
    Y01: str = "Y01"  # d Qy / d Jy


class SecondOrderTerm(StrEnum):
    X20: str = "X20"  # d^2 Qx / (d Jx)^2
    X11: str = "X11"  # d^2 Qx / (d Jx)(d Jy)
    X02: str = "X02"  # d^2 Qx / (d Jy)^2
    Y20: str = "Y20"  # d^2 Qy / (d Jx)^2
    Y11: str = "Y11"  # d^2 Qy / (d Jx)(d Jy)
    Y02: str = "Y02"  # d^2 Qy / (d Jy)^2


@dataclass(slots=True)
class Detuning:
    """ Class holding first and second order detuning values.
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

    def terms(self):
        """ Return names for all set terms."""
        return iter(name for name in self.all_terms() if getattr(self, name) is not None)

    def items(self):
        return iter((name, getattr(self, name)) for name in self.terms())

    @staticmethod
    def all_terms(order: int | None = None) -> tuple[str, ...]:
        """ Return all float-terms.

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

    def __getitem__(self, item):
        """ Convenience wrapper to access terms via `[]` .
        Not set terms will raise a KeyError.
        """
        if item not in self.terms():
            raise KeyError(f"'{item}' is not set in Detuning object.")
        return getattr(self, item)

    def __setitem__(self, item, value):
        """ Convenience wrapper to set terms via `[]` . """
        if item not in self.all_terms():
            raise KeyError(f"'{item}' is not in the available terms of a Detuning object.")
        return setattr(self, item, value)

    def __add__(self, other: Detuning):
        self._check_terms(other)
        return self.__class__(**{term: self[term] + other[term] for term in self.terms()})

    def __sub__(self, other: Detuning):
        self._check_terms(other)
        return self.__class__(**{term: self[term] - other[term] for term in self.terms()})

    def __neg__(self):
        return self.__class__(**{term: -self[term] for term in self.terms()})

    def __mul__(self, other: float | Detuning):
        if isinstance(other, Detuning):
            self._check_terms(other)
            return self.__class__(**{term: self[term] * other[term] for term in self.terms()})
        return self.__class__(**{term: self[term] * other for term in self.terms()})

    def __truediv__(self, other: float | Detuning):
        if isinstance(other, Detuning):
            self._check_terms(other)
            return self.__class__(**{term: self[term] / other[term] for term in self.terms()})
        return self.__class__(**{term: self[term] / other for term in self.terms()})

    def _check_terms(self, other: Detuning):
        not_in_other = [term for term in self.terms() if term not in other.terms()]
        if len(not_in_other):
            raise KeyError(
                f"Term '{not_in_other}' are not in the other detuning object. "
                f"Subtraction not possible."
            )

        not_in_self = [term for term in other.terms() if term not in self.terms()]
        if len(not_in_self):
            LOG.debug(
                f"Term '{not_in_self}' from the other object are not in this "
                f"detuning object. Terms ignored."
            )


@dataclass(slots=True)
class DetuningMeasurement(Detuning):
    """ Class holding first and second order detuning measurement values (i.e. with error)."""
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
                self[term] = MeasureValue(*self[term])

        Detuning.__post_init__(self)

    def get_detuning(self):
        """ Returns a Detuning object with the values (no errors) of this measurement. """
        return Detuning(**{term: self[term].value for term in self.terms()})

    @ classmethod
    def from_detuning(cls, detuning):
        """ Create a DetuningMeasurement from a Detuning object, with zero errors. """
        return cls(**{term: MeasureValue(detuning[term]) for term in detuning.terms()})


@dataclass(slots=True)
class Constraints:
    """ Class for holding detuning contraints.
    These are useful when trying to force a detuning term to have a specific sign,
    but not a specific value.
    Examples of this can be found in Fig. 1 of [DillyControllingLandauDamping2022]_.

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
        for t in self.terms():
            val = getattr(self, t)
            if val[:2] not in ("<=", ">="):
                raise ValueError(f"Unknown constraint {val}, use either `<=` or `>=`.")

    def terms(self) -> Iterator[str]:
        """ Return names for all set terms as iterable. """
        return iter(name for name in self.all_terms() if getattr(self, name) is not None)

    @staticmethod
    def all_terms(order: int | None = None) -> tuple[str, ...]:
        """ Return all float-terms. """
        return Detuning.all_terms(order)

    def __getitem__(self, item: str) -> str:
        if item not in self.terms():
            raise KeyError(f"'{item}' is not set in Constraints object.")
        return getattr(self, item)

    def __setitem__(self, item: str, value: str):
        if item not in self.all_terms():
            raise KeyError(f"'{item}' is not in the available terms of a Constraints object.")
        return setattr(self, item, value)

    def get_leq(self, item: str) -> tuple[int, float]:
        """ Returns a tuple ``(sign, value)`` such that
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
        str_item = self[item]
        sign = 1 if str_item[:2] == "<=" else -1
        value = float(str_item[2:])
        if self.scale:
            value = value * self.scale
        return sign, sign*value


# Default scaling is 1E3 as measurements are usually given in 1E3 m^-1
scaled_detuning = partial(Detuning, scale=1e3)
scaled_contraints = partial(Constraints, scale=1e3)
scaled_detuningmeasurement = partial(DetuningMeasurement, scale=1e3)
