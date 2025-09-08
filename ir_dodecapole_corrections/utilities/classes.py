"""
Classes
-------

Classes used to hold and manipulate data.
"""
from __future__ import annotations

from collections.abc import Iterator
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from omc3.utils.stats import weighted_mean

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class MeasureValue:
    """ Class to hold a value with its error and do basic arithmetics. """
    value: float = 0
    error: float = 0

    def __add__(self, other: float | MeasureValue):
        if isinstance(other, float):
            if other:
                raise NotImplementedError(
                    "Addition of Measurements with scalar values other than 0 are not implemented.")
            return MeasureValue(value=self.value, error=self.error)

        return MeasureValue(value=self.value + other.value, error=np.sqrt(self.error**2 + other.error**2))

    def __radd__(self, other: float):  # make sum work
        if other:
            raise NotImplementedError("Addition of Measurements with scalar values other than 0 are not implemented.")
        return MeasureValue(value=self.value, error=self.error)

    def __sub__(self, other: 'MeasureValue'):
        return MeasureValue(value=self.value - other.value, error=np.sqrt(self.error**2 + other.error**2))

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

    def __repr__(self):
        return str(self)

    def to_list(self):
        return [self.value, self.error]

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
        rms_errors = 0  # TODO
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


@dataclass(slots=True)
class Detuning:
    """ Class holding first and second order detuning values.
    The values are only returned via `__getitem__` or `terms()` if their
    values are set.
    For convenience, the input values are scaled by scale."""
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
    scale: float = 1.

    def __post_init__(self):
        if self.scale:
            for term in self.terms():
                self[term] = self[term] * self.scale

    def terms(self):
        """ Return names for all set terms."""
        return iter(name for name in self.fieldnames() if getattr(self, name) is not None)

    @staticmethod
    def fieldnames(order=None):
        """ Return all float-terms. """
        # return iter(field.name for field in fields(self) if field.type is float)
        mapping = {
            1: ("X10", "X01", "Y10", "Y01"),
            2: ("X20", "X11", "X02", "Y20", "Y11", "Y02"),
        }
        if order:
            return mapping[order]
        return tuple(e for m in mapping.values() for e in m)

    def __getitem__(self, item):
        if item not in self.terms():
            raise KeyError(f"'{item}' is not set in Detuning object.")
        return getattr(self, item)

    def __setitem__(self, item, value):
        if item not in self.fieldnames():
            raise KeyError(f"'{item}' is not in the available terms of a Detuning object.")
        return setattr(self, item, value)

    def __add__(self, other: Detuning):
        self._check_terms(other)
        return self.__class__(**{term: self[term] + other[term] for term in self.terms()})

    def __sub__(self, other: Detuning):
        self._check_terms(other)
        return self.__class__(**{term: self[term] - other[term] for term in self.terms()})

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
    """ Class holding first and second order detuning measurement values (i.e. with error).
    The values are only returned via `__getitem__` or `terms()` if their
    values are set."""
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
    """ Class holding first order detuning contraints.
    Thet are only returned via `__getitem__` or `terms()` if they are set.
    So far only ">=" and "<=" are implemented.
    E.g. X10 = "<=0"
    """
    X10: str | None = None
    X01: str | None = None
    Y10: str | None = None
    Y01: str | None = None
    scale: float = 1

    def __post_init__(self):
        for t in self.terms():
            val = getattr(self, t)
            if val[:2] not in ("<=", ">="):
                raise ValueError(f"Unknown constraint {val}, use either `<=` or `>=`.")

    def terms(self) -> Iterator[str]:
        """ Return names for all set terms as iterable. """
        return iter(name for name in self.fieldnames() if getattr(self, name) is not None)

    def fieldnames(self) -> tuple[str, ...]:
        """ Return all float-terms. """
        return "X10", "X01", "Y10", "Y01"

    def __getitem__(self, item: str) -> str:
        if item not in self.terms():
            raise KeyError(f"'{item}' is not set in Constraints object.")
        return getattr(self, item)

    def __setitem__(self, item: str, value: str):
        if item not in self.fieldnames():
            raise KeyError(f"'{item}' is not in the available terms of a Constraints object.")
        return setattr(self, item, value)

    def get_leq(self, item: str) -> tuple[int, float]:
        """ Returns a tuple (sign, value) such that given the constraint
        is sign*term <= value.

        Example: X10="<=4" returns (1, 4)
                 X01=">=-2" returns (-1, 2)

        Values are rescaled if scale is set.

        Args:
            item (str): term name, e.g. "X10"
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


@dataclass(slots=True)
class TargetData:
    """ Class to hold the Data of a Target.
        Single IPs are converted into a tuple automatically.
        The detuning values should be a dictionary defining the
        beams. Beam 2 and Beam 4 are used interchangeably:
        if only one is defined its values will be returned when
        the other one is requested.
     """
    ips: Sequence[int] | int
    detuning: dict[int, Detuning]
    constraints: dict[int, Constraints] = None
    xing: str = None
    MAIN_XING: str = field(init=False, default='main')

    def __post_init__(self):
        if self.xing is None:
            self.xing = self.MAIN_XING

        if isinstance(self.ips, int):
            self.ips = (self.ips,)

    def beams(self) -> list[int]:
        return list(self.detuning.keys())

    def __getitem__(self, beam):
        try:
            return self.detuning[beam]
        except KeyError as e:
            if beam == 2 and (4 in self.beams()):
                return self.detuning[4]
            if beam == 4 and (2 in self.beams()):
                return self.detuning[2]

            LOG.debug(f"Beam {beam} not defined. Returning empty detuning definition")
            return Detuning()

    def constraints_beams(self) -> list[int]:
        if self.constraints is None:
            return []
        return list(self.constraints.keys())

    def get_contraints(self, beam: int) -> Constraints:
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
    """ Class to hold Target information.

        The data is a TargetData or list of TargetData containing the data for this Target.
        This is done so that the TargetData can be split up depending on which IPs are targeted.
        Single TargetData are converted into a List automatically.
    """
    name: str
    data: Sequence[TargetData] | TargetData

    def __post_init__(self):
        if isinstance(self.data, TargetData):
            self.data = [self.data]
        self.ips = [ip for data in self.data for ip in data.ips]

        if len(self.ips) != len(set(self.ips)):
            LOG.debug(f"Duplicate ips found for {self.name}")
            self.ips = set(self.ips)
