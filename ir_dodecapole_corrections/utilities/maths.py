from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ir_dodecapole_corrections.utilities.classes_detuning import Detuning, DetuningMeasurement


def get_sum(
    meas_a: dict[Any, DetuningMeasurement], meas_b: dict[Any, DetuningMeasurement]
) -> dict[Any, DetuningMeasurement]:
    """ Add the values of two dicts for each entry in the dicts. """
    return {beam: meas_a[beam] + meas_b[beam] for beam in meas_a}


def get_diff(
    meas_a: dict[Any, DetuningMeasurement], meas_b: dict[Any, DetuningMeasurement]
) -> dict[Any, DetuningMeasurement]:
    """ Subtract the values of meas_b from meas_a for each entry in the dicts. """
    return {beam: meas_a[beam] - meas_b[beam] for beam in meas_a}


def get_detuning(meas: dict[Any, DetuningMeasurement]) -> dict[Any, Detuning]:
    """ Get detuning values (i.e. without errors) from the measurement data."""
    return {beam: meas[beam].get_detuning() for beam in meas}
