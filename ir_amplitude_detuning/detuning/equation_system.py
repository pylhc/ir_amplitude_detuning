"""
Equation System
---------------

This module contains the functions to generate the terms to calculate detuning, including feed-down,
and uses them, together with the detuning targets, to build the equation system.
These can then be solved to calculate corrections.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

from ir_amplitude_detuning.detuning.measurements import (
    Constraints,
    Detuning,
    FirstOrderTerm,
    MeasureValue,
    SecondOrderTerm,
)
from ir_amplitude_detuning.utilities.classes_accelerator import (
    Correctors,
    FieldComponent,
    get_fields,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tfs import TfsDataFrame

    from ir_amplitude_detuning.utilities.classes_targets import Target, TargetData

    DetuningTerm: TypeAlias = FirstOrderTerm | SecondOrderTerm
    TwissPerBeam: TypeAlias = dict[int, TfsDataFrame]
    OpticsPerXing: TypeAlias = dict[str, TwissPerBeam]


LOG = logging.getLogger(__name__)

BETA: str = "BET"
ROW_ID: str = "b{beam}{ip}.{term}"


@dataclass(slots=True)
class DetuningCorrectionEquationSystem:
    """ Class to hold the equation system for detuning correction.

    Attributes:
        m (pd.DataFrame): Coefficient matrix
        v (pd.Series): Detuning vector
        m_constr (pd.DataFrame): Coefficient matrix for constraints
        v_constr (pd.Series): Detuning vector for constraints
        v_meas (pd.Series): Detuning vector keeping uncertainties if given
    """
    m: pd.DataFrame
    m_constr: pd.DataFrame
    v: pd.Series
    v_constr: pd.Series
    v_meas: pd.Series

    @classmethod
    def create_empty(cls, columns: Sequence | None = None) -> DetuningCorrectionEquationSystem:
        return cls(
            m = pd.DataFrame(columns=columns, dtype=float),
            v = pd.Series(dtype=float),
            v_meas = pd.Series(dtype=float),  # cannot use dtype MeasureValue
            m_constr = pd.DataFrame(columns=columns, dtype=float),
            v_constr = pd.Series(dtype=float),
        )

    def append_series_to_matrix(self, series: pd.Series) -> None:
        """ Append a series as a new row to the m matrix. """
        self.m = pd.concat([self.m, series.to_frame().T], axis=0)

    def append_series_to_constraints_matrix(self, series: pd.Series) -> None:
        """ Append a series as a new row to the m_constr matrix. """
        self.m_constr = pd.concat([self.m_constr, series.to_frame().T], axis=0)

    def set_value(self, name: str, value: float | MeasureValue) -> None:
        """ Set a value in the values and measurement values (with error if there). """
        self.v.loc[name] = getattr(value, "value", value)
        self.v_meas.loc[name] = MeasureValue.from_value(value)

    def set_constraint(self, name: str, value: float) -> None:
        """ Set a value in the constraint values. """
        self.v_constr.loc[name] = value

    def append_all(self, other: DetuningCorrectionEquationSystem) -> None:
        """ Append all matrices and vectors from another equation system. """
        for field in fields(self):
            attr = field.name
            new_value = pd.concat([getattr(self, attr), getattr(other, attr)], axis=0)
            setattr(self, attr, new_value)


def build_detuning_correction_matrix(
    target: Target,
    ) -> DetuningCorrectionEquationSystem:
    """ Build the full linear equation system of the form M * circuits = detuning.
    In its current form, this builds for decapole (_b5) and dodecapole (_b6) circuits for the ips
    given in the detuning_data (which are the targets).
    Filtering needs to be done afterwards.

    Args:
    """
    full_eqsys = DetuningCorrectionEquationSystem.create_empty(columns=target.correctors)
    for target_data in target.data:
        target_data: TargetData
        eqsys = build_detuning_correction_matrix_per_entry(target_data)
        full_eqsys.append_all(eqsys)
    return full_eqsys


def build_detuning_correction_matrix_per_entry(target_data: TargetData) -> DetuningCorrectionEquationSystem:
    """ Build a part of the full linear equation system of the form M * circuits = detuning,
    for the given TargetData.

    Its building the equation system row-by-row, first for each detuning term, then for each constraint.
    Both beams are appended to the same system.

    Args:

    """
    ips_str = ips2str(target_data.ips)
    correctors = target_data.correctors
    eqsys = DetuningCorrectionEquationSystem.create_empty(columns=correctors)

    for beam in target_data.beams:
        twiss = target_data.optics[beam]
        detuning_data: Detuning = target_data.detuning[beam]
        for term in detuning_data.terms():
            m_row = calculate_matrix_row(beam, twiss, correctors, term)
            m_row.name = ROW_ID.format(beam=beam, ip=ips_str, term=term)

            eqsys.append_series_to_matrix(m_row)
            eqsys.set_value(m_row.name, detuning_data[term])

        constraints: Constraints = target_data.constraints[beam]
        for term in constraints.terms():
            m_row = calculate_matrix_row(beam, twiss, correctors, term)
            m_row.name = ROW_ID.format(beam=beam, ip=ips_str, term=term)

            sign, constraint_val = constraints.get_leq(term)
            eqsys.append_series_to_constraints_matrix(sign*m_row)
            eqsys.set_constraint(m_row.name, constraint_val)
    return eqsys


def calculate_matrix_row(beam: int, twiss: pd.DataFrame, correctors: Correctors, term: DetuningTerm) -> pd.Series:
    """ Get one row of the full matrix for one beam and one detuning term.
    This is a wrapper to select the correct function depending on the order of the term.
    """
    # Check order of amplitude detuning
    order = get_order(term)
    if order not in (1, 2):
        raise NotImplementedError(f"Order must be 1 or 2, got {order}")

    # Check that all fields are defined
    fields = get_fields(correctors)
    if not fields:
        raise ValueError("No detuning correctors defined!")

    if any(field not in list(FieldComponent) for field in fields):
        raise ValueError(f"Field must be one of {list(FieldComponent)}, got {fields}.")

    if order == 2 and FieldComponent.b6 not in fields:
        raise ValueError(f"Term {term} requested, but no b6 correctors defined!")

    # Build row ---
    m = pd.Series(0., index=correctors)

    beam_sign = beam_direction(beam)
    symmetry_sign = magnet_symmetry_sign(beam)

    for corrector in correctors:
        magnet = corrector.magnet

        # skip if magnet not in twiss, e.g. a corrector for a specific beam
        if magnet not in twiss.index:
            LOG.debug(f"Skipping {corrector}, magnet {magnet} not in twiss table.")
            continue

        beta = {p: twiss.loc[magnet, f"{BETA}{p}"] for p in "XY"}
        coeff = get_detuning_coeff(term, beta)

        match order:
            case 1:
                x = beam_sign * twiss.loc[magnet, "X"]                             # changes signs beam 4 -> beam 2
                y = twiss.loc[magnet, "Y"]                                         # same sign in beam 2 and beam 4

                match corrector.field:
                    case FieldComponent.b4:
                        m[corrector] = symmetry_sign * coeff                        # b4 directly contributes
                    case FieldComponent.b5:
                        m[corrector] = x * coeff                                    # b5 feeddown to b4
                    case FieldComponent.b6:
                        m[corrector] = symmetry_sign * 0.5 * (x**2 - y**2) * coeff  # b6 feeddown to b4

            case 2:
                match corrector.field:
                    case FieldComponent.b6:
                        m[corrector] = symmetry_sign * coeff                        # b6 directly contributes
                    case _:
                        continue                                                    # other fields do not contribute
    return m


def get_detuning_coeff(term: DetuningTerm, beta: dict[str, float]) -> float:
    """ Get the coefficient for first and second order amplitude detuning.

    Args:
        term (str): 'X20', 'Y02', 'X11', 'Y20', 'Y11' or 'X02'
        beta (dict[str, float]): Dictionary of planes (uppercase)and values.

    Returns:
        float: The detuning coefficient for the given term, calculated from the betas.
    """
    term = term.upper()
    # First Order ---
    # direct terms:
    if term in (FirstOrderTerm.X10, FirstOrderTerm.Y01):
        return beta[term[0]]**2 / (32 * np.pi)

    # cross term:
    if term in (FirstOrderTerm.X01, FirstOrderTerm.Y10):
        return -beta["X"] * beta["Y"] / (16 * np.pi)

    # Second Order ---
    # direct terms
    if term == SecondOrderTerm.X20:
        return beta["X"]**3 / (384 * np.pi)

    if term == SecondOrderTerm.Y02:
        return -beta["Y"]**3 / (384 * np.pi)

    # Cross- and Diagonal- Terms
    if term in (SecondOrderTerm.X11, SecondOrderTerm.Y20):
        return -beta["X"]**2 * beta["Y"] / (128 * np.pi)

    if term in (SecondOrderTerm.Y11, SecondOrderTerm.X02):
        return beta["X"] * beta["Y"]**2 / (128 * np.pi)

    raise KeyError(f"Unknown Term {term}")


def magnet_symmetry_sign(beam: int) -> int:
    """ Sign to be used for magnets that are anti-symmetric under beam direction
    change, e.g. K4(L) and K6(L) in beam 2 and beam 4 will have opposite sign.

    Args:
        beam (int): Beam number

    Returns:
        int: 1 or -1
    """
    return 1 if beam % 2 else -1


def beam_direction(beam: int) -> int:
    """ Get the direction of the beam.

    Args:
        beam (int): Beam number

    Returns:
        int: 1 or -1
    """
    return -1 if beam == 2 else 1


def ips2str(ips: Sequence[Any]) -> str:
    """ Convert a sequence of IPs into a string.

    Args:
        Sequence (Any): Sequence of IPs

    Returns:
        str: String of concatenated IPs
    """
    if not ips:
        return ''
    return f".ip{''.join(str(ip) for ip in ips)}"


def get_order(term: str) -> int:
    """ Get the order of the detuning, e.g. from X11 -> order 2, Y10 -> order 1.

    Args:
        term (str): 'X20', 'Y02', 'X11', 'Y20', 'Y11' or 'X02'
    """
    return int(term[1]) + int(term[2])
