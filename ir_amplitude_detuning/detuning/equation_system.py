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
ROW_ID: str = "b{beam}.ip{ip}.{term}"


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
    optics: TwissPerBeam | OpticsPerXing,
    target: Target,
    correctors: Correctors
    ) -> DetuningCorrectionEquationSystem:
    """ Build the full linear equation system of the form M * circuits = detuning.
    In its current form, this builds for decapole (_b5) and dodecapole (_b6) circuits for the ips
    given in the detuning_data (which are the targets).
    Filtering needs to be done afterwards.

    Args:
    """
    full_eqsys = DetuningCorrectionEquationSystem.create_empty(columns=correctors)
    for target_data in target.data:
        target_data: TargetData

        if target_data.xing is None:
            use_optics: TwissPerBeam = optics  # should be only one optics per beam
        else:
            try:
                use_optics: TwissPerBeam = optics[target_data.xing]
            except KeyError:
                raise ValueError(f"Optics for xing {target_data.xing} not found in optics dict.")

        use_correctors = [c for c in correctors if (c.ip in target_data.ips or c.ip is None)]

        eqsys = build_detuning_correction_matrix_per_entry(
            optics=use_optics,
            detuning_data=target_data,
            correctors=use_correctors,
        )

        full_eqsys.append_all(eqsys)

    return full_eqsys


def build_detuning_correction_matrix_per_entry(
    optics: TwissPerBeam,
    detuning_data: TargetData,
    correctors: Correctors
    ) -> DetuningCorrectionEquationSystem:
    """ Build the full linear equation system of the form M * circuits = detuning.

    Its building the equation system row-by-row, first for each detuning term, then for each constraint.
    Both beams are appended to the same system.

    Args:

    """
    ips_str = ips2str(detuning_data.ips)
    eqsys = DetuningCorrectionEquationSystem.create_empty(columns=correctors)

    for beam, twiss in optics.items():
        beam_detuning_data = detuning_data.get_detuning(beam)
        for term in beam_detuning_data.terms():
            m_row = calculate_matrix_row(beam, twiss, correctors, term)
            m_row.name = ROW_ID.format(beam=beam, ip=ips_str, term=term)

            eqsys.append_series_to_matrix(m_row)
            eqsys.set_value(m_row.name, beam_detuning_data[term])

        beam_constraints = detuning_data.get_contraints(beam)
        for term in beam_constraints.terms():
            m_row = calculate_matrix_row(beam, twiss, correctors, term)
            m_row.name = ROW_ID.format(beam=beam, ip=ips_str, term=term)

            sign, constraint_val = beam_constraints.get_leq(term)
            eqsys.append_series_to_constraints_matrix(sign*m_row)
            eqsys.set_constraint(m_row.name, constraint_val)
    return eqsys


def calculate_matrix_row(beam: int, twiss: pd.DataFrame, correctors: Correctors, term: DetuningTerm) -> pd.Series:
    """ Get one row of the full matrix for one beam and one detuning term.
    This is a wrapper to select the correct function depending on the order of the term.
    """
    match get_order(term):
        case 1:  # first order detuning -> generated by b4
            return build_b4_row(beam=beam, twiss=twiss, correctors=correctors, term=term)

        case 2:  # second order detuning -> generated by b6
            return build_b6_row(beam=beam, twiss=twiss, correctors=correctors, term=term)

        case order:
            raise NotImplementedError(f"Order {order:d} not implemented.")


def build_b4_row(beam: int, twiss: pd.DataFrame, correctors: Correctors, term: DetuningTerm) -> pd.Series:
    """ Builds one row of the (feed-down to) first order detuning matrix. """
    fields = get_fields(correctors)
    if not fields:
        raise ValueError("No detuning correctors defined!")

    if any(field not in list(FieldComponent) for field in fields):
        raise ValueError(f"Field must be one of {list(FieldComponent)}, got {fields}.")

    m = pd.Series(0., index=correctors)

    beam_sign = -1 if beam == 2 else 1
    symmetry_sign = 1 if beam % 2 else -1  # takes into account that K6 and K6 in B4/B2 have a minus sign as seen from B1

    for corrector in correctors:
        magnet = corrector.magnet

        beta = {p: twiss.loc[magnet, f"{BETA}{p}"] for p in "XY"}
        x = beam_sign * twiss.loc[magnet, "X"]                             # changes signs beam 4 -> beam 2
        y = twiss.loc[magnet, "Y"]                                         # same sign in beam 2 and beam 4
        coeff = get_detuning_coeff(term, beta)

        match corrector.field:
            case FieldComponent.b4:
                m[corrector] = symmetry_sign * coeff                        # b4 directly contributes
            case FieldComponent.b5:
                m[corrector] = x * coeff                                    # b5 feeddown to b4
            case FieldComponent.b6:
                m[corrector] = symmetry_sign * 0.5 * (x**2 - y**2) * coeff  # b6 feeddown to b4
    return m


def build_b6_row(beam: int, twiss: pd.DataFrame, correctors: Correctors, term: DetuningTerm, ips: Sequence[int]):
    """ Builds one row for the second order amplitude matrix.  """
    if FieldComponent.b6 not in get_fields(correctors):
        raise ValueError(f"Term {term} requested, but no b6 correctors defined!")

    m = pd.Series(0., index=correctors)
    symmetry_sign = 1 if beam % 2 else -1  # takes into account that K6L-B4/B2 has a minus sign as seen from B1

    for corrector in correctors:
        if corrector.field != FieldComponent.b6:  # only b6 contributes directly
            continue

        beta = {p: twiss.loc[corrector.magnet, f"{BETA}{p}"] for p in "XY"}
        m[corrector.magnet] = symmetry_sign * get_detuning_coeff(term, beta)
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


def ips2str(ips: Sequence[Any]) -> str:
    """ Convert a sequence of IPs into a string.

    Args:
        Sequence (Any): Sequence of IPs

    Returns:
        str: String of concatenated IPs
    """
    return ''.join(str(ip) for ip in ips)


def get_order(term: str) -> int:
    """ Get the order of the detuning, e.g. from X11 -> order 2, Y10 -> order 1.

    Args:
        term (str): 'X20', 'Y02', 'X11', 'Y20', 'Y11' or 'X02'
    """
    return int(term[1]) + int(term[2])
