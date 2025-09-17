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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ir_dodecapole_corrections.detuning.targets import MeasureValue
from ir_dodecapole_corrections.utilities.classes_accelerator import (
    CIRCUIT,
    Correctors,
    get_fields,
    get_filled_corrector_attributes,
    sort_correctors,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tfs import TfsDataFrame

    from ir_dodecapole_corrections.utilities.classes_targets import Target, TargetData


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
    optics: dict[int, TfsDataFrame],
    target: Target,
    correctors: Correctors
    ) -> DetuningCorrectionEquationSystem:
    """ Build the full linear equation system of the form M * circuits = detuning.
    In its current form, this builds for decapole (_b5) and dodecapole (_b6) circuits for the ips
    given in the detuning_data (which are the targets).
    Filtering needs to be done afterwards.

    Args:
    """
    all_circuits = get_filled_corrector_attributes(target.ips, correctors, attribute=CIRCUIT)
    full_eqsys = DetuningCorrectionEquationSystem.create_empty(columns=all_circuits)
    for target_data in target.data:
        try:
            use_optics = optics[target_data.xing]  # optics for crossing scheme
        except KeyError:
            use_optics = optics  # optics for both beams

        eqsys = build_detuning_correction_matrix_per_entry(
            optics=use_optics,
            detuning_data=target_data,
            correctors=correctors
        )

        full_eqsys.append_all(eqsys)

    return full_eqsys


def build_detuning_correction_matrix_per_entry(
    optics: dict[int, TfsDataFrame],
    detuning_data: TargetData,
    correctors: Correctors
    ) -> DetuningCorrectionEquationSystem:
    """ Build the full linear equation system of the form M * circuits = detuning.

    Its building the equation system row-by-row, first for each detuning term, then for each constraint.
    Both beams are appended to the same system.

    Args:

    """
    ips = detuning_data.ips
    ips_str = ips2str(ips)
    all_circuits = get_filled_corrector_attributes(ips, correctors, attribute=CIRCUIT)
    eqsys = DetuningCorrectionEquationSystem.create_empty(columns=all_circuits)

    for beam, twiss in optics.items():
        for term in detuning_data.get_detuning(beam).terms():
            m_row = calculate_matrix_row(beam, twiss, correctors, term, ips)
            m_row.name = ROW_ID.format(beam=beam, ip=ips_str, term=term)

            eqsys.append_series_to_matrix(m_row)
            eqsys.set_value(m_row.name, detuning_data[beam][term])

        for term in detuning_data.get_contraints(beam).terms():
            m_row = calculate_matrix_row(beam, twiss, correctors, term, ips)
            m_row.name = ROW_ID.format(beam=beam, ip=ips_str, term=term)

            sign, constraint_val = detuning_data.get_contraints(beam).get_leq(term)
            eqsys.append_series_to_constraints_matrix(sign*m_row)
            eqsys.set_constraint(m_row.name, constraint_val)
    return eqsys


def calculate_matrix_row(beam: int, twiss: pd.DataFrame, correctors: Correctors, term: str, ips: Sequence[int]) -> pd.Series:
    """ Get one row of the full matrix for one beam and one detuning term.
    This is a wrapper to select the correct function depending on the order of the term.
    """
    match get_order(term):
        case 1:  # first order detuning -> generated by b4
            return build_b4_row(beam=beam, twiss=twiss, correctors=correctors, term=term, ips=ips)

        case 2:  # second order detuning -> generated by b6
            return build_b6_row(beam=beam, twiss=twiss, correctors=correctors, term=term, ips=ips)

        case order:
            raise NotImplementedError(f"Order {order:d} not implemented.")


def build_b4_row(beam: int, twiss: pd.DataFrame, correctors: Correctors, term: str, ips: Sequence[int]) -> pd.Series:
    """ Builds one row of the (feed-down to) first order detuning matrix. """
    fields = get_fields(correctors)
    if ("b4" not in fields) and ("b5" not in fields) and ("b6" not in fields):
        raise ValueError("No detuning correctors defined!")

    all_circuits = get_filled_corrector_attributes(ips, correctors, attribute=CIRCUIT)
    m = pd.Series(0., index=all_circuits)

    beam_sign = -1 if beam == 2 else 1
    symmetry_sign = 1 if beam % 2 else -1  # takes into account that K6 and K6 in B4/B2 have a minus sign as seen from B1

    for ip in ips:
        for side in "LR":
            for corrector in sort_correctors(correctors):
                magnet = corrector.magnet.format(side=side, ip=ip)
                circuit = corrector.circuit.format(side=side, ip=ip)

                beta = {p: twiss.loc[magnet, f"{BETA}{p}"] for p in "XY"}
                x = beam_sign * twiss.loc[magnet, "X"]                             # changes signs beam 4 -> beam 2
                y = twiss.loc[magnet, "Y"]                                         # same sign in beam 2 and beam 4
                coeff = get_detuning_coeff(term, beta)

                match corrector.field:
                    case "b4":
                        m[circuit] = symmetry_sign * coeff                        # b4 directly contributes
                    case "b5":
                        m[circuit] = x * coeff                                    # b5 feeddown to b4
                    case "b6":
                        m[circuit] = symmetry_sign * 0.5 * (x**2 - y**2) * coeff  # b6 feeddown to b4
    return m


def build_b6_row(beam: int, twiss: pd.DataFrame, correctors: Correctors, term: str, ips: Sequence[int]):
    """ Builds one row for the second order amplitude matrix.  """
    if "b6" not in get_fields(correctors):
        raise ValueError(f"Term {term} requested, but no b6 correctors defined!")

    all_circuits = get_filled_corrector_attributes(ips, correctors, attribute=CIRCUIT)
    m = pd.Series(0., index=all_circuits)
    symmetry_sign = 1 if beam % 2 else -1  # takes into account that K6L-B4/B2 has a minus sign as seen from B1

    for ip in ips:
        for side in "LR":
            for corrector in sort_correctors(correctors):
                if corrector.field != "b6":  # only b6 contributes directly
                    continue

                magnet = corrector.magnet.format(side=side, ip=ip)
                circuit = corrector.circuit.format(side=side, ip=ip)

                beta = {p: twiss.loc[magnet, f"{BETA}{p}"] for p in "XY"}
                m[circuit] = symmetry_sign * get_detuning_coeff(term, beta)
    return m


def get_detuning_coeff(term: str, beta: dict[str, float]) -> float:
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
    if term in ("X10", "Y01"):
        return beta[term[0]]**2 / (32 * np.pi)

    # cross term:
    if term in ("X01", "Y10"):
        return -beta["X"] * beta["Y"] / (16 * np.pi)

    # Second Order ---
    # direct terms
    if term in ("X20",):
        return beta[term[0]]**3 / (384 * np.pi)

    if term in ("Y02",):
        return -beta[term[0]]**3 / (384 * np.pi)

    # Cross- and Diagonal- Terms
    if term in ("X11", "Y20"):
        return -beta["X"]**2 * beta["Y"] / (128 * np.pi)

    if term in ("Y11", "X02"):
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
