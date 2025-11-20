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
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import pandas as pd

from ir_amplitude_detuning.detuning.measurements import (
    Constraints,
    Detuning,
    MeasureValue,
)
from ir_amplitude_detuning.detuning.terms import (
    DetuningTerm,
    FirstOrderTerm,
    SecondOrderTerm,
    get_order,
)
from ir_amplitude_detuning.utilities.correctors import (
    Correctors,
    FieldComponent,
    get_fields,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tfs import TfsDataFrame

    from ir_amplitude_detuning.detuning.targets import Target, TargetData

    TwissPerBeam: TypeAlias = dict[int, TfsDataFrame]
    OpticsPerXing: TypeAlias = dict[str, TwissPerBeam]


LOG = logging.getLogger(__name__)

BETA: str = "BET"
ROW_ID: str = "b{beam}.{label}.{term}"


@dataclass(slots=True)
class DetuningCorrectionEquationSystem:
    r"""Class to hold the equation system for detuning correction.
    The equation system is of the form

    .. math::

        M_{\beta \text{-coefficients}} \times K_NL = V_\text{Detuning}

    as described in the Chapter 7.2.2 of [DillyThesis2024]_.

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
        """Append a series as a new row to the m matrix."""
        self.m = pd.concat([self.m, series.to_frame().T], axis=0)

    def append_series_to_constraints_matrix(self, series: pd.Series) -> None:
        """Append a series as a new row to the m_constr matrix."""
        self.m_constr = pd.concat([self.m_constr, series.to_frame().T], axis=0)

    def set_value(self, name: str, value: float | MeasureValue) -> None:
        """Set a value in the values and measurement values (with error if there)."""
        self.v.loc[name] = getattr(value, "value", value)
        self.v_meas.loc[name] = MeasureValue.from_value(value)

    def set_constraint(self, name: str, value: float) -> None:
        """Set a value in the constraint values."""
        self.v_constr.loc[name] = value

    def append_all(self, other: DetuningCorrectionEquationSystem) -> None:
        """Append all matrices and vectors from another equation system."""
        for field in fields(self):
            attr = field.name
            new_value = pd.concat([getattr(self, attr), getattr(other, attr)], axis=0)
            setattr(self, attr, new_value)

    def fillna(self) -> None:
        """Fill the NaN in the matrices with zeros."""
        self.m = self.m.fillna(0.)
        self.m_constr = self.m_constr.fillna(0.)


def build_detuning_correction_matrix(
    target: Target,
    ) -> DetuningCorrectionEquationSystem:
    """Build the full linear equation system of the form M * circuits = detuning.
    In its current form, this builds for decapole (_b5) and dodecapole (_b6) circuits for the ips
    given in the detuning_data (which are the targets).
    Filtering needs to be done afterwards.

    Args:
        target (Target): The target to build the equation system for.

    Returns:
        DetuningCorrectionEquationSystem: The full equation system as object defined above for the given target,
        build from the individual equation systems for each target data.
    """
    full_eqsys = DetuningCorrectionEquationSystem.create_empty(columns=target.correctors)

    for target_data in target.data:
        target_data: TargetData
        eqsys = build_detuning_correction_matrix_per_entry(target_data)
        full_eqsys.append_all(eqsys)

    full_eqsys.fillna()
    return full_eqsys


def build_detuning_correction_matrix_per_entry(target_data: TargetData) -> DetuningCorrectionEquationSystem:
    """Build a part of the full linear equation system of the form M * circuits = detuning,
    for the given TargetData.

    Its building the equation system row-by-row, first for each detuning term, then for each constraint.
    Both beams are appended to the same system.

    Args:
        target_data (TargetData): The target to build the equation system for.

    Returns:
        DetuningCorrectionEquationSystem: The equation system as object defined above,
        but containing only the rows for the given target_data.
    """
    correctors = target_data.correctors
    eqsys = DetuningCorrectionEquationSystem.create_empty(columns=correctors)

    for beam in target_data.beams:
        twiss = target_data.optics[beam]
        detuning_data: Detuning = target_data.detuning[beam]
        for term in detuning_data.terms():
            m_row = calculate_matrix_row(beam, twiss, correctors, term)
            m_row.name = ROW_ID.format(beam=beam, label=target_data.label, term=term)

            eqsys.append_series_to_matrix(m_row)
            eqsys.set_value(m_row.name, detuning_data[term])

        constraints: Constraints = target_data.constraints[beam]
        for term in constraints.terms():
            m_row = calculate_matrix_row(beam, twiss, correctors, term)
            m_row.name = ROW_ID.format(beam=beam, label=target_data.label, term=term)

            sign, constraint_val = constraints.get_leq(term)
            eqsys.append_series_to_constraints_matrix(sign*m_row)
            eqsys.set_constraint(m_row.name, constraint_val)
    return eqsys


def calculate_matrix_row(beam: int, twiss: pd.DataFrame, correctors: Correctors, term: DetuningTerm) -> pd.Series:
    """Get one row of the full matrix for one beam and one detuning term.
    This is a wrapper to select the correct function depending on the order of the term.
    Feed-down to b4 is calculated as in Eq. (7.2) of [DillyThesis2024]_

    Args:
        beam (int): The beam to calculate the row for.
        twiss (pd.DataFrame): The twiss/optics of the beam.
        correctors (Correctors): The correctors to calculate the row for.
        term (DetuningTerm): The term to calculate the row for.

    Returns:
        pd.Series: The row of the matrix.
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

    symmemtry_sign = beam_symmetry_sign(beam)

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
                x = twiss.loc[magnet, "X"]
                y = twiss.loc[magnet, "Y"]

                match corrector.field:
                    case FieldComponent.b4:
                        m[corrector] = symmemtry_sign * coeff                        # b4 directly contributes
                    case FieldComponent.b5:
                        m[corrector] = x * coeff                                     # b5 feeddown to b4
                    case FieldComponent.b6:
                        m[corrector] = symmemtry_sign * 0.5 * (x**2 - y**2) * coeff  # b6 feeddown to b4

            case 2:
                match corrector.field:
                    case FieldComponent.b6:
                        m[corrector] = symmemtry_sign * coeff                        # b6 directly contributes
                    case _:
                        continue                                                     # other fields do not contribute
    return m


def get_detuning_coeff(term: DetuningTerm, beta: dict[str, float]) -> float:
    """Get the coefficient for first and second order amplitude detuning,
    Eqs. (7.1) and (7.3) of [DillyThesis2024]_ respectively.

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


def beam_symmetry_sign(beam: int) -> int:
    """Sign to be used for magnets whose fields are anti-symmetric under beam direction change,
     e.g. K4(L) and K6(L) in beam 2 will have opposite sign compared to beam 1.

    This is needed, as we calculate the detuning of each beam independently (hence the feed-down
    is calculated with the offset as seen from the beam itself), but have a common magnet (the corrector) to be powered,
    whose field might look different depending on from which side you go through it.

    .. warning::
       This assumes that the magnet powering (which is what we calculate) is implemented with a positive sign
       in beam 1 and - when the field is anti-symmetric - a negative sign in beam 4.

    Args:
        beam (int): Beam number

    Returns:
        int: 1 or -1
    """
    return 1 if beam % 2 else -1
