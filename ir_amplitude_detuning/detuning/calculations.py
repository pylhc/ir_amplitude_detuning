"""
Detuning Calculations
---------------------

Functions to calculate detuning and its corrections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import cvxpy as cvx
import numpy as np
import pandas as pd

from ir_amplitude_detuning.detuning.equation_system import (
    build_detuning_correction_matrix,
    calculate_matrix_row,
)
from ir_amplitude_detuning.detuning.measurements import (
    FirstOrderTerm,
    SecondOrderTerm,
)
from ir_amplitude_detuning.utilities.misc import StrEnum, to_loop

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ir_amplitude_detuning.detuning.equation_system import (
        TwissPerBeam,
    )
    from ir_amplitude_detuning.utilities.classes_accelerator import (
        Correctors,
    )
    from ir_amplitude_detuning.utilities.classes_targets import Target


LOG = logging.getLogger(__name__)

FIELDS: str = "FIELDS"
IP: str = "IP"


class Method(StrEnum):
    cvxpy: str = "cvxpy"
    numpy: str = "numpy"


def calculate_correction(
        target: Target,
        method: Method = Method.cvxpy
    ) -> pd.Series[float]:
    """ Calculates the values for either kcdx or kctx as installed into the Dodecapol corrector.
    Returns a dictionary of circuit names and their settings in KNL values (i.e. needs to be divided by the lenght of the decapole corrector).

    In this function the equation system is named m * x = v, and everything contributing to the left hand side (i.e. the matrix m, or similarly, the contstriaints) is named with m_,
    and everything that contributes to the right hand side (i.e. the detuning values v, or similarly the constraint values) with v_.

    Args:
        target (Target): A Target object defining the target detuning and constraints.
        method (Method): The results of which method used to solve the equation system to be returned.

    Returns:
        pd.Series[float]: A Series of circuit names and their settings in KNL values.
    """
    # Check input ---

    if method not in list(Method):
        raise ValueError(f"Unknown method: {method}. Use one of: {list(Method)}")

    # Build equation system ---

    eqsys = build_detuning_correction_matrix(target)

    # Solve as convex system ---

    x = cvx.Variable(len(eqsys.m.columns))
    cost = cvx.sum_squares(eqsys.m.to_numpy() @ x - eqsys.v)  # ||Mx - v||_2
    if len(eqsys.v_constr):
        # Add constraints
        constr = eqsys.m_constr.to_numpy() @ x <= eqsys.v_constr.to_numpy()
        prob = cvx.Problem(cvx.Minimize(cost), [constr])
    else:
        # No constraints
        prob = cvx.Problem(cvx.Minimize(cost))
    prob.solve()
    if prob.status in ["infeasible", "unbounded"]:
        raise ValueError(f"Optimization failed! Reason: {prob.status}.")

    x_cvxpy = pd.Series(x.value, index=eqsys.m.columns)
    LOG.info(f"Result from cvxpy:\n{x_cvxpy}")

    # Solve via pseudo-inverse ---

    m_inverse = np.linalg.pinv(eqsys.m)
    x_numpy = m_inverse.dot(eqsys.v_meas)
    x_numpy = pd.Series(x_numpy, index=eqsys.m.columns)
    LOG.info(f"Result (with errors) from numpy:\n{x_numpy}")

    if method == Method.cvxpy:
        return x_cvxpy
    return x_numpy


def calc_effective_detuning(optics: TwissPerBeam, values: pd.Series) -> dict[int, pd.DataFrame]:
    """ Build a dataframe that calculates the detuning based on the given optics and corrector values
    individually for the given IPs and corrector fields.

    The detuning is "effective" as it is calculated from the pre-simulated optics.
    In contrast, for an exact detuning calculation the corrector values would need to be individually set,
    detuning gathered per PTC and then and compared to the unset detuning values.
    """
    correctors: Correctors = values.index

    loop_ips: list[Iterable[int]] = to_loop(sorted({c.ip for c in correctors if c.ip is not None}))
    ip_strings: list[str] = [''.join(map(str, ips)) for ips in loop_ips]

    loop_fields: list[str] = to_loop(sorted({c.field for c in correctors}))
    field_strings: list[str] = [''.join(map(str, fields)) for fields in loop_fields]

    dfs = {}
    for beam in optics:
        df = pd.DataFrame(
            index=pd.MultiIndex.from_product([field_strings, ip_strings], names=[FIELDS, IP]),
            columns=list(FirstOrderTerm) + list(SecondOrderTerm),
        )
        for fields, fields_str in zip(loop_fields, field_strings):
            for ips, ip_str in zip(loop_ips, ip_strings):
                filtered_correctors = [c for c in correctors if (c.ip in ips or c.ip is None) and (c.field in fields)]
                for term in df.columns:
                    m = calculate_matrix_row(beam, optics[beam], filtered_correctors, term)
                    detuning = m.dot(values.loc[filtered_correctors])
                    df.loc[(fields_str, ip_str), term] = detuning
        dfs[beam] = df.reset_index()
    return dfs
