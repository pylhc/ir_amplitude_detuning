"""
Detuning Calculations
---------------------

Functions to calculate detuning and its corrections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
from ir_amplitude_detuning.utilities.common import StrEnum, to_loop

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ir_amplitude_detuning.detuning.equation_system import (
        TwissPerBeam,
    )
    from ir_amplitude_detuning.detuning.targets import Target
    from ir_amplitude_detuning.utilities.correctors import (
        Correctors,
    )


LOG = logging.getLogger(__name__)

FIELDS: str = "FIELDS"
IP: str = "IP"


class Method(StrEnum):
    """ Methods to calculate the detuning corrections.
    Use ``cvxpy`` for the convex :class:`~cvxpy.Problem` solver,
    use the ``numpy`` method to calculate the corrections with the
    :func:`pseudo-inverse <numpy.linalg.pinv>`.
    """
    cvxpy: str = "cvxpy"
    numpy: str = "numpy"


def calculate_correction(
        target: Target,
        method: Method = Method.cvxpy
    ) -> pd.Series[float]:
    r""" Calculates the values to power the detuning correctors by solving
    the equation system of the form

    .. math::

        M_{\beta \text{-coefficients}} \times K_NL = V_\text{Detuning}

    the contents of which as described in more detail in the Chapter 7.2.2 of [DillyThesis2024]_ ,
    in particular Eq. (7.4).
    In addition, :class:`~ir_amplitude_detuning.detuning.measurements.Constraints`
    can be added to the correction equation system,
    which are respected when using the :class:`cvxpy.Problem` solver (method ``cvxpy``).

    In fact, this function always calculates the values with the :class:`convex solver<cvxpy.Problem>` (``cvxpy``),
    as well as with the :func:`pseudo-inverse <numpy.linalg.pinv>` method (``numpy``),
    but only returns the results of the method specified in `method`.
    The ``cvxpy`` method has the advantage that the constraints are respected,
    the ``numpy`` method has the advantage that the uncertainties are propagated.
    The function returns a series of correctors and their settings in KNL values, with uncertainties if available.

    In this function everything contributing to the left hand side of the equation system (i.e. the matrix *M*) is named ``m_*``,
    while everything that contributes to the right hand side (i.e. the detuning values *V*, or similarly the constraint values)
    is named ``v_*``.

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
