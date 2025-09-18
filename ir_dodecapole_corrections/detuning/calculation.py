from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeAlias

import cvxpy as cvx
import numpy as np
import pandas as pd
from tfs import TfsDataFrame

from ir_dodecapole_corrections.detuning.equation_system import (
    build_b4_row,
    build_b6_row,
    build_detuning_correction_matrix,
)
from ir_dodecapole_corrections.detuning.measurements import (
    FirstOrderTerm,
    SecondOrderTerm,
)
from ir_dodecapole_corrections.utilities.classes_accelerator import (
    Correctors,
    assert_corrector_fields,
)
from ir_dodecapole_corrections.utilities.misc import StrEnum

if TYPE_CHECKING:
    from ir_dodecapole_corrections.detuning.equation_system import (
        OpticsPerXing,
        TwissPerBeam,
    )
    from ir_dodecapole_corrections.utilities.classes_targets import Target


LOG = logging.getLogger(__name__)


class Method(StrEnum):
    cvxpy: str = "cvxpy"
    numpy: str = "numpy"


def calculate_correction(
        optics: TwissPerBeam | OpticsPerXing,
        target: Target,
        correctors: Correctors,
        method: Method = Method.cvxpy
    ) -> pd.Series[float]:
    """ Calculates the values for either kcdx or kctx as installed into the Dodecapol corrector.
    Returns a dictionary of circuit names and their settings in KNL values (i.e. needs to be divided by the lenght of the decapole corrector).

    In this function the equation system is named m * x = v, and everything contributing to the left hand side (i.e. the matrix m, or similarly, the contstriaints) is named with m_,
    and everything that contributes to the right hand side (i.e. the detuning values v, or similarly the constraint values) with v_.

    Args:
        optics (OpticsPerBeam | OpticsPerBeamPerXing): A dictionary of optics per beam or per xing per beam.
        target (Target): A Target object defining the target detuning and constraints.
        correctors (Correctors): A sequence of correctors to be used.
        method (Method): The results of which method used to solve the equation system to be returned.

    Returns:
        pd.Series[float]: A Series of circuit names and their settings in KNL values.
    """
    # Check input ---

    if method not in list(Method):
        raise ValueError(f"Unknown method: {method}. Use one of: {list(Method)}")
    assert_corrector_fields(correctors)

    # Build equation system ---

    eqsys = build_detuning_correction_matrix(optics, target, correctors)

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


def calc_effective_detuning(optics: TwissPerBeam, corrector_values: pd.Series, correctors: Correctors, ips) -> dict[int, TfsDataFrame]:
    """ Build a dataframe that calculates the detuning based on the given optics and corrector values
    individually for the given IPs and corrector fields.

    The detuning is "effective" as it is only based on the current optics.
    For a full detuning calculation the corrector values would need to be individually set,
    detuning gathered per PTC and then and compared to the unset detuning values.
    """
    loop_ips = [ips] + ([] if len(ips) == 1 else [[ip] for ip in ips])
    ip_strings = ['all' if len(current_ips) > 1 else str(current_ips[0]) for current_ips in loop_ips]
    fields_list = ('b5', 'b6', 'b5b6')

    dfs = [
        TfsDataFrame(
            index=pd.MultiIndex.from_product([fields_list, ip_strings], names=["FIELDS", "IP"]),
            columns=terms_first + terms_second,)
        for _ in enumerate(optics)
    ]
    for current_ips, ip_str in zip(loop_ips, ip_strings):
        m_first = {
            t: pd.DataFrame(
                [
                    build_b4_row(
                        beam,
                        optics[beam],
                        correctors=correctors,
                        term=t,
                        ips=current_ips,
                    )
                    for beam in optics
                ]
            )
            for t in list(FirstOrderTerm)
        }

        m_second = {
            t: pd.DataFrame(
                [
                    build_b6_row(
                        beam,
                        optics[beam],
                        magnet_pattern=DODECAPOLE_CORRECTOR,
                        term=t,
                        ips=current_ips,
                    )
                    for beam in optics
                ]
            )
            for t in terms_second
        }

        for m_order in (m_first, m_second):
            for term, m in m_order.items():
                for fields in fields_list:
                    columns_mask = (m.columns.str.endswith("b5") & ("5" in fields)) | (m.columns.str.endswith("b6") & ("6" in fields))
                    m_filtered = m.loc[:, columns_mask].rename(columns=circuits_map)
                    x_filtered = x.loc[m_filtered.columns]
                    v = m_filtered.dot(x_filtered)
                    for idx, value in enumerate(v):
                        dfs[idx].loc[(fields, ip_str), term] = value
                        # try:
                        #     dfs[idx].loc[(fields, ip_str), term] = value.value
                        # except AttributeError:
                        #     dfs[idx].loc[(fields, ip_str), term] = value
                        # else:
                        #     dfs[idx].loc[(fields, ip_str), f"ERR{term}"] = value.error
    return [df.reset_index() for df in dfs]