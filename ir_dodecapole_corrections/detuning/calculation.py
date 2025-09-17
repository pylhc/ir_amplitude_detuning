from __future__ import annotations

from enum import Enum
import logging
import cvxpy as cvx
import numpy as np
import pandas as pd
from tfs import TfsDataFrame

from ir_dodecapole_corrections.detuning.equation_system import (
    build_b4_row,
    build_b6_row,
    build_detuning_correction_matrix,
)
from ir_dodecapole_corrections.detuning.targets import DetuningMeasurement
from ir_dodecapole_corrections.utilities.classes_accelerator import Correctors, get_fields
from ir_dodecapole_corrections.utilities.classes_targets import Target
from ir_dodecapole_corrections.utilities.misc import StrEnum

optics_per_beam = dict[int, TfsDataFrame]
optics_per_xing = dict[str, optics_per_beam]
AllOptics = optics_per_xing | optics_per_beam

LOG = logging.getLogger(__name__)


class Method(StrEnum):
    cvxpy: str = "cvxpy"
    numpy: str = "numpy"


def calculate_correction(optics: AllOptics, target: Target, correctors: Correctors, method: Method = Method.cvxpy) -> pd.Series[float]:
    """ Calculates the values for either kcdx or kctx as installed into the Dodecapol corrector.
    Returns a dictionary of circuit names and their settings in KNL values (i.e. needs to be divided by the lenght of the decapole corrector).

    In this function the equation system is named m * x = v, and everything contributing to the left hand side (i.e. the matrix m, or similarly, the contstriaints) is named with m_,
    and everything that contributes to the right hand side (i.e. the detuning values v, or similarly the constraint values) with v_.
    """
    if method not in list(Method):
        raise ValueError(f"Unknown method: {method}. Use one of: {list(Method)}")

    fields = get_fields(correctors)
    if ("b4" not in fields) and ("b5" not in fields) and ("b6" not in fields):
        raise ValueError("No detuning correctors defined!")

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

    LOG.info(f"Values from cvxpy: {x.value}")

    # Solve via pseudo-inverse ---
    m_inverse = np.linalg.pinv(eqsys.m)
    x_pseudo = m_inverse.dot(eqsys.v_meas)
    LOG.info(f"Values with errors:\n {x_pseudo}")  # to check against cvxpy values

    if method == Method.cvxpy:
        return pd.Series(x.value, index=eqsys.m.columns)

    return pd.Series(x_pseudo, index=eqsys.m.columns)


def calc_effective_detuning(optics, corrector_values, ips):
    circuits_map = get_knl_to_b5b6_circuit_map_for_correctors(ips)
    terms_first = DetuningMeasurement.fieldnames(order=1)
    terms_second = DetuningMeasurement.fieldnames(order=2)

    loop_ips = [ips] + ([] if len(ips) == 1 else [[ip] for ip in ips])
    ip_strings = ['all' if len(current_ips) > 1 else str(current_ips[0]) for current_ips in loop_ips]
    fields_list = ('b5', 'b6', 'b5b6')

    x = pd.Series(0., index=circuits_map.values())
    x.update(corrector_values)

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
                        magnet_pattern=DODECAPOLE_CORRECTOR,
                        term=t,
                        ips=current_ips,
                    )
                    for beam in optics
                ]
            )
            for t in terms_first
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