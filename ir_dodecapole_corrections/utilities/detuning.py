"""
Calculate Detuning from Feed-down
---------------------------------

This module contains the functions to calculate the detuning stemming
from high orders via feed-down and build the coefficient matrices.
These can then directly used to calculate corrections.

TODO: This should be cleaned up and some additional features added:
a) Make usable for all amplitude detuning and for any corrector: b4, b5, a5, b6, a6
b) Allow for independent correctors, e.g. DECAPOLE and DODECAPOLE correctors
   as in the HL-LHC.
c) A class mapping circuit/corrector magnet, so that you don't need to add
   "_b5" or "_b6"
d) the feed-down matrix calculations should be only for what is given,
   i.e. using the class from a) and determining the feed-down.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import pandas as pd
import tfs
from tfs import TfsDataFrame

from ir_dodecapole_corrections.utilities.classes import (
    DetuningMeasurement,
    Target,
    TargetData,
)
from ir_dodecapole_corrections.utilities.latex import print_correction_and_error_as_latex

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG = logging.getLogger(__name__)

DODECAPOLE_CORRECTOR = "MCTX.3{side}{ip}"
DODECAPOLE_PATTERN = "MCTX.*[15]$"
DODECAPOLE_CIRCUIT = "kctx3.{side}{ip}"
DECAPOLE_CIRCUIT = "kcdx3.{side}{ip}"


# Feed-Down Calculations -------------------------------------------------------

def get_order(term: str):
    """ Get the order of the detuning, e.g. from X11 -> order 2, Y10 -> order 1."""
    return int(term[1]) + int(term[2])


def build_detuning_correction_matrix(optics: dict[int, TfsDataFrame], detuning_data: TargetData, magnet_pattern: str):
    """ Build the full linear equation system of the form M * circuits = detuning.
    In its current form, this builds for decapole (_b5) and dodecapole (_b6) circuits for the ips
    given in the detuning_data (which are the targets).
    Filtering needs to be done afterwards.
    """
    ips = detuning_data.ips
    ips_str = ''.join(str(ip) for ip in ips)
    all_index = get_knl_to_b5b6_circuit_map_for_correctors(ips).keys()
    m = pd.DataFrame(columns=all_index)
    v = pd.Series(dtype=float)
    m_constr = pd.DataFrame(columns=all_index)
    v_constr = pd.Series(dtype=float)

    for beam in optics.keys():
        for term in detuning_data[beam].terms():
            if get_order(term) == 1:
                m_term = build_feeddown_to_b4_row(beam, optics[beam], magnet_pattern=magnet_pattern, term=term, ips=ips)
            else:
                m_term = build_b6_row(beam, optics[beam], magnet_pattern=magnet_pattern, term=term, ips=ips)

            m_term.name = f"b{beam}.ip{ips_str}.{term}"
            m = pd.concat([m , m_term], axis=0)
            v.loc[m_term.name] = detuning_data[beam][term]

        for term in detuning_data.get_contraints(beam).terms():
            if get_order(term) == 1:
                m_term = build_feeddown_to_b4_row(beam, optics[beam], magnet_pattern=magnet_pattern, term=term, ips=ips)
            else:
                m_term = build_b6_row(beam, optics[beam], magnet_pattern=magnet_pattern, term=term, ips=ips)

            sign, constraint_val = detuning_data.get_contraints(beam).get_leq(term)
            m_term.name = f"b{beam}.ip{ips_str}.{term}"
            m_constr = pd.concat([m_constr, sign*m_term], axis=0)
            v_constr.loc[m_term.name] = constraint_val
    return m, v, m_constr, v_constr


def build_feeddown_to_b4_row(beam, twiss, magnet_pattern, term, ips):
    """ Builds one row of the feed-down matrix:
        [K5_L_IP1, K5_R_IP1, K6_L_IP1, K6_R_IP1, K5_L_IP5 ...]
        always in that order.
    """
    all_index = get_knl_to_b5b6_circuit_map_for_correctors(ips).keys()
    m = pd.Series(0., index=all_index)

    beam_sign = -1 if beam == 2 else 1
    coeff_sign_b6 = 1 if beam % 2 else -1  # takes into account that K6L-B4/B2 has a minus sign as seen from B1

    for ip in ips:
        for side in "LR":
            magnet = magnet_pattern.format(side=side, ip=ip)
            beta = {p: twiss.loc[magnet, f"BET{p}"] for p in "XY"}
            x = beam_sign * twiss.loc[magnet, "X"]                             # changes signs beam 4 -> beam 2
            y = twiss.loc[magnet, "Y"]                                         # same sign in beam 2 and beam 4
            coeff = get_detuning_coeff(term, beta)
            m[f"{magnet}_b5"] = x * coeff                                   # b5 feeddown to b4
            m[f"{magnet}_b6"] = coeff_sign_b6 * 0.5 * (x**2 - y**2) * coeff # b6 feeddown to b4
    return m


def build_b6_row(beam: int, twiss: pd.DataFrame, magnet_pattern: str, term: str, ips: Sequence[int]):
    """ Builds one row for second order amplitude matrix:
        [K5_L_IP1, K5_R_IP1, K6_L_IP1, K6_R_IP1, K5_L_IP5 ...]
        always in that order.
    """
    all_index = get_knl_to_b5b6_circuit_map_for_correctors(ips).keys()
    m = pd.Series(0., index=all_index)
    coeff_sign_b6 = 1 if beam % 2 else -1  # takes into account that K6L-B4/B2 has a minus sign as seen from B1

    for ip in ips:
        for side in "LR":
            magnet = magnet_pattern.format(side=side, ip=ip)
            beta = {p: twiss.loc[magnet, f"BET{p}"] for p in "XY"}
            m[f"{magnet}_b5"] = 0  # b5 does not contribute
            m[f"{magnet}_b6"] = coeff_sign_b6 * get_detuning_coeff(term, beta)  # b6
    return m


def get_detuning_coeff(term: str, beta: dict[str, float]):
    """ Get the coefficient for first and second order amplitude detuning.

    Args:
        term: 'X20', 'Y02', 'X11', 'Y20', 'Y11' or 'X02'
        beta: Dictionary of planes and values.

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


def get_all_circuits(ips: Sequence[str], circuits: Sequence[str]) -> list[str]:
    """
    Returns a list of filled-in circuit names for the given IPs and circuit patterns,
    adding 'L' and 'R' for sides.

    For the calculation they need to be in the same order as knl_names,
    which is assured by the order of the loops in this implementation and the calls below.

    Args:
        ips: List of IPs as strings, e.g. ["1", "5"]
        circuits: List of circuit patterns, e.g. ("kcdx3.{side}{ip}", "kctx3.{side}{ip}")
    """
    return [c.format(side=s, ip=ip) for ip in ips for s in "lr" for c in circuits]


def get_all_knl_names(ips: Sequence[str], magnet_patterns: Sequence[str]) -> list[str]:
    """
    Returns a list of filled-in magnet names for the given IPs and magnet patterns,
    adding 'L' and 'R' for sides.

    For the calculation they need to be in the same order as circuits,
    which is assured by the order of the loops in this implementation and the calls below.

    Args:
        ips: List of IPs as strings, e.g. ["1", "5"]
        magnet_patterns: List of magnet patterns, e.g. ("MCTX.3{side}{ip}",)
    """
    return [mp.format(side=s, ip=i) for i in ips for s in "LR" for mp in magnet_patterns]


def get_knl_to_b5b6_circuit_map_for_correctors(ips: Sequence[int]):
    """ Get a map for KNL to corrector name in the current IPs.
    As there is no b5 in LHC the corrector names are MCTX.._b5 and MCTX.._b6 respectively."""
    return dict(
        zip(
            get_all_knl_names(ips, magnet_patterns=(f"{DODECAPOLE_CORRECTOR}_b5", f"{DODECAPOLE_CORRECTOR}_b6")),
            get_all_circuits(ips, circuits=(DECAPOLE_CIRCUIT, DODECAPOLE_CIRCUIT)),
        )
    )


def get_knl_to_b6_circuit_map_for_correctors(ips: Sequence[int]):
    """ Get a map for KNL to dodecapole-corrector name in the current IPs. """
    return dict(
        zip(
            get_all_knl_names(ips, magnet_patterns=(f"{DODECAPOLE_CORRECTOR}_b6",)),
            get_all_circuits(ips, circuits=(DODECAPOLE_CIRCUIT,)),
        )
    )


# Detuning Calculation ---------------------------------------------------------

optics_per_beam = dict[int, TfsDataFrame]
optics_per_xing = dict[str, optics_per_beam]
AllOptics = optics_per_xing | optics_per_beam


def calculate_correction_values_from_feeddown_to_detuning(optics: AllOptics, target: Target, fields: str = 'b5b6') -> dict[str, float]:
    """ Calculates the values for either kcdx or kctx as installed into the Dodecapol corrector.
    Returns a dictionary of circuit names and their settings in KNL values (i.e. needs to be divided by the lenght of the decapole corrector).
    """
    if '5' not in fields and '6' not in fields:
        raise NotImplementedError('Neither b5 nor b6 is in fields.')

    # get a map from corrector name (for now ending in _b5, or _b6)
    # to the actual circuits. This could be skipped if
    # build_full_feeddown_matrix would already return the circuit names
    circuits_map = get_knl_to_b5b6_circuit_map_for_correctors(target.ips)

    m = pd.DataFrame(columns=list(circuits_map.keys()))
    v = pd.Series(dtype=float)
    v_meas = pd.Series(dtype=float)

    m_constr = pd.DataFrame(columns=list(circuits_map.keys()))
    v_constr = pd.Series(dtype=float)

    for target_data in target.data:
        try:
            use_optics = optics[target_data.xing]  # optics for crossing scheme
        except KeyError:
            use_optics = optics  # optics for both beams

        res_tdata = build_detuning_correction_matrix(
            optics=use_optics, detuning_data=target_data,
            magnet_pattern=DODECAPOLE_CORRECTOR
        )
        m = pd.concat([m, res_tdata[0]], axis=0)

        try:  # if this is a measurement value with error
            v = pd.concat([v, res_tdata[1].map(lambda a: getattr(a, "value"))], axis=0)
        except AttributeError:
            v = pd.concat([v, res_tdata[1]], axis=0)
        v_meas = pd.concat([v_meas, res_tdata[1]], axis=0)

        # Add constraints
        m_constr = pd.concat([m_constr, res_tdata[2]], axis=0)
        v_constr = pd.concat([v_constr, res_tdata[3]], axis=0)

    # Select only the used circuits (given by the fields) and filter equation system
    columns_mask = (m.columns.str.endswith("b5") & ("5" in fields)) | (m.columns.str.endswith("b6") & ("6" in fields))
    m = m.loc[:, columns_mask].rename(columns=circuits_map).fillna(0.)
    m_constr = m_constr.loc[:, columns_mask].rename(columns=circuits_map).fillna(0.)

    # Solve as convex system
    x = cp.Variable(len(m.columns))
    cost = cp.sum_squares(m.to_numpy() @ x - v)  # ||Mx - v||_2
    if len(v_constr):
        # Add constraints
        constr = m_constr.to_numpy() @ x <= v_constr.to_numpy()
        prob = cp.Problem(cp.Minimize(cost), [constr])
    else:
        # No constraints
        prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    if prob.status in ["infeasible", "unbounded"]:
        raise ValueError(f"Optimization failed! Reason: {prob.status}.")

    # test against pseudo-inverse solution (without constraints)
    m_inverse = np.linalg.pinv(m)
    x_pseudo = m_inverse.dot(v_meas)
    LOG.info(f"Values from cvxpy: {x.value}")
    LOG.info(f"Values with errors:\n {x_pseudo}")
    print_correction_and_error_as_latex(x_pseudo, m.columns)
    # print(np.abs(x.value - x_pseudo))
    # exit()

    return dict(zip(m.columns, x_pseudo))


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
        tfs.TfsDataFrame(
            index=pd.MultiIndex.from_product([fields_list, ip_strings], names=["FIELDS", "IP"]),
            columns=terms_first + terms_second,)
        for _ in enumerate(optics)
    ]
    for current_ips, ip_str in zip(loop_ips, ip_strings):
        m_first = {
            t: pd.DataFrame(
                [
                    build_feeddown_to_b4_row(
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
