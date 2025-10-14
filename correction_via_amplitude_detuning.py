"""
Main Function
-------------

!!! THIS IS NOT THE MAIN SCRIPT TO BE RUN !!!

it only contains the main simulation function for this specific scenario,
but to set the parameters needed (e.g. the measurement),
see setup_example_2018.py or setup_commish_2022.py.

This module contains the main function to run an LHC simulation with
the given parameters via MAD-X and calculate the corrections based on
the provided targets.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import tfs

from ir_amplitude_detuning.simulation.lhc_simulation import FakeLHCBeam, LHCBeam
from ir_amplitude_detuning.utilities.correctors import CorrectorFillAttributes, Correctors, get_filled_corrector_attributes
from ir_amplitude_detuning.utilities.classes_detuning import MeasureValue
from ir_amplitude_detuning.utilities.detuning import (
    calc_effective_detuning,
    calculate_correction,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ir_amplitude_detuning.detuning.targets import Target

LOG = logging.getLogger(__name__)

def run_lhc_and_calc_correction(outputdirs: dict[int, Path],
         targets: Sequence[Target],
         xing: dict | None = None,  # set to {'scheme': 'top'} below
         optics: str = 'round3030',  # 30cm round optics
         year: int = 2018,  # lhc year
         ):
    """ ."""
    # set mutable defaults ----
    if xing is None:
        xing = {'scheme': 'top'}  # use top-energy crossing scheme

    target_names = [t.name for t in targets]
    if len(target_names) != len(set(target_names)):
        raise KeyError("Some targets have the same names. "
                       "This will lead to outputfiles for these targets to be overwritten. "
                       "Please give them unique names.")

    # Setup LHC for both beams -------------------------------------------------
    lhc_beams: dict[int, LHCBeam] = {}
    for beam, outputdir in outputdirs.items():
        lhc_beam = LHCBeam(
            beam=beam, outputdir=outputdir,
            xing=xing, optics=optics, year=year,
        )
        lhc_beams[beam] = lhc_beam
        lhc_beam.setup_machine()
        lhc_beam.save_nominal()

    # Compensate amplitude detuning --------------------------------------------
    optics = {}
    for beam, lhc_beam in lhc_beams.items():
        lhc_beam.install_circuits_into_mctx()
        optics[beam] = lhc_beam.df_twiss_nominal_ir.copy()

    for target in targets:
        LOG.info(f"Calculating detuning with {fields} for \n{str(target)}")
        id_ = f"{target.name}_{fields}"
        try:
            values = calculate_correction(optics, target=target, fields=fields)
        except ValueError:
            LOG.error(f"Optimization failed for {target.name} and {fields}.")
            values = {}
        dfs_effective_detuning = calc_effective_detuning(optics, values, ips=target.ips)

        for lhc_beam, df in zip(lhc_beams.values(), dfs_effective_detuning):
            tfs.write(lhc_beam.output_path('ampdet_calc', id_), df)
            lhc_beam.set_mctx_circuits_powering(values, id_=id_)
            lhc_beam.check_kctx_limits()
            lhc_beam.reset_detuning_circuits()

    # exit
    for lhc_beam in lhc_beams.values():
        lhc_beam.madx.exit()



def madx_string(key: str, knl: float | MeasureValue, length: str,  kn: float | MeasureValue | None = None,  ):
    knl_string = f"{key} := {getattr(knl, 'value', knl)} / {length};"
    kn_string = ""
    if kn:
        kn_string = f" ! {key} = {getattr(kn, 'value', kn)};"

    return f"{knl_string}{kn_string}"


def madx_settings_out(lhc_out, id_, values):
    madx_command = [f'! Amplitude detuning powering {id_}:', f'! reminder: l.MCTX = {LENGTH_MCTX}']
    for key, knl in values.items():
        madx_command.append(madx_string(key, knl=knl, kn=knl / LENGTH_MCTX))

    LHCBeam.output_path(lhc_out, 'settings', id_, suffix=".madx").write_text("\n".join(madx_command))


def knl_tfs_out(lhc_out: LHCBeam, id_: str, values: pd.Series, correctors: Correctors):
    df = tfs.TfsDataFrame(index=values.index)

    ips = ['1', '5']
    circuit_length_map = {
        circuit: corrector.length
        for corrector in correctors
        for circuit in get_filled_corrector_attributes(
            ips=ips,
            correctors=[corrector],
            attribute=CorrectorFillAttributes.circuit
        )
    }
    circuit_magnet_map = dict(zip(
            get_filled_corrector_attributes(ips=ips, correctors=correctors, attribute=CorrectorFillAttributes.circuit),
            get_filled_corrector_attributes(ips=ips, correctors=correctors, attribute=CorrectorFillAttributes.magnet))
    )

    for circuit, knl in values.items():
        length = circuit_length_map[circuit]
        magnet = circuit_magnet_map[circuit]
        df.headers[f"l.{magnet}"] = length
        try:
            df.loc[magnet, "KNL"] = knl.value
            df.loc[magnet, "ERRKNL"] = knl.error
            df.loc[magnet, "KN"] = knl.value / length
            df.loc[magnet, "ERRKN"] = knl.error / length
        except AttributeError:
            df.loc[magnet, "KNL"] = knl
            df.loc[magnet, "KN"] = knl / length

    tfs.write(LHCBeam.output_path(lhc_out, 'settings', id_), df, save_index="NAME")


def detuning_tfs_out_with_and_without_errors(lhc_out: LHCBeam, id_: str, df: pd.DataFrame):
    """ """
    has_errors = False
    df_errors = df.copy()

    for column in df.columns:
        try:
            values: pd.Series = df[column].apply(MeasureValue.from_value)
        except AttributeError:
            pass  # string column
        else:
            df[column] = values.apply(lambda x: x.value)
            df_errors[column] = df[column]
            df_errors[f"ERR{column}"] = values.apply(lambda x: x.error).fillna(0)
            has_errors = has_errors or df_errors[f"ERR{column}"].any()

    df = df.astype(float, errors='ignore')
    df_errors = df_errors.astype(float, errors='ignore')
    tfs.write(LHCBeam.output_path(lhc_out, 'ampdet_calc', id_), df)
    if has_errors:
        tfs.write(LHCBeam.output_path(lhc_out, 'ampdet_calc_err', id_), df_errors)
