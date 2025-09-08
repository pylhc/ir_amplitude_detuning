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

import tfs

from ir_dodecapole_corrections.simulation.lhc_simulation import FakeLHCBeam, LHCBeam
from ir_dodecapole_corrections.utilities.classes import MeasureValue, Target
from ir_dodecapole_corrections.utilities.detuning import (
    calc_effective_detuning,
    calculate_correction_values_from_feeddown_to_detuning,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

LOG = logging.getLogger(__name__)
LENGTH_MCTX = 0.615


def run_lhc_and_calc_correction(outputdirs: dict[int, Path],
         targets: Sequence[Target],
         field_list: Sequence[str] =('b6',),
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

    for fields in field_list:
        for target in targets:
            LOG.info(f"Calculating detuning with {fields} for \n{str(target)}")
            id_ = f"{target.name}_{fields}"
            try:
                values = calculate_correction_values_from_feeddown_to_detuning(optics, target=target, fields=fields)
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


# Calculate Corrector Settings without running MAD-X again ---------------------

def calculate_from_prerun_optics(
    outputdirs: dict[int, Path],
    inputdirs: dict[int, Path],
    targets: Sequence[Target],
    field_list: Sequence[str] = ("b5", "b6", "b5b6"),
):
    """ Calculate the corrector settings from previously run MAD-X simulations."""
    lhc_beams_in = {b: FakeLHCBeam(beam=b, outputdir=indir) for b, indir in inputdirs.items()}
    lhc_beams_out = {b: FakeLHCBeam(beam=b, outputdir=outdir) for b, outdir in outputdirs.items()}
    optics = {lhc_in.beam: tfs.read(LHCBeam.output_path(lhc_in, 'twiss', 'optics_ir'), index="NAME") for lhc_in in lhc_beams_in.values()}

    for fields in field_list:
        for target in targets:
            id_ = f"{target.name}_{fields}"
            try:
                values = calculate_correction_values_from_feeddown_to_detuning(optics, target=target, fields=fields)
            except ValueError:
                LOG.error(f"Optimization failed for {target.name} and {fields}.")
                values = {}
            dfs_effective_detuning = calc_effective_detuning(optics, values, ips=target.ips)

            for lhc_out, df in zip(lhc_beams_out.values(), dfs_effective_detuning):
                detuning_tfs_out_with_and_without_errors(lhc_out, id_, df)
                madx_settings_out(lhc_out, id_, values)
                knl_tfs_out(lhc_out, id_, values)


def madx_string(key: str, knl: float | MeasureValue, kn: float | MeasureValue | None = None):
    kn_string = ""
    try:
        knl_string = f"{key} := {knl.value} / l.MCTX;"
    except AttributeError:
        knl_string = f"{key} := {knl} / l.MCTX;"

    if kn:
        try:
            kn_string = f" ! {key} = {kn.value};"
        except AttributeError:
            kn_string = f" ! {key} = {kn};"

    return f"{knl_string}{kn_string}"


def madx_settings_out(lhc_out, id_, values):
    madx_command = [f'! Amplitude detuning powering {id_}:', f'! reminder: l.MCTX = {LENGTH_MCTX}']
    for key, knl in values.items():
        madx_command.append(madx_string(key, knl=knl, kn=knl / LENGTH_MCTX))

    LHCBeam.output_path(lhc_out, 'settings', id_, suffix=".madx").write_text("\n".join(madx_command))


def knl_tfs_out(lhc_out, id_, values):
    df = tfs.TfsDataFrame(index=values.keys(), headers={"l.MCTX": LENGTH_MCTX})
    for key, knl in values.items():
        try:
            df.loc[key, "KNL"] = knl.value
        except AttributeError:
            df.loc[key, "KNL"] = knl
            df.loc[key, "KN"] = knl / LENGTH_MCTX
        else:
            df.loc[key, "ERRKNL"] = knl.error
            df.loc[key, "KN"] = knl.value / LENGTH_MCTX
            df.loc[key, "ERRKN"] = knl.error / LENGTH_MCTX

    tfs.write(LHCBeam.output_path(lhc_out, 'settings', id_), df, save_index="NAME")


def detuning_tfs_out_with_and_without_errors(lhc_out, id_, df):
    has_errors = False
    df_errors = df.copy()

    for column in df.columns:
        try:
            values = df[column].apply(MeasureValue.from_value)
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
