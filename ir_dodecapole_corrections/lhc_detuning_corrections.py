"""
LHC Detuning Corrections
------------------------

!!! THIS IS NOT THE MAIN SCRIPT TO BE RUN !!!

it only contains the main simulation function for this specific scenario,
but to set the parameters needed (e.g. the measurement),
see setup_example_2018.py or setup_commish_2022.py.

This module contains the main function to run an LHC simulation with
the given parameters via MAD-X and calculate the corrections based on
the provided targets.

This module is similar to the main function to run and calculate the correction,
but also allows to use different crossing-schemes.
That is, you can specify a crossing scheme per measurement and the
feed-down is calculated based on that scheme.
As the simulation takes a while and multiple measurements might rely on the same
crossing scheme, the optics are calculated and saved first.
They can be then either read or passed to the correction function.

"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import cpymad
import pandas as pd
import tfs

from ir_dodecapole_corrections.simulation.lhc_simulation import FakeLHCBeam, LHCBeam, LHCCorrectors
from ir_dodecapole_corrections.utilities.classes_accelerator import (
    Correctors,
    get_fields,
    get_filled_corrector_attributes,
)
from ir_dodecapole_corrections.utilities.classes_detuning import MeasureValue
from ir_dodecapole_corrections.utilities.classes_targets import Target, TargetData
from ir_dodecapole_corrections.utilities.detuning import (
    calc_effective_detuning,
    calculate_correction,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ir_dodecapole_corrections.utilities.classes_targets import Target

LOG = logging.getLogger(__name__)


def create_optics(
    beams: Sequence[int],
    outputdir: Path,
    xings: dict | None = None,  # default set below
    optics: str = "round3030",  # 30cm round optics
    year: int = 2018,  # lhc year
    tune_x: float = 62.31,  # horizontal tune
    tune_y: float = 60.32,  # vertical tune
) -> dict[str, dict[int, LHCBeam]]:
    """ Main function to run this script."""
    # set mutable defaults ----
    if xings is None:
        xings = {'': {'scheme': 'top'}}  # use top-energy crossing scheme

    # Setup LHC for all crossing-schemes and both beams -------------------------------------------------
    lhc_beams = {}
    for xing_name, xing in xings.items():
        lhc_beams[xing_name] = {}
        for beam in beams:
            xing_outputdir = get_xing_outputdir(outputdir, xing_name, beam)

            lhc_beam = LHCBeam(
                beam=beam,
                outputdir=xing_outputdir,
                xing=xing,
                optics=optics,
                year=year,
                tune_x=tune_x,
                tune_y=tune_y,
            )
            lhc_beam.setup_machine()
            lhc_beam.save_nominal()
            lhc_beams[xing_name][beam] = lhc_beam

    return lhc_beams


def calculate_corrections(
    beams: Sequence[int],
    outputdir: Path,
    targets: Sequence[Target],
    correctors: Correctors,
    main_xing: str = TargetData.MAIN_XING,
    lhc_beams: dict[str, dict[int, LHCBeam]] | None = None,
):
    """ Calculate corrections based on targets and given correctors. """
    lhc_beams_out = {b: FakeLHCBeam(beam=b, outputdir=outputdir) for b in beams}

    for target in targets:
        LOG.info(f"Calculating detuning for \n{str(target)}")
        xings = {t.xing for t in target.data}
        id_ = f"{target.name}_{''.join(get_fields(correctors))}"

        optics = {}
        for xing in xings:
            if lhc_beams is None:
                # Load from tfs
                lhc_beams_in = {b: FakeLHCBeam(beam=b, outputdir=get_xing_outputdir(outputdir, xing, b)) for b in beams}
                optics[xing] = {lhc_in.beam: tfs.read(LHCBeam.output_path(lhc_in, 'twiss', 'optics_ir'), index="NAME") for lhc_in in lhc_beams_in.values()}
            else:
                # Load from memory
                optics[xing] = {beam: lhc_beam.df_twiss_nominal_ir.copy() for beam, lhc_beam in lhc_beams[xing].items()}

        try:
            values = calculate_correction(optics, target=target, correctors=correctors)
        except ValueError:
            LOG.error(f"Optimization failed for {target.name}  ({id_}).")
            values = {}

        # calculate effective detuning
        try:
            main_optics = optics[main_xing]
        except KeyError:
            if lhc_beams is None or main_xing not in lhc_beams:
                # Load from tfs
                lhc_beams_in = {b: FakeLHCBeam(beam=b, outputdir=get_xing_outputdir(outputdir, main_xing, b)) for b in beams}
                main_optics = {lhc_in.beam: tfs.read(LHCBeam.output_path(lhc_in, 'twiss', 'optics_ir'), index="NAME") for lhc_in in lhc_beams_in.values()}
            else:
                main_optics = {beam: lhc_beam.df_twiss_nominal_ir.copy() for beam, lhc_beam in lhc_beams[main_xing].items()}

        dfs_effective_detuning = calc_effective_detuning(main_optics, values, ips=target.ips)

        for lhc_out, df in zip(lhc_beams_out.values(), dfs_effective_detuning):
            detuning_tfs_out_with_and_without_errors(lhc_out, id_, df)
            madx_settings_out(lhc_out, id_, values)
            knl_tfs_out(lhc_out, id_, values)


def get_xing_outputdir(outputdir: Path, xing_name: str, beam: int) -> Path:
    if xing_name == "":
        return outputdir  / f"b{beam}"
    return outputdir / f"{xing_name}_b{beam}"


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
    circuit_length_map = {circuit: corrector.length for corrector in correctors
                   for circuit in get_filled_corrector_attributes(ips=ips, correctors=[corrector], attribute='circuit')}
    circuit_magnet_map = dict(
        zip(get_filled_corrector_attributes(ips=ips, correctors=correctors, attribute='circuit'),
            get_filled_corrector_attributes(ips=ips, correctors=correctors, attribute='magnet'))
    )

    for circuit, knl in values.items():
        length = circuit_length_map[circuit]
        magnet = circuit_magnet_map[circuit]
        try:
            df.loc[magnet, "KNL"] = knl.value
        except AttributeError:
            df.loc[magnet, "KNL"] = knl
            df.loc[magnet, "KN"] = knl / length
        else:
            df.loc[magnet, "ERRKNL"] = knl.error
            df.loc[magnet, "KN"] = knl.value / length
            df.loc[magnet, "ERRKN"] = knl.error / length

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



def check_corrections(
         outputdirs: dict[int, Path],
         lhc_beams: dict[int, LHCBeam] = None,
         xing: dict = None,  # set to {'scheme': 'top'} below
         optics: str = 'round3030',  # 30cm round optics
         year: int = 2018,  # lhc year
         tune_x: float = 62.31,  # horizontal tune
         tune_y: float = 60.32,  # vertical tune
         id_suffix: str = None,  # attach to output id

):
    """ Check the corrections via PTC.
    This installs decapole corrector magnets and reads the corrections
    from the settings file.
    If lhcbeams are given, the output paths will be adapted and these used,
    otherwise new LHCBeams will be set up.
    If an id_suffix is given (can be empty),
    nominal and ptc file will be written, if not, only the ptc file is output.
    """
    if xing is None:
        xing = {'scheme': 'top'}  # use top-energy crossing scheme

    if lhc_beams is None:
        lhc_beams = {}

    for beam, outputdir in outputdirs.items():
        try:
            lhc_beam = lhc_beams[beam]
        except KeyError:
            lhc_beam = LHCBeam(
                beam=beam, outputdir=outputdir,
                xing=xing, optics=optics, year=year,
                tune_x=tune_x, tune_y=tune_y,
            )
            lhc_beam.setup_machine()
            lhc_beam.save_nominal(id_=f"nominal{id_suffix}" if id_suffix else None)
            lhc_beams[beam] = lhc_beam
        else:
            lhc_beam.outputdir = outputdir  # override old outputdir

        lhc_beam.install_circuits_into_mctx()

        for settings in lhc_beam.outputdir.glob("settings.*.madx"):
            id_ = f"{settings.suffixes[-2].strip('.')}{id_suffix or ''}"
            lhc_beam.madx.input(settings.read_text())

            try:
                lhc_beam.match_tune()
                lhc_beam.get_twiss(id_, index_regex=f"{LHCCorrectors.b5.pattern}|{LHCCorrectors.b6.pattern}")
            except cpymad.madx.TwissFailed as e:
                LOG.error("Matching/Twiss failed!")
            else:
                lhc_beam.get_ampdet(id_)

            lhc_beam.check_kctx_limits()
            lhc_beam.reset_detuning_circuits()
    return lhc_beams


# Calculate Corrector Settings without running MAD-X again ---------------------


def calculate_from_prerun_optics(outputdirs: dict[int, Path],
                                 inputdirs: dict[int, Path],
                                 targets: Sequence[Target],
                                 field_list: Sequence[str] = ('b5', 'b6', 'b5b6'),
                                 ):
    lhc_beams_in = {b: FakeLHCBeam(beam=b, outputdir=indir) for b, indir in inputdirs.items()}
    lhc_beams_out = {b: FakeLHCBeam(beam=b, outputdir=outdir) for b, outdir in outputdirs.items()}
    optics = {lhc_in.beam: tfs.read(LHCBeam.output_path(lhc_in, 'twiss', 'optics_ir'), index="NAME") for lhc_in in lhc_beams_in.values()}

    for fields in field_list:
        for target in targets:
            id_ = f"{target.name}_{fields}"
            try:
                values = calculate_correction(optics, target=target, fields=fields)
            except ValueError:
                LOG.error(f"Optimization failed for {target.name} and {fields}.")
                values = {}
            dfs_effective_detuning = calc_effective_detuning(optics, values, ips=target.ips)

            for lhc_out, df in zip(lhc_beams_out.values(), dfs_effective_detuning):
                detuning_tfs_out_with_and_without_errors(lhc_out, id_, df)
                madx_settings_out(lhc_out, id_, values)
                knl_tfs_out(lhc_out, id_, values)
