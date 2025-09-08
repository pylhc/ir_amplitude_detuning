"""
Main Function + Multi Xing
--------------------------

This module is similar to the main function to run and calculate the correction,
but also allows to use different crossing-schemes.
That is, you can specify a crossing scheme per measurement and the
feed-down is calculated based on that scheme.
As the simulation takes a while and multiple measurements might rely on the same
crossing scheme, the optics are calculated and saved first.
They can be then either read or passed to the correction function.

"""
import logging
from collections.abc import Sequence
from pathlib import Path

import cpymad
import tfs

from ir_dodecapole_corrections.correction_via_amplitude_detuning import (
    detuning_tfs_out_with_and_without_errors,
    knl_tfs_out,
    madx_settings_out,
)
from ir_dodecapole_corrections.simulation.lhc_simulation import FakeLHCBeam, LHCBeam
from ir_dodecapole_corrections.utilities.classes import Target, TargetData
from ir_dodecapole_corrections.utilities.detuning import (
    DODECAPOLE_PATTERN,
    calc_effective_detuning,
    calculate_correction_values_from_feeddown_to_detuning,
)

LOG = logging.getLogger(__name__)
LENGTH_MCTX = 0.615


def get_xing_outputdir(outputdir: Path, xing_name: str):
    if xing_name == "":
        return outputdir
    return outputdir.with_name(f"{xing_name}_{outputdir.name}")


def create_optics(
    outputdirs: dict[int, Path],
    xings: dict | None = None,  # set to {'scheme': 'top'} below
    optics: str = "round3030",  # 30cm round optics
    year: int = 2018,  # lhc year
    tune_x: float = 62.31,  # horizontal tune
    tune_y: float = 60.32,  # vertical tune
) -> dict[str, dict[int, LHCBeam]]:
    """ Main function to run this script."""
    # set mutable defaults ----
    if xings is None:
        xings = {TargetData.MAIN_XING: {'scheme': 'top'}}  # use top-energy crossing scheme

    # Setup LHC for all crossing-schemes and both beams -------------------------------------------------
    lhc_beams = {}
    for xing_name, xing in xings.items():
        lhc_beams[xing_name] = {}
        for beam, outputdir in outputdirs.items():
            xing_outputdir = get_xing_outputdir(outputdir, xing_name)

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


def calculate_corrections(outputdirs: dict[int, Path],
                          targets: Sequence[Target],
                          field_list: Sequence[str] =('b6',),
                          main_xing: str = TargetData.MAIN_XING,
                          lhc_beams: dict[str, dict[int, LHCBeam]] | None  = None,
    ):
    """ Calculate corrections based on targets.
    field_list defines which fields to use for feed-down to b4.

    """
    lhc_beams_out = {b: FakeLHCBeam(beam=b, outputdir=outdir) for b, outdir in outputdirs.items()}

    for fields in field_list:
        for target in targets:
            LOG.info(f"Calculating detuning with {fields} for \n{str(target)}")
            id_ = f"{target.name}_{fields}"
            xings = {t.xing for t in target.data}

            optics = {}
            for xing in xings:
                if lhc_beams is None:
                    # Load from tfs
                    lhc_beams_in = {b: FakeLHCBeam(beam=b, outputdir=get_xing_outputdir(folder, xing)) for b, folder in outputdirs.items()}
                    optics[xing] = {lhc_in.beam: tfs.read(LHCBeam.output_path(lhc_in, 'twiss', 'optics_ir'), index="NAME") for lhc_in in lhc_beams_in.values()}
                else:
                    # Load from memory
                    optics[xing] = {beam: lhc_beam.df_twiss_nominal_ir.copy() for beam, lhc_beam in lhc_beams[xing].items()}

            try:
                values = calculate_correction_values_from_feeddown_to_detuning(optics, target=target, fields=fields)  # TODO second order??
            except ValueError:
                LOG.error(f"Optimization failed for {target.name} and {fields}.")
                values = {}

            # calculate effective detuning
            try:
                main_optics = optics[main_xing]
            except KeyError:
                if lhc_beams is None or main_xing not in lhc_beams:
                    # Load from tfs
                    lhc_beams_in = {b: FakeLHCBeam(beam=b, outputdir=get_xing_outputdir(folder, main_xing)) for b, folder in outputdirs.items()}
                    main_optics = {lhc_in.beam: tfs.read(LHCBeam.output_path(lhc_in, 'twiss', 'optics_ir'), index="NAME") for lhc_in in lhc_beams_in.values()}
                else:
                    main_optics = {beam: lhc_beam.df_twiss_nominal_ir.copy() for beam, lhc_beam in lhc_beams[main_xing].items()}

            dfs_effective_detuning = calc_effective_detuning(main_optics, values, ips=target.ips)

            for lhc_out, df in zip(lhc_beams_out.values(), dfs_effective_detuning):
                detuning_tfs_out_with_and_without_errors(lhc_out, id_, df)
                madx_settings_out(lhc_out, id_, values)
                knl_tfs_out(lhc_out, id_, values)


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
                lhc_beam.get_twiss(id_, index_regex=DODECAPOLE_PATTERN)
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
                values = calculate_correction_values_from_feeddown_to_detuning(optics, target=target, fields=fields)
            except ValueError:
                LOG.error(f"Optimization failed for {target.name} and {fields}.")
                values = {}
            dfs_effective_detuning = calc_effective_detuning(optics, values, ips=target.ips)

            for lhc_out, df in zip(lhc_beams_out.values(), dfs_effective_detuning):
                detuning_tfs_out_with_and_without_errors(lhc_out, id_, df)
                madx_settings_out(lhc_out, id_, values)
                knl_tfs_out(lhc_out, id_, values)
