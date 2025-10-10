"""
LHC Detuning Corrections
------------------------

!!! THIS IS NOT THE MAIN SCRIPT TO BE RUN !!!

it only contains the main simulation functions for this specific scenario,
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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import cpymad
import tfs

from ir_amplitude_detuning.detuning.calculations import (
    calc_effective_detuning,
    calculate_correction,
)
from ir_amplitude_detuning.detuning.measurements import MeasureValue
from ir_amplitude_detuning.simulation.lhc_simulation import FakeLHCBeam, LHCBeam, LHCCorrectors
from ir_amplitude_detuning.utilities.classes_accelerator import (
    Corrector,
    Correctors,
    get_fields,
)
from ir_amplitude_detuning.utilities.classes_targets import Target
from ir_amplitude_detuning.utilities.constants import (
    AMPDET_CALC_ERR_ID,
    AMPDET_CALC_ID,
    CIRCUIT,
    ERR,
    KN,
    KNL,
    LENGTH,
    NAME,
    NOMINAL_ID,
    SETTINGS_ID,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd

    from ir_amplitude_detuning.detuning.equation_system import TwissPerBeam
    from ir_amplitude_detuning.utilities.classes_targets import Target


LOG = logging.getLogger(__name__)

LHCBeams: TypeAlias = dict[int, LHCBeam]
LHCBeamsPerXing: TypeAlias = dict[str, LHCBeams]



@dataclass(slots=True)
class CorrectionResults:
    name: str
    series: pd.Series
    dataframe: tfs.TfsDataFrame
    madx: str


def create_optics(
    beams: Sequence[int],
    outputdir: Path,
    output_id: str = '',
    xing: dict[str, dict] | None = None,  # default set below
    optics: str = "round3030",  # 30cm round optics
    year: int = 2018,  # lhc year
    tune_x: float = 62.31,  # horizontal tune
    tune_y: float = 60.32,  # vertical tune
) -> LHCBeams:
    """ Run MAD-X to create optics for all crossing-schemes. """
    # set mutable defaults ----
    if xing is None:
        xing = {'scheme': 'top'}  # use top-energy crossing scheme

    # Setup LHC for both beams -------------------------------------------------
    lhc_beams = {}
    for beam in beams:
        output_subdir = get_label_outputdir(outputdir, output_id, beam)
        lhc_beam = LHCBeam(
            beam=beam,
            outputdir=output_subdir,
            xing=xing,
            optics=optics,
            year=year,
            tune_x=tune_x,
            tune_y=tune_y,
        )
        lhc_beam.setup_machine()
        lhc_beam.save_nominal()
        lhc_beams[beam] = lhc_beam
    return lhc_beams


def calculate_corrections(
    beams: Sequence[int],
    outputdir: Path,
    targets: Sequence[Target],
    ) -> dict[str, CorrectionResults]:
    """ Calculate corrections based on targets and given correctors. """
    results = {}

    for target in targets:
        LOG.info(f"Calculating detuning for \n{str(target)}")


        # Calculate correction ---
        try:
            values = calculate_correction(target)
        except ValueError:
            LOG.error(f"Optimization failed for {target.name}  (fields: {get_fields(target)}.")
            values = {}

        # Save results ---
        madx_command = generate_madx_command(values)
        knl_tfs = generate_knl_tfs(values)

        results[target.name] = CorrectionResults(
            name=target.name,
            series=values,
            dataframe=knl_tfs,
            madx=madx_command,
        )

        lhc_beams_out = {b: FakeLHCBeam(beam=b, outputdir=outputdir) for b in beams}  # to get the file output paths
        for beam in beams:
            lhc_out = lhc_beams_out[beam]
            lhc_out.output_path(SETTINGS_ID, target.name, suffix=".madx").write_text(madx_command)
            tfs.write(lhc_out.output_path(SETTINGS_ID, target.name), knl_tfs, save_index=NAME)

    return results


def get_nominal_optics(beams: LHCBeams | Sequence[int], outputdir: Path | None = None, label: str = '') -> TwissPerBeam:
    """ Return previously generated nominal machine optics as a dictionary of TfsDataFrames per Beam, either directly from the
    LHCBeams objects (if given) or reading from the labeled sub-folder in the output-path.

    Args:

    """
    optics = {}
    for beam in beams:
        if isinstance(beams, dict):
            optics[beam] = beams[beam].df_twiss_nominal.copy()
        else:
            if outputdir is None:
                raise ValueError("outputdir must be provided if beams are not given as LHCBeams.")

            lhc_beam = FakeLHCBeam(beam=beam, outputdir=get_label_outputdir(outputdir, label, beam))
            optics[beam] = tfs.read(lhc_beam.output_path('twiss', NOMINAL_ID), index=NAME)
    return optics


def get_label_outputdir(outputdir: Path, label: str, beam: int) -> Path:
    """ Get the outputdir sub-dir for a given label and beam. """
    if label == "":
        return outputdir  / f"b{beam}"
    return outputdir / f"{label}_b{beam}"


# Correction Output Functions --------------------------------------------------

def generate_madx_command(values: pd.Series) -> str:
    """ Generate a MAD-X command to set the corrector values.

    Args:
        values (pd.Series): The correction values. Assumes the index are the Corrector objects.

    """
    correctors: Correctors = values.index
    length_map = {f"l.{corrector.madx_type}": corrector.length for corrector in correctors if corrector.madx_type is not None}

    madx_command = ['! Amplitude detuning powering:'] + [f'! reminder: {l} = {length_map[l]}' for l in length_map]  # noqa: E741
    for corrector, knl in sorted(values.items()):
        corrector: Corrector
        length_str = corrector.length if corrector.madx_type is None else f"l.{corrector.madx_type}"
        knl_value = getattr(knl, 'value', knl)
        madx_command.append(f"{corrector.circuit} := {knl_value} / {length_str};")
        madx_command.append(f"! {corrector.circuit} = {knl_value / corrector.length};")
    return "\n".join(madx_command)


def generate_knl_tfs(values: pd.Series) -> tfs.TfsDataFrame:
    """ Generate a TFS dataframe with the corrector values.

    Args:
        values (pd.Series): The correction values. Assumes the index are the Corrector objects.
    """
    correctors: Correctors = sorted(values.index)
    df = tfs.TfsDataFrame(index=[c.magnet for c in correctors])

    for corrector, knl in values.items():
        corrector: Corrector
        length = corrector.length
        magnet = corrector.magnet

        df.loc[magnet, CIRCUIT] = corrector.circuit
        df.loc[magnet, LENGTH] = length
        try:
            df.loc[magnet, KNL] = knl.value
            df.loc[magnet, f"{ERR}{KNL}"] = knl.error
            df.loc[magnet, KN] = knl.value / length
            df.loc[magnet, f"{ERR}{KN}"] = knl.error / length
        except AttributeError:
            df.loc[magnet, KNL] = knl
            df.loc[magnet, KN] = knl / length

    return df


# Detuning Check Functions -----------------------------------------------------


# Analytical Check ---

def check_corrections_analytically(outputdir: Path, optics: TwissPerBeam, results: CorrectionResults) -> dict[int, pd.DataFrame]:
    """

    Args:
        optics (TwissPerBeam):
        results (CorrectionResults):
        outputdir (Path):
    """
    effective_detuning = calc_effective_detuning(optics, results.series)

    lhc_beams_out = {b: FakeLHCBeam(beam=b, outputdir=outputdir) for b in optics}  # to get the file output paths
    for beam in optics:
        df_detuning = effective_detuning[beam]
        detuning_tfs_out_with_and_without_errors(lhc_beams_out[beam], results.name, df_detuning)
        LOG.info(f"Detuning check for beam {beam}, {results.name}:\n{df_detuning}\n")


def detuning_tfs_out_with_and_without_errors(lhc_out: LHCBeam | FakeLHCBeam, id_: str, df: pd.DataFrame):
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
            df_errors[f"{ERR}{column}"] = values.apply(lambda x: x.error).fillna(0)
            has_errors = has_errors or df_errors[f"{ERR}{column}"].any()

    tfs.write(lhc_out.output_path(AMPDET_CALC_ID, id_), df)
    if has_errors:
        tfs.write(lhc_out.output_path(AMPDET_CALC_ERR_ID, id_), df_errors)


# PTC Check ---

def check_corrections_ptc(
    outputdir: Path,
    lhc_beams: dict[int, LHCBeam] | None = None,
    # Below only needed if lhc_beams is None ---
    beams: Sequence[int] | None = None,
    xing: dict[str, dict] | None = None,
    optics: str = "round3030",  # 30cm round optics
    year: int = 2018,  # lhc year
    tune_x: float = 62.31,  # horizontal tune
    tune_y: float = 60.32,  # vertical tune
    ):
    """ Check the corrections via PTC.
    This installs decapole corrector magnets and reads the corrections
    from the settings file.
    If lhcbeams are given, the output paths will be adapted and these used,
    otherwise new LHCBeams will be set up.
    If an id_suffix is given (can be empty),
    nominal and ptc file will be written, if not, only the ptc file is output.
    """
    if lhc_beams is None:
        # Setup LHC for both beams ---
        lhc_beams = {}
        if beams is None:
            raise ValueError("Either lhc_beams or beams must be given.")
        if xing is None:
            xing = {'scheme': 'top'}

        for beam in beams:
            lhc_beam = LHCBeam(
                beam=beam,
                outputdir=get_label_outputdir(outputdir, 'tmp_ptc', beam),
                xing=xing,
                optics=optics,
                year=year,
                tune_x=tune_x,
                tune_y=tune_y,
            )
            lhc_beam.setup_machine()
            lhc_beams[beam] = lhc_beam

    # Check Corrections ---
    for lhc_beam in lhc_beams.values():
        lhc_beam.outputdir = outputdir  # override old outputdir

        lhc_beam.install_circuits_into_mctx()
        settings_glob = lhc_beam.output_path(SETTINGS_ID, output_id="*", suffix=".madx").name

        loaded_settings = {NOMINAL_ID: None}  # get nominal to establish a baseline
        for settings_file in lhc_beam.outputdir.glob(settings_glob): # loop over targets
            target_id = settings_file.suffixes[-2].strip(".")
            loaded_settings[target_id] = settings_file.read_text()

        for target_id, settings in loaded_settings.items():
            if settings is not None:
                lhc_beam.madx.input(settings)

            try:
                lhc_beam.match_tune()
                lhc_beam.get_twiss(target_id, index_regex=LHCCorrectors.pattern)
            except cpymad.madx.TwissFailed:
                LOG.error(f"Matching/Twiss failed for target {target_id}!")
            else:
                lhc_beam.get_ampdet(target_id)

            if settings is not None:
                lhc_beam.check_kctx_limits()
                lhc_beam.reset_detuning_circuits()
