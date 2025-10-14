"""
LHC Detuning Corrections
------------------------

.. caution:: THIS FILE CANNOT BE RUN AS A SCRIPT !!!

    It does contain the main simulation functions for the LHC scenarios,
    but to set the parameters needed (e.g. the measurement),
    see the examples.

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
    Method,
    calc_effective_detuning,
    calculate_correction,
)
from ir_amplitude_detuning.detuning.measurements import MeasureValue
from ir_amplitude_detuning.detuning.targets import Target
from ir_amplitude_detuning.simulation.lhc_simulation import FakeLHCBeam, LHCBeam, LHCCorrectors
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
from ir_amplitude_detuning.utilities.correctors import (
    Corrector,
    Correctors,
    get_fields,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd

    from ir_amplitude_detuning.detuning.equation_system import TwissPerBeam
    from ir_amplitude_detuning.detuning.targets import Target


LOG = logging.getLogger(__name__)

LHCBeams: TypeAlias = dict[int, LHCBeam]
LHCBeamsPerXing: TypeAlias = dict[str, LHCBeams]



@dataclass(slots=True)
class CorrectionResults:
    """ Class to store the results of a correction calculation.

    Attributes:
        name (str): The name of the correction target.
        series (pd.Series): Series of the correction KNL values for each corrector.
        dataframe (tfs.TfsDataFrame): Same information as the Series, but in a DataFrame format,
                                      with the corrector-magnet names as index and the other data
                                      split into columns. See :func:`~ir_amplitude_detuning.lhc_detuning_corrections.generate_knl_tfs`.
        madx (str): The MAD-X code to apply the correction.
    """
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
    tune_x: float = 62.28,  # horizontal tune
    tune_y: float = 60.31,  # vertical tune
) -> LHCBeams:
    """ Run MAD-X to create optics for all crossing-schemes.

    The optics are saved in subfolders of the output directory.

    Args:
        beams (Sequence[int]): The beam numbers.
        outputdir (Path): The output directory.
        output_id (str, optional): The output id. Defaults to ''.
        xing (dict[str, dict], optional): The crossing scheme. Defaults to `None`,
                                          which is set to the top-energy collision scheme below.
        optics (str, optional): The optics. Defaults to "round3030".
        year (int, optional): The year. Defaults to 2018.
        tune_x (float, optional): The horizontal tune. Defaults to 62.31.
        tune_y (float, optional): The vertical tune. Defaults to 60.32.

    Returns:
        LHCBeams: The LHC beams, i.e. a dictionary of LHCBeam objects.
    """
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
    method: Method = Method.cvxpy
    ) -> dict[str, CorrectionResults]:
    """ Calculate corrections based on targets and given correctors.

    Args:
        beams (Sequence[int]): The beam numbers to calculate corrections for.
        outputdir (Path): The output directory.
        targets (Sequence[Target]): The targets to calculate corrections for.
        method (Method): The method to use for calculating the corrections (see :func:`ir_amplitude_detuning.detuning.calculations`).

    Returns:
        dict[str, CorrectionResults]: The results for each target.
    """
    results = {}

    for target in targets:
        LOG.info(f"Calculating detuning for \n{str(target)}")


        # Calculate correction ---
        try:
            values = calculate_correction(target, method=method)
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
        beams (LHCBeams | Sequence[int]): The LHCBeams objects or a sequence of beam numbers.
        outputdir (Path): The output directory.
        label (str): The label for the sub-dir (e.g. a name for the optics)
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
    """ Get the outputdir sub-dir for a given label and beam.

    Args:
        outputdir (Path): The output directory.
        label (str): The label for the sub-dir (e.g. a name for the optics)
        beam (int): The beam number.
    """
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
        values (pd.Series): The correction values. Assumes the index are
                            :class:`~ir_amplitude_detuning.utilities.classes_accelerator.Corrector` objects.
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
    """ Calculate the :func:`effective detuning <ir_amplitude_detuning.detuning.calculations.calc_effective_detuning>` for each beam and
    write the results into a `tfs` file.

    Args:
        outputdir (Path): The output directory.
        optics (TwissPerBeam): The machine optics.
        results (CorrectionResults): The calculated correction results.
    """
    effective_detuning = calc_effective_detuning(optics, results.series)

    lhc_beams_out = {b: FakeLHCBeam(beam=b, outputdir=outputdir) for b in optics}  # to get the file output paths
    for beam in optics:
        df_detuning = effective_detuning[beam]
        detuning_tfs_out_with_and_without_errors(lhc_beams_out[beam], results.name, df_detuning)
        LOG.info(f"Detuning check for beam {beam}, {results.name}:\n{df_detuning}\n")


def detuning_tfs_out_with_and_without_errors(lhc_out: LHCBeam | FakeLHCBeam, id_: str, df: pd.DataFrame):
    """  Write out the detuning results, given as :class:`DataFrame` into a
    `tfs` file.
    If the input `DataFrame` contains :class:`~ir_amplitude_detuning.detuning.MeasureValue` objects,
    the values and errors are extracted and two files are written:
    One with only the values and one with the values and errors.

    Args:
        lhc_out (LHCBeam | FakeLHCBeam): LHCBeam object to find the correct output path.
        id_ (str): The identifier (e.g. target name) of the calculation.
        df (pd.DataFrame): The calculated detuning terms.
    """
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
    tune_x: float = 62.28,  # horizontal tune
    tune_y: float = 60.31,  # vertical tune
    ):
    """ Check the corrections via PTC.

    This installs decapole corrector magnets and reads the corrections
    from the settings file.
    If ``lhc_beams`` are given, the output paths will be adapted and these used,
    otherwise new :class:`~ir_amplitude_detuning.lhc_detuning_corrections.LHCBeam` s
    will be set up.

    PTC is run for the ``nominal`` machine as well as all ``settings.*`` files
    found in the output directory.
    The PTC output ids are parsed from the settings file names.

    Args:
        outputdir (Path): Output directory.
        lhc_beams (dict[int, LHCBeam]): Pre-run LHC beams.
        beams (Sequence[int]): Beams (if ``lhc_beams`` is None).
        xing (dict[str, dict]): Crossing scheme (if ``lhc_beams`` is `None`).
        optics (str): Optics (if ``lhc_beams`` is `None`).
        year (int): Year (if ``lhc_beams`` is `None`).
        tune_x (float): Horizontal tune (if ``lhc_beams`` is `None`).
        tune_y (float): Vertical tune (if ``lhc_beams`` is `None`).
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
