"""
Measurement Analysis
--------------------

Functionality to analyse measurement data, using the tools from omc3.
These functions mostly just wrap omc3 functions for convenience and to
transform the data into the format used in this package.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
import tfs
from omc3.amplitude_detuning_analysis import get_kick_and_bbq_df, single_action_analysis
from omc3.optics_measurements.constants import ERR, EXT, KICK_NAME
from omc3.tune_analysis.bbq_tools import OutlierFilterOpt
from omc3.tune_analysis.constants import (
    get_bbq_out_name,
    get_kick_out_name,
    get_odr_header_coeff_corrected,
    get_odr_header_err_coeff_corrected,
)
from omc3.tune_analysis.kick_file_modifiers import read_timed_dataframe

from ir_amplitude_detuning.detuning.measurements import Detuning, DetuningMeasurement, MeasureValue
from ir_amplitude_detuning.detuning.terms import (
    DetuningTerm,
    detuning_term_to_planes,
    get_order,
)
from ir_amplitude_detuning.utilities.common import StrEnum

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class AnalysisOption(StrEnum):
    always: str = "always"
    never: str = "never"
    auto: str = "auto"


def get_beam_from_dir(analysis_dir: Path) -> int:
    """Determine the beam number from the analysis directory.
    Assumes either the directory name is of the form ``B{beam}_``/``b{beam}_`` or
    the parent or the parent directory is of the form ``LHCB{beam}``.

    Args:
        analysis_dir (Path): The analysis directory.

    Returns:
        int: The beam number
    """
    if match := re.match(r"^B(\d)_", analysis_dir.name, flags=re.IGNORECASE):
        return int(match.group(1))

    gui_machine_dir = analysis_dir.absolute().parents[1]  #  LHCB#/Results/Analysis

    if match := re.match(r"^LHCB(\d)", gui_machine_dir.name):
        return int(match.group(1))

    raise ValueError(f"Could not determine beam from {analysis_dir}")


def get_kick_plane_from_dir(analysis_dir: Path) -> str:
    """Determine the kick plane from the analysis directory.
    Assumes the directory name contains either X, Y, H or V (or lowercase)
    separated by dots or underscores on each side, or one side if ending with the letter.

    Args:
        analysis_dir (Path): The analysis directory.

    Returns:
        str: The kick plane (``'x'`` or ``'y'``)
    """
    hv_map = {"h": "x", "v": "y"}

    if match := re.search(r"[._]([HVXY])([._]|$)", analysis_dir.name, flags=re.IGNORECASE):
        plane = match.group(1).lower()
        return hv_map.get(plane, plane)

    raise ValueError(f"Could not determine kick plane from {analysis_dir}")


def create_summary(input_dirs: Iterable[Path], do_analysis: AnalysisOption = AnalysisOption.auto, extract_bbq: bool = False, detuning_order: int = 1) -> pd.DataFrame:
    """Create a summary dataframe from the headers of the kick_ampdet_xy files in the input directories
    which can be used to determine the detuning values from multiple detuning analysis results.

    This function can also run the detuning analysis, in case the given input directories contain the kick files.
    The analysis is run, if either the kick_ampdet_xy file is missing and the ``do_analysis`` flag is set to ``auto``
    of it the flag is set to ``always``.
    In case the analysis is run, the following assumptions are made about the naming scheme of the given folders:

    - They start with ``B{beam}_`` or have a parents parent directory of the form ``LHCB{beam}`` (i.e. the output structure of the GUI).
    - They contain the plane in which the detuning was measured, i.e. the kick was increased,
      separated by dots, underscores or at the end of the name.

    Args:
        input_dirs (Iterable[Path]): The input directories.
        do_analysis (AnalysisOption, optional): Whether to run the detuning analysis.
            'always' will always run the analysis, even if the kick_ampdet_xy file is present,
            'auto' only if it is missing, and 'never' assumes this file to be present.
            Defaults to 'auto.
        extract_bbq (bool, optional): Whether to extract the bbq data even if already present. Defaults to False.
        detuning_order (int, optional): The detuning order. Defaults to 1.

    Returns:
        pd.DataFrame: The detuning summary dataframe
    """
    summary_df = tfs.TfsDataFrame()
    for analysis_dir in input_dirs:
        kick_ampdet_file = analysis_dir / get_kick_out_name()

        if do_analysis == AnalysisOption.always or (do_analysis == AnalysisOption.auto and not kick_ampdet_file.is_file()):
            kick_analysed = do_detuning_analysis(analysis_dir, extract_bbq=extract_bbq, detuning_order=detuning_order)
        else:
            if not kick_ampdet_file.is_file():
                raise ValueError(f"Kick file {kick_ampdet_file} not found!")

            kick_analysed = read_timed_dataframe(analysis_dir / get_kick_out_name())

        summary_df = tfs.concat([summary_df, get_row_from_odr_headers(kick_df=kick_analysed, name=analysis_dir.name)])
    return summary_df


def do_detuning_analysis(analysis_dir: Path, extract_bbq: bool = False, detuning_order: int = 1):
    """Run the detuning analysis on the kick files in the given directory.
    Similar to :func:`omc3.amplitude_detuning_analysis.analyse_with_bbq_corrections`,
    but a bit simplified (e.g. predefined filter, only sigle kick plane) and getting `beam` and `kick_plane` from the directory name.

    Args:
        analysis_dir (Path): The analysis directory.
        extract_bbq (bool, optional): Whether to extract the bbq data even if already present. Defaults to False.
        detuning_order (int, optional): The detuning order. Defaults to 1.

    Returns:
        tfs.TfsDataFrame: The kick dataframe
    """
    kick_x_file = analysis_dir / f"{KICK_NAME}x{EXT}"
    kick_y_file = analysis_dir / f"{KICK_NAME}y{EXT}"
    if not kick_x_file.exists() or not kick_y_file.exists():
        raise ValueError(f"Missing kick files in {analysis_dir}")

    beam = get_beam_from_dir(analysis_dir)
    kick_plane = get_kick_plane_from_dir(analysis_dir)

    extracted_bbq_data = analysis_dir / get_bbq_out_name()
    if extract_bbq or not extracted_bbq_data.exists():
        extracted_bbq_data = None

    kick_df, _ = get_kick_and_bbq_df(
        kick=analysis_dir,
        bbq_in=extracted_bbq_data,
        beam=beam,
        filter_opt=OutlierFilterOpt(window=100, limit=0.0),
    )

    # analyse kick-data file
    return single_action_analysis(kick_df, kick_plane, detuning_order=detuning_order, corrected=True)


def get_terms_and_error_terms(terms: Iterable[DetuningTerm]) -> list[str]:
    """Get the terms and error terms for the given terms.

    Args:
        terms (Iterable[DetuningTerm]): The terms.

    Returns:
        list[str]: The terms and error terms.
    """
    return [f"{prefix}{term}" for term in terms for prefix in ("", ERR)]


def get_row_from_odr_headers(kick_df: tfs.TfsDataFrame, name: str):
    """Turn relevant (order > 1, corrected terms) header items into a single-row dataframe.
    This assumes we always use the BBQ correction, which is the default for the analysis.

    Args:
        kick_df (tfs.TfsDataFrame): The kick dataframe.
        name (str): The name of the analysis.

    Returns:
        pd.DataFrame: A dataframe with the relevant header items as a single row.
    """
    all_terms = Detuning.all_terms()
    df = pd.DataFrame(index=[name], columns=get_terms_and_error_terms(all_terms))
    for term in all_terms:
        order = get_order(term)
        tune_plane, action_plane = detuning_term_to_planes(term)
        if len(set(action_plane)) > 1:
            continue  # dual-action second order detuning is not implemented in omc3 analysis (yet)

        action_plane = action_plane[0]
        coeff = get_odr_header_coeff_corrected(tune_plane, action_plane, order)
        err = get_odr_header_err_coeff_corrected(tune_plane, action_plane, order)

        if coeff in kick_df.headers:
            df.loc[name, term] = kick_df.headers[coeff]

        if err in kick_df.headers:
            df.loc[name, f"{ERR}{term}"] = kick_df.headers[err]

    return df


def get_detuning_from_series(series: pd.Series) -> Detuning | DetuningMeasurement:
    """Get the detuning from the given series.
    If the series contains error terms, a :obj:`DetuningMeasurement` is returned,
    otherwise a :obj:`Detuning` is returned.

    Args:
        series (pd.Series): The series, which has detuning terms as index.

    Returns:
        Detuning | DetuningMeasurement: The detuning object.
    """
    if not any(series.index.str.startswith(ERR)):
        return Detuning(**series)

    return DetuningMeasurement(**{
        term: MeasureValue(series.loc[term], series.get(f"{ERR}{term}", 0.0)) for term in Detuning.all_terms() if term in series
    })
