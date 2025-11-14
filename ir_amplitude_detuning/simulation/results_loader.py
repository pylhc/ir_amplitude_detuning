"""
Simulation Results Loaders
--------------------------

Load and sort the simulated detuning data into handy datastructures.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import tfs

from ir_amplitude_detuning.detuning.calculations import FIELDS, IP
from ir_amplitude_detuning.detuning.measurements import Detuning, DetuningMeasurement, MeasureValue
from ir_amplitude_detuning.utilities.common import BeamDict
from ir_amplitude_detuning.utilities.constants import (
    AMPDET_CALC_ERR_ID,
    AMPDET_CALC_ID,
    AMPDET_ID,
    ERR,
    NOMINAL_ID,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from ir_amplitude_detuning.utilities.correctors import FieldComponent

LOG = logging.getLogger(__name__)


DetuningPerBeam = BeamDict[int, Detuning]


def load_simulation_output_tfs(folder: Path, type_: str, beam: int, id_: str) -> tfs.TfsDataFrame:
    """Load simluation output in tfs form.
    Assumes the simulation writes in the following pattern:
    {type}.{anything}.b{beam}.{id}.tfs

    Args:
        folder (Path): The folder containing the data.
        type_ (str): The type of data to load (e.g. ampet, settings).
        beam (int): The beam number.
        id_ (str): The id of the data (e.g. target name).
    """
    if beam in (2, 4):
        beam = '[24]'
    glob = f"{type_}.*.b{beam}.{id_}.tfs"
    for filename in folder.glob(glob):
        return tfs.read(filename)
    raise FileNotFoundError(f"No file matching '{glob}' in {folder}.")


def get_detuning_from_ptc_output(df: pd.DataFrame,  terms: Sequence[str] = Detuning.all_terms()) -> Detuning:
    """Convert PTC output to a Series.

    Args:
        df (DataFrame): DataFrame as given by PTC.
        terms (Sequence[str]): Terms to extract
    """
    results = Detuning()
    for term in terms:
        value = df.query(
            f'NAME == "ANH{term[0]}" and '
            f'ORDER1 == {term[1]} and ORDER2 == {term[2]} '
            f'and ORDER3 == 0 and ORDER4 == 0'
        )["VALUE"].to_numpy()[0]
        results[term] = value
    LOG.debug(f"Extracted detuning values:\n{results}")
    return results


def load_ptc_detuning(folder: Path, beam: int, id_: str) -> Detuning:
    """Load detuning data from PTC output for the given beam and target.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number.
        id_ (str): The id of the data (target name).
    """
    df = load_simulation_output_tfs(folder=folder, type_=AMPDET_ID, beam=beam, id_=id_)
    return get_detuning_from_ptc_output(df)


def convert_dataframe_to_dict(df: pd.DataFrame) -> dict[str, Detuning | DetuningMeasurement]:
    """Convert a dataframe containing detuning-term columns into a dictionary of Detuning objects,
    sorted by the index of the dataframe.

    Args:
        df (pd.Dataframe): Dataframe to be converted.
    """
    error_columns = df.columns.str.startswith(ERR)

    # without errors
    if not any(error_columns):
        return {key: Detuning(**series) for key, series in df.iterrows()}

    # with errors
    result = {}
    for key, series in df.iterrows():
        values = series[~error_columns]
        measure_values = pd.Series(
            {term: MeasureValue(value, series.loc[f"{ERR}{term}"]) for term, value in values.items()}
        )
        result[key] = DetuningMeasurement(**measure_values)
    return result


def get_calculated_detuning_for_ip(
    folder: Path,
    beam: int,
    id_: str,
    ip: str,
    errors: bool = False
    ) -> dict[str, Detuning]:
    """Load and sort the detuning data for a given IP.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number.
        id_ (str): The id of the data (target name).
        ip (str): The IP(s) to load. If multiple can be given as a single string, e.g. "15",
            as this is how the data should be stored in the dataframe.
        errors (bool, optional): Whether to load the errors or not.

    Returns:
        pd.DataFrame: The detuning data for the given IP in a dictionary, sorted by the different fields in the file.
    """
    type_ = AMPDET_CALC_ID if not errors else AMPDET_CALC_ERR_ID
    df = load_simulation_output_tfs(folder=folder, type_=type_, beam=beam, id_=id_)
    ip_mask = df[IP] == ip
    if sum(ip_mask) == 0:
        raise ValueError(f"No data for IP {ip} in {folder} for beam {beam} and id {id_}.")
    df_ip = df.loc[ip_mask, :].drop(columns=[IP]).set_index(FIELDS, drop=True)
    return convert_dataframe_to_dict(df_ip)



def get_calculated_detuning_for_field(
    folder: Path,
    beam: int,
    id_: str,
    field: Iterable[FieldComponent] | FieldComponent | str,
    errors: bool = False,
    ) -> dict[str, Detuning]:
    """Load and sort the detuning data for a given set of fields.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number.
        id_ (str): The id of the data (target name).
        field (Iterable[FieldComponent] | FieldComponent):
            The field(s) to load. If multiple are given they will be converted into a single string, e.g. "b5b6",
            as this is how the data should be stored in the dataframe.
        errors (bool, optional): Whether to load the errors or not.

    Returns:
        dict[str, Detuning]: The Detuning data in a dictionary, sorted by the different IPs in the file.
    """
    type_ = AMPDET_CALC_ID if not errors else AMPDET_CALC_ERR_ID
    df = load_simulation_output_tfs(folder=folder, type_=type_, beam=beam, id_=id_)

    if not isinstance(field, str):
        field = ''.join(sorted(field))

    fields_mask = df[FIELDS] == field
    if sum(fields_mask) == 0:
        raise ValueError(f"No data for fields {field} in {folder} for beam {beam} and id {id_}.")

    df_fields = df.loc[fields_mask, :].drop(columns=[FIELDS]).set_index(IP, drop=True)
    return convert_dataframe_to_dict(df_fields)


def get_detuning_change_ptc(
    folder: Path,
    ids: Iterable[str],
    beams: Iterable[int],
    ):
    """Load the detuning data from PTC simulations for the given set of ids (target names)
    and return their change with respect to the nominal values.

    Args:
        folder (Path): The folder containing the data.
        ids (str): The ids of the data (target names).
        beams (int): The beam numbers.

    """
    ptc_data = {id_: BeamDict({beam: load_ptc_detuning(folder, beam, id_) for beam in beams}) for id_ in ids}
    nominal_data = BeamDict({beam: load_ptc_detuning(folder, beam, NOMINAL_ID) for beam in beams})
    for id_ in ids:
        ptc_data[id_] = ptc_data[id_] - nominal_data
    return ptc_data
