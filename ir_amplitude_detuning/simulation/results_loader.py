"""
Simulation Results Loaders
--------------------------

Load and sort the simulated detuning data into handy datastructures.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tfs

from ir_amplitude_detuning.detuning.calculations import FIELDS, IP
from ir_amplitude_detuning.detuning.measurements import Detuning
from ir_amplitude_detuning.utilities.constants import AMPDET_CALC_ID, AMPDET_ID, NOMINAL_ID
from ir_amplitude_detuning.utilities.common import BeamDict, dict_diff

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import pandas as pd

    from ir_amplitude_detuning.utilities.classes_accelerator import FieldComponent

LOG = logging.getLogger(__name__)


DetuningPerBeam = dict[int, Detuning]


def load_simulation_output_tfs(folder: Path, type_: str, beam: int, id_: str) -> tfs.TfsDataFrame:
    """ Load simluation output in tfs form.
    Assumes the simulation writes in the following pattern:
    {type}.{anything}.b{beam}.{id}.tfs

    Args:
        folder (Path): The folder containing the data.
        type_ (str): The type of data to load (e.g. ampet, settings).
        beam (int): The beam number.
        id_ (str): The id of the data (e.g. target name).
    """
    glob = f"{type_}.*.b{beam}.{id_}.tfs"
    for filename in folder.glob(glob):
        return tfs.read(filename)
    raise FileNotFoundError(f"No file matching '{glob}' in {folder}.")


def get_detuning_from_ptc_output(df: pd.DataFrame,  terms: Sequence[str] = Detuning.all_terms()) -> Detuning:
    """ Convert PTC output to a Series.

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
    """ Load detuning data from PTC output for the given beam and target.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number.
        id_ (str): The id of the data (target name).
    """
    df = load_simulation_output_tfs(folder, AMPDET_ID,beam, id_)
    return get_detuning_from_ptc_output(df)


def convert_dataframe_to_dict(df: pd.DataFrame) -> dict[str, Detuning]:
    """ Convert a dataframe containing detuning-term columns into a dictionary of Detuning objects,
    sorted by the index of the dataframe.

    Args:
        df (pd.Dataframe): Dataframe to be converted.
    """
    return {key: Detuning(**series) for key, series in df.iterrows()}


def get_calculated_detuning_for_ip(folder: Path, beam: int, id_: str, ip: str) -> dict[str, Detuning]:
    """ Load and sort the detuning data for a given IP.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number.
        id_ (str): The id of the data (target name).
        ip (str): The IP(s) to load. If multiple can be given as a single string, e.g. "15",
            as this is how the data should be stored in the dataframe.

    Returns:
        pd.DataFrame: The detuning data for the given IP in a dictionary, sorted by the different fields in the file.
    """
    df = load_simulation_output_tfs(folder, AMPDET_CALC_ID, beam, id_)
    ip_mask = df[IP] == ip
    if sum(ip_mask) == 0:
        raise ValueError(f"No data for IP {ip} in {folder} for beam {beam} and id {id_}.")
    df_ip = df.loc[ip_mask, :].set_index(FIELDS, drop=True)
    return convert_dataframe_to_dict(df_ip)



def get_calculated_detuning_for_field(folder: Path, beam: int, id_: str, field: Iterable[FieldComponent] | FieldComponent | str
    ) -> dict[str, Detuning]:
    """ Load and sort the detuning data for a given set of fields.

    Args:
        folder (Path): The folder containing the data.
        beam (int): The beam number.
        id_ (str): The id of the data (target name).
        field (Iterable[FieldComponent] | FieldComponent):
            The field(s) to load. If multiple are given they will be converted into a single string, e.g. "b5b6",
            as this is how the data should be stored in the dataframe.

    Returns:
        dict[str, Detuning]: The Detuning data in a dictionary, sorted by the different IPs in the file.
    """
    df = load_simulation_output_tfs(folder, beam, id_, AMPDET_CALC_ID)

    if not isinstance(field, str):
        field = ''.join(sorted(field))

    fields_mask = df[FIELDS] == field
    if sum(fields_mask) == 0:
        raise ValueError(f"No data for fields {field} in {folder} for beam {beam} and id {id_}.")

    df_fields = df.loc[fields_mask, :].set_index(IP, drop=True)
    return convert_dataframe_to_dict(df_fields)


def get_detuning_change_ptc(
    folder: Path,
    ids: Iterable[str],
    beams: Iterable[int],
    ):
    """ Load the detuning data from PTC simulations for the given set of ids (target names)
    and return their change with respect to the nominal values.

    Args:
        folder (Path): The folder containing the data.
        ids (str): The ids of the data (target names).
        beams (int): The beam numbers.

    """
    ptc_data = {id_: {beam: load_ptc_detuning(folder, beam, id_) for beam in beams} for id_ in ids}
    nominal_data = {beam: load_ptc_detuning(folder, beam, NOMINAL_ID) for beam in beams}
    for id_ in ids:
        ptc_data[id_] = BeamDict.from_dict(dict_diff(ptc_data[id_], nominal_data))
    return ptc_data
