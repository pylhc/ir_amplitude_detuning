"""
Common Detuning Functions
-------------------------

"""
import logging
from collections.abc import Sequence

import pandas as pd

from ir_amplitude_detuning.detuning.measurements import FirstOrderTerm

LOG = logging.getLogger(__name__)


def get_detuning_from_ptc_output(df: pd.DataFrame,  terms: Sequence[str] = tuple(FirstOrderTerm)) -> pd.Series:
    """ Convert PTC output to a Series.

    Args:
        df (DataFrame): DataFrame as given by PTC.
        terms (Sequence[str]): Terms to extract
    """
    results = pd.Series(dtype=float, index=terms)
    for term in terms:
        value = df.query(
            f'NAME == "ANH{term[0]}" and '
            f'ORDER1 == {term[1]} and ORDER2 == {term[2]} '
            f'and ORDER3 == 0 and ORDER4 == 0'
        )["VALUE"].to_numpy()[0]
        results.loc[term] = value
    LOG.debug(f"Current Detuning Values:\n{results}")
    return results
