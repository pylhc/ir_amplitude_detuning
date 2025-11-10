"""
Setup for MD6863 (2022)
-----------------------

In this example, the dodecapole corrections are calculated based on
the measurements performed during MD6863 in 2022.

This setup is the most complicated one of the examples given in this package,
as we are not only calculating the correction, but also extract the data directly
from the omc3 output directories.
In fact, you can modify this example easily below, to even run the detuning
analysis in omc3 first.

To achieve this automation, a naming scheme for the output-directories is assumed (see :func:`get_config_from_name`),
that allows this script to sort the measurements into the different machine settings used,
these are:

- With full crossing scheme in IP1 and IP5
- With flat crossing scheme in IP1 and IP5
- With positive crossing scheme in IP5
- With negative crossing scheme in IP5

The naming scheme is as follows:

``b$BEAM_1_$XING1_5_$XING5_ampdet_$PLANE_b6_$CORR``

Where:
- ``$BEAM`` is the beam number
- ``$XING1`` and ``$XING5`` are the IP1 and IP5 crossing schemes, respectively, in signed-integer murad or 'off' for flat
- ``$PLANE`` is the plane of the kick, either 'H' or 'V' or 'X' or 'Y'
- ``$CORR`` is the whether there is b6 correction, either 'in' or 'out'

The resulting data is then used to calculate the correction.

You can find the detunig as well in https://gitlab.cern.ch/jdilly/lhc_amplitude_detuning_summary/
and Table 7.2 of [DillyThesis2024]_ ; there are minor differences due to different analysis settings.

Some more information can be found in Chapter 7.4.1 of [DillyThesis2024]_ .
In particular, Table 7.3 contains, in the "MD6863" rows,
the results of the corrections performed here.

The resulting detuning values are depicted in Figures 7.5, 7.7 and 7.10.
"""
# data labels as used in the detuning summary
from functools import cache
import logging
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence, TypeAlias

import pandas as pd
from ir_amplitude_detuning.detuning.calculations import Method

from ir_amplitude_detuning.detuning.targets import Target, TargetData
from ir_amplitude_detuning.lhc_detuning_corrections import LHCBeams, calculate_corrections, get_nominal_optics, Method
from ir_amplitude_detuning.simulation.lhc_simulation import LHCCorrectors
from ir_amplitude_detuning.simulation.results_loader import DetuningPerBeam
from ir_amplitude_detuning.utilities.common import Container, StrEnum, dict_diff
from ir_amplitude_detuning.utilities.correctors import fill_corrector_masks
from ir_amplitude_detuning.utilities.logging import log_setup
from ir_amplitude_detuning.utilities.measurement_analysis import (
    AnalysisOption,
    create_summary,
    get_detuning_from_series,
)


LOG = logging.getLogger(__name__)

class Labels(StrEnum):
    flat: str = "flat"
    full: str = "full"
    ip5p: str = "ip5p"
    ip5m: str = "ip5m"
    corrected: str = "corrected"

@dataclass
class DetuningSchemes:
    flat: DetuningPerBeam | None = None
    full: DetuningPerBeam | None = None
    ip5p: DetuningPerBeam | None = None
    ip5m: DetuningPerBeam | None = None
    corrected: DetuningPerBeam | None = None

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def items(self):
        return iter((field.name, getattr(self, field.name)) for field in fields(self))


class LabelSchemes(Container):
    flat: tuple[str, str] = ("2022 MD6863 w/o $b_6$", "flat-orbit")
    full: tuple[str, str] = ("2022 MD6863 w/o $b_6$", r"IP1\&5 xing @ $\mp$\qty{160}{\micro\radian}")
    ip5p: tuple[str, str] = ("2022 MD6863 w/o $b_6$", r"IP5 xing @ $+$\qty{160}{\micro\radian}")
    ip5m: tuple[str, str] = ("2022 MD6863 w/o $b_6$", r"IP5 xing @ $-$\qty{160}{\micro\radian}")
    corrected: tuple[str, str] = ("2022 MD6863 w/ $b_6$", r"IP1\&5 xing @ $\mp$\qty{160}{\micro\radian}")


# Fill in special values for crossing if needed:
class XingSchemes(Container):
    flat: dict[str, float] = {'scheme': 'flat'}
    full: dict[str, float] = {'scheme': 'flat', 'on_x1_v': -160, 'on_x5_h': 160}
    ip5p: dict[str, float] = {'scheme': 'flat', 'on_x5_h': 160}
    ip5m: dict[str, float] = {'scheme': 'flat', 'on_x5_h': -160}


# Define Machine Data
# -------------------
class LHCSimParams2022(Container):
    beams: tuple[int, int] = 1, 4
    year: int = 2022
    outputdir: Path = Path("2022_md6863")
    tune_x: float = 62.28  # horizontal tune
    tune_y: float = 60.31  # vertical tune


# Get Detuning Data
# -----------------
# Note: These functions often use the naming scheme as in the ``md6863_data``
#       folder and are hence quite specific to this data set.
@dataclass
class MeasurementConfig:
    """Configuration for a measurement.
    Used to quickly organize the data from the ``md6863_data`` folder into the
    different "schemes" as defined in this file.
    """
    beam: int
    xing: str
    kick_plane: str
    b6corr: bool

    def __hash__(self):
        return hash((self.beam, self.xing, self.kick_plane, self.b6corr))


def format_detuning_per_scheme(detuning: DetuningSchemes):
    """Format the detuning data for printing.

    Args:
        detuning (DetuningPerScheme): The detuning data per scheme and beam.
    """
    parts = ["\nLoaded detuning data for MD6863 [10^-3 m^-1]:"]
    for scheme, beams in detuning.items():
        indent = " " * 4
        parts.append(f"{indent}{scheme}")
        for beam, measured in beams.items():
            indent = " " * 8
            parts.append(f"{indent}Beam {beam}: ")
            for name, value in measured.items():
                indent = " " * 12
                parts.append(f"{indent}{name} = {value*1e-3: 5.1f}")
    parts.append("")
    return "\n".join(parts)


def get_config_from_name(name: str) -> MeasurementConfig:
    """Create a MeasurementConfig from a measurement name, as used in the ``md6863_data`` folder.
    The `MeasurementConfig` contains the beam number, the kick plane and the xing scheme as well as the b6 correction
    and is only used to quickly organize the data into the different "schemes" as defined in this file.

    Args:
        name (str): The name of the measurement (i.e. the folder name).

    Returns:
        MeasurementConfig: The measurement configuration.
    """
    if match := re.match(r"^b(?P<beam>\d)_1_(?P<ip1>off|[+-]\d+)_5_(?P<ip5>off|[+-]\d+)_AmpDet_(?P<plane>[HV])_b6_(?P<corr>[^_]+)$", name.lower(), flags=re.IGNORECASE):
        xing_map = {(XingSchemes[name].get("on_x1_v"), XingSchemes[name].get("on_x5_h")): name for name in XingSchemes}
        ip1= None if match.group("ip1") == "off" else int(match.group("ip1"))
        ip5= None if match.group("ip5") == "off" else int(match.group("ip5"))
        return MeasurementConfig(
            beam=int(match.group("beam")),
            kick_plane={"h": "x", "v": "y"}[match.group("plane")],
            xing=xing_map[(ip1, ip5)],
            b6corr=match.group("corr") == "in",
        )
    raise ValueError(f"Could not determine measurement configuration from {name}")


def extract_data_for_both_planes_and_beams(summary: pd.DataFrame, xing: str, b6corr: bool) -> DetuningPerBeam:
    """Extract the data for both planes and beams from the summary DataFrame.

    Args:
        summary (pd.DataFrame): The summary DataFrame.
        xing (str): The xing scheme.
        b6corr (bool): Whether to apply the b6 correction.

    Returns:
        DetuningPerBeam: A dictionary of detuning data, merged for both planes, by beams as keys.
    """
    beams = {}
    for beam in (1, 2):
        rows_xy = [MeasurementConfig(beam=beam, xing=xing, kick_plane=plane, b6corr=b6corr) for plane in "xy"]
        summary_xy = summary.loc[rows_xy, :]
        merged = summary_xy.max(skipna=True).dropna()
        merged_test = summary_xy.min(skipna=True).dropna()

        if any(merged != merged_test):
            raise ValueError(
                "Detuning data is inconsistent. There are non-matching values in the same entry for both kick planes."
                " Expected are ``NaN`` values in at least one of the planes."
                f" Something is wrong with the data for {rows_xy[0]} or {rows_xy[1]}.")

        beams[beam] = get_detuning_from_series(merged).apply_acdipole_correction()
    return beams


def convert_summary_to_detuning(summary: pd.DataFrame) -> DetuningSchemes:
    """Convert the summary DataFrame to a dictionary of detuning data.

    Args:
        summary (pd.DataFrame): The summary DataFrame.

    Returns:
        DetuningPerScheme: A dictionary of detuning data per scheme and beam.
    """
    summary.index = [get_config_from_name(name) for name in summary.index]

    detuning = DetuningSchemes()
    for xing in XingSchemes:
        detuning[xing] = extract_data_for_both_planes_and_beams(summary, xing=xing, b6corr=False)
    detuning["corrected"] = extract_data_for_both_planes_and_beams(summary, xing="full", b6corr=True)

    return detuning


@cache
def get_detuning_data(redo_analysis: bool = False) -> DetuningSchemes:
    """Extract the detuning measurement values from the analysed data in the `md6863_data` folder.

    The values are automatically corrected for the influence of forced oscillations (see [DillyAmplitudeDetuning2023]_).
    This data is presented in Table 7.2 of [DillyThesis2024]_ ; the values might be slightly
    different due to different analysis settings, but should be within errorbar.

    As the raw kick-data and BBQ data is also present in these folders,
    you can choose to re-analyse the data for detuning values by setting ``redo_analysis`` to ``True``,
    otherwise simply the already analysed ``kick_ampdet_xy.tfs`` files are loaded.
    To change the analysis settings, you need to manually edit :func:`~ir_amplitude_detuning.utilities.measurement_analysis.do_detuning_analysis`.

    Args:
        redo_analysis (AnalysisOption, optional): Whether to re-analyse the data.
            Defaults to 'never'.
    """
    summary = create_summary(
        input_dirs=(Path(__file__).parent / "md6863_data").glob("B*"),
        do_analysis=AnalysisOption.always if redo_analysis else AnalysisOption.never,
    )

    detuning = convert_summary_to_detuning(summary)
    LOG.info(format_detuning_per_scheme(detuning))

    return detuning


@cache
def get_targets(lhc_beams: LHCBeams | None = None) -> Sequence[Target]:
    """Define the targets to be used.

    Here:
        Calculate the values for the dodecapole correctors in the LHC to compensate
        for the shift in measured detuning from the flat to the full crossing scheme
        (i.e. crossing active in IP1 and IP5) and from flat to the IP5 crossing schemes.

        The defined targets are as in Chapter 7.4.1 of [DillyThesis2024]_ ,
        named there "w/o IP5" (here: "global") and "w/ IP5" (here: "local_and_global").

    Note:
    The detuning target should be the opposite of the measured detuning,
    such that the calculated correction compensates the measured detuning.
    This is why here it is "flat-xing".
    """
    if lhc_beams is None:
        lhc_beams = LHCSimParams2022.beams

    meas2022 = get_detuning_data()

    # Compensate the global contribution
    target_global = TargetData(
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(1, 5)),
        detuning=dict_diff(meas2022.flat, meas2022.full),
        optics=get_nominal_optics(lhc_beams, outputdir=LHCSimParams2022.outputdir, label=Labels.full),
    )

    # Compensate the IP5 contribution at positive crossing
    target_ip5p = TargetData(
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(5, )),
        detuning=dict_diff(meas2022.flat, meas2022.ip5p),
        optics=get_nominal_optics(lhc_beams, outputdir=LHCSimParams2022.outputdir, label=Labels.ip5p),
    )

    # Compensate the IP5 contribution at negative crossing
    target_ip5m = TargetData(
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(5, )),
        detuning=dict_diff(meas2022.flat, meas2022.ip5m),
        optics=get_nominal_optics(lhc_beams, outputdir=LHCSimParams2022.outputdir, label=Labels.ip5m),
    )

    return [
        Target(
            name="global",
            data=[target_global]
        ),
        Target(
            name="local_and_global",
            data=[target_global, target_ip5p, target_ip5m]
        ),
    ]


def do_correction(lhc_beams: LHCBeams | None = None):
    """Calculate the dodecapole corrections for the LHC for the set targets.

    Also calculates the individual contributions per corrector order and IP to
    the individual detuning terms.
    """
    results = calculate_corrections(
        beams=LHCSimParams2022.beams,
        outputdir=LHCSimParams2022.outputdir,
        targets=get_targets(lhc_beams),
        method=Method.numpy,  # No constraints, so calculate with errors
    )

    optics = get_nominal_optics(lhc_beams or LHCSimParams2022.beams, outputdir=LHCSimParams2022.outputdir)

    for values in results.values():
        check_corrections_analytically(
            outputdir=LHCSimParams2018.outputdir,
            optics=optics,
            results=values,
        )













if __name__ == "__main__":
    log_setup()
    lhc_beams = None  # in case you want to skip the simulation
    # lhc_beams = simulation()
    # do_correction(lhc_beams=lhc_beams)
    get_detuning_data()