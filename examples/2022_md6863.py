"""
Setup for MD6863 (2022)
-----------------------

`Source on github <https://github.com/pylhc/ir_amplitude_detuning/blob/master/examples/2022_md6863.py>`_.

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

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from ir_amplitude_detuning.detuning.calculations import Method
from ir_amplitude_detuning.detuning.measurements import DetuningMeasurement
from ir_amplitude_detuning.detuning.targets import Target, TargetData
from ir_amplitude_detuning.detuning.terms import FirstOrderTerm
from ir_amplitude_detuning.lhc_detuning_corrections import (
    LHCBeams,
    calculate_corrections,
    check_corrections_analytically,
    check_corrections_ptc,
    create_optics,
    get_nominal_optics,
)
from ir_amplitude_detuning.plotting.correctors import plot_correctors
from ir_amplitude_detuning.plotting.detuning import PlotSetup, plot_measurements
from ir_amplitude_detuning.plotting.utils import OtherColors, get_color_for_field, get_color_for_ip
from ir_amplitude_detuning.simulation.lhc_simulation import LHCCorrectors
from ir_amplitude_detuning.simulation.results_loader import (
    DetuningPerBeam,
    get_calculated_detuning_for_field,
)
from ir_amplitude_detuning.utilities.common import BeamDict, Container, StrEnum
from ir_amplitude_detuning.utilities.correctors import FieldComponent, fill_corrector_masks
from ir_amplitude_detuning.utilities.logging import log_setup
from ir_amplitude_detuning.utilities.measurement_analysis import (
    AnalysisOption,
    create_summary,
    get_detuning_from_series,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


LOG = logging.getLogger(__name__)


class Labels(StrEnum):
    """Labels for the different measurement configurations (as enum to avoid typos)."""
    flat: str = "flat"
    full: str = "full"
    ip5p: str = "ip5p"
    ip5m: str = "ip5m"
    corrected: str = "corrected"


@dataclass
class DetuningData:
    """Dataclass to hold the detuning data per scheme/configuration."""
    flat: DetuningPerBeam | None = None
    full: DetuningPerBeam | None = None
    ip5p: DetuningPerBeam | None = None
    ip5m: DetuningPerBeam | None = None
    corrected: DetuningPerBeam | None = None

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def items(self):
        return iter((field.name, getattr(self, field.name)) for field in fields(self))

class XingSchemes(Container):
    """Crossing schemes used in the measurements."""
    flat: dict[str, float] = {'scheme': 'flat'}
    full: dict[str, float] = {'scheme': 'flat', 'on_x1_v': -160, 'on_x5_h': 160}
    ip5p: dict[str, float] = {'scheme': 'flat', 'on_x5_h': 160}
    ip5m: dict[str, float] = {'scheme': 'flat', 'on_x5_h': -160}


class LHCSimParams2022(Container):
    """LHC simulation parameters for 2022 MD6863."""
    beams: tuple[int, int] = 1, 4
    year: int = 2022
    outputdir: Path = Path("2022_md6863")
    tune_x: float = 62.28  # horizontal tune
    tune_y: float = 60.31  # vertical tune


# Get Detuning Data
# -----------------
# Note: The functions below often use the naming scheme as in the ``md6863_data``
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


def format_detuning_per_scheme(detuning: DetuningData):
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
    beams = BeamDict()
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


def convert_summary_to_detuning(summary: pd.DataFrame) -> DetuningData:
    """Convert the summary DataFrame to a dictionary of detuning data.

    Args:
        summary (pd.DataFrame): The summary DataFrame.

    Returns:
        DetuningPerScheme: A dictionary of detuning data per scheme and beam.
    """
    summary.index = [get_config_from_name(name) for name in summary.index]

    detuning = DetuningData()
    for xing in XingSchemes:
        detuning[xing] = extract_data_for_both_planes_and_beams(summary, xing=xing, b6corr=False)
    detuning["corrected"] = extract_data_for_both_planes_and_beams(summary, xing="full", b6corr=True)

    return detuning


@cache
def get_detuning_data(redo_analysis: bool = False) -> DetuningData:
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

# Steps of correction calculation ------------------------------------------------------------------

def get_targets(lhc_beams_per_setup: dict[Labels, LHCBeams] | None = None) -> Sequence[Target]:
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
    if lhc_beams_per_setup is None:
        lhc_beams_per_setup = dict.fromkeys(Labels, LHCSimParams2022.beams)

    meas2022 = get_detuning_data()

    # Compensate the global contribution
    target_global = TargetData(
        label=Labels.full,
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(1, 5)),
        detuning=meas2022.flat - meas2022.full,
        optics=get_nominal_optics(
            lhc_beams_per_setup[Labels.full],
            outputdir=LHCSimParams2022.outputdir,
            label=Labels.full
        ),
    )

    # Compensate the IP5 contribution at positive crossing
    target_ip5p = TargetData(
        label=Labels.ip5p,
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(5, )),
        detuning=meas2022.flat - meas2022.ip5p,
        optics=get_nominal_optics(
            lhc_beams_per_setup[Labels.ip5p],
            outputdir=LHCSimParams2022.outputdir,
            label=Labels.ip5p
        ),
    )

    # Compensate the IP5 contribution at negative crossing
    target_ip5m = TargetData(
        label=Labels.ip5m,
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(5, )),
        detuning=meas2022.flat - meas2022.ip5m,
        optics=get_nominal_optics(
            lhc_beams_per_setup[Labels.ip5m],
            outputdir=LHCSimParams2022.outputdir,
            label=Labels.ip5m
        ),
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


def simulation() -> dict[str, LHCBeams]:
    """Create LHC all optics with their respective crossing schemes.

    Here:

         - Flat orbit.
         - IP1 and IP5 crossing active.
         - IP5 positive crossing only (IP1 flat).
         - IP5 negative crossing only (IP1 flat).

    """
    optics = {}
    for scheme in XingSchemes:
        optics[scheme]  = create_optics(
            **LHCSimParams2022,
            xing=XingSchemes[scheme],
            output_id=scheme
        )
    return optics


def do_correction(lhc_beams_per_setup: dict[Labels, LHCBeams] | None = None):
    """Calculate the dodecapole corrections for the LHC for the set targets.

    Also calculates the individual contributions per corrector order and IP to
    the individual detuning terms.
    """
    results = calculate_corrections(
        beams=LHCSimParams2022.beams,
        outputdir=LHCSimParams2022.outputdir,
        targets=get_targets(lhc_beams_per_setup),  # calculate corrections for these targets
        method=Method.numpy,  # No constraints, so calculate with errors
    )

    # Get full-crossing nominal optics for analytical checks
    optics = get_nominal_optics(
        lhc_beams_per_setup or LHCSimParams2022.beams,
        outputdir=LHCSimParams2022.outputdir,
        label=Labels.full
    )

    for values in results.values():  # per target
        check_corrections_analytically(
            outputdir=LHCSimParams2022.outputdir,
            optics=optics,
            results=values,
        )


def check_correction(lhc_beams_per_setup: dict[Labels, LHCBeams] | None = None):
    """Check the corrections via PTC. (Not used for plotting here)."""
    check_corrections_ptc(
        lhc_beams=lhc_beams_per_setup[Labels.full] if lhc_beams_per_setup is not None else None,
        **LHCSimParams2022,  # apart form outputdir only used if lhc_beams is None
    )


# Plotting ---------------------------------------------------------------------

ID_MAP: dict[str, str] = {
    "global": "Global",
    "local_and_global": "Local & Global",
}


def plot_corrector_strengths():
    """Plot the corrector strengths for the different targets.
    These are similar to the green and red values in Fig. 7.12 of [DillyThesis2024]_ .
    """
    outputdir = LHCSimParams2022.outputdir

    fig = plot_correctors(
        outputdir,
        ids=ID_MAP,
        field=FieldComponent.b6,
        ncol=1,
        beam=1,  # does not matter as the same correctors are used for both beams
    )
    fig.axes[0].set_xlim([-1.4, 4.2])
    fig.axes[0].set_ylim([-3.5, 0.2])
    fig.savefig(outputdir / "plot.b6_correctors.ip15.pdf")


def plot_measurement_comparison():
    """Plot the measured detuning values."""
    style_adaptions = {
        "figure.figsize": [6.4, 4.0],
        "legend.handletextpad": 0.4,
    }
    meas2022 = get_detuning_data()

    for beam in (1, 2):
        setup = [
            PlotSetup(
                label="flat orbit",
                measurement=meas2022.flat[beam],
                color=OtherColors.flat,
            ),
            PlotSetup(
                label="full X-ing",
                measurement=meas2022.full[beam],
                color=get_color_for_ip('15'),
            ),
            PlotSetup(
                label="IP5 +160μrad",
                measurement=meas2022.ip5p[beam],
                color=get_color_for_ip('5'),
            ),
            PlotSetup(
                label="IP5 -160μrad",
                measurement=meas2022.ip5m[beam],
                color=get_color_for_field(FieldComponent.b6),
            ),
            PlotSetup(
                label="corrected",
                measurement=meas2022.corrected[beam],
                color=get_color_for_ip('1'),
            )
        ]
        fig = plot_measurements(
            setup,
            ylim=[-70, 70],
            ncol=3,
            transpose_legend=True,
            manual_style=style_adaptions,
            terms=[FirstOrderTerm.X10, FirstOrderTerm.X01, FirstOrderTerm.Y01],
            average=True,
        )
        fig.savefig(LHCSimParams2022.outputdir / f"plot.ampdet_measured.all.b{beam}.pdf")

def plot_target_comparison():
    """Plot the detuning to be compensated (inverse of target) and
    how close the different simulation results get.

    As we have two targets here, multiple plots are created:

     - Comparison with measured detuning values (Fig. 7.5 in [DillyThesis2024]_):
       During MD only the "global" target correction could be applied,
       as this correction was calculated while the IP5 measurements were still ongoing.
       Therefore we can only compare the "global" target here to measurements after correction.
     - Comparison of individual contributions from IP1 and IP5 (Fig. 7.10 in [DillyThesis2024]_):
       Here we can compare the calculated contributions from both corrections (i.e. the two targets)
       to the measured detuning differences from full crossing, the individual contributions from IP1 and IP5
       and the contribution from IP5 dodecapoles only.
       In Fig. 7.10 of [DillyThesis2024]_ only the result from the "local_and_global" target is shown,
       as this is the more complete correction. This function however creates plots for both targets.

    Note:
        We always show the detuning difference from flat to the respective crossing scheme,
        i.e. the expected detuning coming from the crossing scheme change.

        - AmpDet from IP5+ = Ampdet at IP5 positive crossing - AmpDet at flat
        - AmpDet from Full = AmpDet at full crossing - AmpDet at flat = AmpDet from IP1 + AmpDet from IP5
        - AmpDet from IP1 = AmpDet from Full - AmpDet from IP5
        - AmpDet from IP5 dodecapoles =

           | 0.5 * (AmpDet from IP5+ + AmpDet from IP5-) =
           | - 0.5 * (AmpDet at IP5+ + AmpDet at IP5- - 2 * AmpDet at flat)
           | (See Eqs. 7.6 - 7.8 in [DillyThesis2024]_)
    """
    style_adaptions = {
        "figure.figsize": [6.0, 4.0],
        "legend.handletextpad": 0.4,
    }

    meas2022 = get_detuning_data()
    targets = get_targets()

    # Get detuning differences
    # measured
    global_detuning = meas2022.full - meas2022.flat
    ip5p_detuning = meas2022.ip5p - meas2022.flat
    ip5m_detuning = meas2022.ip5m - meas2022.flat
    corrected_detuning = meas2022.corrected - meas2022.flat

    # inferred
    ip1_detuning =  meas2022.full - meas2022.ip5p
    ip5b6_detuning = 0.5 * (ip5p_detuning + ip5m_detuning)

    for beam in (1, 2):
        for target in targets:
            simulation = get_calculated_detuning_for_field(
                folder=LHCSimParams2022.outputdir,
                beam=beam,
                id_=target.name,
                field=FieldComponent.b6,
                errors=False,
            )

            # to add `Detuning` to `DetuningMeasurement` we need to convert the former
            compensation_global = DetuningMeasurement.from_detuning(simulation["15"])

            # Plot comparison of all contributions (Fig. 7.10 in [DillyThesis2024]_) ---
            setup = [
                PlotSetup(
                    label="full X-ing",
                    measurement=global_detuning[beam],
                    simulation=-simulation['15'],
                    color=get_color_for_ip('15'),
                ),
                PlotSetup(
                    label="from IP5",
                    measurement=ip5p_detuning[beam],
                    simulation=-simulation['5'],
                    color=get_color_for_ip('5'),
                ),
                PlotSetup(
                    label="from IP5 $b6$",
                    measurement=ip5b6_detuning[beam],
                    simulation=-simulation['5'],  # same as above
                    color=get_color_for_field(FieldComponent.b6),
                ),
                PlotSetup(
                    label="from IP1",
                    measurement=ip1_detuning[beam],
                    simulation=-simulation['1'],
                    color=get_color_for_ip('1'),
                ),
                PlotSetup(
                    label="estimated",
                    measurement=global_detuning[beam] + compensation_global,
                    color=OtherColors.estimated,
                )
            ]
            fig = plot_measurements(
                setup,
                ylim=[-70, 70],
                ncol=3,
                manual_style=style_adaptions,
                terms=[FirstOrderTerm.X10, FirstOrderTerm.X01, FirstOrderTerm.Y01],
                is_shift=True,
                average=True,
            )
            fig.savefig(LHCSimParams2022.outputdir / f"plot.ampdet_compensation.{target.name}.b{beam}.pdf")

            if target.name == "global":
                # for global target we have a measurement to compare to (Fig. 7.5 in [DillyThesis2024]_) ---
                setup = [
                    PlotSetup(
                        label="w/o $b_6$ corr.",
                        measurement=global_detuning[beam],
                        simulation=-simulation['15'],
                        color=get_color_for_ip('15'),
                    ),
                    PlotSetup(
                        label="estimated",
                        measurement=global_detuning[beam] + compensation_global,
                        color=OtherColors.estimated,
                    ),
                    PlotSetup(
                        label="w/ $b_6$ corr.",
                        measurement=corrected_detuning[beam],
                        color=get_color_for_ip('1'),
                    ),
                ]
                fig = plot_measurements(
                    setup,
                    ylim=[-70, 70],
                    ncol=3,
                    manual_style=style_adaptions,
                    terms=[FirstOrderTerm.X10, FirstOrderTerm.X01, FirstOrderTerm.Y01],
                    is_shift=True,
                    average=True,
                )
                fig.savefig(LHCSimParams2022.outputdir / f"plot.ampdet_compensation_and_measured_corrected.{target.name}.b{beam}.pdf")


# Run --------------------------------------------------------------------------

if __name__ == '__main__':
    log_setup()
    lhc_beams = None  # in case you want to skip the simulation
    # lhc_beams = simulation()
    do_correction(lhc_beams_per_setup=lhc_beams)
    # check_correction(lhc_beams_per_setup=lhc_beams)
    plot_corrector_strengths()
    plot_target_comparison()
    plot_measurement_comparison()
