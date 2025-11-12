"""
Setup Commissioning 2022
------------------------

`Source on github  <https://github.com/pylhc/ir_amplitude_detuning/blob/master/examples/2022_commissioning.py>`_.

In this example, the dodecapole corrections are calculated based on
the measurements performed during the commissioning in 2022.

This data has been analyzed via the amplitude detuning analysis tool of omc3
and the resulting detuning values have been entered manually below to be used here.


You can find the data in https://gitlab.cern.ch/jdilly/lhc_amplitude_detuning_summary/
and Table 7.2 of [DillyThesis2024]_ .
Some more information can be found in Chapter 7.4.1 of the same document.
In particular, Table 7.3 contrains, in the "Commissioning 2022" rows,
the results of the correction performed here.
The values are slightly different, due to the use of the wrong tunes (.31, .32) in the simulation.
As you can see from the plots, this does not change the trends:
The resulting detuning values are depicted in Figure 7.4 (blue and orange values).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ir_amplitude_detuning.detuning.calculations import Method
from ir_amplitude_detuning.detuning.measurements import scaled_detuningmeasurement
from ir_amplitude_detuning.detuning.targets import (
    Target,
    TargetData,
)
from ir_amplitude_detuning.detuning.terms import (
    FirstOrderTerm,
)
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
from ir_amplitude_detuning.plotting.utils import OtherColors, get_color_for_ip
from ir_amplitude_detuning.simulation.lhc_simulation import LHCCorrectors
from ir_amplitude_detuning.simulation.results_loader import (
    DetuningPerBeam,
    get_calculated_detuning_for_field,
    get_detuning_change_ptc,
)
from ir_amplitude_detuning.utilities.common import BeamDict, Container
from ir_amplitude_detuning.utilities.correctors import (
    FieldComponent,
    fill_corrector_masks,
)
from ir_amplitude_detuning.utilities.logging import log_setup

if TYPE_CHECKING:
    from collections.abc import Sequence


# Define Machine Data
# -------------------
class LHCSimParams2022(Container):
    beams: tuple[int, int] = 1, 4
    year: int = 2022
    outputdir: Path = Path("2022_commissioning")
    xing: dict[str, str | float] = {'scheme': 'flat', 'on_x1_v': -150, 'on_x5_h': 150}  # scheme: all off ("flat") apart from IP1 and IP5
    tune_x: float = 62.28  # horizontal tune
    tune_y: float = 60.31  # vertical tune


# Measurement data in 10^3 m^-1; keys represent beam, 2 or 4 will make no difference
class Meas2022(Container):
    flat: DetuningPerBeam = BeamDict({
        1: scaled_detuningmeasurement(X10=(-15.4, 0.9), X01=(33.7, 1), Y01=(-8.4, 0.5)),
        2: scaled_detuningmeasurement(X10=(-8.7, 0.7), X01=(13, 2), Y01=(10, 0.9)),
    })
    full: DetuningPerBeam = BeamDict({
        1: scaled_detuningmeasurement(X10=(20, 4), X01=(43, 4), Y01=(-10, 3)),
        2: scaled_detuningmeasurement(X10=(26, 0.8), X01=(-27, 4), Y01=(18, 7)),
    })

# Steps of calculations --------------------------------------------------------

def get_targets(lhc_beams: LHCBeams | None = None) -> Sequence[Target]:
    """Define the targets to be used.

    Here:

        Calculate the values for the dodecapole correctors in the LHC to compensate
        for the shift in measured detuning from the flat to the full crossing scheme
        (i.e. crossing active in IP1 and IP5).
        The optics used are only with crossing scheme in IP1 and IP5 active,
        assuming zero detuning at flat-orbit in the simulation.

    Note:
        The detuning target should be the opposite of the measured detuning,
        such that the calculated correction compensates the measured detuning.
        This is why here it is "flat-full".
    """
    if lhc_beams is None:
        lhc_beams = LHCSimParams2022.beams

    return [
        Target(
            name="X10X01Y01_IP15",
            data=[
                TargetData(
                    correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(1, 5)),
                    detuning=Meas2022.flat - Meas2022.full,
                    optics=get_nominal_optics(lhc_beams, outputdir=LHCSimParams2022.outputdir),
                ),
            ]
        ),
    ]


def simulation():
    """Create LHC optics with the set crossing scheme.

    Here:
        IP1 and IP5 crossing active.
    """
    return create_optics(**LHCSimParams2022)


def do_correction(lhc_beams: LHCBeams | None = None):
    """Calculate the dodecapole corrections for the LHC for the set targets.

    Also calculates the individual contributions per corrector order and IP to
    the individual detuning terms.
    """
    results = calculate_corrections(
        beams=LHCSimParams2022.beams,
        outputdir=LHCSimParams2022.outputdir,
        targets=get_targets(lhc_beams),
        method=Method.numpy,  # as we do not define any constraints, we can use numpy and get errorbars on the results

    )

    check_corrections_analytically(
        outputdir=LHCSimParams2022.outputdir,
        optics=get_nominal_optics(lhc_beams or LHCSimParams2022.beams, outputdir=LHCSimParams2022.outputdir),
        results=list(results.values())[0],  # single target
    )


def check_correction(lhc_beams: LHCBeams | None = None):
    """Check the corrections via PTC."""
    check_corrections_ptc(
        lhc_beams=lhc_beams,
        **LHCSimParams2022,  # apart form outputdir only used if lhc_beams is None
    )


# Plotting ---------------------------------------------------------------------

def plot_detuning_comparison():
    """Plot the measured detuning values.
    As well as the target (i.e. the detuning that should be compensated) and
    the reached detuning values by the correction.
    """
    target = get_targets()[0]  # only one target here
    ptc_diff = get_detuning_change_ptc(
        LHCSimParams2022.outputdir,
        ids=[target.name],
        beams=LHCSimParams2022.beams
    )
    for beam in (1, 2):
        setup = [
            PlotSetup(
                label="flat orbit",
                measurement=Meas2022.flat[beam],
                color=OtherColors.flat,
            ),
            PlotSetup(
                label="full X-ing",
                measurement=Meas2022.full[beam],
                color=get_color_for_ip('15'),
            ),
            PlotSetup(
                label="delta",
                measurement=-target.data[0].detuning[beam],
                simulation=-ptc_diff[target.name][beam],
            ),
            PlotSetup(
                label="expected",
                measurement=-(target.data[0].detuning[beam] - ptc_diff[target.name][beam]),  # keep order to keep errorbars
            ),
        ]
        style_adaptions = {
            "figure.figsize": [7.0, 3.0],
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.0,
        }
        fig = plot_measurements(
            setup,
            ylim=[-55, 55],
            average=True,
            ncol=4,
            manual_style=style_adaptions,
            terms=[FirstOrderTerm.X10, FirstOrderTerm.X01, FirstOrderTerm.Y01],
        )
        fig.savefig(LHCSimParams2022.outputdir / f"plot.ampdet_comparison.b{beam}.pdf")


def plot_simulation_comparison():
    """Plot the target detuning and how close the different simulation results are.

    Note, that we here show the actual target.
    In the other examples we usually plot the detuning change from flat to full crossing,
    i.e. the inverse of the target here.

    Shows:

    - Target vs. PTC
    - Target vs. calculated contributions from IP1 & IP5
    - PTC vs. calculated contributions from IP1 & IP5
    - Calculated contributions from IP1 & IP5 vs. IP1

    """
    target = get_targets()[0]  # only one target here
    ptc_diff = get_detuning_change_ptc(
        LHCSimParams2022.outputdir,
        ids=[target.name],
        beams=LHCSimParams2022.beams
    )
    for beam in (1, 2):
        calculated = get_calculated_detuning_for_field(
            folder=LHCSimParams2022.outputdir,
            beam=beam,
            id_=target.name,
            field=FieldComponent.b6,
            errors=True,
            )

        setup = [
            PlotSetup(
                label="Target & PTC",
                measurement=target.data[0].detuning[beam],
                simulation=ptc_diff[target.name][beam],
                color='#d62728',  # blue already used for IP15
            ),
            PlotSetup(
                label="Target & Calc. IP15",
                measurement=target.data[0].detuning[beam],
                simulation=calculated['15'],
            ),
            PlotSetup(
                label="Calculated IP15",
                measurement=calculated['15'],  # plot marker with errors
                simulation=calculated['15'],   # plot bar
                color=get_color_for_ip('15'),
            ),
            PlotSetup(
                label="Calculated IP1",
                measurement=calculated['1'],
                simulation=calculated['1'],
                color=get_color_for_ip('1'),
            ),
            PlotSetup(
                label="Calculated IP5",
                measurement=calculated['5'],
                simulation=calculated['5'],
                color=get_color_for_ip('5'),
            )
        ]
        style_adaptions = {
            "figure.figsize": [7.0, 4.0],
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.0,
        }
        fig = plot_measurements(
            setup,
            ylim=[-55, 55],
            ncol=2,
            manual_style=style_adaptions,
            terms=[FirstOrderTerm.X10, FirstOrderTerm.X01, FirstOrderTerm.Y01],
            transpose_legend=True,
            is_shift=True,
        )
        fig.savefig(LHCSimParams2022.outputdir / f"plot.ampdet_sim_comparison.b{beam}.pdf")



def plot_corrector_strengths():
    """Plot the corrector strengths."""
    outputdir = LHCSimParams2022.outputdir
    target = get_targets()[0]  # only one target here
    ips = '15'
    fig = plot_correctors(
        outputdir,
        ids={target.name: "Feed-Down Correction"},
        corrector_pattern=LHCCorrectors.b6.circuit_pattern.format(side="[LR]", ip=f"[{ips}]").replace(".", r"\."),
        field=FieldComponent.b6,
        beam=1,  # does not matter as the same correctors are used for both beams
    )
    fig.savefig(outputdir / f"plot.b6_correctors.ip{ips}.pdf")


# Run --------------------------------------------------------------------------

if __name__ == '__main__':
    log_setup()
    lhc_beams = None  # in case you want to skip the simulation
    # lhc_beams = simulation()
    do_correction(lhc_beams=lhc_beams)
    check_correction(lhc_beams=lhc_beams)
    plot_detuning_comparison()
    plot_simulation_comparison()
    plot_corrector_strengths()
