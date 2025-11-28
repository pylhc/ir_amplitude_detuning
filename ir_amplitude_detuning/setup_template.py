"""
Template
--------

This is a template script, that can be filled with new measurement data
and then be run to output corrections and plots.

It is very similar in structure to the Commissioning 2022 example.
There are `TODO` markers all over the template to help you find where
to plug in your data.

Step 1: Fill in Measured Data (optionally: and desired Constraints)
Step 2: Define your correction targets in `get_targets()` function.
Step 3: Define the things you want to plot.
Step 4: Run the script.

See the "examples" folder for alredy filled-in examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ir_amplitude_detuning.detuning.calculations import Method
from ir_amplitude_detuning.detuning.measurements import Constraints, scaled_detuningmeasurement
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


class LHCSimParams(Container):
    """LHC simulation parameters.

    TODO: Fill in your actual machine setup!
    """
    beams: tuple[int, int] = 1, 4
    year: int = 2026
    outputdir: Path = Path("temp_output")
    xing: dict[str, str | float] = {'scheme': 'flat', 'on_x1_v': -150, 'on_x5_h': 150}  # scheme: all off ("flat") apart from IP1 and IP5
    tune_x: float = 62.28  # horizontal tune
    tune_y: float = 60.31  # vertical tune


class MeasuredDetuning(Container):
    """Measured detuning values for different crossing schemes in 10^3 m^-1.
    Note: Keys are beam numbers, 2 and 4 can be used interchangeably (but consistently) here.

    TODO: Fill in what you measured!
    """
    flat: DetuningPerBeam = BeamDict({
        1: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
        2: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
    })
    full: DetuningPerBeam = BeamDict({
        1: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
        2: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
    })

class CorrectionConstraints(Container):
    """Constraints to be used in the calculations.

    TODO: Add contstraints you want to use! Example here, is to force the cross-term to be negative.
    """
    negative_crossterm: BeamDict[int, Constraints] = BeamDict({b: Constraints(X01="<=0") for b in (1, 2)})

# Steps of calculations --------------------------------------------------------

def get_targets(lhc_beams: LHCBeams | None = None) -> Sequence[Target]:
    """Define the targets to be used.

    Note:
        The detuning target should be the opposite of the measured detuning,
        such that the calculated correction compensates the measured detuning.
        This is why here it is "flat-full".

    TODO: Define your own targets, using the measurement and constraints from above.
    """
    if lhc_beams is None:
        lhc_beams = LHCSimParams.beams

    return [
        Target(
            name="global_correction",
            data=[
                TargetData(
                    label="ips15",
                    correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(1, 5)),
                    detuning=MeasuredDetuning.flat - MeasuredDetuning.full,
                    optics=get_nominal_optics(lhc_beams, outputdir=LHCSimParams.outputdir),
                ),
            ]
        ),
        Target(
            name="global_correction_constrained",
            data=[
                TargetData(
                    label="ips15",
                    correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(1, 5)),
                    detuning=MeasuredDetuning.flat - MeasuredDetuning.full,
                    constraints=CorrectionConstraints.negative_crossterm,
                    optics=get_nominal_optics(lhc_beams, outputdir=LHCSimParams.outputdir),
                ),
            ]
        ),
    ]


def simulation():
    """Create LHC optics with the set crossing scheme.

    Here:
        IP1 and IP5 crossing active.
    """
    return create_optics(**LHCSimParams)


def do_correction(lhc_beams: LHCBeams | None = None):
    """Calculate the dodecapole corrections for the LHC for the set targets.

    Also calculates the individual contributions per corrector order and IP to
    the individual detuning terms.
    """
    results = calculate_corrections(
        beams=LHCSimParams.beams,
        outputdir=LHCSimParams.outputdir,
        targets=get_targets(lhc_beams),
        method=Method.numpy,  # as we do not define any constraints, we can use numpy and get errorbars on the results

    )

    check_corrections_analytically(
        outputdir=LHCSimParams.outputdir,
        optics=get_nominal_optics(lhc_beams or LHCSimParams.beams, outputdir=LHCSimParams.outputdir),
        results=list(results.values())[0],  # single target
    )


def check_correction(lhc_beams: LHCBeams | None = None):
    """Check the corrections via PTC."""
    check_corrections_ptc(
        lhc_beams=lhc_beams,
        **LHCSimParams,  # apart form outputdir only used if lhc_beams is None
    )


# Plotting ---------------------------------------------------------------------
# TODO: Think of nicer plotting functions!

def plot_detuning_comparison():
    """Plot the measured detuning values.
    As well as the target (i.e. the detuning that should be compensated) and
    the reached detuning values by the correction.
    """
    target = get_targets()[0]  # only one target here
    ptc_diff = get_detuning_change_ptc(
        LHCSimParams.outputdir,
        ids=[target.name],
        beams=LHCSimParams.beams
    )
    for beam in (1, 2):
        setup = [
            PlotSetup(
                label="flat orbit",
                measurement=MeasuredDetuning.flat[beam],
                color=OtherColors.flat,
            ),
            PlotSetup(
                label="full X-ing",
                measurement=MeasuredDetuning.full[beam],
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
        fig.savefig(LHCSimParams.outputdir / f"plot.ampdet_comparison.b{beam}.pdf")


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
        LHCSimParams.outputdir,
        ids=[target.name],
        beams=LHCSimParams.beams
    )
    for beam in (1, 2):
        calculated = get_calculated_detuning_for_field(
            folder=LHCSimParams.outputdir,
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
        fig.savefig(LHCSimParams.outputdir / f"plot.ampdet_sim_comparison.b{beam}.pdf")



def plot_corrector_strengths():
    """Plot the corrector strengths."""
    outputdir = LHCSimParams.outputdir
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
