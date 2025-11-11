"""
Setup 2018 Data
---------------

Example for a filled template based on the 2018 measurements from commissioning
and MD3311.

You can find the data in https://gitlab.cern.ch/jdilly/lhc_amplitude_detuning_summary/
and Table 7.2 of [DillyThesis2024]_ .
The extensive simulations studies are discussed in Chapter 7.3 of the same document
and partially recreated here.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ir_amplitude_detuning.detuning.calculations import Method
from ir_amplitude_detuning.detuning.measurements import (
    Constraints,
    FirstOrderTerm,
    scaled_detuningmeasurement,
)
from ir_amplitude_detuning.detuning.targets import (
    Target,
    TargetData,
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
from ir_amplitude_detuning.plotting.utils import get_color_for_ip
from ir_amplitude_detuning.simulation.lhc_simulation import LHCCorrectors
from ir_amplitude_detuning.simulation.results_loader import (
    DetuningPerBeam,
    get_calculated_detuning_for_field,
)
from ir_amplitude_detuning.utilities.common import BeamDict, Container
from ir_amplitude_detuning.utilities.correctors import (
    FieldComponent,
    fill_corrector_masks,
)
from ir_amplitude_detuning.utilities.logging import log_setup

if TYPE_CHECKING:
    from collections.abc import Sequence


class LHCSimParams2018(Container):
    """LHC simulation parameters for 2018 MD3311."""
    beams: tuple[int, int] = 1, 4
    year: int = 2018
    outputdir: Path = Path("2018_md3311")
    xing: dict[str, str | float] = {'scheme': 'top'}  # scheme: crossing scheme of top-energy collisions
    tune_x: float = 62.28  # horizontal tune
    tune_y: float = 60.31  # vertical tune


class Meas2018(Container):
    """Measured detuning values for different crossing schemes in 10^3 m^-1.
    Note: Keys are beam numbers, 2 and 4 can be used interchangeably (but consistently) here.
    """
    flat: DetuningPerBeam = BeamDict({
        1: scaled_detuningmeasurement(X10=(0.8, 0.5), Y01=(-3, 1)),
        2: scaled_detuningmeasurement(X10=(-7.5, 0.5), Y01=(6, 1)),
    })
    full: DetuningPerBeam = BeamDict({
        1: scaled_detuningmeasurement(X10=(34, 1), Y01=(-38, 1)),
        2: scaled_detuningmeasurement(X10=(-3, 1), Y01=(13, 3)),
    })
    ip5: DetuningPerBeam = BeamDict({
        1: scaled_detuningmeasurement(X10=(56, 6), Y01=(3, 2)),
        2: scaled_detuningmeasurement(X10=(1.5, 0.5), Y01=(12, 1)),
    })
    ip1: DetuningPerBeam = None  # IP1 was not measured, calculated below

# IP1 was not measured, but we can infer from the difference to the IP5 contribution (for plotting)
Meas2018.ip1 = Meas2018.full - Meas2018.ip5 + Meas2018.flat


class Const2018(Container):
    """Constraints to be used in the calculations."""
    negative_crossterm: BeamDict[int, Constraints] = BeamDict({b: Constraints(X01="<=0") for b in (1, 2)})


# Steps of correction calculation ------------------------------------------------------------------

def get_targets(lhc_beams: LHCBeams | None = None) -> Sequence[Target]:
    """Define the targets to be used.

    Here:
        Calculate the values for the dodecapole correctors in the LHC to compensate
        for the shift in measured detuning from the flat to the full crossing scheme
        (i.e. crossing active in IP1 and IP5) and from flat to the IP5 crossing scheme.

        The defined targets are as in Scenarios D, G and approximately I
        in Figure 7.1 of [DillyThesis2024]_ .

    Note:
    The detuning target should be the opposite of the measured detuning,
    such that the calculated correction compensates the measured detuning.
    This is why here it is "flat-full".
    """
    if lhc_beams is None:
        lhc_beams = LHCSimParams2018.beams

    optics = get_nominal_optics(lhc_beams, outputdir=LHCSimParams2018.outputdir)


    # Compensate the global contribution to X10 and Y01 using the
    # decapole correctors in IP1 and IP5.
    target_global = TargetData(
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(1, 5)),
        detuning=Meas2018.flat - Meas2018.full,
        optics=optics,
    )

    # Compensate the global contribution to X10 and Y01,
    # while constraining the crossterm, using the
    # decapole correctors in IP1 and IP5 a.
    target_global_constrained = TargetData(
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(1, 5)),
        detuning=Meas2018.flat - Meas2018.full,
        constraints=Const2018.negative_crossterm,
        optics=optics,
    )

    # Compensate the IP5 contribution to X10 and Y01 using the
    # decapole correctors in IP5 only.
    target_ip5 = TargetData(
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(5, )),
        detuning=Meas2018.flat - Meas2018.ip5,
        optics=optics,  # can use same optics, as the xing in IP5 is the same
    )

    return [
        Target(
            name="global",  # scenario D
            data=[target_global]
        ),
        Target(
            name="local_and_global",  # scenario G
            data=[target_global, target_ip5]
        ),
        Target(
            name="local_and_global_constrained",  # similar to I
            data=[target_global_constrained, target_ip5]
        ),
    ]

def simulation():
    """Create LHC optics with the set crossing scheme.

    Here:
        IP1 and IP5 crossing active.
    """
    return create_optics(**LHCSimParams2018)


def do_correction(lhc_beams: LHCBeams | None = None):
    """Calculate the dodecapole corrections for the LHC for the set targets.

    Also calculates the individual contributions per corrector order and IP to
    the individual detuning terms.
    """
    results = calculate_corrections(
        beams=LHCSimParams2018.beams,
        outputdir=LHCSimParams2018.outputdir,
        targets=get_targets(lhc_beams),  # calculate corrections for these targets
        method=Method.auto,  # some have constraints, some not. Let the solver decide.
    )

    optics = get_nominal_optics(lhc_beams or LHCSimParams2018.beams, outputdir=LHCSimParams2018.outputdir)

    for values in results.values():  # per target
        check_corrections_analytically(
            outputdir=LHCSimParams2018.outputdir,
            optics=optics,
            results=values,
        )


def check_correction(lhc_beams: LHCBeams | None = None):
    """Check the corrections via PTC. (Not used for plotting here)."""
    check_corrections_ptc(
        lhc_beams=lhc_beams,
        **LHCSimParams2018,  # apart form outputdir only used if lhc_beams is None
    )

# Plotting ---------------------------------------------------------------------

ID_MAP: dict[str, str] = {
    "global": "Global",
    "local_and_global": "Local & Global",
    "local_and_global_constrained": "Local & Global (constrained)"
}


def plot_corrector_strengths():
    """Plot the corrector strengths for the different targets."""
    outputdir = LHCSimParams2018.outputdir

    fig = plot_correctors(
        outputdir,
        ids=ID_MAP,
        field=FieldComponent.b6,
        ncol=1,
        lim=[-5.1, 1.1],
        beam=1,  # does not matter as the same correctors are used for both beams
    )
    fig.savefig(outputdir / "plot.b6_correctors.ip15.pdf")


def plot_detunig_compensation():
    """Plot the detuning to be compensated (inverse of target) and
    how close the different simulation results get.

    Shows:
        How well the expected detuning from the corrector powering will
        match global detuning as well as the individual contributions from the IPs,
        for each target individually:

        You can see from the plots, that for the first target the global contribution
        is well matched, yet the individual IPs overshoot (but conpensate each other).

        When trying to match the IPs we loose some global accuracy and get also a
        small amount of positive crossterm.

        Constraining the crossterm to be negative, does not seem to be possible
        and hence the correction tries to keep it at zero, by compensating
        the contributions from IP1 and IP5 exactly.
        This comes at the cost of even worse accuracy for the other terms,
        globally and per IP.
    """
    style_adaptions = {
        "figure.figsize": [6.0, 4.0],
        "legend.handletextpad": 0.4,
    }

    targets = get_targets()

    global_detuning = Meas2018.full - Meas2018.flat
    ip5_detuning = Meas2018.ip5 - Meas2018.flat
    ip1_detuning = Meas2018.ip1 - Meas2018.flat

    for beam in (1, 2):
        calculated = {
            target.name: get_calculated_detuning_for_field(
                folder=LHCSimParams2018.outputdir,
                beam=beam,
                id_=target.name,
                field=FieldComponent.b6,
                errors=False,
            )
            for target in targets
        }

        for target in targets:
            simulation = calculated[target.name]
            setup = [
                PlotSetup(
                    label="Global",
                    measurement=global_detuning[beam],
                    simulation=-simulation['15'],
                    color=get_color_for_ip('15'),
                ),
                PlotSetup(
                    label="IP5",
                    measurement=ip5_detuning[beam],
                    simulation=-simulation['5'],
                    color=get_color_for_ip('5'),
                ),
                PlotSetup(
                    label="IP1",
                    measurement=ip1_detuning[beam],
                    simulation=-simulation['1'],
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
            )
            fig.savefig(LHCSimParams2018.outputdir / f"plot.ampdet_compensation.{target.name}.b{beam}.pdf")

# Run --------------------------------------------------------------------------

if __name__ == '__main__':
    log_setup()
    lhc_beams = None  # in case you want to skip the simulation
    lhc_beams = simulation()
    do_correction(lhc_beams=lhc_beams)
    check_correction(lhc_beams=lhc_beams)
    plot_corrector_strengths()
    plot_detunig_compensation()
