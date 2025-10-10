"""
Setup Commissioning 2022
------------------------

In this example, the dodecapole corrections are calculated based on
the measurements performed during the commissioning in 2022.

This data has been analyzed via the amplitude detuning analysis tool of omc3
and the resulting detuning values have been entered manually below to be used here.

You can find the data in https://gitlab.cern.ch/jdilly/lhc_amplitude_detuning_summary/ .
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ir_amplitude_detuning.detuning.measurements import scaled_detuningmeasurement
from ir_amplitude_detuning.lhc_detuning_corrections import (
    LHCBeams,
    calculate_corrections,
    check_corrections_analytically,
    check_corrections_ptc,
    create_optics,
    get_nominal_optics,
)
from ir_amplitude_detuning.plotting.correctors import plot_correctors
from ir_amplitude_detuning.plotting.detuning import MeasurementSetup, plot_measurements
from ir_amplitude_detuning.plotting.utils import get_full_target_labels
from ir_amplitude_detuning.simulation.lhc_simulation import LHCCorrectors
from ir_amplitude_detuning.simulation.results_loader import get_detuning_change_ptc
from ir_amplitude_detuning.utilities.classes_accelerator import (
    FieldComponent,
    fill_corrector_masks,
)
from ir_amplitude_detuning.utilities.classes_targets import (
    Target,
    TargetData,
)
from ir_amplitude_detuning.utilities.common import Container, dict_diff
from ir_amplitude_detuning.utilities.logging import log_setup

if TYPE_CHECKING:
    from collections.abc import Sequence


# Define Machine Data
# -------------------
class LHCSimulationParameters(Container):
    beams: tuple[int, int] = 1, 4
    year: int = 2022
    outputdir: Path = Path("commissioning2022")
    xing: dict[str, str | float] = {'scheme': 'flat', 'on_x1_v': -150, 'on_x5_h': 150}  # scheme: all off ("flat") apart from IP1 and IP5
    tune_x: float = 62.28  # horizontal tune
    tune_y: float = 60.31  # vertical tune


# Fill in measurement data in 10^3 m^-1
# dictionary keys represent beam, 2 or 4 will make no difference
MEAS_FLAT = {
    1: scaled_detuningmeasurement(X10=(-15.4, 0.9), X01=(33.7, 1), Y01=(-8.4, 0.5)),
    2: scaled_detuningmeasurement(X10=(-8.7, 0.7), X01=(13, 2), Y01=(10, 0.9)),
}

MEAS_FULL = {
    1: scaled_detuningmeasurement(X10=(20, 4), X01=(43, 4), Y01=(-10, 3)),
    2: scaled_detuningmeasurement(X10=(26, 0.8), X01=(-27, 4), Y01=(18, 7)),
}


# Steps of calculations --------------------------------------------------------

def get_targets(lhc_beams: LHCBeams | None = None) -> Sequence[Target]:
    """ Define the targets to be used.

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
        lhc_beams = LHCSimulationParameters.beams

    targets = [
        Target(
            name="X10X01Y01_IP15",
            data=[
                TargetData(
                    correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(1, 5)),
                    detuning=dict_diff(MEAS_FLAT, MEAS_FULL),
                    optics=get_nominal_optics(lhc_beams, outputdir=LHCSimulationParameters.outputdir),
                ),
            ]
        ),
    ]
    return targets  # noqa: R504


def simulation():
    """ Create LHC optics with the set crossing scheme.

    Here:
    IP1 and IP5 crossing active.
    """
    return create_optics(**LHCSimulationParameters)


def do_correction(lhc_beams: LHCBeams | None = None):
    """ Calculate the dodecapole corrections for the LHC for the set targets.

    Also calculates the individual contributions per corrector order and IP to
    the individual detuning terms.
    """
    results = calculate_corrections(
        beams=LHCSimulationParameters.beams,
        outputdir=LHCSimulationParameters.outputdir,
        targets=get_targets(lhc_beams),
    )

    check_corrections_analytically(
        outputdir=LHCSimulationParameters.outputdir,
        optics=get_nominal_optics(lhc_beams or LHCSimulationParameters.beams, outputdir=LHCSimulationParameters.outputdir),
        results=list(results.values())[0],  # single target
    )


def check_correction(lhc_beams: LHCBeams | None = None):
    """ Check the corrections via PTC. """
    check_corrections_ptc(
        lhc_beams=lhc_beams,
        **LHCSimulationParameters,  # apart form outputdir only used if lhc_beams is None
    )


def plot_detuning_comparison():
    """ Plot the measured detuning values.
    As well as the target (i.e. the detuning that should be compensated) and the reached detuning values by the correction."""
    target = get_targets()[0]  # only one target here
    ptc_diff = get_detuning_change_ptc(
        LHCSimulationParameters.outputdir,
        ids=[target.name],
        beams=LHCSimulationParameters.beams
    )
    for beam in (1, 2):
        setup = [
            MeasurementSetup(
                label="Flat Orbit",
                measurement=MEAS_FLAT[beam],
            ),
            MeasurementSetup(
                label="Full X-ing",
                measurement=MEAS_FULL[beam],
            ),
            MeasurementSetup(
                label="Delta",
                measurement=-target.data[0].detuning[beam],
                simulation=-ptc_diff[target.name][beam],
            ),
            MeasurementSetup(
                label="Expected",
                measurement=-(target.data[0].detuning[beam] - ptc_diff[target.name][beam]),  # keep order to keep errorbars
            ),
        ]
        style_adaptions = {
            "figure.figsize": [7.0, 3.0],
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.0,
        }
        fig = plot_measurements(setup, ylim=[-55, 55], average=True, ncol=4, manual_style=style_adaptions)
        fig.savefig(LHCSimulationParameters.outputdir / f"plot.ampdet_comparison.b{beam}.pdf")



def plot_corrector_strengths():
    outputdir = LHCSimulationParameters.outputdir
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



if __name__ == '__main__':
    log_setup()
    lhc_beams = None  # in case you want to skip the simulation
    lhc_beams = simulation()
    do_correction(lhc_beams=lhc_beams)
    check_correction(lhc_beams=lhc_beams)
    plot_detuning_comparison()
    plot_corrector_strengths()
