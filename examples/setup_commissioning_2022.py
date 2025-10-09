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
    check_corrections_ptc,
    check_corrections_analytically,
    create_optics,
    get_nominal_optics,
)
from ir_amplitude_detuning.plotting.compare_measurements import plot_measurements, MeasurementSetup

from ir_amplitude_detuning.plotting.detuning import (
    plot_detuning_ips,
)
from ir_amplitude_detuning.plotting.correctors import plot_correctors
from ir_amplitude_detuning.simulation.lhc_simulation import LHCCorrectors
from ir_amplitude_detuning.utilities import latex
from ir_amplitude_detuning.utilities.classes_accelerator import (
    FieldComponent,
    fill_corrector_masks,
)
from ir_amplitude_detuning.utilities.classes_targets import (
    Target,
    TargetData,
)
from ir_amplitude_detuning.utilities.logging import log_setup
from ir_amplitude_detuning.utilities.misc import get_diff, detuning_short_to_planes

if TYPE_CHECKING:
    from collections.abc import Sequence


# Define Machine Data
# -------------------

class LHCSimulationParameters:
    beams: tuple[int, int] = 1, 4
    year: int = 2022
    outputdir: Path = Path("commissioning2022")
    xing: dict[str, str | float] = {'scheme': 'flat', 'on_x1_v': -150, 'on_x5_h': 150}  # scheme: all off ("flat") apart from IP1 and IP5
    tune_x: float = 62.28,  # horizontal tune
    tune_y: float = 60.31,  # vertical tune

    @classmethod
    def as_dict(cls) -> dict[str, int | float | str]:
        return {k: getattr(cls, k) for k in cls.__annotations__}


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
                    detuning=get_diff(MEAS_FLAT, MEAS_FULL),
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
    return create_optics(**LHCSimulationParameters.as_dict())


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
        **LHCSimulationParameters.as_dict(),  # apart form outputdir only used if lhc_beams is None
    )


def plot_measured():
    """ Plot the measured detuning values. """
    diff = get_diff(MEAS_FULL, MEAS_FLAT)
    for beam in (1, 2):
        setup = [
            MeasurementSetup(
                label="Flat Orbit",
                measurement=MEAS_FLAT[beam],
            ),
            MeasurementSetup(
                label="Full Crossing",
                measurement=MEAS_FULL[beam],
            ),
            MeasurementSetup(
                label="Difference",
                measurement=diff[beam],
            ),
        ]
        fig = plot_measurements(setup)
        fig.savefig(LHCSimulationParameters.outputdir / f"measurements_b{beam}.pdf")


def get_plot_labels(nchar: int = 13, scale: float = 1e-3) -> dict[str, str]:
    targets = get_targets()

    ids = [targets[0].name]
    target_data = targets[0].data[0]  # only one in this setup

    scaled_values = {
        term: (target_data.detuning[1][term]*scale, target_data.detuning[2][term]*scale) for term in target_data.detuning[1].terms()
    }
    labels = [
        "\n".join([
            f"${latex.dqd2j(*detuning_short_to_planes(term))}$ = {f'{values[0].value: 2.1f} | {values[1].value: 2.1f}'.center(nchar)}"
            for term, values in scaled_values.items()
        ])
    ]
    return dict(zip(ids, labels))


def plot_corrector_strengths():
    outputdir = LHCSimulationParameters.outputdir
    ips = '15'
    fig = plot_correctors(
        outputdir,
        ids=get_plot_labels(),
        corrector_pattern=LHCCorrectors.b6.circuit_pattern.format(side="[LR]", ip=f"[{ips}]").replace(".", r"\."),
        field=FieldComponent.b6,
        beam=1,  # does not matter as the same correctors are used for both beams
    )
    fig.savefig(outputdir / f"correctors_ip{ips}.pdf")


def plotting():
    outputdir = LHCSimulationParameters.outputdir

    targets = get_targets()
    ids = [f"{target.name}_b6" for target in targets]
    scaled = [{term: targets[0].data[0][b][term]*1e-3 for term in terms} for b in (1, 2)]
    labels = [
        "\n".join([
            f"{latex.dqd2j(term[0], action_map[term[1:]])} = {f'{scaled[0][term]:.1f} | {scaled[1][term]}'.center(nchar)}"
            for term in terms
            ]
        )
    ]

    output_id = "_corrections"
    plot_detuning_ips(
        outputdir,
        ids=ids,
        labels=labels,
        fields="b6",
        size=[6, 3.9],
        measurement=measurement,
        beams=(1, 4),
        ylims={1: [-62, 62], 2: [-3, 3]},
        tickrotation=0,
        output_id=output_id,
        alternative="separate",  # "separate", "normal"
        delta=True,
    )



if __name__ == '__main__':
    log_setup()
    lhc_beams = None
    # lhc_beams = simulation()
    # do_correction(lhc_beams=lhc_beams)
    # check_correction(lhc_beams=lhc_beams)
    # plot_measured()
    plot_corrector_strengths()
    # plotting()
