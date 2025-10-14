"""
Setup 2018 Data
---------------

Example for a filled template based on the 2018 measurements from commissioning
and MD3311.

You can find the data in https://gitlab.cern.ch/jdilly/lhc_amplitude_detuning_summary/ .
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ir_amplitude_detuning.detuning.calculations import Method
from ir_amplitude_detuning.detuning.measurements import FirstOrderTerm, scaled_detuningmeasurement
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
from ir_amplitude_detuning.plotting.utils import get_color_for_ip
from ir_amplitude_detuning.simulation.lhc_simulation import LHCCorrectors
from ir_amplitude_detuning.simulation.results_loader import (
    get_calculated_detuning_for_field,
    get_detuning_change_ptc,
)
from ir_amplitude_detuning.utilities.classes_accelerator import (
    FieldComponent,
    fill_corrector_masks,
)
from ir_amplitude_detuning.utilities.classes_targets import (
    Target,
    TargetData,
)
from ir_amplitude_detuning.utilities.common import Container, dict_diff, dict_sum
from ir_amplitude_detuning.utilities.logging import log_setup

if TYPE_CHECKING:
    from collections.abc import Sequence


# Define Machine Data
# -------------------
class LHCSimulationParameters(Container):
    beams: tuple[int, int] = 1, 4
    year: int = 2018
    outputdir: Path = Path("2018_md3311")
    xing: dict[str, str | float] = {'scheme': 'top'}  # scheme: crossing scheme of top-energy collisions
    tune_x: float = 62.28  # horizontal tune
    tune_y: float = 60.31  # vertical tune


# Fill in measurement data in 10^3 m^-1
# dictionary keys represent beam, 2 or 4 will make no difference
MEAS_FLAT = {
    1: scaled_detuningmeasurement(X10=(0.8, 0.5), Y01=(-3, 1)),
    2: scaled_detuningmeasurement(X10=(-7.5, 0.5), Y01=(6, 1)),
}
MEAS_FULL = {
    1: scaled_detuningmeasurement(X10=(34, 1), Y01=(-38, 1)),
    2: scaled_detuningmeasurement(X10=(-3, 1), Y01=(13, 3)),
}
MEAS_IP5 = {
    1: scaled_detuningmeasurement(X10=(56, 6), Y01=(3, 2)),
    2: scaled_detuningmeasurement(X10=(1.5, 0.5), Y01=(12, 1)),
}

# IP1 was not measured, but we can infer from the difference to the IP5 contribution
MEAS_IP1 = dict_sum(dict_diff(MEAS_FULL, MEAS_IP5), MEAS_FLAT)



# Steps of calculations --------------------------------------------------------

def get_targets(lhc_beams: LHCBeams | None = None) -> Sequence[Target]:
    """ Define the targets to be used.

    Here:
        Calculate the values for the dodecapole correctors in the LHC to compensate
        for the shift in measured detuning from the flat to the full crossing scheme
        (i.e. crossing active in IP1 and IP5) and from flat to the IP5 crossing scheme.

        The defined targets are as in Scenarios D and G in Figure 7.1 of [DillyThesis2024]_.

    Note:
    The detuning target should be the opposite of the measured detuning,
    such that the calculated correction compensates the measured detuning.
    This is why here it is "flat-full".
    """
    if lhc_beams is None:
        lhc_beams = LHCSimulationParameters.beams

    optics = get_nominal_optics(lhc_beams, outputdir=LHCSimulationParameters.outputdir)


    # Compensate the global contribution using the
    # decapole correctors in IP1 and IP5.
    target_global = TargetData(
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(1, 5)),
        detuning=dict_diff(MEAS_FLAT, MEAS_FULL),
        optics=optics,
    )

    # Compensate the IP5 contribution using the
    # decapole correctors in IP5 only.
    target_ip5 = TargetData(
        correctors=fill_corrector_masks([LHCCorrectors.b6], ips=(5, )),
        detuning=dict_diff(MEAS_FLAT, MEAS_IP5),
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
    ]


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
        method=Method.numpy,  # as we do not define any constraints, we can use numpy and get errorbars on the results

    )

    optics = get_nominal_optics(lhc_beams or LHCSimulationParameters.beams, outputdir=LHCSimulationParameters.outputdir)

    for values in results.values():
        check_corrections_analytically(
            outputdir=LHCSimulationParameters.outputdir,
            optics=optics,
            results=values,
        )

# Run --------------------------------------------------------------------------

if __name__ == '__main__':
    log_setup()
    lhc_beams = None  # in case you want to skip the simulation
    # lhc_beams = simulation()
    do_correction(lhc_beams=lhc_beams)
    # check_correction(lhc_beams=lhc_beams)
    # plot_detuning_comparison()
    # plot_corrector_strengths()
    # plot_simulation_comparison()