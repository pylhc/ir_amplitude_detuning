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

from ir_dodecapole_corrections.lhc_detuning_corrections import (
    calculate_corrections,
    create_optics,
)
from ir_dodecapole_corrections.plotting.detuning import (
    plot_correctors,
    plot_detuning_ips,
)
from ir_dodecapole_corrections.simulation.lhc_simulation import LHCCorrectors
from ir_dodecapole_corrections.utilities import latex
from ir_dodecapole_corrections.utilities.classes_detuning import scaled_detuningmeasurement
from ir_dodecapole_corrections.utilities.classes_targets import (
    Target,
    TargetData,
)
from ir_dodecapole_corrections.utilities.logging import log_setup
from ir_dodecapole_corrections.utilities.maths import get_diff

if TYPE_CHECKING:
    from collections.abc import Sequence


# Define Machine Data
# -------------------

# Output Path
OUTPUT = Path("commissioning2022")

# Fill in special values for crossing if needed:
XINGS = {'': {'scheme': 'flat', 'on_x1_v': -150, 'on_x5_h': 150}}  # scheme: all off ("flat") apart from IP1 and IP5

# Fill in measurement data in 10^3 m^-1:
MEAS_FLAT = {
    1: scaled_detuningmeasurement(X10=(-15.4, 0.9), X01=(33.7, 1), Y01=(-8.4, 0.5)),
    2: scaled_detuningmeasurement(X10=(-8.7, 0.7), X01=(13, 2), Y01=(10, 0.9)),
}
MEAS_FULL = {
    1: scaled_detuningmeasurement(X10=(20, 4), X01=(43, 4), Y01=(-10, 3)),
    2: scaled_detuningmeasurement(X10=(26, 0.8), X01=(-27, 4), Y01=(18, 7)),
}


# Steps of calculations --------------------------------------------------------

def get_targets() -> Sequence[Target]:
    """ Define the targets to be used.

    Here: Simple global detuning correction based on the difference between
    the flat and full crossing scheme measurements.
    """
    detuning_flat = MEAS_FLAT
    detuning_full = MEAS_FULL
    targets = [
        Target(
            name="X10X01Y01_global",
            data=[
                TargetData(
                    ips=(1, 5),
                    detuning=get_diff(detuning_flat, detuning_full),
                    xing='',
                ),
            ]
        ),
    ]
    return targets  # noqa: R504


def simulation():
    return create_optics(
        beams=(1, 4),
        outputdir=OUTPUT,
        xings=XINGS,
        year=2022
    )


def do_correction(lhc_beams=None):
    calculate_corrections(
        beams=(1, 4),
        outputdir=OUTPUT,
        targets=get_targets(),
        lhc_beams=lhc_beams,
        correctors=(LHCCorrectors.b6,),
        main_xing=''
    )


def plotting():
    action_map = {"10": 'x', "01": 'y'}

    nchar = 10
    measurement = get_diff(MEAS_FLAT, MEAS_FULL)
    measurement[4] = measurement.pop(2)

    targets = get_targets()
    ids = [f"{target.name}_b6" for target in targets]
    terms = ("X10", "X01", "Y01")
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
        OUTPUT,
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
    corr_size = [4.80, 4.00]

    plot_correctors(OUTPUT, ids=ids, labels=labels, size=corr_size, corrector_pattern=r'kctx3\.[lr][15]', order="6", output_id=f'{output_id}_b6')
    plot_correctors(OUTPUT, ids=ids, labels=labels, size=corr_size, corrector_pattern=r'kctx3\.[lr]5', order="6", output_id=f'{output_id}_b6_ip5')
    plot_correctors(OUTPUT, ids=ids, labels=labels, size=corr_size, corrector_pattern=r'kctx3\.[lr]1', order="6", output_id=f'{output_id}_b6_ip1')


if __name__ == '__main__':
    log_setup()
    lhc_beams = None
    lhc_beams = simulation()
    do_correction(lhc_beams=lhc_beams)
    # plotting()
