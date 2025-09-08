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

from ir_dodecapole_corrections.correction_via_amplitude_detuning import run_lhc_and_calc_correction
from ir_dodecapole_corrections.correction_via_amplitude_detuning_multi_xing import (
    calculate_corrections,
)
from ir_dodecapole_corrections.plotting.detuning import (
    plot_correctors,
    plot_detuning_ips,
)
from ir_dodecapole_corrections.utilities import latex
from ir_dodecapole_corrections.utilities.classes import (
    Target,
    TargetData,
    scaled_detuningmeasurement,
)
from ir_dodecapole_corrections.utilities.logging import log_setup
from ir_dodecapole_corrections.utilities.maths import get_diff

if TYPE_CHECKING:
    from collections.abc import Sequence


# Define Machine Data
# -------------------

# Output Path
output = Path("commissioning2022")

# Fill in special values for crossing if needed:
xing = {'scheme': 'flat', 'on_x1': -150, 'on_x5': 150}

# Fill in measurement data in 10^3 m^-1:
meas_flat = {
    1: scaled_detuningmeasurement(X10=(-15.4, 0.9), X01=(33.7, 1), Y01=(-8.4, 0.5)),
    2: scaled_detuningmeasurement(X10=(-8.7, 0.7), X01=(13, 2), Y01=(10, 0.9)),
}
meas_full = {
    1: scaled_detuningmeasurement(X10=(20, 4), X01=(43, 4), Y01=(-10, 3)),
    2: scaled_detuningmeasurement(X10=(26, 0.8), X01=(-27, 4), Y01=(18, 7)),
}


# Steps of calculations --------------------------------------------------------

def get_targets() -> Sequence[Target]:
    # detuning_flat = get_detuning(meas_flat)
    # detuning_full = get_detuning(meas_full)
    detuning_flat = meas_flat
    detuning_full = meas_full
    targets = [
        Target(
            name="X10X01Y01_global",
            data=[
                TargetData(
                    ips=(1, 5),
                    detuning=get_diff(detuning_flat, detuning_full),
                    xing='',  # label for output path
                ),
            ]
        ),
    ]
    return targets  # noqa: R504


def simulation():
    paths = {i: output / f"b{i}" for i in (1, 4)}
    run_lhc_and_calc_correction(paths, xing=xing, targets=get_targets(), year=2022)


def do_correction_only():
    paths = {i: output / f"b{i}" for i in (1, 4)}
    calculate_corrections(paths, targets=get_targets(), main_xing='')


def plotting():
    action_map = {"10": 'x', "01": 'y'}

    nchar = 10
    measurement = get_diff(meas_flat, meas_full)
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
        output,
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

    plot_correctors(output, ids=ids, labels=labels, size=corr_size, corrector_pattern=r'kctx3\.[lr][15]', order="6", output_id=f'{output_id}_b6')
    plot_correctors(output, ids=ids, labels=labels, size=corr_size, corrector_pattern=r'kctx3\.[lr]5', order="6", output_id=f'{output_id}_b6_ip5')
    plot_correctors(output, ids=ids, labels=labels, size=corr_size, corrector_pattern=r'kctx3\.[lr]1', order="6", output_id=f'{output_id}_b6_ip1')


if __name__ == '__main__':
    log_setup()
    simulation()
    # do_correction_only()
    # plotting()
