"""
Example for a filled template based on the 2018 measurements.
"""
from pathlib import Path
from typing import Any, Dict

from ir_dodecapole_corrections.utilities.logging import log_setup
from ir_dodecapole_corrections.correction_via_amplitude_detuning import run_lhc_and_calc_correction as run_lhc_and_calc_correction
from ir_dodecapole_corrections.utilities.classes import (Detuning, DetuningMeasurement, Target,
                                                         TargetData, scaled_detuningmeasurement)
from ir_dodecapole_corrections.utilities.plotting import plot_correctors, plot_detuning_ips


def get_sum(meas_a, meas_b):
    return {beam: meas_a[beam] + meas_b[beam] for beam in meas_a.keys()}

def get_diff(meas_a, meas_b):
    return {beam: meas_a[beam] - meas_b[beam] for beam in meas_a.keys()}


# Define Machine Data
# -------------------

# Output Path
output = Path("2018_xing_b1b4")


# Fill in special values for crossing if needed:
xing = {'scheme': 'top'}

# Fill in measurement data in 10^3 m^-1, (value, error):
meas_flat = {
    1: scaled_detuningmeasurement(X10=(0.8, 0.5), Y01=(-3, 1)),
    2: scaled_detuningmeasurement(X10=(-7.5, 0.5), Y01=(6, 1)),
}
meas_full = {
    1: scaled_detuningmeasurement(X10=(34, 1), Y01=(-38, 1)),
    2: scaled_detuningmeasurement(X10=(-3, 1), Y01=(13, 3)),
}
meas_ip5 = {
    1: scaled_detuningmeasurement(X10=(56, 6), Y01=(3, 2)),
    2: scaled_detuningmeasurement(X10=(1.5, 0.5), Y01=(12, 1)),
}

# If one or the other was not measured (for local corrections in IP)

meas_ip1 = get_sum(get_diff(meas_full, meas_ip5), meas_flat)
# meas_ip5 = get_sum(get_diff(meas_full, meas_ip1), meas_flat)


# Steps of calculations --------------------------------------------------------

def ltx_dqd2j(tune, action, power=1):
    if power == 1:
        return f"$Q_{{{tune},{action}}}$"
    return f"$Q_{{{tune},{action}^{{{power}}}}}$"


def get_detuning(meas: Dict[Any, DetuningMeasurement]) ->  Dict[Any, Detuning]:
    return {beam: meas[beam].get_detuning() for beam in meas.keys()}


def get_targets():
    detuning_flat = get_detuning(meas_flat)
    detuning_full = get_detuning(meas_full)
    detuning_ip1 = get_detuning(meas_ip1)
    detuning_ip5 = get_detuning(meas_ip5)
    targets = [
        Target(
            name="X10Y01_local_global",
            data=[
                # GLOBAL CORRECTION
                TargetData(
                    ips=(1, 5),
                    detuning=get_diff(detuning_flat, detuning_full),
                    # constraints={
                    #     1: ScaledConstraints(X01="<=0"),
                    #     2: ScaledConstraints(X01="<=0"),
                    # }
                ),
                # LOCAL CORRECTION
                # TargetData(
                #     ips=(1, ),
                #     detuning=get_diff(detuning_flat, detuning_ip1),
                #     # constraints={
                #     #     1: ScaledConstraints(X01="<=0"),
                #     #     2: ScaledConstraints(X01="<=0"),
                #     # }
                # ),
                TargetData(
                    ips=(5, ),
                    detuning=get_diff(detuning_flat, detuning_ip5),
                    # constraints={
                    #     1: ScaledConstraints(X01="<=0"),
                    #     2: ScaledConstraints(X01="<=0"),
                    # }
                ),
            ]
        ),
        # # TARGET TEMPLATE
        # Target(
        #     name="X10X01Y01_",
        #     data=[
        #         TargetData(
        #             ips=(1, ),
        #             detuning={
        #                 1: ScaledDetuning(X10=0, X01=0, Y01=0),
        #                 2: ScaledDetuning(X10=0, X01=0, Y01=0),
        #             },
        #             constraints={
        #                 1: ScaledConstraints(X01="<=0"),
        #                 2: ScaledConstraints(X01="<=0"),
        #             }
        #         ),
        #     ]
        # )
    ]
    return targets


def simulation():
    paths = {i: output / f"b{i}" for i in (1, 4)}
    run_lhc_and_calc_correction(paths, xing=xing, targets=get_targets())


def plotting():
    action_map = {"10": 'x', "01": 'y'}

    nchar = 10
    measurement = get_diff(meas_flat, meas_full)
    measurement[4] = measurement.pop(2)
    measurement = {k: meas*1e-3 for k, meas in measurement.items()}

    targets = get_targets()
    ids = [f"{target.name}_b6" for target in targets]
    labels = [
            f"{ltx_dqd2j('x', 'x')} = {'55.2,-22 | 9,-4.5'.center(nchar)}"
    ]

    output_id = f"_corrections"
    plot_detuning_ips(
        output,
        ids=ids,
        labels=labels,
        fields="b6",
        size=[20., 7.],
        measurement=measurement,
        beams=(1, 4),
        ylims={1: [-62, 62], 2: [-3, 3]},
        tickrotation=0,
        output_id=output_id,
        alternative="separate",  # "separate", "normal"
    )
    corr_size = [6.00, 6.90]

    plot_correctors(output, ids=ids, labels=labels, size=corr_size, corrector_pattern='kctx3\.[lr][15]', order="6", output_id=f'{output_id}_b6')
    plot_correctors(output, ids=ids, labels=labels, size=corr_size, corrector_pattern='kctx3\.[lr]5', order="6", output_id=f'{output_id}_b6_ip5')
    plot_correctors(output, ids=ids, labels=labels, size=corr_size, corrector_pattern='kctx3\.[lr]1', order="6", output_id=f'{output_id}_b6_ip1')


if __name__ == '__main__':
    log_setup()
    simulation()
    plotting()
