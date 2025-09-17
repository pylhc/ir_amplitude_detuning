"""
Template
--------

This is a template script, that can be filled with new measurement data
and then be run to output corrections and plots.

Step 1: Fill in Measured Data
Step 2: Define your correction targets in `get_targets()` function.
Step 3: Define the labels for your targets in the `plotting()` function.

See the "examples" folder for alredy filled-in examples.
"""
from collections.abc import Sequence
from pathlib import Path

from ir_dodecapole_corrections.lhc_detuning_corrections import (
    calculate_corrections,
)
from ir_dodecapole_corrections.plotting.detuning import (
    plot_correctors,
    plot_detuning_ips,
)
from ir_dodecapole_corrections.utilities import latex
from ir_dodecapole_corrections.utilities.classes import (
    Detuning,  # noqa: F401 -> used in EXAMPLE
    Target,
    TargetData,
    scaled_detuningmeasurement,
)
from ir_dodecapole_corrections.utilities.logging import log_setup
from ir_dodecapole_corrections.utilities.maths import get_detuning, get_diff

# Define Machine Data
# -------------------

# Output Path
output = Path("newmeas_xing_b1b4")

# Fill in special values for crossing if needed:
# scheme can be e.g. 'inj' for injection scheme, 'top' for top-energy scheme or 'flat' for flat-orbit.
# You can add additional keys for e.g. on_x1, on_x5, if needed.
# See cpymad_lhc.ir_orbit for details.
xing = {'scheme': 'top'}

# Fill in measurement data in 10^3 m^-1:
meas_flat = {
    1: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
    2: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
}
meas_full = {
    1: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
    2: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
}
meas_ip5 = {
    1: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
    2: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
}
meas_ip1 = {
    1: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
    2: scaled_detuningmeasurement(X10=(0, 0), X01=(0, 0), Y01=(0, 0)),
}

# If one or the other was not measured (for local corrections in IP)

# meas_ip1 = get_sum(get_diff(meas_full, meas_ip5), meas_flat)
# meas_ip5 = get_sum(get_diff(meas_full, meas_ip1), meas_flat)


# Steps of calculations --------------------------------------------------------

def get_targets() -> Sequence[Target]:
    # # BASIC TARGET TEMPLATE/EXAMPLE
    # # ---------------------
    # # They are given as a list of different targets,
    # # which will be matched separately, e.g for testing different constraints
    # targets = [
    #     Target(
    #         name="X10X01Y01",  # identifyier for this target (name of your choice)
    #         data=[  # all of these TargetDatas are matched together
    #             TargetData(
    #                 ips=(1,),  # ip's that contribute to this detuning
    #                 detuning={
    #                     # dict of Detuning Values per beam.
    #                     1: Detuning(X10=0e3, X01=0e3, Y01=0e3),
    #                     2: Detuning(X10=0e3, X01=0e3, Y01=0e3),
    #                 },
    #                 constraints={
    #                     # dict of Constraints per beam.
    #                     1: Detuning(X01="<=0e3"),  # example constraint
    #                     2: Detuning(X01="<=0e3"),
    #                 },
    #                 xing='',  # label for the crossing scheme, if needed
    #             ),
    #         ]
    #     ),
    # ]

    # TARGET EXAMPLE using measured data from above
    detuning_flat = get_detuning(meas_flat)  # value without errors, as we cannot include errors yet
    detuning_full = get_detuning(meas_full)
    detuning_ip1 = get_detuning(meas_ip1)
    detuning_ip5 = get_detuning(meas_ip5)
    targets = [
        Target(
            name="X10X01Y01_local_global",
            data=[
                # GLOBAL CORRECTION, i.e. both IPs
                TargetData(
                    ips=(1, 5),
                    detuning=get_diff(detuning_flat, detuning_full),  # change with crossing scheme
                    # constraints={
                    #     1: scaled_constraints(X01="<=0"),
                    #     2: scaled_constraints(X01="<=0"),
                    # }
                ),
                # LOCAL CORRECTION, i.e. from the measurement with only one IP in crossing
                TargetData(
                    ips=(1, ),
                    detuning=get_diff(detuning_flat, detuning_ip1),
                    # constraints={
                    #     1: scaled_constraints(X01="<=0"),
                    #     2: scaled_constraints(X01="<=0"),
                    # }
                ),
                TargetData(
                    ips=(5, ),
                    detuning=get_diff(detuning_flat, detuning_ip5),
                    # constraints={
                    #     1: scaled_constraints(X01="<=0"),
                    #     2: scaled_constraints(X01="<=0"),
                    # }
                ),
            ]
        ),
    ]
    return targets


def simulation():
    paths = {i: output / f"b{i}" for i in (1, 4)}
    calculate_corrections(paths, xing=xing, targets=get_targets())


def plotting():
    action_map = {"10": 'x', "01": 'y'}

    nchar = 10
    measurement = get_diff(meas_flat, meas_full)
    measurement[4] = measurement.pop(2)
    measurement = {k: meas*1e-3 for k, meas in measurement.items()}

    targets = get_targets()
    ids = [f"{target.name}_b6" for target in targets]
    labels = [
        f"{latex.dqd2j('x', 'x')} = {'55.2,-22 | 9,-4.5'.center(nchar)}",
    ]

    output_id = "_corrections"
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

    plot_correctors(output, ids=ids, labels=labels, size=corr_size, corrector_pattern=r'kctx3\.[lr][15]', order="6", output_id=f'{output_id}_b6')
    plot_correctors(output, ids=ids, labels=labels, size=corr_size, corrector_pattern=r'kctx3\.[lr]5', order="6", output_id=f'{output_id}_b6_ip5')
    plot_correctors(output, ids=ids, labels=labels, size=corr_size, corrector_pattern=r'kctx3\.[lr]1', order="6", output_id=f'{output_id}_b6_ip1')


if __name__ == '__main__':
    log_setup()
    simulation()
    plotting()
