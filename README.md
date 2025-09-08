# Corrections from amplitude detuning

This package provides an easy way to match the LHC IRs to generate
desired amplitude detuning, including feed-down.

## Main

The `main` function to run can be found in `correction_via_amplitude detuning`
and requires the folloging parameters:

- outputdirs: `Dict[int, Path]` A dictionary (keys are beam numbers and define which beams to use!),
            containing the output path for the simulation data.
- targets: `Sequence[Target]` A list of `Target`s, which are the matching inputs
        for this simulations, i.e. they contain the detuning data.
        Each target is evaluated individually and can be identified by its
        name in the output data.
        The data is hereby given as `Detuning` instances, containing the detuning
        values for X10 (dQx / d2Jx), X01 (dQx / d2Jy or dQy / d2Jx) and Y01 (dQy / d2Jy).
        A typical construction for the targets looks like this:

        Target(
            name="X10X01Y01",  # identifyier for this target
            data=[             # all of these TargetDatas are matched together
                TargetData(
                    ips=(1, ),   # ip's that contribute to this detuning
                    detuning={
                        # dict of Detuning Values per beam.
                        # The `scaled_` functions simply return `Detuning` objects with all values * 1E3
                        1: scaled_detuning(X10=0, X01=0, Y01=0),  # insert your Beam 1 detuning to compensate here
                        2: scaled_detuning(X10=0, X01=0),         # insert your Beam 2 detuning to compensate here
                    },
                    constraints={
                        # dict of Constraints per beam.
                        # `scaled_` as above for `Constraint`. Only `<=` or `>=` possible.
                        1: scaled_constraints(X01="<=0"),             # insert your Beam 1 constraints here
                        2: scaled_constraints(X01="<=0", Y01=">=0"),  # insert your Beam 2 constraints here
                    }
                ),
            ]
        )

- field_list: `Sequence[str]` Sequence of the field orders used to apply the amplitude deutuning,
            e.g. `("b4", "b5b6", "b6" )` to use the octupole correctors, then the decapole and dodecapole correctors
            and then the dodecapole correctors only. Only normal correctors are implemented so far.
            Decapole correctors are installed into the dodecapole correctors for now.
- xing: `dict` The crossing scheme. See `cpymad_lhc.ir_orbit.orbit_setup`. If `None` set to `{'scheme': 'top'}`.
- optics: `Union[str,Path]` A name for the optics (see `lhc_simulation.get_optics_path()`)
                            or the path to the optics file to be used. Defaults to 'round3030'
- year: `int` The LHC year. The main difference being, that after 2020 an `acc-models-lhc` symlink is created,
            and the appropriate sequence filenames are chosen. Defaults to `2018`.

### Output

In the respective output directory, the following files can be found.
If there is multiple files, they are usually identifyable by their *output_id*,
which is either the name given to the `Targets` or `nominal` for the machine without any targets applied.

- ampdet.lhc.b#.*output_id*.tfs: Output of PTC containing the amplitude detuning data after applying the found matching.
- twiss.lhc.b#.*output_id*.tfs: Output of the `TWISS` command, containing the optics.
- settings.lhc.b#.*output_id*.tfs: Table containing the corrector settings to apply to match the amplitude detuning for the given target.
- settings.lhc.b#.*output_id*.madx:  Same as above, but as a madx command, assigning the values to the circuits.
- full_output.log: logging output. In the log of the last beam, also the logging from the intermediate python can be found.
- madx_commads.log: Madx commands used in this run.

A plotting script is also provided to visualize the results from `ampdet` and `settings`.

## Template and Examples

A template for the setup can be found in `setup_template.py`, a runnable example in `setup_example_2018`.
Actual usages are in the remaining `setup_*.py` files.
