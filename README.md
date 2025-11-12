# Corrections from Amplitude Detuning

This package provides tools to examine measured amplitude detuning and calculate corrections based on the measurements.
There is also a lot of handy functionality to assess the quality of correction and compare different correction schemes.

The heart of the package is the creation of a linear equation system (optionally with constraints),
which reflects the contribution of the different corrector magnets to amplitude terms, including feed-down effects, on the
left hand side, and the measured amplitude detuning values on the right hand side.
This equation system is then solved via least-squares minimization to find the optimal corrector settings
to correct the measured amplitude detuning.

To create this equation system, the user can define different `Target`s.
Each `Target` has a name and a list of `TargetData` instances,
in which the user can define individual detuning contributions to be corrected,
e.g. using different correctors, optics, measured values and constraints.
All target data within one target are matched together, i.e. transformed into a single equation system.
`Target`s themselves are treated independently, so that the user can compare which correction approach worked best.

A detailed description of the underlying theory can be found in [Dilly - Thesis 2024](https://edoc.hu-berlin.de/handle/18452/28999).

## Installation

The package can be installed via pip:

```bash
pip install ir-amplitude-detuning
```

## Examples

Several examples for different LHC years and measurement setups can be found in the `examples/` folder.
These will not be installed with the package, so please clone the repository if you want to use them.

The examples used here, are particularly tailored to correct the detuning changes
with and without the crossing scheme in the LHC IRs, i.e. to compensate the detuning from feed-down effects.
They only use normal-oriented dodecapole correctors.
In principle, the package can be used to correct also other detuning contributions, and use e.g. octuople or decapole correctors.
Only normal-oriented correctors are implemented so far.

The examples focus on the correction of first-order amplitude detuning terms (dQ/d2J),
but correction of second-order terms is also implemented.

### Setup Template

A template for the setup can be found in `setup_template.py`, which can be used as a starting point for your own setups.
As it is only a template, it might be easier to look at the examples for actual setups to understand how to use the package.

The important part for correction is the defintion of the `Targets` and the call to the `calculate_corrections` function,
yet usually the optics for the machine need to be simulated first.

The worklow is usually as follows:

1. Simulate the machine and get the optics for the given setups (in exampels: different crossing schemes).
   The setup parameters are defined in the `SimParams` container.<br>
   _This step is optional if you already have the optics (twiss) as DataFrames at hand._

1. Plug in the measured amplitude detuning values into the `Measurements` container.<br>
   _These could be directly put into the targets, but usually it's easier to define them separately._

1. Define correction targets: apart from the `name`, these consist of the detuning values
   to be corrected. These are be grouped into different `TargetData` instances, which can e.g. contain
   different setups of the machine (crossing schemes, optics, etc.,) given as twiss `DataFrames`
   loaded from the simulation output.

1. Run the correction calculation with the defined targets and corrector field orders,
   this gives the corrector settings per `Target` to apply to the machine.

1. Assess the correction quality by calculating the contributions analytically.
   This is fast and allows to compare different sources (e.g. different corrector field orders or different IPs)
   at a single `Target` in addition to comparing the different `Target`s with each other.

1. Assess the correction quality by re-simulating the machine with the applied corrector settings
   (implemented: Using Mad-X/PTC).
   This is slower, but gives a more realistic picture of the correction quality,
   as all non-linear effects and feed-downs are taken into account.

1. Plot the results to visualize the correction quality, e.g. by comparing the detuning before and after correction
   or how well the correction matches the target detuning.

A lot of details about the individual steps can be found in the examples.

### 2022 Commissioning Example

An example for the 2022 commissioning measurements can be found in
[`examples/2022_commissioning.py`](https://github.com/pylhc/ir_amplitude_detuning/blob/master/examples/commissioning_2022.py).
This example is the most basic one and implements only a single `Target`:

- Target 1: Correcting the detuning change between full crossing scheme enabled and flat orbit (a single `TargetData`).

### 2018 MD3311 Example

A slightly more complex example for the 2018 MD3311 measurements can be found in
[`examples/2018_md3311.py`](https://github.com/pylhc/ir_amplitude_detuning/blob/master/examples/md3311_2018.py).
Here, three different TArgets are defined:

- Target 1: Correcting the detuning change between full crossing scheme enabled and flat orbit.
- Target 2: As Target 1, but additionally targeting the detuning stemming from the crossing in IP5 only.
- Target 3: As Target 2, but with additional constraints on the cross-term to stay below zero.

### 2022 MD6863 Example

In the example [`examples/2022_md6863.py`](https://github.com/pylhc/ir_amplitude_detuning/blob/master/examples/md6863_2022.py),
the measurements from MD6863 in 2022 are directly loaded from the analysed detuning output files from `omc3`,
which makes already the first step more complex in code and requires a well defined naming scheme for the files.

Two different `Target`s are defined here:

- Target 1: Correcting the detuning change between full crossing scheme enabled and flat orbit.
- Target 2: As above, but targeting also the detuning stemming from IR5 only, with positive and negative crossing angles in IP5.

Target 2 allows to better differenciate between the decapole and dodecapole contributions,

### Output

In the respective output directory and its sub-directories, the following files can be found.

Files of the same type are identifyable by their _output_id_,
which is either the name given to the `Target`s or `nominal` for the machine without any targets applied.

- ampdet.lhc.b#._output_id_.tfs: Output of PTC containing amplitude detuning data (after applying the found corrections or `nominal`).
- ampdet_calc.lhc.b#._output_id_.tfs: Output of containing amplitude detuning data calcualted from the found corrections.
- settings.lhc.b#._output_id_.tfs: Table containing the corrector settings to match the amplitude detuning for the given target,
  as well as additional information if applicable, such as error-bar, KN and KNL value, length of the magnet, magnet name and circuit name.
- settings.lhc.b#._output_id_.madx: A madx command, assigning the calculated corrector values to the circuits.
- twiss.lhc.b#._output_id_.tfs: Output of the `TWISS` command, containing the optics (after applying the found correction or `nominal`).
- plot.*.pdf: Different plots comparing the detuning before and after correction, as well as the calculated detuning from the applied corrector settings.

The sub-directories are the output directories of the individual machine-setup simulations per beam.
They contain the beam in their name (e.g. `b1`) and if multiple machine setups are required, they are additionally labeled.
In these sub-directories the following files can be found:

- ampdet.lhc.b#.nominal.tfs: Output of PTC containing the amplitude detuning data.
- twiss.lhc.b#.nominal.tfs: Output of the `TWISS` command, containing the optics.
- twiss.lhc.b#.optics_ir.tfs: Output of the `TWISS` command, containing only the IR optics (not used, only for user convenience).
- full_output.log: Logging output. In the log of the last run beam, also the logging from the intermediate python can be found.
- madx_commads.log: Mad-X commands used in this run.

## Different Machines and Codes

The package is designed to be extensible to different machines and simulation codes,
yet so far only the LHC is implemented in Mad-X via `cpymad`.
To add a different machine, you will need to implement a workflow that creates the optics as `DataFrame`s.
The easiest way is to stick to the same output format and naming conventions as are used in the LHC simulation part of this package.
A big concession to the LHC layout is the assumption of IPs and beams, which might not be present in other machines.
You should be able to work around this by simply assigning `None` to the `ip` attribute of correctors and using only beam 1
(beam 2 and beam 4 trigger some special behaviour in the calculations to take beam direction and mad-x conventions into account).
**Beware**: This has not been tested yet!
