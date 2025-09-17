"""
This is the whole analysis, correction and plotting run for
the MD6863 measurements, including IP5.

It also checks the influence of second order detuning and calculates corrections
without this influence.

Running order:
 - Do fits based on measurement data (kick.tfs files)
 - Calculate corrections based on fits (i.e. detuning)
 - Calculate second order detuning from corrections and subtract
   from measurements -> new fits -> new detuning values -> new corrections
 - plot measurements comparison

"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tfs
from generic_parser import DotDict
from matplotlib import pyplot as plt
from omc3.amplitude_detuning_analysis import get_kick_and_bbq_df, single_action_analysis
from omc3.plotting.plot_amplitude_detuning import _plot_2d
from omc3.plotting.utils import colors as pcolors
from omc3.plotting.utils import style as pstyle
from omc3.tune_analysis.bbq_tools import OutlierFilterOpt
from omc3.tune_analysis.constants import (CORRECTED, get_action_col, get_kick_out_name,
                                          get_natq_corr_col)
from omc3.tune_analysis.kick_file_modifiers import read_timed_dataframe
from tfs.tools import significant_digits

from ir_dodecapole_corrections.lhc_detuning_corrections import (
    calculate_corrections, check_corrections, create_optics)
from ir_dodecapole_corrections.utilities.classes import (Detuning, DetuningMeasurement,
                                                         MeasureValue, Target, TargetData)
from ir_dodecapole_corrections.utilities.latex import (
    print_correction_and_error_as_latex
)
from ir_dodecapole_corrections.simulation.lhc_simulation import get_detuning_from_ptc_output
from ir_dodecapole_corrections.utilities.plot_utils_measurements_comparison import (
    MeasurementSetup, get_ylabel, plot_measurements)
from ir_dodecapole_corrections.utilities.plotting import get_calc_detuning_ips, get_detuning
from ir_dodecapole_corrections.utilities.logging import log_setup

LOGGER = logging.getLogger(__name__)

SO_TERMS = ["X20", "X11", "X02", "Y20", "Y11", "Y02"]


# Naming scheme specific functions ---------------------------------------------

# Define Machine Data
# -------------------


# data labels as used in the detuning summary
DATA_LABELS = {
    "1_OFF_5_OFF_B6_OUT": "meas_flat",
    "1_-160_5_+160_B6_OUT": "meas_full",
    "1_OFF_5_+160_B6_OUT": "meas_ip5p",
    "1_OFF_5_-160_B6_OUT": "meas_ip5m",
    "1_-160_5_+160_B6_IN": "meas_full_corr",
}


LATEX_LABELS = {
    "1_OFF_5_OFF_B6_OUT": ("2022 MD6863 w/o $b_6$", "flat-orbit"),
    "1_-160_5_+160_B6_OUT": ("2022 MD6863 w/o $b_6$", r"IP1\&5 xing @ $\mp$\qty{160}{\micro\radian}"),
    "1_OFF_5_+160_B6_OUT": ("2022 MD6863 w/o $b_6$", r"IP5 xing @ $+$\qty{160}{\micro\radian}"),
    "1_OFF_5_-160_B6_OUT": ("2022 MD6863 w/o $b_6$", r"IP5 xing @ $-$\qty{160}{\micro\radian}"),
    "1_-160_5_+160_B6_IN": ("2022 MD6863 w/ $b_6$", r"IP1\&5 xing @ $\mp$\qty{160}{\micro\radian}"),
}

# Fill in special values for crossing if needed:
XINGS = {
    'full': {'scheme': 'flat', 'on_x1': -160, 'on_x5': 160},
    'ip5+': {'scheme': 'flat', 'on_x5': 160},
    'ip5-': {'scheme': 'flat', 'on_x5': -160},
}


MAIN_XING = "full"  # main xing to test the correction on
SIMULATION_ID = "full_and_ip5pm"  # id of the target (we only use a single target here)


def get_all_results_dirs() -> List[Path]:
    measurement_data = Path("/afs/cern.ch/work/j/josch/temp.bbgui_output/2022-06-24_md6863")
    return (
            list((measurement_data / "LHCB1" / "Results").glob("*")) +
            list((measurement_data / "LHCB2" / "Results").glob("*"))
    )


def get_simulation_path() -> Tuple[Path, str]:
    return Path(), ""


def merge_summary_data(summary_df):
    """ Merge the data together based on names.
    This is highly specific to the current results folder. """
    summary_merged_df = tfs.TfsDataFrame(columns=summary_df.columns)
    for name in summary_df.index:
        if "_H_" in name:
            continue

        new_name = get_normalized_name(name)
        hor_name = name.replace("_V_", "_H_")
        v_data, h_data = summary_df.loc[name, :], summary_df.loc[hor_name, :]

        assert all(np.logical_xor(v_data, h_data))
        summary_merged_df.loc[new_name, :] = v_data + h_data
    return summary_merged_df


# Assuming anlysis names follow conventions:

def get_normalized_name(name: str):
    return name.replace("AmpDet_H_", "").replace("AmpDet_V_", "").replace("2022-06-24_", "").upper()


def get_beam_from_name(name: str) -> int:
    return int(re.search("(^|_)B(\d)_", name).group(2))


def get_kick_plane_from_name(name: str) -> str:
    return {"H": "X", "V": "Y"}[re.search("_([HV])_", name).group(1)]


def check_action(action):
    if len(action) == 2 and action[0] != action[1]:
        raise NotImplementedError(f"Not implemented for derivative {action}")


def column_name(tune, action):
    check_action(action)
    return f"ODR_dQ{tune}d2J{action[0]}_CORRCOEFF{len(action)}"


def err_column_name(tune, action):
    check_action(action)
    return f"ODR_dQ{tune}d2J{action[0]}_ERRCORRCOEFF{len(action)}"


# DETUNING ANALYSIS FROM MEASUREMENT DATA --------------------------------------


def do_full_detuning_analysis(output_dir: Path):
    """ Load results data from analysed output folders (i.e. the kick-files and bbq_tfs)
    and collect result in a single dataframe."""
    output_dir.mkdir(exist_ok=True, parents=True)
    all_results_paths = get_all_results_dirs()
    # print("\n".join(str(p) for p in ALL_RESULTS))
    summary_df = tfs.TfsDataFrame()
    for analysis in all_results_paths:
        beam = get_beam_from_name(analysis.name)
        kick_plane = get_kick_plane_from_name(analysis.name)
        # kick_df = read_timed_dataframe(analysis / get_kick_out_name())
        kick_df, _ = get_kick_and_bbq_df(kick=analysis, bbq_in=analysis / "bbq_ampdet.tfs",
                                         beam=beam,
                                         filter_opt=OutlierFilterOpt(window=100, limit=0.),
                                         )

        # analyse kick-data file
        kick_analysed = single_action_analysis(kick_df, kick_plane, detuning_order=1, corrected=True)
        summary_df = tfs.concat([summary_df, get_summary_row(kick_df=kick_analysed, name=analysis.name)])

    save_summary(output_dir, summary_df)


def get_summary_row(kick_df, name):
    """ Turn relevant (order > 1, corrected terms) header items into Series. """
    return pd.DataFrame(
        {coeff: kick_df.headers[coeff]
         for coeff in kick_df.headers.keys()
         if CORRECTED in coeff and not coeff.endswith("0")
         },
        index=[name]
    )


def get_summary_name(prefix: str = ""):
    return f"detuning_analysis_{prefix}data.tfs"


def save_summary(output_dir: Path, dataframe: tfs.TfsDataFrame, prefix: str = ""):
    tfs.write(output_dir / get_summary_name(prefix), merge_summary_data(dataframe.fillna(0)), save_index="Result")

# CALCULATE CORRECTIONS --------------------------------------------------------


def get_sum(meas_a, meas_b):
    return {beam: meas_a[beam] + meas_b[beam] for beam in meas_a.keys()}


def get_diff(meas_a, meas_b):
    return {beam: meas_a[beam] - meas_b[beam] for beam in meas_a.keys()}


# Fill in measurement data in 10^3 m^-1:
def get_measured_detuning_values(output_dir: Path, prefix: str = ""):
    """ Get the detuning values from summary file (see redo_all_from_scratch.py). """
    data = {}
    df = tfs.read(output_dir / get_summary_name(prefix), index="Result")
    for meas, label in DATA_LABELS.items():
        data[label] = {}
        for idx, beam in enumerate((1, 2)):
            beam_label = f"B{beam}_{meas}"
            beam_data = {}
            plane2nums = {"X": "10", "Y": "01"}

            for tune, action in (("X", "X"), ("Y", "Y"), ("Y", "X"), ("X", "Y")):
                acd_scale = 0.5 if (tune == action) else 1
                value_col, error_col = column_name(tune, action), err_column_name(tune, action)
                value, error = df.loc[beam_label, value_col], df.loc[beam_label, error_col]
                beam_data[f"{tune}{plane2nums[action]}"] = MeasureValue(value * acd_scale, error * acd_scale)
                # value_str, error_str = significant_digits(value * scale * acd_scale, error * scale * acd_scale)
            beam_data["X01"] = (beam_data.pop("X01") + beam_data.pop("Y10")) * 0.5
            data[label][beam] = DetuningMeasurement(**beam_data)
    return data


# Steps of calculations --------------------------------------------------------

def ltx_dqd2j(tune, action, power=1):
    if power == 1:
        return f"$Q_{{{tune},{action}}}$"
    return f"$Q_{{{tune},{action}^{{{power}}}}}$"


# def get_detuning(meas: Dict[Any, DetuningMeasurement]) -> Dict[Any, Detuning]:
#     return {beam: meas[beam].get_detuning() for beam in meas.keys()}


def get_targets(output_dir: Path, prefix: str = "") -> Sequence[Target]:
    data = get_measured_detuning_values(output_dir, prefix)
    detuning_flat = data["meas_flat"]
    detuning_full = data["meas_full"]
    detuning_ip5p = data["meas_ip5p"]
    detuning_ip5m = data["meas_ip5m"]
    targets = [
        Target(
            name=f"{prefix}{SIMULATION_ID}",
            data=[
                TargetData(
                    ips=(1, 5),
                    detuning=get_diff(detuning_flat, detuning_full),
                    xing='full',
                ),
                TargetData(
                    ips=(5,),
                    detuning=get_diff(detuning_flat, detuning_ip5p),
                    xing='ip5+',
                ),
                TargetData(
                    ips=(5,),
                    detuning=get_diff(detuning_flat, detuning_ip5m),
                    xing='ip5-',
                ),
            ]
        ),
    ]
    return targets


def simulation(output_dir: Path):
    paths = {i: output_dir / f"b{i}" for i in (1, 4)}
    lhc_beams = create_optics(paths, xings=XINGS, year=2022, tune_x=62.28, tune_y=60.31)

    calculate_corrections(paths, targets=get_targets(output_dir), main_xing=MAIN_XING, lhc_beams=lhc_beams)
    check_corrections(paths, lhc_beams[MAIN_XING])

    for beams in lhc_beams.values():
        for lhcbeam in beams.values():
            lhcbeam.madx.exit()


def do_all_corrections(output_dir: Path):
    paths = {i: output_dir / f"b{i}" for i in (1, 4)}
    # calculate_corrections(paths, targets=get_targets(output_dir), main_xing='full')
    calculate_corrections(paths, targets=get_targets(output_dir, prefix="sub_"), main_xing=MAIN_XING)
    calculate_corrections(paths, targets=get_targets(output_dir, prefix="so_"), main_xing=MAIN_XING)
    check_corrections(paths, id_suffix="_ptc_recheck", year=2022, tune_x=62.28, tune_y=60.31)


# SUBTRACT SECOND ORDER DETUNING -----------------------------------------------

def do_all_plots_and_subtractions(output_dir: Path):
    all_results_paths = get_all_results_dirs()
    summary_sub_df = tfs.TfsDataFrame()
    summary_so_df = tfs.TfsDataFrame()
    summary_df = tfs.read_tfs(output_dir / get_summary_name(), index="Result")

    for idx, analysis in enumerate(all_results_paths):
        kick_df, kick_df_sub, kick_df_so = check_second_order_detuning(analysis, simulation=output_dir, sim_id=SIMULATION_ID)
        summary_sub_df = tfs.concat([summary_sub_df, get_summary_row(kick_df_sub, name=analysis.name)])
        summary_so_df = tfs.concat([summary_so_df, get_summary_row(kick_df_so, name=analysis.name)])

        kick_df.headers.update(summary_df.loc[get_normalized_name(analysis.name)])
        plot_measurement_fit_with_second_order(output_dir / "plot_second_order" / get_normalized_name(analysis.name),
                                               kick_df, kick_df_sub, kick_df_so,
                                               kick_plane=get_kick_plane_from_name(analysis.name))

    save_summary(output_dir, summary_sub_df, prefix="sub_")
    save_summary(output_dir, summary_so_df, prefix="so_")


def check_second_order_detuning(analysis: Path, simulation: Path, sim_id: str):
    """ Remove second order detuning influence from kick data as well
    as perform second order fit on the kick data.

    Args:
        analysis (Path): Path to the analysis folder
        simulation (Path): Path to the simulation folder to get the PTC detuning data from
        sim_id (str): id to look for in simulation folder

    Returns:

    """
    beam = get_beam_from_name(analysis.name)
    kick_plane = get_kick_plane_from_name(analysis.name)

    # get detuning ptc (opposite sign, as this is what is introduced by correction)
    detuning_terms = -get_corrected_detuning_ptc(
        folder=simulation,
        beam=beam if beam == 1 else 4,
        id_=f"{sim_id}_b6",
        columns=SO_TERMS,
        main_xing=MAIN_XING,
    )

    # load kick_ampdet.xy file
    kick_df = read_timed_dataframe(analysis / get_kick_out_name())

    # calculate expected tune change from second order
    kick_subtracted = subtract_detuning(kick_df, detuning_terms)

    # analyse kick-data file
    kick_subtracted_analysed = single_action_analysis(kick_subtracted, kick_plane, detuning_order=1, corrected=True)

    kick_so_fit = tfs.TfsDataFrame(kick_df.to_dict())
    kick_so_fit = single_action_analysis(kick_so_fit, kick_plane, detuning_order=2, corrected=True)

    return kick_df, kick_subtracted_analysed, kick_so_fit


def plot_measurement_fit_with_second_order(output: Path, kick_df, kick_df_subtracted, kick_df_so, kick_plane):
    """ Plot all three fits into one single plot. """
    output.parent.mkdir(parents=True, exist_ok=True)
    opt = DotDict(
        plane=kick_plane,
        kicks=[kick_df, kick_df_subtracted, kick_df_so],
        labels=["original", "2nd order sub.", "2nd order fit"],
        tune_scale=-3,
        detuning_order=2,
        action_unit="m",
        action_plot_unit="um",
        x_lim=[0, 0.016],
        y_lim=[-1.5, 1.5],
        correct_acd=True,
        output=output,
        bbq_corrected=(True,)
    )

    manual_style = {
        # "font.size": 22.5,
        # "figure.constrained_layout.use": False,
        "figure.figsize": [9.50, 4.50],
        "lines.linestyle": "none",
        "figure.subplot.left": 0.12,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.right": 0.99,
        "figure.subplot.top": 0.77,
    }
    pstyle.set_style("standard", manual_style)

    for tune_plane in "XY":
        figs = _plot_2d(tune_plane, opt)
    # plt.show()
    # exit()
    for fig in figs.values():
        plt.close(fig)


def get_corrected_detuning_ptc(folder: Path, beam: int, id_: str, columns: Sequence[str], main_xing: str = None) -> pd.Series:
    """ Get the detuning terms from the PTC difference between id_ and nominal.
    If the state of the machine at "id_" is with errors,
    this returns the detuning as INTROCUDED by the errors.
    On the other hand, if this is a machine with corrections, this is
    still the detuning introduced by the correction, i.e. the opposite sign  of
    what is compensated.
    """
    nominal_beam = f"b{beam}"
    if main_xing:
        nominal_beam = f"{main_xing}_b{beam}"

    df = tfs.read(folder / f"b{beam}" / f"ampdet.lhc.b{beam}.{id_}.tfs")
    df_nominal = tfs.read(folder / nominal_beam / f"ampdet.lhc.b{beam}.nominal.tfs")
    detuning = pd.Series(get_detuning_from_ptc_output(df, log=False, terms=columns))
    detuning_nominal = pd.Series(get_detuning_from_ptc_output(df_nominal, log=False, terms=columns))
    return detuning - detuning_nominal


def subtract_detuning(kick_df: tfs.TfsDataFrame, detuning_terms: pd.Series):
    """ Subtract the detuning as given in the detuning terms from the measurement data in kick_df.
    The detuning terms are all assumed to be the "free" tune derivatives.
    AC-Dipole and taylor coefficients are calculated for these. """
    kick_df = tfs.TfsDataFrame(kick_df.to_dict())

    col_jx = get_action_col("X")
    col_jy = get_action_col("Y")

    for tune_plane in "XY":
        col_q = get_natq_corr_col(tune_plane)
        # col_qerr = get_corr_natq_err_col(plane)
        # col_j = get_action_col(plane)
        # col_jerr = get_action_err_col(plane)

        plane_terms = detuning_terms.loc[detuning_terms.index.str.upper().str.startswith(tune_plane)]

        for kick in kick_df.index:
            action = (kick_df.loc[kick, col_jx], kick_df.loc[kick, col_jy])
            detuning = calculate_detuning(
                terms=plane_terms,
                action=action,
                plane=tune_plane,
                acd=True
            )
            kick_df.loc[kick, col_q] = kick_df.loc[kick, col_q] - detuning
    return kick_df


def calculate_detuning(terms: pd.Series, action: Tuple[float, float], plane: str, acd: bool = True):
    """ Calculate the total change in tune on the measurement, based on the given 'detuning' term,
    i.e. the derivative of the tune with action. """
    detuning = 0
    for term in terms.index:
        if term[0] not in plane:
            continue

        # detuning derivative term
        exp_x, exp_y = int(term[1]), int(term[2])
        det = terms[term] * (action[0]**exp_x) * (action[1]**exp_y)

        # coefficient (Taylor)
        if exp_x == 2 or exp_y == 2:
            det = det * 0.5

        # AC-Dipole influence
        if acd:
            if exp_x == 1 and exp_y == 1:
                det = det * 2

            if term[0].upper() == "X" and exp_x == 2:
                det = det * 3

            if term[0].upper() == "Y" and exp_y == 2:
                det = det * 3

        LOGGER.info(f"{term}: {det:.1e}")

        detuning = detuning + det
    LOGGER.info(f"Total: {detuning:.1e}")
    return detuning


# LATEX Printing ---------------------------------------------------------------


def spaces(n: int) -> str:
    return n * " "


def latex_column(tune, action):
    return f"$Q_{{{tune.lower()},{action.lower()}}}$"


def print_measurements_latex_table(output_data: Path, prefix: str = ""):
    """ Print the measurement fit results as a latex table.
    The prefix indicates if this should print the default ones (empty)
    the ones with second order subtracted ("sub_") or the ones
    that include second order in the fit ("so_")

    """
    df = tfs.read(output_data / get_summary_name(prefix), index="Result")
    column_order = (("X", "X"), ("Y", "X"), ("X", "Y"), ("Y", "Y"))
    n_first_order = len(column_order)
    n_second_order = 0
    if prefix == "so_":
        column_order = (*column_order, ("X", "XX"), ("X", "YY"), ("Y", "XX"), ("Y", "YY"))
        n_second_order = len(column_order) - n_first_order

    latex_str = []
    # HEADER
    latex_str.append(spaces(45) + " & " + " & ".join(f"{latex_column(tune, action):21s}" for tune, action in column_order) + r"\\")
    latex_str.append(spaces(46) + n_first_order * r"& \unitcellfo           " + n_second_order * r"& \unitcellso           " + r"\\")
    latex_str.append(r"\midrule[1pt]")

    # Entries
    for meas, label in LATEX_LABELS.items():
        for idx, beam in enumerate((1, 2)):
            beam_label = f"B{beam}_{meas}"
            beam_str = "\\bblue" if beam == 1 else "\\bred"
            entries = []
            for tune, action in column_order:
                scale = 1e-3 if len(action) == 1 else 1e-11
                acd_scale = 1
                if len(action) == 1 and tune == action:
                    acd_scale = 0.5
                if len(action) == 2 and tune == action[0]:
                    acd_scale = 1/3
                value_col, error_col = column_name(tune, action), err_column_name(tune, action)
                value, error = df.loc[beam_label, value_col], df.loc[beam_label, error_col]
                value_str, error_str = significant_digits(value*scale*acd_scale, error*scale*acd_scale)
                # entries.append(f'{f"{value_str}({error_str})":10s}')
                entries.append(f'{f"{beam_str}{{{value_str}}}{{{error_str}}}":20s}')

            latex_str.append(f"{label[idx]:45s} & " + "  & ".join(entries) + r"\\")
        latex_str.append("\midrule")

    print("\n".join(latex_str))


def print_measurements_latex_table_diff(output_dir: Path, prefix: str = "sub_"):
    """ Print the latex table for the detuning measurements again,
    this time only for the subtracted or second order one,
    with additional columns indicating
    the difference to the "normal" analysis. """
    df_default = tfs.read(output_dir / get_summary_name(prefix=""), index="Result")
    df = tfs.read(output_dir / get_summary_name(prefix=prefix), index="Result")

    column_order = (("X", "X"), ("Y", "X"), ("X", "Y"), ("Y", "Y"))
    n_first_order = len(column_order)
    n_second_order = 0
    if prefix == "so_":
        pass
        # column_order = (*column_order, ("X", "XX"), ("X", "YY"), ("Y", "XX"), ("Y", "YY"))
        # n_second_order = len(column_order) - n_first_order

    latex_str = []
    # HEADER
    latex_str.append(spaces(45) + " & " + (" & $\Delta$ " + spaces(1) + "& ").join(f"{latex_column(tune, action):21s}" for tune, action in column_order) + r" & $\Delta$ \\")
    latex_str.append(spaces(46) + n_first_order * (r"& \unitcellfo" + spaces(11) + "&" + spaces(11)) + (n_second_order * r"& \unitcellso" + spaces(10)) + r"\\")
    latex_str.append(r"\midrule[1pt]")

    # Entries
    for meas, label in LATEX_LABELS.items():
        for idx, beam in enumerate((1, 2)):
            beam_label = f"B{beam}_{meas}"
            beam_str = "\\bblue" if beam == 1 else "\\bred"
            entries = []
            for tune, action in column_order:
                scale = 1e-3 if len(action) == 1 else 1e-11
                acd_scale = 1
                if len(action) == 1 and tune == action:
                    acd_scale = 0.5
                if len(action) == 2 and tune == action[0]:
                    acd_scale = 1/3
                value_col, error_col = column_name(tune, action), err_column_name(tune, action)
                value, error = df.loc[beam_label, value_col], df.loc[beam_label, error_col]
                value_default, error_default = df_default.loc[beam_label, value_col], df.loc[beam_label, error_col]
                value_diff, error_diff = value - value_default, (error**2 + error_default**2)**0.5

                value_str, error_str = significant_digits(value*scale*acd_scale, error*scale*acd_scale)
                value_diff_str, error_diff_str = significant_digits(value_diff*scale*acd_scale, error_diff*scale*acd_scale)

                # entries.append(f'{f"{value_str}({error_str})":10s}')
                entries.append(f'{f"{beam_str}{{{value_str}}}{{{error_str}}}":20s}')
                digits = len(value_diff_str.split(".")[-1]) if "." in value_diff_str else 0
                entries.append(f'{f"{float(value_diff_str):+.{digits}f}":8s}')

            latex_str.append(f"{label[idx]:45s} & " + "  & ".join(entries) + r"\\")
        latex_str.append("\midrule")

    print("\n".join(latex_str))


def print_corrections_latex_table(output_dir: Path, prefix: str = ""):
    """ Print the corrections in latex format to be put in a table. """
    id_ = f"{SIMULATION_ID}_b6"
    beam = 1
    corrections = tfs.read(output_dir / f"b{beam}" / f"settings.lhc.b{beam}.{prefix}{id_}.tfs", index="NAME")
    values = np.array([MeasureValue(r["KNL"], r["ERRKNL"]) for _, r in corrections.iterrows()])
    print_correction_and_error_as_latex(values, corrections.index)


# PLOT CORRECTION COMPARISON ---------------------------------------------------

def average_crossterms_all_measurements(data: Dict[str, Dict[int, DetuningMeasurement]]):
    data_out = {}
    for measurement, meas_beams in data.items():
        data_out[measurement] = average_crossterms_all_beams(meas_beams)
    return data_out


def average_crossterms_all_beams(data: Dict[int, DetuningMeasurement]):
    data_out = {}
    for beam, meas in data.items():
        if meas.X01 and meas.Y10:
            av_cross = (meas.X01 + meas.Y10) / 2
        else:
            av_cross = meas.X01 or meas.Y10
        data_out[beam] = DetuningMeasurement(X10=meas.X10, X01=av_cross, Y01=meas.Y01)
    return data_out


def get_simulated_detuning_data(folder):
    id_ = f"{SIMULATION_ID}_b6"
    beams = 1, 4
    # PTC data (as Dict[int: Detuning]):
    data = {beam: Detuning(**get_detuning(folder / f"b{beam}", beam, id_)) for beam in beams}
    nominal_data = {beam: Detuning(**get_detuning(folder / f"{MAIN_XING}_b{beam}", beam, 'nominal')) for beam in beams}
    # Subtract Nominal Value:
    for beam in data.keys():
        data[beam] = data[beam] - nominal_data[beam]

    # Calculated Data by ips/fields (as Dict[int, DataFrame]):
    calculated_data = {beam: get_calc_detuning_ips(folder / f"b{beam}", beam, id_) for beam in beams}

    return data, calculated_data


def plot_comparison(output_dir: Path):
    """
    To Plot:
      - dF = Full - Flat
      - d5 = IP5+ - Flat
      - d1 = Full - IP5+
      - b6 contribution: (IP5+ + IP5- - 2Flat) / 2
    From:
    d5 = 5 - 0
    dF = F - 0 = d1 + d5
    d1 = 1 - 0
    d1 = F - 0 - d5 = F - 0 - 5 + 0 = F - 5
    """
    data = get_measured_detuning_values(output_dir, prefix="")
    data = average_crossterms_all_measurements(data)
    data_ptc, data_calc = get_simulated_detuning_data(folder=output_dir)
    plot_terms = ["X10", "X01", "Y01"]

    manual_style = {
        # "font.size": 22.5,
        # "figure.constrained_layout.use": False,
        "figure.figsize": [6.50, 3.0],
        "figure.subplot.left": 0.12,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.right": 0.99,
        "figure.subplot.top": 0.77,
        "errorbar.capsize": 5,
        "lines.marker": "x",
        "lines.markersize": 4,
        "axes.grid": False,
        "ytick.minor.visible": True,
    }

    for beam in (1, 2):
        sim_beam = beam
        if beam == 2:
            sim_beam = 4

        full = (data["meas_full"][beam] - data["meas_flat"][beam])
        ip5 = (data["meas_ip5p"][beam] - data["meas_flat"][beam])
        ip1 = (data["meas_full"][beam] - data["meas_ip5p"][beam])
        from_ip5_b6 = (data["meas_ip5p"][beam] + data["meas_ip5m"][beam]) * 0.5 - data["meas_flat"][beam]
        est = full + DetuningMeasurement.from_detuning(data_ptc[sim_beam])
        meas = [
            MeasurementSetup(measurement=full, label="Full Xing"),
            MeasurementSetup(measurement=ip5, label="from IP5"),
            MeasurementSetup(measurement=from_ip5_b6, label="from IP5 $b_6$"),
            MeasurementSetup(measurement=ip1, label="from IP1"),
            MeasurementSetup(measurement=est, label="estimated"),
        ]
        fig = plot_measurements(meas,
                                manual_style=manual_style,
                                rescale=3,
                                ylabel=get_ylabel(rescale=3, delta=True),
                                ylim=[-70, 70],
                                add_rms=True,
                                )
        ax = fig.gca()

        bar_width = 0.33
        stack_width = 0.15 * bar_width
        measurement_width = 1 / (len(meas) + 1)
        color_all = pcolors.get_mpl_color(0)
        color_ip5 = pcolors.get_mpl_color(1)
        color_ip5_b6 = pcolors.get_mpl_color(2)
        color_ip1 = pcolors.get_mpl_color(3)
        scale = 1e-3

        for idx_term, term in enumerate(plot_terms):
            # Plot calculated Data:
            calc = {f: data_calc[sim_beam].loc[f, term]*scale for f in ("5", "1", "all")}

            x_pos = idx_term + 1 * measurement_width
            ax.bar(x_pos, -calc["all"], stack_width, bottom=0, label=f"_{beam}.all", color=color_all, alpha=0.3)

            x_pos = idx_term + 2 * measurement_width
            ax.bar(x_pos, -calc["5"], stack_width, bottom=0, label=f"_{beam}.ip5", color=color_ip5, alpha=0.3)

            x_pos = idx_term + 3 * measurement_width
            ax.bar(x_pos, -calc["5"], stack_width, bottom=0, label=f"_{beam}.ip5_b6", color=color_ip5_b6, alpha=0.3)

            x_pos = idx_term + 4 * measurement_width
            ax.bar(x_pos, -calc["1"], stack_width, bottom=0, label=f"_{beam}.ip1", color=color_ip1, alpha=0.3)

        fig.savefig(output_dir / f"plot.detuning_ip5_ip5b6_ip1_and_corr.b{beam:d}.pdf")

        text = ""
        for term in full.terms():
            text += (
                f"{term}:\n"
                f"full: {str(full[term])}\n"
                f"ip5: {str(ip5[term])}\n"
                f"ip1: {str(ip1[term])}\n"
                f"ip5_b6: {str(from_ip5_b6[term])}\n"
                f"est: {str(est[term])}\n"
            )
        text += (
            f"RMS:\n"
            f"full: {str(MeasureValue.weighted_rms([full[t] for t in full.terms()]))}\n"
            f"ip5: {str(MeasureValue.weighted_rms([ip5[t] for t in ip5.terms()]))}\n"
            f"ip1: {str(MeasureValue.weighted_rms([ip1[t] for t in ip1.terms()]))}\n"
            f"ip5_b6: {str(MeasureValue.weighted_rms([from_ip5_b6[t] for t in from_ip5_b6.terms()]))}\n"
            f"est: {str(MeasureValue.weighted_rms([est[t] for t in est.terms()]))}\n"
        )
        # LOG.info("\n" + text)
        (output_dir / f"data.detuning_change_corrections.b{beam:d}.txt").write_text(text)


if __name__ == '__main__':
    log_setup()
    output_dir = Path("md6863_with_ip5_inj_tunes_and_second_order")

    # redo full analysis and save detuning results in output dir
    do_full_detuning_analysis(output_dir)

    # calculate main correction
    # simulation(output_dir)

    # create new detuning fits with second order subtracted
    do_all_plots_and_subtractions(output_dir)

    # calculate new corrections based on these modified fits
    # do_all_corrections(output_dir)

    # print_latex_table(output_dir, prefix="")
    # print_latex_table(output_dir, prefix="sub_")
    # print_latex_table(output_dir, prefix="so_")
    # plot_comparison(output_dir)
    # plt.show()

    # print_corrections_latex_table(output_dir)
    # print_corrections_latex_table(output_dir, "sub_")
    # print_corrections_latex_table(output_dir, "so_")

    # print_measurements_latex_table_diff(output_dir, prefix="sub_")
