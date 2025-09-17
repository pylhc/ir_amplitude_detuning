"""
Run a cpymad MAD-X simulation for the LHC optics (2018) without errors.
In addition, extra functionality is added to install kcdx decapole correctors
into the MCTX and assign powering for decapole and dodecapole circuits.

The class ``LHCBeam`` is setting up and running cpymad.
This class can be useful for a lot of different studies, by extending
it with extra functionality.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import cpymad.madx
import tfs
from cpymad.madx import Madx
from cpymad_lhc.corrector_limits import LimitChecks
from cpymad_lhc.coupling_correction import correct_coupling
from cpymad_lhc.general import (
    amplitude_detuning_ptc,
    get_k_strings,
    get_lhc_sequence_filename_and_bv,
    get_tfs,
    match_tune,
)
from cpymad_lhc.ir_orbit import log_orbit, orbit_setup
from cpymad_lhc.logging import MADXCMD, MADXOUT, cpymad_logging_setup
from optics_functions.coupling import closest_tune_approach, coupling_via_cmatrix
from tfs import TfsDataFrame

from ir_dodecapole_corrections.utilities.classes_accelerator import Corrector

LOG = logging.getLogger(__name__)  # setup in main()
LOG_LEVEL = logging.DEBUG

ACC_MODELS = "acc-models-lhc"

PATHS = {
    "db5": Path("/afs/cern.ch/eng/lhc/optics/V6.503"),
    "optics2016": Path("/afs/cern.ch/eng/lhc/optics/runII/2016"),
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "optics_repo": Path("/afs/cern.ch/eng/acc-models/lhc"),
    ACC_MODELS: Path(ACC_MODELS),
}


def pathstr(key: str, *args: str) -> str:
    """ Wrapper to get the path (as string! Because MADX wants strings)
    with the base from the dict ``PATHS``.

    Args:
        key (str): Key for the base-path in ``PATHS``.
        args (str): Path parts to attach to the base.

    Returns:
        str: Full path with the base from  given ``key``.
    """
    return str(PATHS[key].joinpath(*args))


def get_optics_path(year: int, name: str | Path):
    """ Get optics by name, i.e. a collection of optics path-strings to the optics files.

     Args:
         year (int): Year of the optics
         name (str, Path): Name for the optics or a path to the optics file.

    Returns:
        str: Path to the optics file.
     """
    if isinstance(name, Path):
        return str(name)

    optics_map = {
        2018: {
            'inj': pathstr("optics2018", "PROTON", "opticsfile.1"),
            'flat6015': pathstr("optics2018", 'MDflatoptics2018', 'opticsfile_flattele60cm.21'),
            'round3030': pathstr("optics2018", "PROTON", "opticsfile.22_ctpps2"),
        },
        2022: {
            'round3030': pathstr(ACC_MODELS, "strengths", "ATS_Nominal", "2022", "squeeze", "ats_30cm.madx")
        }
    }
    return optics_map[year][name]


def get_wise_path(seed: int):
    """ Get the wise errordefinition file by seed-number.

    Args:
        seed (int): Seed for the error realization.

    Returns:
        str: Path to the wise errortable file.
    """
    return pathstr('wise', f"WISE.errordef.{seed:04d}.tfs")


def drop_allzero_columns(df: TfsDataFrame) -> TfsDataFrame:
    """ Drop columns that contain only zeros, to save harddrive space.

    Args:
        df (TfsDataFrame): DataFrame with all data

    Returns:
        TfsDataFrame: DataFrame with only non-zero columns.
    """
    return df.loc[:, (df != 0).any(axis="index")]


def get_detuning_from_ptc_output(df, beam=None, log=True, terms=("X10", "Y01", "X01")):
    """ Convert PTC output to DataFrame. """
    results = dict.fromkeys(terms)
    if log:
        LOG.info("Current Detuning Values" + ("" if not beam else f" in Beam {beam}"))
    for term in terms:
        value = df.query(
            f'NAME == "ANH{term[0]}" and '
            f'ORDER1 == {term[1]} and ORDER2 == {term[2]} '
            f'and ORDER3 == 0 and ORDER4 == 0'
        )["VALUE"].to_numpy()[0]
        if log:
            LOG.info(f"  {term:<3s}: {value}")
        results[term] = value
    return results


class LHCCorrectors:
    """ Container for the corrector definitions used in the LHC.

        These correctors are installed into the MCTX and powered via kcdx3 and kctx3 circuits.
        The length is set to 0.615 m, which is the length of the MCTs.
        The pattern is used to find the correctors in the MAD-X sequence.
    """
    b5 = Corrector(
        field = "b5",
        length=0.615,
        magnet="MCTX.3{side}{ip}",  # installed into the MCTX
        circuit="kcdx3.{side}{ip}",
        pattern="MCTX.*[15]$",
    )
    b6 = Corrector(
        field = "b6",
        length=0.615,
        magnet="MCTX.3{side}{ip}",
        circuit="kctx3.{side}{ip}",
        pattern="MCTX.*[15]$",
    )


@dataclass()
class LHCBeam:
    """ Object containing all the information about the machine setup and
    performing the MAD-X commands to run the simulation. """
    beam: int
    outputdir: Path
    xing: dict
    optics: str
    year: int = 2018
    thin: bool = False
    tune_x: float = 62.31
    tune_y: float = 60.32
    chroma: float = 3
    emittance: float = 7.29767146889e-09
    n_particles: float = 1.0e10   # number of particles in beam
    # Placeholders (set in functions)
    df_twiss_nominal: TfsDataFrame = field(init=False)
    df_twiss_nominal_ir: TfsDataFrame = field(init=False)
    df_ampdet_nominal: TfsDataFrame = field(init=False)
    # Constants
    ACCEL: ClassVar[str] = 'lhc'
    TWISS_COLUMNS = ['NAME', 'KEYWORD', 'S', 'X', 'Y', 'L', 'LRAD',
                     'BETX', 'BETY', 'ALFX', 'ALFY', 'DX', 'DY', 'MUX', 'MUY',
                     'R11', 'R12', 'R21', 'R22'] + get_k_strings()
    ERROR_COLUMNS = ["NAME", "DX", "DY"] + get_k_strings()

    def __post_init__(self):
        """ Setup the MADX, output dirs and logging as well as additional instance parameters. """
        self.outputdir.mkdir(exist_ok=True, parents=True)
        self.madx = Madx(**cpymad_logging_setup(level=LOG_LEVEL,  # sets also standard loggers
                                                command_log=self.outputdir/'madx_commands.log',
                                                full_log=self.outputdir/'full_output.log'))
        self.logger = {key: logging.getLogger(key).handlers for key in ("", MADXOUT, MADXCMD)}  # save logger to reinstate later
        self.madx.globals.mylhcbeam = self.beam  # used in macros

        # Define Sequence to use
        self.seq_name, self.seq_file, self.bv_flag = get_lhc_sequence_filename_and_bv(self.beam, accel="lhc" if self.year < 2020 else "hllhc")  # `hllhc` just for naming of the sequence file, i.e. without _as_built

    # Output Helper ---
    def output_path(self, type_: str, output_id: str, dir_: Path | None = None, suffix: str = ".tfs") -> Path:
        """ Returns the output path for standardized tfs names in the default output directory.

        Args:
            type_ (str): Type of the output file (e.g. 'twiss', 'errors', 'ampdet')
            output_id (str): Name of the output (e.g. 'nominal')
            dir_ (Path): Override default directory.
            suffix (str): suffix of the output file.

        Returns:
            Path: Path to the output file
         """
        if dir_ is None:
            dir_ = self.outputdir
        return dir_ / f'{type_}.lhc.b{self.beam:d}.{output_id}{suffix}'

    def get_twiss(self, output_id=None, index_regex=r"BPM|M|IP", **kwargs) -> TfsDataFrame:
        """ Uses the ``twiss`` command to get the current optics in the machine
        as TfsDataFrame.

        Args:
            output_id (str): ID to use in the output (see ``output_path``).
                             If not given, no output is written.
            index_regex (str): Filter DataFrame index (NAME) by this pattern.

        Returns:
            TfsDataFrame: DataFrame containing the optics.
        """
        kwargs['chrom'] = kwargs.get('chrom', True)
        kwargs['centre'] = kwargs.get('centre', True)
        self.madx.twiss(sequence=self.seq_name, **kwargs)
        df_twiss = self.get_last_twiss(index_regex=index_regex)
        if output_id is not None:
            self.write_tfs(df_twiss, 'twiss', output_id)
        return df_twiss

    def get_last_twiss(self, index_regex=r"BPM|M|IP") -> TfsDataFrame:
        """ Returns the twiss table of the last calculated twiss.

        Args:
            index_regex (str): Filter DataFrame index (NAME) by this pattern.

        Returns:
            TfsDataFrame: DataFrame containing the optics.
        """
        return get_tfs(self.madx.table.twiss, columns=self.TWISS_COLUMNS, index_regex=index_regex)

    def get_ampdet(self, output_id: str) -> TfsDataFrame:
        """ Write out current amplitude detuning via PTC.

        Args:
            output_id (str): ID to use in the output (see ``output_path``).
                             If not given, no output is written.

        Returns:
            TfsDataFrame: Containing the PTC output data.
        """
        file = None
        if output_id is not None:
            file = self.output_path('ampdet', output_id)
            LOG.info(f"Calculating amplitude detuning for {output_id}.")
        df_ampdet = amplitude_detuning_ptc(self.madx, ampdet=2, chroma=4, file=file)
        get_detuning_from_ptc_output(df_ampdet, beam=self.beam)
        return df_ampdet

    def write_tfs(self, df: TfsDataFrame, type_: str, output_id: str):
        """ Write the given TfsDataFrame with the standardized name (see ``output_path``)
        and the index ``NAME``.

        Args:
            df (TfsDataFrame): DataFrame to write.
            type_ (str): Type of the output file (see ``output_path``)
            output_id (str): Name of the output (see ``output_path``)
        """
        tfs.write(self.output_path(type_, output_id), drop_allzero_columns(df), save_index="NAME")

    # Wrapper ---
    def log_orbit(self):
        """ Log the current orbit. """
        log_orbit(self.madx, accel=self.ACCEL, year=self.year)

    def closest_tune_approach(self, df: TfsDataFrame | None = None):
        """ Calculate and print out the closest tune approach from the twiss
        DataFrame given. If no frame is given, it gets the current twiss.

        Args:
            df (TfsDataFrame): Twiss DataFrame.
        """
        if df is None:
            df = self.get_twiss()
        df_coupling = coupling_via_cmatrix(df)
        closest_tune_approach(df_coupling, qx=self.tune_x, qy=self.tune_y)

    def correct_coupling(self):
        """ Correct the current coupling in the machine. """
        correct_coupling(self.madx,
                         accel=self.ACCEL, sequence=self.seq_name,
                         qx=self.tune_x, qy=self.tune_y,
                         dqx=self.chroma, dqy=self.chroma)

    def match_tune(self):
        """ Match the machine to the preconfigured tunes. """
        match_tune(self.madx,
                   accel=self.ACCEL, sequence=self.seq_name,
                   qx=self.tune_x, qy=self.tune_y,
                   dqx=self.chroma, dqy=self.chroma)

    def reinstate_loggers(self):
        """ Set the saved logger handlers to the current logger. """
        for name, handlers in self.logger.items():
            logging.getLogger(name).handlers = handlers

    def get_other_beam(self):
        """ Return the respective other beam number. """
        return 1 if self.beam == 4 else 4

    # Main ---

    def setup_machine(self):
        """ Nominal machine setup function.
        Initialized the beam and applies optics, crossing. """
        self.reinstate_loggers()
        madx = self.madx  # shorthand
        mvars = madx.globals  # shorthand

        # Load Macros
        madx.call(pathstr("optics2018", "toolkit", "macro.madx"))

        # Lattice Setup ---------------------------------------
        # Load Sequence
        if self.year > 2019:  # after 2019, use acc-models
            acc_models_path = PATHS[ACC_MODELS]
            if acc_models_path.exists():
                acc_models_path.unlink()
            acc_models_path.symlink_to(pathstr("optics_repo", str(self.year)))
            madx.call(pathstr(ACC_MODELS, self.seq_file))
        else:
            madx.call(pathstr("optics2018", self.seq_file))

        # Slice Sequence
        if self.thin:
            mvars.slicefactor = 4
            madx.beam()
            madx.call(pathstr("optics2018", "toolkit", "myslice.madx"))
            madx.beam()
            madx.use(sequence=self.seq_name)
            madx.makethin(sequence=self.seq_name, style="teapot", makedipedge=True)

        # Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
        madx.seqedit(sequence=self.seq_name)
        madx.flatten()
        madx.cycle(start="IP3")
        madx.endedit()

        # Define Optics and make beam
        madx.call(get_optics_path(self.year, self.optics))
        if self.optics == 'inj':
            mvars.NRJ = 450.000  # not defined in injection optics.1 but in the others

        madx.beam(sequence=self.seq_name, bv=self.bv_flag,
                  energy="NRJ", particle="proton", npart=self.n_particles,
                  kbunch=1, ex=self.emittance, ey=self.emittance)

        # Setup Orbit
        orbit_vars = orbit_setup(madx, accel='lhc', year=self.year, **self.xing)

        madx.use(sequence=self.seq_name)

    def save_nominal(self, id_="nominal"):
        """ Save nominal machine into Dataclass slots and (if `id_` is not None) output to tfs. """
        self.reinstate_loggers()

        # Save Nominal
        self.match_tune()
        self.df_twiss_nominal = self.get_twiss(id_)
        self.df_ampdet_nominal = self.get_ampdet(id_)
        self.log_orbit()

        # Save nominal optics in IR+Correctors for ir nl correction
        self.df_twiss_nominal_ir = self.get_last_twiss(index_regex="M(QS?X|BX|BRC|C[SOT]S?X)")
        if id_ is not None:
            ir_id = 'optics_ir' + ("" if id_ == "nominal" else f"_{id_}")
            self.write_tfs(self.df_twiss_nominal_ir, 'twiss', ir_id)

    def install_circuits_into_mctx(self):
        """ Installs kcdx and (and reinstalls kctx) into the Dodecapole Correctors.

        This allows for decapole and dodecapole correction with the MCTX magnets for test purposes.
        """
        self.reinstate_loggers()
        beam_sign_str = "-" if self.beam == 4 else ""
        for ip in (1, 5):
            for side in "LR":
                magnet = LHCCorrectors.b6.magnet.format(side=side, ip=ip)
                magnet_b5 = LHCCorrectors.b5.magnet.format(side=side, ip=ip)

                assert magnet_b5 == magnet, "Magnet name for b5 and b6 must be the same as we install b5 corrector in same magnet!"

                deca_circuit = LHCCorrectors.b5.circuit.format(side=side.lower(), ip=ip)
                dodeca_circuit = LHCCorrectors.b6.circuit.format(side=side.lower(), ip=ip)

                self.madx.input(f"{magnet}, KNL := {{0, 0, 0, 0, {deca_circuit}*l.MCTX, {beam_sign_str}{dodeca_circuit}*l.MCTX}}, polarity=+1;")
                self.madx.globals[deca_circuit] = 0
                self.madx.globals[dodeca_circuit] = 0

    def reset_detuning_circuits(self):
        """ Reset all kcdx and kctx circuits (to zero). """
        for circuit in (LHCCorrectors.b5.circuit, LHCCorrectors.b6.circuit):
            for ip in (1, 5):
                for side in "LR":
                    self.madx.globals[circuit.format(side=side.lower(), ip=ip)] = 0

    def set_mctx_circuits_powering(self, knl_values: dict[str, str | float], id_: str = ''):
        """ Set the knl_values at the corrector circuits and write them out.
        Try to also match tune and run twiss and ptc and output data. """
        self.reinstate_loggers()
        id_ = id_ if id_ else 'w_ampdet'
        magnet_l = "l.MCTX"
        magnet_length = self.madx.globals[magnet_l]
        df = tfs.TfsDataFrame(index=list(knl_values.keys()), columns=["VALUE", "KNL"], headers={magnet_l: magnet_length})

        madx_command = [f'! Amplitude detuning powering {id_}:', f'! reminder: {magnet_l} = {magnet_length}']
        for key, knl in knl_values.items():
            value = knl / magnet_length
            madx_command.append(f"{key} := {knl} / {magnet_l};  ! {key} = {value};")
            df.loc[key, "VALUE"] = value
            df.loc[key, "KNL"] = knl
            self.madx.globals[key] = value

        self.output_path('settings', id_, suffix=".madx").write_text("\n".join(madx_command))
        tfs.write(self.output_path('settings', id_), df, save_index="CIRCUIT")

        try:
            self.match_tune()
            self.get_twiss(id_, index_regex=LHCCorrectors.b6.pattern)
        except cpymad.madx.TwissFailed as e:
            LOG.error("Matching/Twiss failed!")
            return None
        else:
            return self.get_ampdet(id_)

    def check_kctx_limits(self):
        """ Check the corrector kctx limits."""
        self.reinstate_loggers()
        checks = LimitChecks(madx=self.madx, beam=self.beam,
                             limit_to_max=False,
                             values_dict={"MCTX1": 'kmax_MCTX'})
        checks.run_checks()
        if not checks.success:
            # raise ValueError("One or more strengths are out of its limits, see log.")
            pass


@dataclass()
class FakeLHCBeam:
    """ Mock of LHCBeam to use in calculations without the functions noticing.
    Used in the main-functions to load tfs-files without running MAD-X again.
    """
    beam: int
    outputdir: Path

    def __post_init__(self):
        self.outputdir.mkdir(exist_ok=True, parents=True)
