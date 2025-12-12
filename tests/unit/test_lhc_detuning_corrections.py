from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import tfs
from pandas.testing import assert_frame_equal, assert_series_equal

from ir_amplitude_detuning.detuning.measurements import MeasureValue
from ir_amplitude_detuning.lhc_detuning_corrections import (
    CorrectionResults,
    calculate_corrections,
    check_corrections_analytically,
    check_corrections_ptc,
    create_optics,
    detuning_tfs_out_with_and_without_errors,
    generate_knl_tfs,
    generate_madx_command,
    get_label_outputdir,
    get_nominal_optics,
    get_optics,
)
from ir_amplitude_detuning.simulation.lhc_simulation import ACC_MODELS, FakeLHCBeam
from ir_amplitude_detuning.utilities.constants import (
    AMPDET_CALC_ERR_ID,
    AMPDET_CALC_ID,
    CIRCUIT,
    ERR,
    KN,
    KNL,
    LENGTH,
    NAME,
    NOMINAL_ID,
    SETTINGS_ID,
)
from ir_amplitude_detuning.utilities.correctors import Corrector, FieldComponent

# ============================================================================
# Tests for get_optics
# ============================================================================

class TestGetOptics:
    """Test cases for the get_optics function."""

    def test_get_optics_2018(self):
        """Test getting optics for year 2018."""
        result = get_optics(2018)
        assert isinstance(result, str)
        assert result.startswith(str(ACC_MODELS))
        assert "PROTON" in result
        assert result.endswith("opticsfile.22_ctpps2")

    def test_get_optics_2022(self):
        """Test getting optics for year 2022."""
        result = get_optics(2022)
        assert isinstance(result, str)
        assert result.startswith(str(ACC_MODELS))
        assert result.endswith("ats_30cm.madx")

    def test_get_optics_invalid_year(self):
        """Test that invalid year raises KeyError."""
        with pytest.raises(KeyError):
            get_optics(2020)

    def test_get_optics_return_type(self):
        """Test that get_optics always returns a string."""
        for year in [2018, 2022]:
            result = get_optics(year)
            assert isinstance(result, str)


# ============================================================================
# Tests for CorrectionResults dataclass
# ============================================================================

class TestCorrectionResults:
    """Test cases for the CorrectionResults dataclass."""

    def test_correction_results_creation(self):
        """Test creating a CorrectionResults instance."""
        series = pd.Series([1.0, 2.0], index=[Mock(), Mock()])
        df = tfs.TfsDataFrame()
        madx_cmd = "K1 := 1.0;"

        result = CorrectionResults(
            name="test_target",
            series=series,
            dataframe=df,
            madx=madx_cmd
        )

        assert result.name == "test_target"
        assert len(result.series) == 2
        assert isinstance(result.dataframe, tfs.TfsDataFrame)
        assert result.madx == madx_cmd


# ============================================================================
# Tests for get_label_outputdir
# ============================================================================

class TestGetLabelOutputdir:
    """Test cases for the get_label_outputdir function."""

    def test_get_label_outputdir_empty_label(self, tmp_path):
        """Test with empty label."""
        result = get_label_outputdir(tmp_path, "", 1)
        assert result == tmp_path / "b1"

    def test_get_label_outputdir_different_beams(self, tmp_path):
        """Test with different beam numbers."""
        result1 = get_label_outputdir(tmp_path, "label", 1)
        result2 = get_label_outputdir(tmp_path, "label", 4)

        assert result1 == tmp_path / "label_b1"
        assert result2 == tmp_path / "label_b4"


# ============================================================================
# Tests for generate_madx_command
# ============================================================================

class TestGenerateMadxCommand:
    """Test cases for the generate_madx_command function."""

    def test_generate_madx_command_single_corrector(self):
        """Test generating MAD-X command with a single corrector."""
        corrector = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="MCTX.1L1",
            circuit="kctx1.l1",
            madx_type="MCTX"
        )
        series = pd.Series([2.0], index=[corrector])

        result = generate_madx_command(series)

        assert "kctx1.l1 := 2.0 / l.MCTX;" in result
        assert "! Amplitude detuning powering:" in result

    def test_generate_madx_command_multiple_correctors(self):
        """Test generating MAD-X command with multiple correctors."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="MCTX.1L1",
                     circuit="kctx1.l1", madx_type="MCTX"),
            Corrector(field=FieldComponent.b4, length=0.3, magnet="MCTX.2L1",
                     circuit="kctx2.l1", madx_type="MCTQ")
        ]
        series = pd.Series([2.0, 3.0], index=correctors)

        result = generate_madx_command(series)

        assert "kctx1.l1 := 2.0 / l.MCTX;" in result
        assert "kctx2.l1 := 3.0 / l.MCTQ;" in result
        assert "! reminder: l.MCTQ = 0.3" in result
        assert "! reminder: l.MCTX = 0.5" in result
        assert "! Amplitude detuning powering:" in result

    def test_generate_madx_command_without_madx_type(self):
        """Test generating MAD-X command for corrector without madx_type."""
        corrector = Corrector(
            field=FieldComponent.b4,
            length=0.5,
            magnet="MCTX.1L1",
            circuit="kctx1.l1",
            madx_type=None
        )
        series = pd.Series([2.0], index=[corrector])

        result = generate_madx_command(series)

        assert "kctx1.l1 := 2.0 / 0.5;" in result


# ============================================================================
# Tests for generate_knl_tfs
# ============================================================================

class TestGenerateKnlTfs:
    """Test cases for the generate_knl_tfs function."""

    def test_generate_knl_tfs_single_corrector_with_value(self):
        """Test generating KNL TFS with a single corrector (simple value)."""
        corrector = Corrector(
            field=FieldComponent.b4, length=0.5, magnet="MCTX.1L1",
            circuit="kctx1.l1", madx_type="MCTX"
        )
        series = pd.Series([2.0], index=[corrector])

        result = generate_knl_tfs(series)

        assert isinstance(result, tfs.TfsDataFrame)
        assert "MCTX.1L1" in result.index
        assert result.loc["MCTX.1L1", CIRCUIT] == "kctx1.l1"
        assert result.loc["MCTX.1L1", LENGTH] == 0.5
        assert result.loc["MCTX.1L1", KNL] == 2.0
        assert result.loc["MCTX.1L1", KN] == 4.0  # 2.0 / 0.5
        assert set(result.columns) == {CIRCUIT, LENGTH, KNL, KN}

    def test_generate_knl_tfs_with_measure_value(self):
        """Test generating KNL TFS with MeasureValue objects."""
        corrector = Corrector(
            field=FieldComponent.b4, length=0.5, magnet="MCTX.1L1",
            circuit="kctx1.l1", madx_type="MCTX"
        )
        measure_value = MeasureValue(value=2.0, error=0.1)
        series = pd.Series([measure_value], index=[corrector])

        result = generate_knl_tfs(series)

        assert result.loc["MCTX.1L1", KNL] == 2.0
        assert result.loc["MCTX.1L1", f"{ERR}{KNL}"] == 0.1
        assert result.loc["MCTX.1L1", KN] == 4.0
        assert result.loc["MCTX.1L1", f"{ERR}{KN}"] == 0.2

        assert set(result.columns) == {CIRCUIT, LENGTH, KNL, KN, f"{ERR}{KN}", f"{ERR}{KNL}"}


    def test_generate_knl_tfs_multiple_correctors(self):
        """Test generating KNL TFS with multiple correctors."""
        correctors = [
            Corrector(field=FieldComponent.b4, length=0.5, magnet="MCTX.1L1",
                     circuit="kctx1.l1", madx_type="MCTX"),
            Corrector(field=FieldComponent.b4, length=0.3, magnet="MCTX.2L1",
                     circuit="kctx2.l1", madx_type="MCTX")
        ]
        series = pd.Series([2.0, 3.0], index=correctors)

        result = generate_knl_tfs(series)

        assert len(result) == 2
        assert "MCTX.1L1" in result.index
        assert "MCTX.2L1" in result.index

        expected_columns = [CIRCUIT, LENGTH, KNL, KN]
        for col in expected_columns:
            assert col in result.columns


# ============================================================================
# Tests for detuning_tfs_out_with_and_without_errors
# ============================================================================

class TestDetuningTfsOutWithAndWithoutErrors:
    """Test cases for the detuning_tfs_out_with_and_without_errors function."""

    def test_detuning_output_simple_values(self, tmp_path):
        """Test output with simple float values."""
        lhc_out = FakeLHCBeam(beam=1, outputdir=tmp_path)

        df = pd.DataFrame({
            "strings": ["a", "b", "c"],
            'dQ1': [1.0, 2.0, 3.0],
            'dQ2': [0.5, 1.5, 2.5]
        })

        detuning_tfs_out_with_and_without_errors(lhc_out, "test_id", df)
        assert lhc_out.output_path(AMPDET_CALC_ID, "test_id").exists()
        assert not lhc_out.output_path(AMPDET_CALC_ERR_ID, "test_id").exists()

        df_out = tfs.read(lhc_out.output_path(AMPDET_CALC_ID, "test_id"))
        assert_frame_equal(df, df_out, check_frame_type=False)


    def test_detuning_output_with_measure_values(self, tmp_path):
        """Test output with MeasureValue objects."""
        lhc_out = FakeLHCBeam(beam=1, outputdir=tmp_path)

        df = pd.DataFrame({
            "strings": ["a", "b"],
            'dQ1': [MeasureValue(1.0, 0.1), MeasureValue(2.0)],
            'dQ2': [MeasureValue(5.0, 0.1), MeasureValue(0.5, 0.05)]
        })

        detuning_tfs_out_with_and_without_errors(lhc_out, "test_id", df)

        assert lhc_out.output_path(AMPDET_CALC_ID, "test_id").exists()
        assert lhc_out.output_path(AMPDET_CALC_ERR_ID, "test_id").exists()

        df_out = tfs.read(lhc_out.output_path(AMPDET_CALC_ID, "test_id"))
        df_out_err = tfs.read(lhc_out.output_path(AMPDET_CALC_ERR_ID, "test_id"))

        for col in df.columns:
            data = df[col]
            if col != "strings":
                assert all(df_out_err[f"{ERR}{col}"] == data.apply(lambda x: x.error).fillna(0.))
                data = data.apply(lambda x: x.value)

            assert all(df_out[col] == data)
            assert all(df_out_err[col] == data)



# ============================================================================
# Tests for get_nominal_optics
# ============================================================================

class TestGetNominalOptics:
    """Test cases for the get_nominal_optics function."""

    def test_get_nominal_optics_from_lhc_beams_dict(self):
        """Test getting nominal optics from LHCBeams dictionary."""
        beam1_df = tfs.TfsDataFrame()
        beam1_df['x'] = [1.0, 2.0]

        beam4_df = tfs.TfsDataFrame()
        beam4_df['x'] = [3.0, 4.0]

        mock_beam1 = Mock()
        mock_beam1.df_twiss_nominal = beam1_df

        mock_beam4 = Mock()
        mock_beam4.df_twiss_nominal = beam4_df

        lhc_beams = {1: mock_beam1, 4: mock_beam4}

        result = get_nominal_optics(lhc_beams)

        assert 1 in result
        assert 4 in result
        assert len(result[1]) == 2
        assert len(result[4]) == 2
        assert_frame_equal(result[1], beam1_df)
        assert_frame_equal(result[4], beam4_df)

    def test_get_nominal_optics_requires_outputdir_for_sequence(self, tmp_path):
        """Test that outputdir is required when beams is a sequence."""
        with pytest.raises(ValueError, match="outputdir must be provided"):
            get_nominal_optics([1, 4], outputdir=None)

    def test_get_nominal_optics_from_file(self, tmp_path):
        """Test reading nominal optics from files."""
        # Create fake nominal twiss file
        beam1_dir = tmp_path / "b1"
        beam1_dir.mkdir()

        fake_df = tfs.TfsDataFrame()
        fake_df.index.name = NAME
        fake_df['x'] = [1.0, 2.0]

        nominal_file = FakeLHCBeam(beam=1, outputdir=beam1_dir).output_path("twiss", NOMINAL_ID)
        tfs.write(nominal_file, fake_df, save_index=NAME)

        result = get_nominal_optics([1], outputdir=tmp_path)

        assert 1 in result
        assert_frame_equal(result[1], fake_df)


# ============================================================================
# Tests for check_corrections_ptc
# ============================================================================

class TestCheckCorrectionsPtc:
    """Test cases for the check_corrections_ptc function."""

    def test_ptc_check_corrections_fail_no_files(self, tmp_path):
        """Test that FileNotFoundError is raised when no settings files exist."""
        fake_beam = FakeLHCBeam(1, tmp_path)
        fake_beam.install_circuits_into_mctx = Mock()
        beams = {1: fake_beam}

        with pytest.raises(FileNotFoundError) as e:
            check_corrections_ptc(
                outputdir=tmp_path,
                lhc_beams=beams,
            )
        assert "No settings files found" in str(e)
        fake_beam.install_circuits_into_mctx.assert_called_once()

    def test_ptc_check_corrections_requires_beams_info(self, tmp_path):
        """Test that either lhc_beams or beams parameter must be provided."""
        with pytest.raises(ValueError, match="Either lhc_beams or beams must be given"):
            check_corrections_ptc(outputdir=tmp_path)

    def test_ptc_check_corrections_with_lhc_beams(self, tmp_path):
        """Test PTC check with pre-created LHC beams."""
        # Create a settings file so the test gets past the FileNotFoundError
        class MockMadx:
            def __init__(self):
                self.commands = ""

            def input(self, commands: str):
                self.commands += commands

        fake_beam = FakeLHCBeam(1, tmp_path)
        fake_beam.madx = MockMadx()
        fake_beam.match_tune = Mock()
        fake_beam.get_twiss = Mock()
        fake_beam.get_ampdet = Mock()
        fake_beam.check_kctx_limits = Mock()
        fake_beam.reset_detuning_circuits = Mock()
        fake_beam.install_circuits_into_mctx = Mock()

        settings_text = "K1 := 1.0;"
        settings_file = fake_beam.output_path(SETTINGS_ID, "my_settings", suffix=".madx")
        settings_file.write_text(settings_text)

        beams = {1: fake_beam}

        check_corrections_ptc(outputdir=tmp_path, lhc_beams=beams)

        assert fake_beam.madx.commands == settings_text
        assert fake_beam.install_circuits_into_mctx.call_count == 1
        assert fake_beam.match_tune.call_count == 2
        assert fake_beam.get_twiss.call_count == 2
        assert fake_beam.get_ampdet.call_count == 2
        assert fake_beam.check_kctx_limits.call_count == 1
        assert fake_beam.reset_detuning_circuits.call_count == 1


# ============================================================================
# Tests for create_optics (mocked)
# ============================================================================

class TestCreateOptics:
    """Test cases for the create_optics function."""

    @patch('ir_amplitude_detuning.lhc_detuning_corrections.LHCBeam')
    def test_create_optics(self, mock_lhc_beam_class, tmp_path):
        """Test creating optics for multiple beams."""
        mock_beam_instance = Mock()
        mock_beam_instance.setup_machine = Mock()
        mock_beam_instance.save_nominal = Mock()
        mock_lhc_beam_class.return_value = mock_beam_instance

        result = create_optics(beams=[1, 4], outputdir=tmp_path)

        assert 1 in result
        assert 4 in result
        assert mock_lhc_beam_class.call_count == 2
        assert mock_beam_instance.setup_machine.call_count == 2
        assert mock_beam_instance.save_nominal.call_count == 2

        call_kwargs = mock_lhc_beam_class.call_args[1]
        assert call_kwargs["optics"] == get_optics(2018)
        assert call_kwargs["xing"] == {'scheme': 'top'}

    @patch('ir_amplitude_detuning.lhc_detuning_corrections.LHCBeam')
    def test_create_optics_with_custom_parameters(self, mock_lhc_beam_class, tmp_path):
        """Test creating optics with custom parameters."""
        mock_beam_instance = Mock()
        mock_beam_instance.setup_machine = Mock()
        mock_beam_instance.save_nominal = Mock()
        mock_lhc_beam_class.return_value = mock_beam_instance

        class Params:
            beams=[1]
            outputdir=tmp_path
            output_id="test_id"
            year=2022
            xing={"test": 22}
            tune_x=62.0
            tune_y=60.0


        create_optics(
            beams=Params.beams,
            outputdir=Params.outputdir,
            output_id=Params.output_id,
            year=Params.year,
            tune_x=Params.tune_x,
            tune_y=Params.tune_y,
        )

        # Verify LHCBeam was called with the correct parameters
        call_kwargs = mock_lhc_beam_class.call_args[1]
        assert call_kwargs['year'] == Params.year
        assert call_kwargs['tune_x'] == Params.tune_x
        assert call_kwargs['tune_y'] == Params.tune_y
        assert call_kwargs["optics"] == get_optics(Params.year)
        assert call_kwargs["xing"] == {'scheme': 'top'}
        assert call_kwargs["outputdir"] == tmp_path / f"{Params.output_id}_b{Params.beams[0]}"


# ============================================================================
# Tests for calculate_corrections (mocked)
# ============================================================================

class TestCalculateCorrections:
    """Test cases for the calculate_corrections function."""

    @patch('ir_amplitude_detuning.lhc_detuning_corrections.calculate_correction')
    def test_calculate_corrections_success_and_fail(
        self, mock_calc_correction, tmp_path
    ):
        """Test successful calculation of corrections."""
        # Setup mocks ---
        corrector = Corrector(
            field=FieldComponent.b4, length=0.5, magnet="MCTX.1L1",
            circuit="kctx1.l1", madx_type="MCTX"
        )
        series = pd.Series([2.0], index=[corrector])

        mock_calc_correction.side_effect = [series, ValueError(), series]

        mock_target0 = Mock()
        mock_target0.name = "test_target"

        mock_target1 = Mock()
        mock_target1.name = "failing_target"
        mock_target1.correctors = [corrector]

        mock_target2 = Mock()
        mock_target2.name = "other_target"

        targets = [mock_target0, mock_target1, mock_target2]

        # Run ---
        results = calculate_corrections(beams=[1, 4], outputdir=tmp_path, targets=targets)

        # Checks ---
        assert len(results) == len(targets) - 1
        assert mock_calc_correction.call_count == len(targets)

        for idx, target in enumerate(targets):
            is_failing = "failing" in target.name
            if is_failing:
                assert target.name not in results
            else:
                assert target.name in results

                result = results[target.name]
                assert isinstance(result, CorrectionResults)
                assert result.name == target.name
                assert_series_equal(result.series, series)
                assert isinstance(result.dataframe, tfs.TfsDataFrame)
                assert isinstance(result.madx, str)

            calc_args = mock_calc_correction.mock_calls[idx]
            assert calc_args.args[0] == target
            assert "method" in calc_args.kwargs

            for beam in (1, 4):
                file_path = tmp_path / f"{SETTINGS_ID}.lhc.b{beam}.{target.name}.tfs"
                file_path_madx = file_path.with_suffix(".madx")
                if is_failing:
                    assert not file_path.is_file()
                    assert not file_path_madx.is_file()
                else:
                    assert file_path.is_file()
                    tfs.read_tfs(file_path)
                    assert file_path_madx.is_file()

    def test_calculate_corrections_no_targets(self):
        result = calculate_corrections(beams=[], outputdir=None, targets=[])
        assert result == {}


# ============================================================================
# Tests for check_corrections_analytically
# ============================================================================

class TestCheckCorrectionsAnalytically:
    """Test cases for the check_corrections_analytically function."""

    @patch('ir_amplitude_detuning.lhc_detuning_corrections.calc_effective_detuning')
    @patch('ir_amplitude_detuning.lhc_detuning_corrections.detuning_tfs_out_with_and_without_errors')
    def test_analytical_check(self, mock_detuning_tfs_out, mock_calc_detuning, tmp_path):
        """Test analytical detuning check."""
        mock_results = {1: "res1", 4: "res4"}
        mock_calc_detuning.return_value = mock_results

        optics = {1: None, 4: None}
        results = CorrectionResults(
            name="test_target",
            series=pd.Series(),
            dataframe=tfs.TfsDataFrame(),
            madx="K1 := 2.0;"
        )

        check_corrections_analytically(tmp_path, optics, results)

        # Verify calculation of effective detuning was called once
        assert mock_calc_detuning.call_count == 1
        assert mock_calc_detuning.mock_calls[0].args[0] is optics
        assert mock_calc_detuning.mock_calls[0].args[1] is results.series

        # Verify detuning output was called for each beam (function itself tested above)
        assert mock_detuning_tfs_out.call_count == 2
        for idx, beam in enumerate((1, 4)):
            assert mock_detuning_tfs_out.mock_calls[idx].args[0].beam == beam
            assert mock_detuning_tfs_out.mock_calls[idx].args[1] == results.name
            assert mock_detuning_tfs_out.mock_calls[idx].args[2] == mock_results[beam]
