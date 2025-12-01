from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
import tfs
from omc3.optics_measurements.constants import EXT, KICK_NAME
from omc3.tune_analysis.constants import get_bbq_out_name, get_kick_out_name

from ir_amplitude_detuning.detuning.measurements import Detuning, DetuningMeasurement
from ir_amplitude_detuning.detuning.terms import FirstOrderTerm, SecondOrderTerm
from ir_amplitude_detuning.utilities.constants import ERR
from ir_amplitude_detuning.utilities.measurement_analysis import (
    AnalysisOption,
    create_summary,
    do_detuning_analysis,
    get_beam_from_dir,
    get_detuning_from_series,
    get_kick_plane_from_dir,
    get_row_from_odr_headers,
    get_terms_and_error_terms,
)
from tests.conftest import INPUTS_DIR

if TYPE_CHECKING:
    from pathlib import Path

MEASUREMENT_INPUTS_DIR = INPUTS_DIR / "measurement"

# ============================================================================
# Tests for AnalysisOption Enum
# ============================================================================

class TestAnalysisOption:
    """Test cases for the AnalysisOption enum."""

    def test_analysis_option_values(self):
        """Test that all analysis option values are correct."""
        assert len(list(AnalysisOption))  == 3  # adapt if more are added
        assert AnalysisOption.always == "always"
        assert AnalysisOption.never == "never"
        assert AnalysisOption.auto == "auto"


# ============================================================================
# Tests for get_beam_from_dir
# ============================================================================

class TestGetBeamFromDir:
    """Test cases for the get_beam_from_dir function."""

    def test_get_beam_from_dir_with_prefix(self, tmp_path: Path):
        """Test beam extraction from directory with [Bb]{beam}_ prefix."""
        analysis_dir = tmp_path / "B1_some_analysis"
        analysis_dir.mkdir()
        beam = get_beam_from_dir(analysis_dir)
        assert beam == 1

    def test_get_beam_from_dir_with_prefix_beam2(self, tmp_path: Path):
        """Test beam extraction with beam 2."""
        analysis_dir = tmp_path / "b2_analysis_dir"
        analysis_dir.mkdir()
        beam = get_beam_from_dir(analysis_dir)
        assert beam == 2

    def test_get_beam_from_dir_with_parent_directory(self, tmp_path: Path):
        """Test beam extraction from parent directory LHCB{beam} structure."""
        # Create directory structure: LHCB1/Results/Analysis/some_analysis
        lhc_dir = tmp_path / "LHCB1"
        results_dir = lhc_dir / "Results"
        analysis_dir = results_dir / "some_analysis"
        analysis_dir.mkdir(parents=True)
        beam = get_beam_from_dir(analysis_dir)
        assert beam == 1

    def test_get_beam_from_dir_with_parent_directory_beam2(self, tmp_path: Path):
        """Test beam extraction with beam 2 from parent directory."""
        # Create directory structure: LHCB2/Results/Analysis/analysis
        lhc_dir = tmp_path / "LHCB2"
        results_dir = lhc_dir / "Results"
        analysis_dir = results_dir / "analysis"
        analysis_dir.mkdir(parents=True)

        beam = get_beam_from_dir(analysis_dir)
        assert beam == 2

    def test_get_beam_from_dir_prefix_takes_precedence(self, tmp_path: Path):
        """Test that B{beam}_ prefix takes precedence over parent directory."""
        # Create directory structure with both prefix and parent directory
        lhc_dir = tmp_path / "LHCB2"
        results_dir = lhc_dir / "Results"
        analysis_base_dir = results_dir / "Analysis"
        analysis_dir = analysis_base_dir / "B1_analysis"
        analysis_dir.mkdir(parents=True)

        # Should return 1 from the prefix, not 2 from parent
        beam = get_beam_from_dir(analysis_dir)
        assert beam == 1

    def test_get_beam_from_dir_invalid_directory(self, tmp_path: Path):
        """Test that invalid directory raises ValueError."""
        analysis_dir = tmp_path / "invalid_directory_name"
        analysis_dir.mkdir()
        with pytest.raises(ValueError, match="Could not determine beam"):
            get_beam_from_dir(analysis_dir)


# ============================================================================
# Tests for get_kick_plane_from_dir
# ============================================================================

class TestGetKickPlaneFromDir:
    """Test cases for the get_kick_plane_from_dir function."""

    @pytest.mark.parametrize("dir_name,expected", [
        ("analysis_X_data", "x"),
        ("analysis_Y_data", "y"),
        ("analysis_H_data", "x"),
        ("analysis_V_data", "y"),
        ("analysis_x_data", "x"),
        ("analysis_y_data", "y"),
        ("analysis_h_data", "x"),
        ("analysis_v_data", "y"),
        ("analysis.X.data", "x"),
        ("analysis.Y.data", "y"),
        ("data_X", "x"),
        ("data_Y", "y"),
        ("end_with_H", "x"),
        ("end_with_V", "y"),
    ])
    def test_get_kick_plane_from_dir_valid(self, tmp_path, dir_name, expected):
        """Test kick plane extraction with various directory naming patterns."""
        analysis_dir = tmp_path / dir_name
        analysis_dir.mkdir()
        plane = get_kick_plane_from_dir(analysis_dir)
        assert plane == expected

    def test_get_kick_plane_from_dir_invalid_directory(self, tmp_path: Path):
        """Test that invalid directory raises ValueError."""
        analysis_dir = tmp_path / "invalid_directory_without_plane"
        analysis_dir.mkdir()
        with pytest.raises(ValueError, match="Could not determine kick plane"):
            get_kick_plane_from_dir(analysis_dir)

    def test_get_kick_plane_from_dir_with_multiple_matches(self, tmp_path: Path):
        """Test that first match is used when multiple planes present."""
        # Should match the first occurrence (note: maybe should throw error?)
        analysis_dir = tmp_path / "analysis_X_and_Y_data"
        analysis_dir.mkdir()
        plane = get_kick_plane_from_dir(analysis_dir)
        assert plane == "x"


# ============================================================================
# Tests for get_terms_and_error_terms
# ============================================================================

class TestGetTermsAndErrorTerms:
    """Test cases for the get_terms_and_error_terms function."""

    def test_get_terms_and_error_terms_empty(self):
        """Test with empty terms list."""
        result = get_terms_and_error_terms([])
        assert result == []

    def test_get_terms_and_error_terms_single(self):
        """Test with single term."""
        result = get_terms_and_error_terms([FirstOrderTerm.X01])
        assert FirstOrderTerm.X01 in result
        assert f"{ERR}{FirstOrderTerm.X01}" in result
        assert len(result) == 2

    def test_get_terms_and_error_terms_multiple(self):
        """Test with multiple terms."""
        so_terms = list(SecondOrderTerm)
        result = get_terms_and_error_terms(so_terms)
        assert len(result) == 2 * len(so_terms)
        for term in so_terms:
            assert term in result
            assert f"{ERR}{term}" in result


# ============================================================================
# Tests for get_row_from_odr_headers
# ============================================================================

class TestGetRowFromOdrHeaders:
    """Test cases for the get_row_from_odr_headers function."""

    def test_get_row_from_odr_headers_empty_headers( self):
        """Test with empty headers."""
        kick_df = tfs.TfsDataFrame()
        result = get_row_from_odr_headers(kick_df, "test_analysis")
        assert isinstance(result, pd.DataFrame)
        assert len(result == 1)
        assert result.iloc[0].isna().all()
        assert result.index[0] == "test_analysis"

    @pytest.mark.parametrize("plane", ("x", "y", "xy"))
    def test_get_row_from_odr_headers_with_values(self, plane):
        """Test with values in headers."""
        kick_df = get_kick_ampdet_header_tdf(plane)
        result = get_row_from_odr_headers(kick_df, "test_analysis")
        for idx, (tune_plane, order) in enumerate((("X", 1), ("Y", 1), ("X", 2), ("Y", 2)), start=1):  # order as in get-func
            if "x" in plane:
                assert result.loc["test_analysis", f"{tune_plane}{order}0"] == idx
                assert result.loc["test_analysis", f"{ERR}{tune_plane}{order}0"] == idx/10
            else:
                assert np.isnan(result.loc["test_analysis", f"{tune_plane}{order}0"])
                assert np.isnan(result.loc["test_analysis", f"{ERR}{tune_plane}{order}0"])

            if "y" in plane:
                assert result.loc["test_analysis", f"{tune_plane}0{order}"] == idx
                assert result.loc["test_analysis", f"{ERR}{tune_plane}0{order}"] == idx/10
            else:
                assert np.isnan(result.loc["test_analysis", f"{tune_plane}0{order}"])
                assert np.isnan(result.loc["test_analysis", f"{ERR}{tune_plane}0{order}"])

            # assert dual-action planes always skipped:
            assert np.isnan(result.loc["test_analysis", f"{tune_plane}11"])
            assert np.isnan(result.loc["test_analysis", f"{ERR}{tune_plane}11"])


# ============================================================================
# Tests for get_detuning_from_series
# ============================================================================

class TestGetDetuningFromSeries:
    """Test cases for the get_detuning_from_series function."""

    def test_get_detuning_from_series_without_errors(self):
        """Test with series without error terms."""
        series = pd.Series({FirstOrderTerm.X01: 1.5, SecondOrderTerm.Y11: 2.5})

        result = get_detuning_from_series(series)
        assert isinstance(result, Detuning)
        assert len(list(result.terms())) == 2
        assert result.X01 == 1.5
        assert result.Y11 == 2.5

    def test_get_detuning_from_series_with_errors(self):
        """Test with series containing error terms."""
        series = pd.Series({
            FirstOrderTerm.X10: 1.0,
            f"{ERR}{FirstOrderTerm.X10}": 0.1,
            SecondOrderTerm.Y20: 2.0,
            f"{ERR}{SecondOrderTerm.Y20}": 0.2,
            FirstOrderTerm.Y01: 3.0,  # only value
            f"{ERR}{FirstOrderTerm.X01}": 0.4,  # only error
        })

        result = get_detuning_from_series(series)
        assert isinstance(result, DetuningMeasurement)
        assert len(list(result.terms())) == 3
        assert result.X10.value == 1.0
        assert result.X10.error == 0.1
        assert result.Y20.value == 2.0
        assert result.Y20.error == 0.2
        assert result.Y01.value == 3.0
        assert result.Y01.error == 0.0
        assert result.X01 is None  # only error is skipped


# ============================================================================
# Tests for do_detuning_analysis
# ============================================================================

class TestDoDetuningAnalysis:
    """Test cases for the do_detuning_analysis function."""
    kick_x =  f"{KICK_NAME}x{EXT}"
    kick_y =  f"{KICK_NAME}y{EXT}"

    @patch('ir_amplitude_detuning.utilities.measurement_analysis.get_beam_from_dir')
    def test_do_detuning_analysis_missing_files(self, mock_get_beam, tmp_path: Path):
        """Test that missing kick files raise ValueError."""

        kick_x = tmp_path / self.kick_x
        kick_y = tmp_path / self.kick_y

        # trigger return
        def raises_key(analysis_dir):
            raise KeyError("End of Test")
        mock_get_beam.side_effect = raises_key

        # No files
        with pytest.raises(ValueError, match="Missing kick files"):
            do_detuning_analysis(tmp_path)

        # x file exists
        kick_x.touch()
        with pytest.raises(ValueError, match="Missing kick files"):
            do_detuning_analysis(tmp_path)
        kick_x.unlink()

        # y file exists
        kick_y.touch()
        with pytest.raises(ValueError, match="Missing kick files"):
            do_detuning_analysis(tmp_path)
        kick_y.unlink()

        # both exist
        kick_x.touch()
        kick_y.touch()
        with pytest.raises(KeyError, match="End of Test"):
            do_detuning_analysis(tmp_path)

    @pytest.mark.parametrize("with_bbq", (False, "extract", "file"))
    @patch('ir_amplitude_detuning.utilities.measurement_analysis.single_action_analysis')
    @patch('ir_amplitude_detuning.utilities.measurement_analysis.OutlierFilterOpt')
    @patch('ir_amplitude_detuning.utilities.measurement_analysis.get_kick_and_bbq_df')
    @patch('ir_amplitude_detuning.utilities.measurement_analysis.get_kick_plane_from_dir')
    @patch('ir_amplitude_detuning.utilities.measurement_analysis.get_beam_from_dir')
    def test_do_detuning_analysis_success(
        self,
        mock_beam,
        mock_plane,
        mock_kick_and_bbq_df,
        mock_outlier_filter_opt,
        mock_analysis,
        with_bbq,
        tmp_path: Path,
    ):
        """Test successful detuning analysis."""
        # Prepare ---
        # Create files
        (tmp_path / self.kick_x).touch()
        (tmp_path / self.kick_y).touch()
        bbq_file = tmp_path / get_bbq_out_name()
        if with_bbq:  # no matter if "file" or "extract"
            bbq_file.touch()

        # Mock returns
        mock_beam.return_value = 1234
        mock_plane.return_value = "test_plane"

        mock_filter_opt = Mock()
        mock_outlier_filter_opt.return_value = mock_filter_opt

        mock_kick_df = Mock()
        mock_kick_and_bbq_df.return_value = (mock_kick_df, Mock())

        mock_result = Mock()
        mock_analysis.return_value = mock_result


        # Run ---
        result = do_detuning_analysis(tmp_path, extract_bbq=(with_bbq == "extract"))

        # Check ---
        assert result is mock_result
        mock_beam.assert_called_once_with(tmp_path)
        mock_plane.assert_called_once_with(tmp_path)

        if with_bbq != "file":  # even if present, should be re-extraced
            bbq_file = None

        mock_outlier_filter_opt.assert_called_once()
        mock_kick_and_bbq_df.assert_called_once_with(
            kick=tmp_path,
            bbq_in=bbq_file,
            beam=1234,
            filter_opt=mock_filter_opt
        )

        mock_analysis.assert_called_once_with(
            mock_kick_df,
            "test_plane",
            detuning_order=1,  # reasonable default
            corrected=True,
        )


# ============================================================================
# Tests for create_summary
# ============================================================================

class TestCreateSummary:
    """Test cases for the create_summary function."""

    @patch('ir_amplitude_detuning.utilities.measurement_analysis.read_timed_dataframe')
    @patch('ir_amplitude_detuning.utilities.measurement_analysis.do_detuning_analysis')
    def test_create_summary_never_analyze(self, mock_do_analysis, mock_read, tmp_path: Path):
        """Test create_summary with AnalysisOption.never."""
        # Prepare ---
        kick_file = tmp_path / get_kick_out_name()

        def return_tdf(*args, **kwargs):
            count = mock_read.call_count + mock_do_analysis.call_count
            return get_kick_ampdet_header_tdf("x" if count % 2 else "y")

        mock_read.side_effect = return_tdf
        mock_do_analysis.side_effect = return_tdf

        # Run without file existing ---
        with pytest.raises(ValueError, match="Kick file .+ not found"):
            result = create_summary(
                [tmp_path],
                do_analysis=AnalysisOption.never
            )

        # Run with file existing ---
        kick_file.touch()
        result = create_summary(
            [tmp_path],
            do_analysis=AnalysisOption.never
        )

        # Check ---
        assert isinstance(result, tfs.TfsDataFrame)
        assert mock_do_analysis.call_count == 0
        assert mock_read.call_count == 1

    @patch('ir_amplitude_detuning.utilities.measurement_analysis.read_timed_dataframe')
    @patch('ir_amplitude_detuning.utilities.measurement_analysis.do_detuning_analysis')
    def test_create_summary_always_analyze(self, mock_do_analysis, mock_read, tmp_path: Path):
        """Test create_summary with AnalysisOption.never."""
        # Prepare ---
        kick_file = tmp_path / get_kick_out_name()
        kick_file.touch()

        def return_tdf(*args, **kwargs):
            count = mock_read.call_count + mock_do_analysis.call_count
            return get_kick_ampdet_header_tdf("x" if count % 2 else "y")

        mock_read.side_effect = return_tdf
        mock_do_analysis.side_effect = return_tdf

        # Run with file existing ---
        result = create_summary(
            [tmp_path],
            do_analysis=AnalysisOption.always
        )

        # Check ---
        assert isinstance(result, tfs.TfsDataFrame)
        assert mock_do_analysis.call_count == 1
        assert mock_read.call_count == 0

    @patch('ir_amplitude_detuning.utilities.measurement_analysis.read_timed_dataframe')
    @patch('ir_amplitude_detuning.utilities.measurement_analysis.do_detuning_analysis')
    def test_create_summary_auto_analyze(self, mock_do_analysis, mock_read, tmp_path: Path):
        """Test create_summary with AnalysisOption.never."""
        # Prepare ---
        analysis_dirs = (
            tmp_path / "analysis_1",
            tmp_path / "analysis_2",
            tmp_path / "analysis_3",
            tmp_path / "analysis_4",
        )
        put_files = (0, 3)
        with_files = []
        without_files = []
        for idx, ana_dir in enumerate(analysis_dirs):
            ana_dir.mkdir()
            if idx in put_files:
                (ana_dir / get_kick_out_name()).touch()
                with_files.append(ana_dir)
            else:
                without_files.append(ana_dir)

        def return_tdf(*args, **kwargs):
            count = mock_read.call_count + mock_do_analysis.call_count
            return get_kick_ampdet_header_tdf("x" if count % 2 else "y")

        mock_read.side_effect = return_tdf
        mock_do_analysis.side_effect = return_tdf

        # Run with file existing ---
        result = create_summary(
            analysis_dirs,
            do_analysis=AnalysisOption.auto,
            extract_bbq=True,  # to pass on
            detuning_order=2,  # to pass on
        )

        # Check ---
        assert isinstance(result, tfs.TfsDataFrame)

        # Check concat as expected
        assert len(result) == 4
        assert all(term in result.columns for term in list(FirstOrderTerm) + list(SecondOrderTerm))
        assert result.isna().any().any()
        assert (~result.isna()).any().any()

        # Check calls as expected
        assert all(ana_dir.name in result.index for ana_dir in analysis_dirs)
        assert mock_do_analysis.call_count == len(without_files)
        assert mock_read.call_count == len(with_files)

        for analysis_call, ana_dir in zip(mock_do_analysis.call_args_list, without_files):
            assert analysis_call == call(ana_dir, extract_bbq=True, detuning_order=2)

        for analysis_call, ana_dir in zip(mock_read.call_args_list, with_files):
            assert analysis_call == call(ana_dir / get_kick_out_name())


# Helper Utils -----------------------------------------------------------------

def get_kick_ampdet_header_tdf(plane: str):
    """Returns a TfsDataFrame with the ODR headers, from the detuning
    analysis, as expected. Separated by kick-plane as this is how it is
    usually done in omc3 (only kicks in one plane). But for testing you
    can also request all entries to be present via plane="xy". """
    if len(plane) == 1:
        plane = plane.upper()
        return tfs.TfsDataFrame(
            headers={
                f"ODR_dQXd2J{plane}_CORRCOEFF1": 1.0,     # X10 or X01
                f"ODR_dQXd2J{plane}_ERRCORRCOEFF1": 0.1,  # ERR -^
                f"ODR_dQYd2J{plane}_CORRCOEFF1": 2.0,     # Y10 or Y01
                f"ODR_dQYd2J{plane}_ERRCORRCOEFF1": 0.2,  # ERR -^
                f"ODR_dQXd2J{plane}_CORRCOEFF2": 3.0,     # X20 or X02
                f"ODR_dQXd2J{plane}_ERRCORRCOEFF2": 0.3,  # ERR -^
                f"ODR_dQYd2J{plane}_CORRCOEFF2": 4.0,     # Y20 or Y02
                f"ODR_dQYd2J{plane}_ERRCORRCOEFF2": 0.4,  # ERR -^
            }
        )

    df_x = get_kick_ampdet_header_tdf("x")
    df_y = get_kick_ampdet_header_tdf("y")
    df_x.headers.update(df_y.headers)
    return df_x
