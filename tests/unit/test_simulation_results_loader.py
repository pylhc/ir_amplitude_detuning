from __future__ import annotations

from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import tfs
from pandas.testing import assert_frame_equal

# Import the module under test
from ir_amplitude_detuning.simulation import results_loader as rl


class TestLoadSimulationOutputTfs:
    """Tests for :func:`load_simulation_output_tfs`."""

    def test_load_finds_file_and_returns_tfs_dataframe(self, tmp_path):
        # Arrange
        dummy_df = pd.DataFrame({"A": [1, 2]})
        tfs.write(tmp_path / "ampdet.lhc.b1.myid.tfs", dummy_df)

        # Act
        result = rl.load_simulation_output_tfs(
            folder=tmp_path,
            type_="ampdet",
            beam=1,
            id_="myid"
        )

        # Assert
        pd.testing.assert_frame_equal(result, dummy_df, check_like=True, check_frame_type=False)

    def test_load_finds_file_beam4_and_returns_tfs_dataframe(self, tmp_path):
        # Arrange
        dummy_df = pd.DataFrame({"A": [1, 2]})
        tfs.write(tmp_path / "ampdet.lhc.b2.myid.tfs", dummy_df)

        for beam in (2, 4):
            # Act
            result = rl.load_simulation_output_tfs(
                folder=tmp_path,
                type_="ampdet",
                beam=2,
                id_="myid"
            )

            # Assert
            pd.testing.assert_frame_equal(result, dummy_df, check_like=True, check_frame_type=False)

    def test_load_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            rl.load_simulation_output_tfs(
                folder=Path("/empty"),
                type_="ampdet",
                beam=1,
                id_="missing"
            )


class TestGetDetuningFromPtc:
    """Tests for :func:`get_detuning_from_ptc_output` and :func:`load_ptc_detuning`."""

    ptc_data_frame = pd.DataFrame({
            "NAME": ["ANHX", "ANHY"],
            "ORDER1": [1, 0],
            "ORDER2": [0, 1],
            "ORDER3": [0, 0],
            "ORDER4": [0, 0],
            "VALUE": [3.14, 2.71],
        })
    terms = ["X10", "Y01"]

    def assert_result(self, result):
        assert result[self.terms[0]] == 3.14
        assert result[self.terms[1]] == 2.71

    def test_get_detuning_from_ptc_output(self):
        # Create a DataFrame that contains the rows we will query

        # Use a custom term list to keep the test simple

        result = rl.get_detuning_from_ptc_output(self.ptc_data_frame, terms=self.terms)
        self.assert_result(result)


    def test_load_ptc_detuning(self, tmp_path, monkeypatch):
        tfs.write(tmp_path / f"{rl.AMPDET_ID}.lhc.b1.myid.tfs", self.ptc_data_frame)

        monkeypatch.setattr(
            rl, "get_detuning_from_ptc_output", partial(rl.get_detuning_from_ptc_output, terms=self.terms)
        )

        result = rl.load_ptc_detuning(tmp_path, beam=1, id_="myid")
        self.assert_result(result)


class TestConvertDataframeToDict:
    """Tests for :func:`convert_dataframe_to_dict`."""

    def test_without_error_columns(self):
        # Simple DF – no column starts with the error prefix
        df = pd.DataFrame(
            {
                "X01": [0.1, 0.2],
                "X10": [0.3, 0.4]
            },
            index=["row1", "row2"]
        )
        result = rl.convert_dataframe_to_dict(df)

        # Expect a dict with two entries, each a Detuning containing the row values
        assert isinstance(result, dict)
        assert set(result.keys()) == {"row1", "row2"}
        for key, val in result.items():
            assert isinstance(val, rl.Detuning)
            assert set(df.columns) == set(val.terms())
            for col in df.columns:
                assert df.loc[key, col] == val[col]

    def test_with_error_columns(self):
        # Columns that start with the error prefix (ERR) are present
        df = pd.DataFrame(
            {
                "X01": [0.1, 0.2],
                "X10": [0.3, 0.4],
                f"{rl.ERR}X01": [0.5, 0.9],
                f"{rl.ERR}X10": [0.2, 0.7],
            },
            index=["row1", "row2"]
        )
        result = rl.convert_dataframe_to_dict(df)

        # Result should be a dict of DetuningMeasurement objects
        assert isinstance(result, dict)
        assert set(result.keys()) == {"row1", "row2"}
        for key, val in result.items():
            assert isinstance(val, rl.DetuningMeasurement)
            assert len(list(val.terms())) == 2
            for term in val.terms():
                assert df.loc[key, term] == val[term].value
                assert df.loc[key, f"{rl.ERR}{term}"] == val[term].error


class TestGetCalculatedDetuningForIp:
    """Tests for :func:`get_calculated_detuning_for_ip`."""

    def setup_class(self):
        monkeypatch = pytest.MonkeyPatch()

        self.data_frame = pd.DataFrame(
            {
                rl.IP: ["1", "1", "2", "2"],
                rl.FIELDS: ["b4", "b6", "b4", "b6"],
                "VAL": [0.11, 0.22, 0.33, 0.44],
                "ERRVAL": [0.01, 0.02, 0.03, 0.04],
            }
        )
        self.df_expected = self.data_frame.iloc[2:, 1:].set_index(rl.FIELDS, drop=True)

        mock_load = Mock()
        mock_load.return_value = self.data_frame
        monkeypatch.setattr(rl, "load_simulation_output_tfs", mock_load)
        self.mock_load = mock_load

        mock_convert = Mock()
        mock_convert.return_value = "returned"
        monkeypatch.setattr(rl, "convert_dataframe_to_dict", mock_convert)
        self.mock_convert = mock_convert

    def test_ip_filtering_and_conversion(self):

        result = rl.get_calculated_detuning_for_ip(
            folder=Path("/some"),
            beam=1,
            id_="test",
            ip="2",          # we request the second IP
            errors=True      # ask for error version – triggers use of ERR columns
        )
        assert self.mock_convert.call_count == 1
        assert result == self.mock_convert.return_value

        # Verify that the mock was called with the correct parameters
        self.mock_load.assert_called_once_with(folder=Path("/some"), type_=rl.AMPDET_CALC_ERR_ID, beam=1, id_="test")

        # Ensure that the DataFrame was filtered correctly (only IP2 row kept)
        df_passed = self.mock_convert.call_args.args[0]
        assert df_passed.shape[0] == 2
        assert_frame_equal(df_passed, self.df_expected, check_like=True)

    def test_no_errors(self):
        rl.get_calculated_detuning_for_ip(
            folder=Path("/some"),
            beam=2,
            id_="test",
            ip="2",          # we request the second IP
            errors=False      # No errors
        )
        self.mock_load.assert_called_with(folder=Path("/some"), type_=rl.AMPDET_CALC_ID, beam=2, id_="test")

    def test_raises_error(self):
        with pytest.raises(ValueError) as e:
            rl.get_calculated_detuning_for_ip(
                folder=Path("/some"),
                beam=1,
                id_="test",
                ip="9",
                errors=False,
            )
        assert "No data for IP 9" in str(e)


class TestGetCalculatedDetuningForField:
    """Tests for :func:`get_calculated_detuning_for_field`."""

    def setup_class(self):
        monkeypatch = pytest.MonkeyPatch()

        self.data_frame = pd.DataFrame(
            {
                rl.FIELDS: ["b4", "b6", "b4", "b4b6"],
                rl.IP: ["1", "1", "2", "2"],
                "VAL": [0.11, 0.22, 0.33, 0.44],
                "ERRVAL": [0.01, 0.02, 0.03, 0.04],
            }
        )

        self.df_expected_b4 = self.data_frame.iloc[[0,2], 1:].set_index(rl.IP, drop=True)
        self.df_expected_b4b6 = self.data_frame.iloc[-1:, 1:].set_index(rl.IP, drop=True)

        mock_load = Mock()
        mock_load.return_value = self.data_frame
        monkeypatch.setattr(rl, "load_simulation_output_tfs", mock_load)
        self.mock_load = mock_load

        mock_convert = Mock()
        mock_convert.return_value = "returned"
        monkeypatch.setattr(rl, "convert_dataframe_to_dict", mock_convert)
        self.mock_convert = mock_convert

    def test_dual_field_filtering(self):
        self.mock_convert.reset_mock()

        result = rl.get_calculated_detuning_for_field(
            folder=Path("/data"),
            beam=2,
            id_="foo",
            field=["b4", "b6"],   # will become 'b4b6'
            errors=False
        )
        self.mock_load.assert_called_with(folder=Path("/data"), type_=rl.AMPDET_CALC_ID, beam=2, id_="foo")
        assert self.mock_convert.call_count == 1
        assert result == self.mock_convert.return_value
        df_passed = self.mock_convert.call_args.args[0]
        assert len(df_passed) == 1
        assert_frame_equal(df_passed, self.df_expected_b4b6, check_like=True)

    def test_single_field_filtering(self):
        self.mock_convert.reset_mock()

        result = rl.get_calculated_detuning_for_field(
            folder=Path("/data"),
            beam=2,
            id_="foo",
            field="b4",
            errors=True
        )
        self.mock_load.assert_called_with(folder=Path("/data"), type_=rl.AMPDET_CALC_ERR_ID, beam=2, id_="foo")

        assert self.mock_convert.call_count == 1
        assert result == self.mock_convert.return_value
        df_passed = self.mock_convert.call_args.args[0]
        assert len(df_passed) == 2
        assert_frame_equal(df_passed, self.df_expected_b4, check_like=True)


class TestGetDetuningChangePtc:
    """Smoke‑test for :func:`get_detuning_change_ptc` with heavy mocking."""

    def test_calls_load_ptc_and_performs_subtraction(self, monkeypatch):
        values = {
            "a": 34,
            "b": 23,
            rl.NOMINAL_ID: 3,
        }
        ids = list(values.keys())[:2]
        beams = (1, 2)
        folder_ = Path("/data")

        def mock_loader(folder, beam, id_):
            assert folder == folder_
            assert beam in beams
            return values[id_]

        mock_load = Mock(side_effect=mock_loader)
        monkeypatch.setattr(rl, "load_ptc_detuning", mock_load)

        # Replace BeamDict with a trivial dict subclass that implements __sub__
        class SimpleBeamDict:
            def __init__(self, arg: dict):
                self.value = list(arg.values())[0]

            def __sub__(self, other: SimpleBeamDict):
                return self.value - other.value

        monkeypatch.setattr(rl, "BeamDict", SimpleBeamDict)

        # Test the function ---
        detuning = rl.get_detuning_change_ptc(
            folder=Path("/data"),
            beams=beams,
            ids=ids,
        )

        # Asserts ---
        assert detuning == {k: values[k] - values[rl.NOMINAL_ID] for k in "ab"}
        assert mock_load.call_count == len(values) * len(beams)
