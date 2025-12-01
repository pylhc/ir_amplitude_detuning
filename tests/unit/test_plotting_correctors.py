"""Tests for the plotting.correctors module."""
from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.lines import Line2D
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
import tfs
from matplotlib import pyplot as plt

from ir_amplitude_detuning.plotting.correctors import (
    get_corrector_strengths,
    get_labels,
    get_settings_file,
    pair_correctors,
    plot_correctors,
)
from ir_amplitude_detuning.utilities.constants import CIRCUIT, ERR, KNL, SETTINGS_ID

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# Tests for plot_correctors
# ============================================================================

class TestPlotCorrectors:
    """Test cases for the plot_correctors function."""

    def test_plot_correctors_basic(self, tmp_path: Path):
        """Test basic plotting functionality with two ids."""
        # Create settings files --
        ids = ["id1", "id2"]
        for id_ in ids:
            settings_file = tmp_path / f"{SETTINGS_ID}.test.b1.{id_}.tfs"
            df = tfs.TfsDataFrame(
                {
                    CIRCUIT: ["KCTX3.L1", "KCTX3.R1", "KCSX3.L5", "KCSX3.R5"],
                    KNL: [0.1, 0.2, 0.15, 0.25],
                }
            )
            tfs.write(settings_file, df)

        # Run ---
        fig = plot_correctors(
            tmp_path, beam=1, ids=ids, field="b4", corrector_pattern=".*"
        )

        # Checks ----
        # Check plot layput correct
        scale = 1e-3  # default

        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "$K_4L$ Left [10$^3$ m$^{-3}$]"
        assert ax.get_ylabel() == "$K_4L$ Right [10$^3$ m$^{-3}$]"

        # Check data correct
        data_ips = {"1": [0.1, 0.2], "5": [0.15, 0.25]}

        for ip, data_ip in data_ips.items():
            lines_ip = [line for line in ax.lines if f"${ip}$" in line.get_marker()]
            assert len(lines_ip) == len(ids)
            for idx, line in enumerate(lines_ip):
                assert ids[idx] in line.get_label()
                data_plot = line.get_data()
                assert data_plot[0] == pytest.approx(data_ip[0]*scale)
                assert data_plot[1] == pytest.approx(data_ip[1]*scale)
        plt.close(fig)

    def test_plot_correctors_multiple_ids_with_dict(self, tmp_path: Path):
        """Test plotting with multiple ids using dictionary labels."""
        ids = {"id1": "Target A", "id2": "Target B"}
        for id_ in ids:
            settings_file = tmp_path / f"{SETTINGS_ID}.test.b2.{id_}.tfs"
            df = tfs.TfsDataFrame(
                {
                    CIRCUIT: ["KCTX3.L1", "KCTX3.R1"],
                    KNL: [0.1, 0.2],
                }
            )
            tfs.write(settings_file, df)

        fig = plot_correctors(
            tmp_path,
            beam=2,
            ids=ids,
            field="a6",
            corrector_pattern=".*",
            rescale=0,  # no rescale this time
        )
        # Checks ----
        # Check plot layput correct
        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "$K_6SL$ Left [m$^{-5}$]"
        assert ax.get_ylabel() == "$K_6SL$ Right [m$^{-5}$]"

        # Check data correct
        data_ips = {"1": [0.1, 0.2],}

        for ip, data_ip in data_ips.items():
            lines_ip = [line for line in ax.lines if f"${ip}$" in line.get_marker()]
            assert len(lines_ip) == len(ids)
            for id_, line in zip(ids.values(), lines_ip):
                assert id_ in line.get_label()
                data_plot = line.get_data()
                assert data_plot[0] == pytest.approx(data_ip[0])
                assert data_plot[1] == pytest.approx(data_ip[1])

        assert fig is not None
        ax = fig.axes[0]
        # Check that the plot has data points
        assert len(ax.lines) > 0
        plt.close(fig)

    @pytest.mark.parametrize("scale", (0, 2, 3))
    def test_plot_correctors_with_errors_and_scales(self, tmp_path: Path, scale):
        """Test plotting with error bars."""
        settings_file = tmp_path / f"{SETTINGS_ID}.test.b1.id1.tfs"
        df = tfs.TfsDataFrame(
            {
                CIRCUIT: ["KCTX3.L1", "KCTX3.R1", "KCSX3.L5", "KCSX3.R5"],
                KNL: [0.1, 0.2, 0.15, 0.25],
                f"{ERR}{KNL}": [0.01, 0.02, 0.015, 0.025],
            }
        )
        tfs.write(settings_file, df)
        df = df.set_index(CIRCUIT)

        # Run ---
        fig = plot_correctors(
            tmp_path, beam=1, ids=["id1"], field="b4", corrector_pattern=".*", rescale=scale,
        )

        # Check ---
        factor = 10**-scale

        assert fig is not None
        ax = fig.axes[0]

        # Check the values for the lines (again)
        for ip, left, right in ((1, "KCTX3.L1", "KCTX3.R1"), (5, "KCSX3.L5", "KCSX3.R5")):
            lines_ip = [line for line in ax.lines if f"${ip}$" in line.get_marker()]
            assert len(lines_ip) == 1  # only one id
            line = lines_ip[0]
            assert "id1" in line.get_label()
            data_plot = line.get_data()
            assert data_plot[0] == pytest.approx(factor * df.loc[left, KNL])
            assert data_plot[1] == pytest.approx(factor * df.loc[right, KNL])

        # Check that error rectangles are added as patches in the correct places with the right size
        assert len(ax.patches) == 2  # number of IPs
        for patch, (left, right) in zip(ax.patches, (("KCTX3.L1", "KCTX3.R1"), ("KCSX3.L5", "KCSX3.R5"))):
            x, y = patch.get_xy()
            assert x == pytest.approx(factor * (df.loc[left, KNL] - df.loc[left, f"{ERR}{KNL}"]))
            assert y == pytest.approx(factor * (df.loc[right, KNL] - df.loc[right, f"{ERR}{KNL}"]))
            assert patch.get_width() == pytest.approx(factor * 2 * df.loc[left, f"{ERR}{KNL}"])
            assert patch.get_height() == pytest.approx(factor * 2 * df.loc[right, f"{ERR}{KNL}"])

        plt.close(fig)


    def test_plot_correctors_pattern_filtering(self, tmp_path: Path):
        """Test that corrector pattern filters correctly in the plot."""
        settings_file = tmp_path / f"{SETTINGS_ID}.test.b1.id1.tfs"
        df = tfs.TfsDataFrame(
            {
                CIRCUIT: ["KCTX3.L1", "KCTX3.R1", "KCSX3.L5", "KCSX3.R5"],
                KNL: [0.1, 0.2, 0.15, 0.25],
            }
        )
        df.index.name = CIRCUIT
        tfs.write(settings_file, df)

        fig = plot_correctors(
            tmp_path, beam=1, ids=["id1"], field="b4", corrector_pattern="KCTX.*"
        )

        assert fig is not None
        ax = fig.axes[0]
        # Should only plot data for KCTX correctors (IP1)
        lines_ip1 = [line for line in ax.lines if "$1$" in line.get_marker()]
        assert len(lines_ip1) == 1  # only one id

        lines_ip5 = [line for line in ax.lines if "$5$" in line.get_marker()]
        assert not len(lines_ip5)  # filtered
        plt.close(fig)


# ============================================================================
# Tests for get_settings_file
# ============================================================================

class TestGetSettingsFile:
    """Test cases for the get_settings_file function."""

    def test_get_settings_file_finds_file(self, tmp_path: Path):
        """Test that get_settings_file finds the correct file."""
        # Create a matching file
        settings_file = tmp_path / f"{SETTINGS_ID}.b4.b1.test_id.tfs"
        settings_file.touch()

        result = get_settings_file(tmp_path, beam=1, id_="test_id")
        assert result == settings_file

    def test_get_settings_file_finds_file_beam2(self, tmp_path: Path):
        """Test that get_settings_file finds files for beam 2."""
        settings_file = tmp_path / f"{SETTINGS_ID}.myfield.b2.my_target.tfs"
        settings_file.touch()

        result = get_settings_file(tmp_path, beam=2, id_="my_target")
        assert result == settings_file

    def test_get_settings_file_file_not_found(self, tmp_path: Path):
        """Test that FileNotFoundError is raised when file is not found."""
        with pytest.raises(FileNotFoundError, match="No file matching"):
            get_settings_file(tmp_path, beam=1, id_="nonexistent")

    def test_get_settings_file_multiple_files_returns_first(self, tmp_path: Path):
        """Test that the first matching file is returned when multiple exist."""
        # Create multiple matching files
        file1 = tmp_path / f"{SETTINGS_ID}.a.b1.test_id.tfs"
        file2 = tmp_path / f"{SETTINGS_ID}.b.b1.test_id.tfs"
        file1.touch()
        file2.touch()

        result = get_settings_file(tmp_path, beam=1, id_="test_id")
        # Should return one of them (glob order may vary)
        assert result in (file1, file2)


# ============================================================================
# Tests for get_corrector_strengths
# ============================================================================

class TestGetCorrectorStrengths:
    """Test cases for the get_corrector_strengths function."""

    def test_get_corrector_strengths_without_errors(self, tmp_path: Path):
        """Test extraction of corrector strengths without error columns."""
        # Create a settings TFS file
        settings_file = tmp_path / f"{SETTINGS_ID}.test.b1.id1.tfs"
        df = tfs.TfsDataFrame(
            {
                CIRCUIT: ["KCTX3.L1", "KCTX3.R1", "KCSX3.L1", "KCSX3.R1"],
                KNL: [0.1, 0.2, 0.15, 0.25],
            }
        )
        df.index.name = CIRCUIT
        tfs.write(settings_file, df)

        result_knl, result_err = get_corrector_strengths(
            tmp_path, beam=1, id_="id1", corrector_pattern=".*"
        )

        assert isinstance(result_knl, pd.Series)
        assert result_err is None
        assert len(result_knl) == 4
        assert_series_equal(df.set_index(CIRCUIT, drop=True)[KNL], result_knl)

    def test_get_corrector_strengths_with_errors(self, tmp_path: Path):
        """Test extraction of corrector strengths with error columns."""
        settings_file = tmp_path / f"{SETTINGS_ID}.test.b2.id2.tfs"
        df = tfs.TfsDataFrame(
            {
                CIRCUIT: ["KCTX3.L1", "KCTX3.R1"],
                KNL: [0.1, 0.2],
                f"{ERR}{KNL}": [0.01, 0.02],
            }
        )
        df.index.name = CIRCUIT
        tfs.write(settings_file, df)

        result_knl, result_err = get_corrector_strengths(
            tmp_path, beam=2, id_="id2", corrector_pattern=".*"
        )

        assert isinstance(result_knl, pd.Series)
        assert isinstance(result_err, pd.Series)
        df_indexed = df.set_index(CIRCUIT, drop=True)
        assert_series_equal(df_indexed[KNL], result_knl)
        assert_series_equal(df_indexed[f"{ERR}{KNL}"], result_err)

    def test_get_corrector_strengths_pattern_filtering(self, tmp_path: Path):
        """Test that corrector pattern filters correctly."""
        settings_file = tmp_path / f"{SETTINGS_ID}.test.b1.id1.tfs"
        df = tfs.TfsDataFrame(
            {
                CIRCUIT: ["KCTX3.L1", "KCTX3.R1", "KCSX3.L1", "KCSX3.R1"],
                KNL: [0.1, 0.2, 0.15, 0.25],
            }
        )
        df.index.name = CIRCUIT
        tfs.write(settings_file, df)

        result_knl, _ = get_corrector_strengths(
            tmp_path, beam=1, id_="id1", corrector_pattern="KCTX.*"
        )

        assert len(result_knl) == 2
        assert "KCTX3.L1" in result_knl.index
        assert "KCSX3.L1" not in result_knl.index

    def test_get_corrector_strengths_case_insensitive_pattern(self, tmp_path: Path):
        """Test that pattern matching is case-insensitive."""
        settings_file = tmp_path / f"{SETTINGS_ID}.test.b1.id1.tfs"
        df = tfs.TfsDataFrame(
            {
                CIRCUIT: ["KCTX3.L1", "KCTX3.R1"],
                KNL: [0.1, 0.2],
            }
        )
        df.index.name = CIRCUIT
        tfs.write(settings_file, df)

        result_knl, _ = get_corrector_strengths(
            tmp_path, beam=1, id_="id1", corrector_pattern="kctx.*"
        )

        assert len(result_knl) == 2

    def test_get_corrector_strengths_no_matches(self, tmp_path: Path):
        """Test that AttributeError is raised when no correctors match."""
        settings_file = tmp_path / f"{SETTINGS_ID}.test.b1.id1.tfs"
        df = tfs.TfsDataFrame(
            {
                CIRCUIT: ["KCTX3.L1"],
                KNL: [0.1],
            }
        )
        df.index.name = CIRCUIT
        tfs.write(settings_file, df)

        with pytest.raises(AttributeError, match="No matching corrector"):
            get_corrector_strengths(
                tmp_path, beam=1, id_="id1", corrector_pattern="NOMATCH.*"
            )


# ============================================================================
# Tests for pair_correctors
# ============================================================================

class TestPairCorrectors:
    """Test cases for the pair_correctors function."""

    def test_pair_correctors_basic(self):
        """Test basic corrector pairing."""
        correctors = ["KCTX3L1", "KCTX3R1", "KCSX3L5", "KCSX3R5"]
        result = pair_correctors(correctors)

        assert "1" in result
        assert "5" in result
        # Last character is the IP, second-to-last is L/R indicator
        assert result["1"]["l"] == "KCTX3L1"
        assert result["1"]["r"] == "KCTX3R1"
        assert result["5"]["l"] == "KCSX3L5"
        assert result["5"]["r"] == "KCSX3R5"


# ============================================================================
# Tests for get_labels
# ============================================================================

class TestGetLabels:
    """Test cases for the get_labels function."""

    def test_get_labels_b4_no_rescale(self):
        """Test label generation for B4 field without rescaling."""
        xlabel, ylabel = get_labels("b4", rescale=0)

        assert "$K_4L$" in xlabel
        assert "$K_4L$" in ylabel
        assert "Left" in xlabel
        assert "Right" in ylabel
        assert "[m$^{-3}$]" in xlabel
        assert "[m$^{-3}$]" in ylabel

    def test_get_labels_a6_with_rescale(self):
        """Test label generation for B6 field with rescaling."""
        xlabel, ylabel = get_labels("a6", rescale=3)

        assert "$K_6SL$" in xlabel
        assert "$K_6SL$" in ylabel
        assert "10$^3$" in xlabel
        assert "10$^3$" in ylabel
        assert "[10$^3$ m$^{-5}$]" in xlabel
        assert "[10$^3$ m$^{-5}$]" in ylabel

    def test_get_labels_different_rescales(self):
        """Test label generation with different rescale values."""
        _, ylabel1 = get_labels("b4", rescale=0)
        _, ylabel2 = get_labels("b4", rescale=2)
        _, ylabel3 = get_labels("b4", rescale=-1)

        assert "10$^0$" not in ylabel1
        assert "10$^2$" in ylabel2
        assert "10$^-1$" in ylabel3
