import pytest

from ir_amplitude_detuning.lhc_detuning_corrections import check_corrections_ptc
from ir_amplitude_detuning.simulation.lhc_simulation import FakeLHCBeam


def test_ptc_check_corrections_fail_no_files(tmp_path):
    class CallChecker:
        n_calls: int = 0

    def call_me():
        CallChecker.n_calls += 1

    fake_beam = FakeLHCBeam(1, tmp_path)
    fake_beam.install_circuits_into_mctx = call_me
    beams = {1: fake_beam}
    with pytest.raises(FileNotFoundError) as e:
        check_corrections_ptc(
            outputdir=tmp_path,
            lhc_beams=beams,
        )
    assert "No settings files found" in str(e)
    assert CallChecker.n_calls == 1
