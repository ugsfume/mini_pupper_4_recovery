"""The numpy policy reproduces the torch actor (the export the robot runs)."""
import numpy as np

from mp4_recovery.policy_file import NumpyPolicy, export


def test_numpy_equals_torch(tmp_path):
    import torch
    from mp4_recovery.gpu.actor import mlp
    net = mlp(45, 12, (128, 128, 64))
    sd = {f"mlp.{k}": v for k, v in net.state_dict().items()}
    sd["distribution.std_param"] = torch.full((12,), 0.1)
    ck = tmp_path / "m.pt"
    torch.save({"actor_state_dict": sd}, ck)
    path = export(ck, tmp_path / "exp", scale=[0.1] * 12)
    p = NumpyPolicy(path)
    d = np.load(path)
    assert np.abs(p(d["fixture_obs"]) - d["fixture_act"]).max() < 1e-5
    with torch.no_grad():
        x = np.random.default_rng(5).normal(size=(10, 45)).astype(np.float32)
        assert np.abs(p(x) - net(torch.from_numpy(x)).numpy()).max() < 1e-5




def test_band_and_kind_meta_round_trip(tmp_path):
    """The export writes the policy's kind and band (12 floats); the runner's reading gives them back, and the
    default is the walking band, kind walk."""
    import json
    import pytest
    import torch
    from mp4_recovery.contract import BAND_DEG, policy_band_deg, policy_kind
    from mp4_recovery.gpu.actor import mlp
    sd = {f"mlp.{k}": v for k, v in mlp(45, 12, (64, 32)).state_dict().items()}
    ck = tmp_path / "m.pt"
    torch.save({"actor_state_dict": sd}, ck)
    walk = NumpyPolicy(export(ck, tmp_path / "walk", scale=[0.1] * 12))
    assert walk.meta["kind"] == "walk" and walk.meta["band_deg"] == BAND_DEG.tolist()
    getup = NumpyPolicy(export(ck, tmp_path / "getup", scale=[0.5] * 12, band_deg=[30, 90, 90], kind="getup"))
    meta = json.loads(json.dumps(getup.meta))
    assert policy_kind(meta) == "getup" and meta["band_deg"] == [30.0, 90.0, 90.0] * 4
    assert policy_band_deg(meta).tolist() == [30.0, 90.0, 90.0] * 4
    with pytest.raises(ValueError):
        export(ck, tmp_path / "bad", scale=[0.1] * 12, kind="crawl")


def test_getup_phase_meta(tmp_path):
    """A phased get-up exports meta ``getup_phase_s``; an unphased one has none; a walker cannot carry it."""
    import pytest
    import torch
    from mp4_recovery.gpu.actor import mlp
    sd = {f"mlp.{k}": v for k, v in mlp(45, 12, (64, 32)).state_dict().items()}
    ck = tmp_path / "m.pt"
    torch.save({"actor_state_dict": sd}, ck)
    ph = NumpyPolicy(export(ck, tmp_path / "ph", scale=[1.57] * 12, band_deg=[90.0], kind="getup", getup_phase_s=4.0))
    assert ph.meta["getup_phase_s"] == 4.0
    plain = NumpyPolicy(export(ck, tmp_path / "plain", scale=[1.57] * 12, band_deg=[90.0], kind="getup"))
    assert "getup_phase_s" not in plain.meta
    with pytest.raises(ValueError):
        export(ck, tmp_path / "bad", scale=[0.1] * 12, getup_phase_s=4.0)
