"""Observation normalisation for the get-up network (``mp4_recovery.obs_norm``; ``train --actor-hidden
--critic-hidden --obs-norm --obs-norm-freeze``)."""
from pathlib import Path

import torch
from tensordict import TensorDict

from mp4_recovery import obs_norm

REPO = Path(__file__).resolve().parents[2]


def _model(hidden=(32, 16)):
    from rsl_rl.models.mlp_model import MLPModel
    obs = TensorDict({"actor": torch.zeros(2, 45)}, batch_size=[2])
    return MLPModel(obs, {"actor": ["actor"]}, "actor", 12, hidden_dims=hidden, activation="elu",
                    obs_normalization=True,
                    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"})


def test_state_matches_rsl_rl_normaliser_and_stats():
    m = _model()
    x = torch.randn(4096, 45) * torch.linspace(0.1, 3.0, 45) + torch.linspace(-2, 2, 45)
    m.train()
    m.update_normalization(TensorDict({"actor": x}, batch_size=[4096]))      # rsl_rl's first update: the batch's
    mean, std = obs_norm.stats(x)
    sd = obs_norm.state(mean, std, 4096)
    for k, v in sd.items():
        assert torch.allclose(m.state_dict()[k].float(), v.float(), atol=1e-4), k
    assert obs_norm.read(sd)[0].shape == (45,) and obs_norm.read({"mlp.0.weight": 0}) is None
    assert obs_norm.hidden_dims(m.state_dict()) == (32, 16)


def test_freeze_updates_skips_the_first_iterations():
    m = _model()
    m.train()
    sd0 = obs_norm.state(torch.zeros(45), torch.ones(45) * 2.0, 10)
    m.load_state_dict({**m.state_dict(), **sd0})
    st = obs_norm.freeze_updates(m, 2)                                 # 2 iterations x 24 steps
    batch = TensorDict({"actor": torch.randn(64, 45) + 5.0}, batch_size=[64])
    for _ in range(48):
        m.update_normalization(batch)
    assert torch.equal(m.obs_normalizer._mean, sd0["obs_normalizer._mean"]) and st["skipped"] == 48
    m.update_normalization(batch)
    assert float(m.obs_normalizer._mean.mean()) > 1.0 and int(m.obs_normalizer.count) == 10 + 64
    never = _model()
    never.train()
    obs_norm.freeze_updates(never, -1)
    for _ in range(100):
        never.update_normalization(batch)
    assert int(never.obs_normalizer.count) == 0


