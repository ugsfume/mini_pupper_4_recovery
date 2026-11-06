"""The contract's torch implementation in the environment against the numpy one on the robot."""
import numpy as np
import torch

from mp4_recovery.contract import Contract, VelocityFilter
from mp4_recovery.gpu.mdp import TorchContract, velocity_from_hist

C = Contract.load()


def test_torch_matches_numpy_quantisation():
    tc = TorchContract("cpu")
    rng = np.random.default_rng(2)
    rel = rng.uniform(-0.6, 0.6, (500, 12))
    dd_np, q_np = zip(*(C.quantise_rel(r) for r in rel))
    dd_t, q_t = tc.quantise_rel(torch.tensor(rel, dtype=torch.float32))
    assert np.array_equal(np.array(dd_np), dd_t.numpy().astype(np.int64))
    assert np.abs(np.array(q_np) - q_t.numpy()).max() < 1e-5


def test_tick_centre_requantises_to_itself():
    """Every tick, for every servo's offset: the target at the tick's centre goes back to exactly that tick, in numpy
    and in torch (the dd's AT32 angle, converted back, would move ticks below 511 by one)."""
    ticks = np.tile(np.arange(1024)[:, None], (1, 12))
    assert np.array_equal(C.ticks_from_servo_deg(C.servo_deg_of_tick(ticks)), ticks)
    tc = TorchContract("cpu")
    dd_t, _ = tc.quantise_rel(torch.tensor(C.rel_from_ticks(ticks), dtype=torch.float32))
    assert np.array_equal(dd_t.numpy().astype(np.int64), C.dd_from_ticks(ticks))


def test_sim_velocity_code_matches_numpy_filter():
    vf = VelocityFilter()
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(0, 0.01, (30, 12)), axis=0)
    hist = None
    for k in range(30):
        vf.push(x[k])
        row = torch.tensor(x[k], dtype=torch.float32)[None, None]
        hist = row.expand(1, 6, 12).clone() if hist is None else torch.cat([hist[:, 1:], row], 1)
        assert np.abs(vf.velocity() - velocity_from_hist(hist)[0].numpy()).max() < 1e-4
