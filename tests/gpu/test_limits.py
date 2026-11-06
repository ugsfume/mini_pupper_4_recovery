"""The measured joint limits (limits.py) and the stance offset: the packaged file, the model's joint
ranges, the decoder's clamps (numpy = torch), the offset (torch = numpy), the export meta,
the evaluation plant; then on the GPU: the ranges in the compiled model, sim and runner targets for the same actions,
and the get-up's starts inside the bounds."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mp4_recovery import limits as L
from mp4_recovery.robot import sku2
from mp4_recovery.contract import (ABD_OUT_SIGN, ABDUCTION, ActionDecoder, Bounds, Contract, policy_bounds,
                                      stand_offset_rel)
from mp4_recovery.gpu.mdp import TorchBounds, TorchContract

REPO = Path(__file__).resolve().parents[2]
C = Contract.load()
TICK = 3100 / 10240                                     # one firmware tick, deg


# ------------------------------------------------------------------------------------------------------------- CPU


def test_hand_ranges_reproduce_the_files_model_ranges():
    """The mapping physical -> model (stand + radians(sign x value)) gives the file's own model_joint_range_rad."""
    hand = L.joint_ranges(hand_only=True)
    for leg in L.load()["legs"]:
        for name, r in leg["model_joint_range_rad"].items():
            if not name.endswith("_note"):
                assert np.allclose(hand[name], r, atol=6e-5), (name, hand[name], r)
    assert L.abd_out_sign() == ABD_OUT_SIGN


def test_ranges_take_the_narrower_of_hand_and_powered_per_side():
    phys = L.physical_ranges()
    assert phys["FL"]["abd_out"] == (-13.8, 25.9)          # inward: powered 13.8 < hand 16.7; outward: hand
    assert phys["FR"]["abd_out"] == (-21.6, 23.3)          # powered went further both ways: hand
    assert phys["RR"]["abd_out"] == (-13.8, 23.6)
    assert phys["RL"]["abd_out"] == (-16.9, 28.2)
    jr = L.joint_ranges()
    assert np.allclose(np.degrees(jr["FL_hip"]), (-25.9, 13.8))       # FL: outward is - in the model
    assert np.allclose(np.degrees(jr["FR_hip"]), (-21.6, 23.3))
    for name, (lo, hi) in jr.items():
        hand = L.joint_ranges(hand_only=True)[name]
        assert hand[0] <= lo + 1e-12 and hi <= hand[1] + 1e-12


def test_apply_limits_sets_the_ranges_on_the_spec():
    spec = sku2.get_spec()
    got = L.apply_limits(spec, margin_deg=1.0)
    m = spec.compile()
    for name, (lo, hi) in L.joint_ranges().items():
        j = m.joint(name)
        assert m.jnt_limited[j.id] == 1
        assert np.allclose(m.jnt_range[j.id], (lo + math.radians(1), hi - math.radians(1)))
        assert np.allclose(got[name], m.jnt_range[j.id])
    m0 = sku2.get_spec().compile()                          # default: no ranges
    assert not any(m0.jnt_limited[m0.joint(n).id] for n in L.joint_ranges())
    m1 = sku2.get_spec(limits=True).compile()
    assert all(m1.jnt_limited[m1.joint(n).id] for n in L.joint_ranges())
    stand = dict(zip(C.joint_order, C.stand_joint_rad))
    for name, (lo, hi) in L.joint_ranges().items():         # the stand is inside every range
        assert lo < stand[name] < hi


def _deg_rel(b, i):
    return math.degrees(b.lo[i] - C.stand_rel_rad[i]), math.degrees(b.hi[i] - C.stand_rel_rad[i])


def test_decoder_bounds_abduction_and_tracks():
    w, g = L.decoder_bounds("walk"), L.decoder_bounds("getup")
    phys = L.physical_ranges()
    for li, leg in enumerate(L.LEGS):
        i = ABDUCTION[li]
        s = ABD_OUT_SIGN[li]
        a_in, a_out = phys[leg]["abd_out"]
        lo, hi = _deg_rel(w, i)
        out_w, in_w = (hi, lo) if s > 0 else (-lo, -hi)
        assert abs(out_w - min(20.0, a_out - 3.0)) < 1e-9 and abs(in_w - (a_in + 3.0)) < 1e-9, (leg, lo, hi)
        lo, hi = _deg_rel(g, i)
        out_g, in_g = (hi, lo) if s > 0 else (-lo, -hi)
        assert abs(out_g - (a_out - 2.0)) < 1e-9 and abs(in_g - (a_in + 2.0)) < 1e-9, leg
    # hips and knees: 20 deg inside the pot's track (dd 0 / 3100) at the contract's stand; the notes' table
    room = {"FR": (163, 147), "FL": (119, 191), "RR": (180, 130), "RL": (162, 148)}       # hip: dd up / dd down
    for li, leg in enumerate(L.LEGS):
        for j in (1, 2):
            i = 3 * li + j
            for b in (w, g):
                dd = C.quantise_rel(np.where(np.arange(12) == i, b.lo - C.stand_rel_rad, 0.0))[0][i], \
                    C.quantise_rel(np.where(np.arange(12) == i, b.hi - C.stand_rel_rad, 0.0))[0][i]
                assert min(dd) >= 200 - 2 and max(dd) <= 2900 + 2, (leg, j, dd)
        up, dn = room[leg]
        lo, hi = _deg_rel(w, 3 * li + 1)
        assert sorted([round(abs(lo)), round(abs(hi))]) == sorted([up - 20, dn - 20]), (leg, lo, hi)


def _random_rel(n, seed=0, deg=150.0):
    return np.radians(np.random.default_rng(seed).uniform(-deg, deg, (n, 12)))


def test_clamp_keeps_targets_inside_ranges_tracks_and_the_knee_thigh_range():
    for mode in L.MODES:
        b = L.decoder_bounds(mode)
        rel = _random_rel(4000, 1)
        out = b.clamp_rel(rel, C.stand_rel_rad)
        model = C.stand_rel_rad + out
        assert np.all(model >= b.lo - 1e-12) and np.all(model <= b.hi + 1e-12)
        v = model.reshape(-1, 4, 3)
        calf = (v[:, :, 2] - b.kc[:, 0] * v[:, :, 1]) / b.kc[:, 1]          # the implied knee - thigh angle
        lo_k, hi_k = b.lo.reshape(4, 3)[:, 2], b.hi.reshape(4, 3)[:, 2]
        for li in range(4):
            ok = (calf[:, li] >= b.calf_lo[li] - 1e-9) & (calf[:, li] <= b.calf_hi[li] + 1e-9)
            # where the coupling's interval lies inside the knee's track the calf is in its range
            t = v[:, li, 1]
            a, c = b.kc[li, 0] * t + b.kc[li, 1] * b.calf_lo[li], b.kc[li, 0] * t + b.kc[li, 1] * b.calf_hi[li]
            feasible = (np.maximum(a, c) <= hi_k[li]) & (np.minimum(a, c) >= lo_k[li])
            assert np.all(ok[feasible]), (mode, li)
            assert feasible.mean() > 0.2, (mode, li, feasible.mean())
        # a target already inside the bounds is unchanged, bit for bit
        again = b.clamp_rel(out, C.stand_rel_rad)
        assert np.array_equal(again, out)
        assert np.array_equal(b.clamp_rel(np.zeros(12), C.stand_rel_rad), np.zeros(12))      # the stand is inside


def test_knee_follows_the_hip():
    """The knee servo is the action: with the hip turned, the same knee - thigh angle needs the knee servo turned
    with it; a knee left at the stand is clamped back into the coupling."""
    b = L.decoder_bounds("walk")
    for li in range(4):
        rel = np.zeros(12)
        rel[3 * li + 1] = math.radians(100.0)               # hip 100 deg, knee servo at the stand: knee - thigh -100
        out = b.clamp_rel(rel, C.stand_rel_rad)
        m = (C.stand_rel_rad + out).reshape(4, 3)[li]
        calf = (m[2] - b.kc[li, 0] * m[1]) / b.kc[li, 1]
        assert b.calf_lo[li] - 1e-9 <= calf <= b.calf_hi[li] + 1e-9
        assert min(abs(calf - b.calf_lo[li]), abs(calf - b.calf_hi[li])) < 1e-9     # moved to the knee's stop
        assert abs(out[3 * li + 2]) > math.radians(10.0)    # the knee servo moved with the hip
        assert out[3 * li + 1] == rel[3 * li + 1]           # the hip is inside its track: not moved


def test_torch_bounds_equal_numpy():
    for mode in L.MODES:
        b = L.decoder_bounds(mode)
        env = {leg: {"abd": [-0.3, 0.0, 0.3], "lo": [-1.0, -2.0, -0.5], "hi": [1.5, 2.0, 1.0]} for leg in ("FR", "RL")}
        for hip_env in (None, env):
            bb = Bounds.from_meta({**b.to_meta(), "hip_envelope": hip_env})
            rel = _random_rel(3000, 2)
            want = bb.clamp_rel(rel, C.stand_rel_rad)
            tc = TorchContract("cpu")
            tb = TorchBounds(bb, tc.stand_phys)
            got = tb.clamp_rel(torch.tensor(rel, dtype=torch.float32)).numpy()
            assert np.abs(got - want).max() < 2e-5, (mode, hip_env is None)


def test_bounds_meta_round_trip_and_checks():
    b = L.decoder_bounds("getup")
    m = json.loads(json.dumps(b.to_meta()))
    b2 = policy_bounds({"bounds": m})
    assert np.array_equal(b2.lo, b.lo) and np.array_equal(b2.hi, b.hi) and b2.mode == "getup"
    assert policy_bounds({}) is None and policy_bounds({"bounds": None}) is None
    for bad in ({**m, "frame": "rel"}, {**m, "lo": m["hi"]}, {**m, "calf_range": m["calf_range"][:3]}):
        with pytest.raises((ValueError, IndexError)):
            Bounds.from_meta(bad)


def test_decoder_without_bounds_or_offset_is_unchanged():
    rng = np.random.default_rng(7)
    a, b = ActionDecoder(C, 0.25), ActionDecoder(C, 0.25, bounds=None, stand_offset_abd_deg=0.0)
    for k in range(50):
        raw = rng.normal(0, 1, 12)
        x, y = a(raw), b(raw)
        assert np.array_equal(x["dd"], y["dd"]) and np.array_equal(x["rel"], y["rel"])


def test_decoder_bounds_clamp_and_keep_prev_consistent():
    b = L.decoder_bounds("walk")
    d = ActionDecoder(C, [0.6] * 12, bounds=b)
    for _ in range(10):
        out = d(np.ones(12))
    model = C.stand_rel_rad + out["rel"]
    assert np.all(model >= b.lo - math.radians(TICK + 0.01)) and np.all(model <= b.hi + math.radians(TICK + 0.01))
    assert np.array_equal(d.prev_rel, out["rel"]) and np.allclose(out["action"], out["rel"] / 0.6)
    # the outward abductions stop at +20 deg (the walking cap), not at the band's 20 + 14
    for li, i in enumerate(ABDUCTION):
        if ABD_OUT_SIGN[li] > 0:
            assert abs(math.degrees(out["rel"][i]) - 20.0) <= TICK + 0.01


def test_stand_offset_signs_match_the_limits_file():
    off = stand_offset_rel(12.0)
    # + on FR and RL, - on FL and RR, abductions only
    want = np.radians([12, 0, 0, -12, 0, 0, -12, 0, 0, 12, 0, 0])
    assert np.allclose(off, want)
    phys = L.physical_ranges()
    jr = L.joint_ranges()
    for li, leg in enumerate(L.LEGS):                        # outward: towards the larger-outward end of the range
        lo, hi = jr[f"{leg}_hip"]
        out_end = hi if ABD_OUT_SIGN[li] > 0 else lo
        assert abs(abs(math.degrees(out_end)) - phys[leg]["abd_out"][1]) < 1e-9


def test_offset_decoder_shifts_targets_and_observation():
    d0, d = ActionDecoder(C, 0.25), ActionDecoder(C, 0.25, stand_offset_abd_deg=10.0, bounds=L.decoder_bounds("walk"))
    raw = np.zeros(12)
    x0, x = d0(raw), d(raw)
    off = stand_offset_rel(10.0)
    assert np.abs(x["rel"] - (x0["rel"] + off)).max() <= math.radians(TICK + 0.01)
    assert np.allclose(x["rel_policy"], x["rel"] - off) and np.allclose(d.obs_offset, off)
    sd = d.stand_dd()
    assert not np.array_equal(sd, C.stand_dd) and np.array_equal(sd[[1, 2, 4, 5, 7, 8, 10, 11]],
                                                                  C.stand_dd[[1, 2, 4, 5, 7, 8, 10, 11]])
    d.reset(off)                                            # a measured pose at the offset stand: policy rel 0
    assert np.abs(d.prev_rel).max() < 1e-12


def test_torch_offset_equals_a_shifted_stand():
    """``TorchContract(stand_offset_abd_deg)``'s stand is the stand plus the offset."""
    for deg in (8.0, 10.0, 12.0):
        tc = TorchContract("cpu", deg)
        off = math.radians(deg)
        want = torch.tensor(C.stand_rel_rad, dtype=torch.float32) + torch.tensor(
            [off, 0, 0, -off, 0, 0, -off, 0, 0, off, 0, 0], dtype=torch.float32)
        assert torch.equal(tc.stand, want)
        assert torch.equal(TorchContract("cpu").stand, torch.tensor(C.stand_rel_rad, dtype=torch.float32))


def test_export_meta_carries_bounds_and_offset(tmp_path):
    from mp4_recovery.gpu.actor import mlp
    from mp4_recovery.policy_file import NumpyPolicy, export, with_meta
    from mp4_recovery.deploy.runner import band_refusal
    sd = {f"mlp.{k}": v for k, v in mlp(45, 12, (64, 32)).state_dict().items()}
    ck = tmp_path / "m.pt"
    torch.save({"actor_state_dict": sd}, ck)
    plain = NumpyPolicy(export(ck, tmp_path / "plain", scale=[0.25] * 12))
    assert plain.meta["limits"] is False and plain.meta["bounds"] is None and plain.meta["stand_offset_abd_deg"] == 0.0
    assert band_refusal(plain.meta) is None
    b = L.decoder_bounds("walk").to_meta()
    new = NumpyPolicy(export(ck, tmp_path / "new", scale=[0.25] * 12, bounds=b, stand_offset_abd_deg=10.0))
    assert new.meta["limits"] is True and policy_bounds(new.meta).mode == "walk"
    assert new.meta["stand_offset_abd_deg"] == 10.0 and band_refusal(new.meta) is None
    re = NumpyPolicy(with_meta(tmp_path / "plain", tmp_path / "re", limits=True, bounds=b, stand_offset_abd_deg=8.0))
    assert re.meta["stand_offset_abd_deg"] == 8.0 and "source_policy" not in re.meta and len(re.meta["source_policy_sha256"]) == 64
    assert all(np.array_equal(x, y) for (x, _), (y, _) in zip(re.layers, plain.layers))
    # the runner's rules: a get-up export without bounds; an offset without bounds; another calibration's bounds
    g = NumpyPolicy(export(ck, tmp_path / "g", scale=[1.57] * 12, band_deg=[90.0], kind="getup"))
    assert "without decoder bounds" in band_refusal(g.meta)
    no_bounds = {k: v for k, v in g.meta.items() if k not in ("bounds", "limits", "stand_offset_abd_deg")}
    assert band_refusal(no_bounds) is None                  # a get-up exported without the limits: its 90 deg band
    assert "stand offset" in band_refusal({**plain.meta, "stand_offset_abd_deg": 10.0})
    other = json.loads(json.dumps(b))
    other["info"]["contract_stand_dd"][0] += 5
    assert "another calibration" in band_refusal({**plain.meta, "bounds": other})


def test_eval_plant_flags_and_meta():
    plain = {"limits": False, "stand_offset_abd_deg": 0.0, "widen_limits_deg": None}
    assert L.eval_plant({}) == plain
    meta = {"limits": True, "stand_offset_abd_deg": 10.0}
    assert L.eval_plant(meta) == {**plain, "limits": True, "stand_offset_abd_deg": 10.0}
    assert L.eval_plant(meta, False, 0.0) == plain
    assert L.eval_plant({}, True, 12) == {**plain, "limits": True, "stand_offset_abd_deg": 12.0}
    wide = {"limits": True, "bounds": {"info": {"widen_limits_deg": 60.0}}}
    assert L.eval_plant(wide)["widen_limits_deg"] == 60.0
    from mp4_recovery.eval import omni_eval as oe
    pol = SimpleNamespace(scale=[0.25] * 12, meta={})
    kw = oe.env_kwargs(pol, 8, "flat")
    assert kw["limits"] is False and kw["stand_offset_abd_deg"] == 0.0
    kw = oe.env_kwargs(pol, 8, "flat", limits=True, stand_offset_abd=10.0)
    assert kw["limits"] is True and kw["stand_offset_abd_deg"] == 10.0


def test_terrain_and_getup_options_wire_the_limits():
    from mp4_recovery.gpu import terrain_task as tt
    from mp4_recovery.gpu.env_cfg import describe
    from mp4_recovery.gpu.getup_task import GetupOptions, make_getup_env_cfg
    off = tt.make_terrain_env_cfg(tt.TerrainOptions(terrain="flat", rows=2))
    assert off.actions["servo"].bounds is None and off.actions["servo"].stand_offset_abd_deg == 0.0
    on = tt.make_terrain_env_cfg(tt.TerrainOptions(terrain="flat", rows=2, limits=True,
                                                   stand_offset_abd_deg=10.0))
    assert on.actions["servo"].bounds["mode"] == "walk" and on.actions["servo"].stand_offset_abd_deg == 10.0
    m = on.scene.entities["robot"].spec_fn().compile()
    assert all(m.jnt_limited[m.joint(n).id] for n in L.joint_ranges())
    g = make_getup_env_cfg(GetupOptions(limits=True))
    assert g.actions["servo"].bounds["mode"] == "getup"
    mg = g.scene.entities["robot"].spec_fn().compile()
    assert all(mg.jnt_limited[mg.joint(n).id] for n in L.joint_ranges())
    d = describe(GetupOptions(limits=True))
    assert d["options"]["limits"] is True and d["limits"]["limits_sha256"] == L.LIMITS_SHA256
    assert "limits" not in describe(tt.TerrainOptions())


# ------------------------------------------------------------------------------------------------------------- GPU
def _terrain_env(n, **kw):
    from mjlab.envs import ManagerBasedRlEnv
    from mp4_recovery.gpu import terrain_task as tt
    opt = tt.TerrainOptions(num_envs=n, terrain="flat", rows=2, noisy=False, randomise=False, **kw)
    cfg = tt.make_terrain_env_cfg(opt)
    cfg.seed = 5
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def test_sim_and_runner_decode_the_same_targets():
    """The simulator's decoder (ServoAction) and the runner's (ActionDecoder) on the same raw actions, with the
    limits and a 10 deg stand offset: the same dd (to a tick where float32 and float64 round differently), the same
    applied action; the targets inside the bounds."""
    from mp4_recovery.gpu import mdp
    n = 8
    env = _terrain_env(n, limits=True, stand_offset_abd_deg=10.0, action_scale=(0.25, 0.5, 0.5) * 4)
    env.reset()
    servo = mdp.servo_action(env)
    m = env.sim.mj_model
    for name, (lo, hi) in L.joint_ranges().items():
        j = m.joint("robot/" + name)
        assert m.jnt_limited[j.id] and np.allclose(m.jnt_range[j.id], (lo, hi), atol=1e-6)
    b = L.decoder_bounds("walk")
    decs = [ActionDecoder(C, servo.scale.cpu().numpy(), bounds=b, stand_offset_abd_deg=10.0) for _ in range(n)]
    g = torch.Generator(device="cuda:0").manual_seed(0)
    worst = 0.0
    for k in range(40):
        raw = 2.0 * torch.randn(n, 12, device="cuda:0", generator=g)
        servo.process_actions(raw)
        tgt = (servo.target_model - servo.tc.stand_phys).cpu().numpy()       # physical rel
        act = servo.applied_action.cpu().numpy()
        for i in range(n):
            out = decs[i](raw[i].cpu().numpy())
            worst = max(worst, float(np.abs(np.degrees(out["rel"] - tgt[i])).max()))
            assert np.abs(out["action"] - act[i]).max() * 0.25 <= math.radians(TICK + 0.01), (k, i)
            decs[i].prev_rel = servo.prev_rel[i].cpu().numpy().astype(float)     # stay on the sim's tick
        model = servo.target_model.cpu().numpy()
        assert np.all(model >= b.lo - math.radians(TICK + 0.02)) and np.all(model <= b.hi + math.radians(TICK + 0.02))
    assert worst <= TICK + 0.01, worst
    env.close()


def test_offset_flag_equals_a_shifted_stand():
    """A shifted ``TorchContract.stand`` and ``stand_offset_abd_deg`` give the same targets for the same actions; without
    quantisation exactly, with it within a tick and the dd's 0.1 deg each way (the option quantises the physical target,
    as the robot does; the shifted stand quantises the policy's)."""
    from mp4_recovery.gpu import mdp
    n = 4
    g = torch.Generator(device="cuda:0").manual_seed(1)
    raws = [torch.randn(n, 12, device="cuda:0", generator=g) for _ in range(20)]

    def targets(patch: bool, quantise: bool):
        orig = mdp.TorchContract.__init__
        if patch:
            def init(self, *a, **k):
                orig(self, *a, **k)
                o = math.radians(12.0)
                self.stand = self.stand + torch.tensor([o, 0, 0, -o, 0, 0, -o, 0, 0, o, 0, 0], dtype=self.stand.dtype,
                                                       device=self.stand.device)
            mdp.TorchContract.__init__ = init
        try:
            env = _terrain_env(n, stand_offset_abd_deg=0.0 if patch else 12.0)
            servo = mdp.servo_action(env)
            servo.cfg.quantise = quantise
            env.reset()
            out = []
            for r in raws:
                servo.process_actions(r)
                out.append(servo.target_model.clone())
            env.close()
        finally:
            mdp.TorchContract.__init__ = orig
        return torch.stack(out)

    for q in (False, True):
        a, b = targets(True, q), targets(False, q)
        err = float(torch.rad2deg((a - b).abs()).max())
        assert err < (1e-4 if not q else 2 * (TICK + 0.1)), (q, err)      # two grids: a tick and the dd's 0.1 deg each


def test_getup_starts_are_inside_the_bounds():
    from mjlab.envs import ManagerBasedRlEnv
    from mp4_recovery.gpu import mdp
    from mp4_recovery.gpu.env_cfg import noise_profile
    from mp4_recovery.gpu.getup_task import GetupOptions, make_getup_env_cfg
    opt = GetupOptions(num_envs=32, noise_end=noise_profile(), limits=True)
    cfg = make_getup_env_cfg(opt)
    cfg.seed = 3
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    act = mdp.servo_action(env)
    b = L.decoder_bounds("getup")
    model = (act.tc.stand_phys + act.true_rel()).cpu().numpy()
    assert np.all(model >= b.lo - 2e-3) and np.all(model <= b.hi + 2e-3)
    tgt = act.target_model.cpu().numpy()
    assert np.all(tgt >= b.lo - math.radians(TICK + 0.02)) and np.all(tgt <= b.hi + math.radians(TICK + 0.02))
    env.close()






def test_sequence_with_limits_and_a_walker_offset(tmp_path):
    """sequence --limits --stand-offset-abd 10: the walker's offset and walk bounds, the get-up's getup bounds and no
    offset, per robot by mode; a forced GETUP hands back and the robots walk again."""
    import sys as _sys
    _sys.path.insert(0, str(REPO / "tools"))
    from mp4_recovery.eval import sequence as sc
    from test_sequence import WALKER, zero_getup
    g = zero_getup(tmp_path / "getup")
    out = tmp_path / "sim.json"
    rc = sc.main(["--walker", str(WALKER), "--getup", str(g), "--robots", "4", "--seconds", "5", "--kick", "0",
                  "--force-getup-t", "1.0", "--seed", "11", "--json", str(out), "--limits", "--stand-offset-abd", "10"])
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["limits"] is True and d["walker_stand_offset_abd_deg"] == 10.0
    assert d["robots"]["getups"] == [1] * 4 and d["summary"]["walking_again"] == 1.0
