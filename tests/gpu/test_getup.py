"""The get-up task: start slices and their ramp, the collision pairs, the seeded decoder and the band, the terminations,
and the reward's signs on scripted robots (CPU tests first, then the GPU ones)."""
import math

import mujoco
import numpy as np
import torch

from mp4_recovery.robot import sku2
from mp4_recovery.contract import Contract
from mp4_recovery.gpu.getup_task import (MIX_END, MIX_START, SLICES, GetupOptions, getup_collisions, getup_scale,
                                        sample_orientation, start_mix)



def _tilt_deg(q):
    """Tilt from vertical of (w, x, y, z) quaternions: the angle between the body z axis and the world z axis."""
    w, x, y, z = q.unbind(-1)
    return torch.rad2deg(torch.acos((1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)))


# ------------------------------------------------------------------------------------------------------------- CPU
def test_start_mix_ramps_from_the_easy_mix_to_the_full_one():
    assert torch.allclose(start_mix(0, 1000), torch.tensor(MIX_START))
    assert torch.allclose(start_mix(1000, 1000), torch.tensor(MIX_END))
    assert torch.allclose(start_mix(5000, 1000), torch.tensor(MIX_END))
    assert torch.allclose(start_mix(500, 1000), 0.5 * (torch.tensor(MIX_START) + torch.tensor(MIX_END)))
    for it in (0, 250, 1000):
        assert abs(float(start_mix(it, 1000).sum()) - 1.0) < 1e-6
    assert SLICES == ("crouch", "partial", "side", "nose", "back", "random", "midfall")
    assert float(start_mix(0, 1000)[6]) == 0.0 and float(start_mix(5000, 1000)[6]) == 0.0    # midfall off by default
    six = (0.4, 0.3, 0.2, 0.1, 0.0, 0.0)
    assert torch.allclose(start_mix(0, 1000, six, six), torch.tensor(six + (0.0,)))          # a 6-value mix: padded
    o = GetupOptions(midfall=0.2)
    assert abs(o.mix_start[6] - 0.2) < 1e-9 and abs(sum(o.mix_start) - 1.0) < 1e-9 and abs(o.mix_end[6] - 0.2) < 1e-9
    assert abs(o.mix_end[0] / o.mix_end[2] - MIX_END[0] / MIX_END[2]) < 1e-9


def test_each_slice_draws_its_orientations():
    torch.manual_seed(0)
    n = 20000
    lo_hi = {0: (0, 30), 1: (30, 80), 2: (70, 110), 3: (60, 90), 4: (150, 180)}
    for sid, (lo, hi) in lo_hi.items():
        t = _tilt_deg(sample_orientation(torch.full((n,), sid)))
        assert float(t.min()) >= lo - 0.01 and float(t.max()) <= hi + 0.01, (sid, float(t.min()), float(t.max()))
    q = sample_orientation(torch.full((n,), 1))          # partial roll: about x only, both signs
    assert float(q[:, 2].abs().max()) < 1e-6 and 0.4 < float((q[:, 1] > 0).float().mean()) < 0.6
    q = sample_orientation(torch.full((n,), 3))          # nose / tail down: about y only, both signs
    assert float(q[:, 1].abs().max()) < 1e-6 and 0.4 < float((q[:, 2] > 0).float().mean()) < 0.6
    cos = torch.cos(torch.deg2rad(_tilt_deg(sample_orientation(torch.full((n,), 5)))))
    assert abs(float(cos.mean())) < 0.03 and abs(float((cos ** 2).mean()) - 1 / 3) < 0.02    # uniform: cos ~ U(-1, 1)
    q, axis = sample_orientation(torch.full((n,), 6), return_axis=True)      # mid-fall: 40-70 deg about the axis
    t = _tilt_deg(q)
    assert float(t.min()) >= 40.0 - 0.01 and float(t.max()) <= 70.0 + 0.01
    assert float(axis[:, 2].abs().max()) == 0.0 and torch.allclose(axis.norm(dim=-1), torch.ones(n))
    half = torch.deg2rad(t) / 2                            # q = (cos t/2, axis sin t/2)
    assert torch.allclose(q[:, 1:], axis * torch.sin(half)[:, None], atol=1e-5)
    ex = torch.tensor([[math.pi / 2, 0.0], [0.0, math.radians(75)], [math.pi, 0.0]])
    t = _tilt_deg(sample_orientation(torch.full((3,), -1), ex))
    assert torch.allclose(t, torch.tensor([90.0, 75.0, 180.0]), atol=1e-3)


def test_getup_scale_covers_the_band():
    assert getup_scale((90, 90, 90)) == (1.57,) * 12
    s = getup_scale((90, 150, 150))
    assert s[0] == 1.57 and abs(s[1] - math.radians(150)) < 1e-9 and abs(s[2] - math.radians(150)) < 1e-9
    assert GetupOptions().action_scale == (1.57,) * 12
    assert GetupOptions().band_deg == (90.0,) * 12 and GetupOptions().kind == "getup"     # export_policy.py's keys


def _collides(m, g1, g2):
    """MuJoCo's contype / conaffinity test and its parent-child filter."""
    ok = (m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])
    b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
    family = b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1
    return bool(ok) and not family


def _stand(m, d, extra=None):
    c = Contract.load()
    d.qpos[:] = 0.0
    d.qpos[:3] = [0.0, 0.0, 0.2]
    d.qpos[3] = 1.0
    for n, v in zip(c.joint_order, c.stand_joint_rad):
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v + (extra or {}).get(n, 0.0)
    mujoco.mj_forward(m, d)
    return [(m.body(m.geom_bodyid[d.contact[i].geom1]).name, m.body(m.geom_bodyid[d.contact[i].geom2]).name)
            for i in range(d.ncon)]


def test_getup_collision_pairs_and_the_stand_is_overlap_free():
    spec = sku2.get_spec(strip_visuals=True, self_collision=True)
    info = getup_collisions(spec)
    assert all(abs(v - 0.0421) < 1e-3 for v in info["thigh_capsule_m"].values()), info   # >= 30 mm kept
    m = spec.compile()
    d = mujoco.MjData(m)
    g = lambda name: m.geom(name).id
    can = {leg: next(i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(f"{leg}_hip").id) for leg in sku2.LEGS}
    base = [i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body("base").id]
    assert _collides(m, g("FR_calf_collision"), g("FL_calf_collision"))            # leg <-> leg
    assert _collides(m, g("FR_foot"), g("RR_thigh_collision"))
    assert _collides(m, g("FR_thigh_collision"), g("FL_thigh_collision"))
    assert _collides(m, g("FR_calf_collision"), can["FL"])                         # leg <-> another leg's hip can
    assert _collides(m, g("FR_thigh_collision"), can["RR"])
    assert _collides(m, g("FR_calf_collision"), can["FR"])                         # its own can: not parent-child
    assert all(_collides(m, g(f"{leg}_thigh_collision"), b) for leg in sku2.LEGS for b in base)   # thigh <-> base
    assert all(_collides(m, g(f"{leg}_calf_collision"), b) for leg in sku2.LEGS for b in base)    # the walker's
    assert not _collides(m, g("FR_thigh_collision"), can["FR"])                    # parent-child, filtered
    assert not _collides(m, g("FR_thigh_collision"), g("FR_calf_collision"))
    assert not _collides(m, can["FR"], can["FL"])
    assert _stand(m, d) == []                                                     # the stand: no self-contact
    crossed = _stand(m, d, {"FR_hip": -math.radians(45), "FL_hip": math.radians(45)})   # both front legs inward
    assert ("FR_calf", "FL_calf") in crossed or ("FL_calf", "FR_calf") in crossed, crossed
    # stiffer contacts for every robot geom but the feet (priority 2 over the feet's 1 and the floor's 0)
    from mp4_recovery.gpu.getup_task import CONTACT_SOLREF, stiff_contacts
    spec2 = sku2.get_spec(strip_visuals=True, self_collision=True)
    getup_collisions(spec2)
    assert stiff_contacts(spec2) == 2 + 4 + 8                                    # 2 base boxes, 4 cans, 8 capsules
    m2 = spec2.compile()
    for i in range(m2.ngeom):
        foot = m2.geom(i).name.endswith("_foot")
        assert m2.geom_priority[i] == (1 if foot else 2)
        assert tuple(m2.geom_solref[i]) == ((0.02, 1.0) if foot else CONTACT_SOLREF)    # feet untouched (get_spec's)
    # the walker's model is unchanged: no leg <-> leg, no thigh <-> base
    w = sku2.get_spec(strip_visuals=True, self_collision=True).compile()
    assert not any(w.geom_contype[i] & 24 or w.geom_conaffinity[i] & 24 for i in range(w.ngeom))
    assert not _collides(w, w.geom("FR_calf_collision").id, w.geom("FL_calf_collision").id)
    assert not _collides(w, w.geom("FR_thigh_collision").id, [i for i in range(w.ngeom)
                                                            if w.geom_bodyid[i] == w.body("base").id][0])


# ------------------------------------------------------------------------------------------------------------- GPU
def _env(n, **kw):
    from mjlab.envs import ManagerBasedRlEnv
    from mp4_recovery.gpu.env_cfg import noise_profile
    from mp4_recovery.gpu.getup_task import make_getup_env_cfg
    opt = GetupOptions(num_envs=n, noise_end=noise_profile(), **kw)
    cfg = make_getup_env_cfg(opt)
    cfg.seed = 3
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def test_start_proportions_at_iterations_0_and_1000_and_overlap_free_spawns():
    from mp4_recovery.gpu import mdp
    from mp4_recovery.gpu.getup_task import contact_penetration, tilt_deg
    n = 1024
    env = _env(n, fixed_iter=0)
    term = env.event_manager.get_term_cfg("getup_start").func
    lo_hi = {0: (0, 30), 1: (30, 80), 2: (70, 110), 3: (60, 90), 4: (150, 180)}
    for it, mix in ((0, MIX_START), (1000, MIX_END)):
        env.event_manager.get_term_cfg("getup_start").params["fixed_iter"] = it
        torch.manual_seed(it)
        env.reset()
        share = torch.bincount(term.slice_id, minlength=len(SLICES)).float() / n
        assert torch.allclose(share.cpu(), torch.tensor(mix), atol=0.06), (it, share.tolist())
        tilt = tilt_deg(env)
        for sid, (lo, hi) in lo_hi.items():
            sel = term.slice_id == sid
            if bool(sel.any()):
                assert float(tilt[sel].min()) >= lo - 0.1 and float(tilt[sel].max()) <= hi + 0.1, (it, sid)
        floor, self_ = contact_penetration(env)
        assert float(torch.maximum(floor, self_).max()) <= 0.002 + 1e-6
        rel = mdp.servo_action(env).true_rel()
        assert float(rel.abs().max()) <= math.radians(45.0) + 1e-3          # stand + U(+-0.5 band)
        h = term.spawn_height                       # 0.06-0.10 m, raised where a leg reached into the floor
        assert float(h.min()) >= 0.06 - 1e-6 and float(h.max()) < 0.25
    assert term.unresolved == 0
    env.close()


def test_time_out_is_the_only_termination_and_the_decoder_is_seeded():
    from mp4_recovery.gpu import mdp
    env = _env(64, episode_s=1.0)
    assert list(env.termination_manager.active_terms) == ["time_out", "blowup", "nonfinite"]
    assert all(env.termination_manager.get_term_cfg(t).time_out for t in ("time_out", "blowup", "nonfinite"))
    assert env.observation_manager.group_obs_dim["actor"] == (45,)
    assert env.observation_manager.group_obs_dim["critic"] == (53,)            # 45 + lin vel 3 + height 1 + feet 4
    obs, _ = env.reset()
    assert float(obs["actor"][:, 6:9].abs().max()) == 0.0                     # the command, zeroed
    act = mdp.servo_action(env)
    meas = act.tc.measure_rel(act.true_rel())
    assert float(meas.abs().max()) > math.radians(10)                         # fallen starts, not the stand
    q_deg = 3100.0 / 10240.0                                                  # one firmware tick (deg)
    assert torch.equal(act.prev_rel, meas.clamp(-act.band, act.band))          # contract.ActionDecoder.reset(rel0)
    assert torch.allclose(act.target_model, act.tc.stand + act.prev_rel)
    assert torch.allclose(act.applied_action, act.prev_rel / act.scale)
    assert torch.allclose(obs["actor"][:, 33:45], act.prev_rel / act.scale)  # the previous action observed
    hold = act.applied_action.clone()
    env.step(hold)                                                            # hold: the first target is the pose
    assert float(torch.rad2deg(act.target_model - act.tc.stand - meas).abs().max()) <= q_deg + 0.1 + 1e-3
    g = torch.Generator(device="cuda:0").manual_seed(0)
    resets = []
    for k in range(1, 60):
        _, _, term, timeout, _ = env.step(torch.randn(64, 12, device="cuda:0", generator=g))
        resets.append((k + 1, int(term.sum()), int(timeout.sum())))
    assert all(t == 0 for _, t, _ in resets)                                  # no termination but the time-out
    assert [(k, to) for k, _, to in resets if to] == [(50, 64)], resets       # 1 s = 50 steps, every env at once
    env.close()


def test_band_and_scale_are_applied_per_joint():
    from mp4_recovery.gpu import mdp
    env = _env(8, band_deg=(45.0, 150.0, 60.0), start_joints="stand", spawn_z=(0.08, 0.08), fixed_iter=0)
    env.reset()
    act = mdp.servo_action(env)
    assert np.allclose(act.scale.cpu().numpy(), np.array(getup_scale((45, 150, 60))))
    for sign in (1.0, -1.0):
        for _ in range(16):                                                    # 300 deg at 20 deg/step: 15 steps
            env.step(torch.full((8, 12), sign, device="cuda:0"))
        # the band, on the firmware's tick grid (its 0..1023 clamp caps some hips below 150 deg)
        _, want = act.tc.quantise_rel(sign * torch.deg2rad(torch.tensor([45.0, 150.0, 60.0] * 4, device="cuda:0")))
        got = act.prev_rel
        assert float((got - want[None]).abs().max()) < math.radians(0.35), (sign, torch.rad2deg(got[0]).tolist())
        assert float(torch.rad2deg(want).abs()[[0, 2, 3, 5, 6, 8, 9, 11]].min()) > 44.0
    env.close()
    from mp4_recovery.contract import BAND_DEG
    assert mdp.ServoActionCfg().band_deg is None and not mdp.ServoActionCfg().seed_from_measured   # the walker's
    assert list(BAND_DEG) == [20.0, 60.0, 60.0] * 4


def test_self_collisions_on_the_gpu():
    """MJWarp makes the contacts: none at the stand, the two front shins when both legs abduct inward 45 deg."""
    from mp4_recovery.gpu import mdp
    from mp4_recovery.gpu.getup_task import spawn_overlaps
    env = _env(4, start_joints="stand", fixed_iter=0)
    env.reset()
    robot, act = env.scene["robot"], mdp.servo_action(env)
    pos = env.scene.env_origins.clone()
    pos[:, 2] += 0.2                                                          # in the air: no floor contact
    root = torch.zeros(4, 13, device="cuda:0")
    root[:, :3], root[:, 3] = pos, 1.0
    for extra, want in ((0.0, False), (math.radians(45), True)):
        robot.write_root_state_to_sim(root)
        rel = torch.zeros(4, 12, device="cuda:0")
        rel[:, 0], rel[:, 3] = -extra, extra                                   # FR and FL abduction, both inward
        q = act.joint_state_from_rel(rel)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=act._jid_all)
        env.sim.forward()
        floor, legs = spawn_overlaps(env, tol=0.0)
        assert float(floor.max()) == 0.0
        assert bool(legs[:, 0].all()) == want and bool(legs[:, 1].all()) == want, legs
    env.close()




def test_bounded_gaussian_clamps_the_std():
    from mp4_recovery.gpu.getup_task import BoundedGaussianDistribution
    d = BoundedGaussianDistribution(3, init_std=1.0, std_min=0.05, std_max=0.5)
    d.update(torch.zeros(2, 3))
    assert torch.allclose(d._distribution.stddev, torch.full((2, 3), 0.5))
    with torch.no_grad():
        d.std_param.fill_(0.01)
    d.update(torch.zeros(2, 3))
    assert torch.allclose(d._distribution.stddev, torch.full((2, 3), 0.05))






def test_phase_observation():
    env = _env(4, phase_s=4.0, start_joints="stand", fixed_iter=0)
    obs, _ = env.reset()
    assert torch.allclose(obs["actor"][:, 6:9], torch.tensor([0.0, 0.0, 1.0], device="cuda:0").expand(4, 3), atol=1e-6)
    for _ in range(50):
        obs, *_ = env.step(torch.zeros(4, 12, device="cuda:0"))
    assert torch.allclose(obs["actor"][:, 6:9], torch.tensor([0.25, 1.0, 0.0], device="cuda:0").expand(4, 3),
                          atol=1e-5)
    env.close()


def test_search_sequence_unpack():
    from mp4_recovery.gpu import getup_search as gs
    x = torch.tensor([[2.0] * 24 + [0.1, 3.0]])                      # k = 2: postures clipped, holds clipped and capped
    post, hold = gs.unpack(x, 2)
    assert float(post.max()) == 1.0
    assert torch.allclose(hold, torch.tensor([[0.3, 1.5]]))
    post, hold = gs.unpack(torch.tensor([[0.0] * 60 + [1.5] * 5]), 5)
    assert abs(float(hold.sum()) - gs.MAX_SEQ_S) < 1e-5


def test_blowup_guard_resets_a_nonfinite_or_flying_world():
    env = _env(4, start_joints="stand", fixed_iter=0)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    root = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w, torch.zeros(4, 6, device="cuda:0")], -1)
    root[0, 7] = float("nan")                                           # env 0: a NaN velocity
    root[1, 7] = 50.0                                                   # env 1: flying at 50 m/s
    robot.write_root_state_to_sim(root)
    obs, _, term, timeout, _ = env.step(torch.zeros(4, 12, device="cuda:0"))
    assert timeout[:2].tolist() == [True, True] and not bool(timeout[2:].any()) and not bool(term.any())
    assert bool(torch.isfinite(obs["actor"]).all()) and bool(torch.isfinite(obs["critic"]).all())
    assert env.blowups == 2 and env.episode_length_buf[:2].tolist() == [0, 0]
    for _ in range(5):
        obs, *_ = env.step(torch.zeros(4, 12, device="cuda:0"))
    assert bool(torch.isfinite(obs["actor"]).all()) and env.blowups == 2
    env.close()






def test_end_on_success_terminates_at_the_hand_back():
    env = _env(4, start_joints="stand", spawn_z=(0.075, 0.075), fixed_iter=0, end_on_success=True, success_gyro=None)
    term = env.event_manager.get_term_cfg("getup_start").func
    term._exact = torch.zeros(4, 2, device="cuda:0")                    # upright on its feet
    term._fixed_random = torch.zeros(4, dtype=torch.bool, device="cuda:0")
    env.reset()
    ended, bonus = None, torch.zeros(4, device="cuda:0")
    for k in range(80):
        _, r, terminated, timeout, _ = env.step(torch.zeros(4, 12, device="cuda:0"))
        if bool(terminated.any()) and ended is None:
            ended = k + 1
            assert bool(terminated.all()) and not bool(timeout.any())
            bonus = r.clone()
    assert ended is not None and 25 <= ended <= 60, ended
    assert bool((bonus > 9.0).all()), bonus                             # the +10 on the terminating step
    env.close()


def test_midfall_starts_rotate_into_the_fall_without_penetration():
    """Mid-fall starts: near the stand, tilted 40-70 deg, rotating 2-6 rad/s further about the tilt axis, the base
    moving as that rotation about the feet, no geom touching the floor at the spawn; the fall then continues."""
    from mp4_recovery.gpu import mdp
    from mp4_recovery.gpu.getup_task import STAND_Z, contact_penetration, tilt_deg
    n = 256
    env = _env(n, fixed_iter=0, midfall=1.0)
    term = env.event_manager.get_term_cfg("getup_start").func
    torch.manual_seed(5)
    env.reset()
    assert bool((term.slice_id == SLICES.index("midfall")).all())
    tilt0 = tilt_deg(env).clone()
    assert float(tilt0.min()) >= 40.0 - 0.1 and float(tilt0.max()) <= 70.0 + 0.1
    floor, self_ = contact_penetration(env)
    assert float(floor.max()) == 0.0 and float(self_.max()) <= 0.002 + 1e-6 and term.unresolved == 0
    assert float(mdp.servo_action(env).true_rel().abs().max()) <= math.radians(8.0) + 1e-3
    robot = env.scene["robot"]
    w = robot.data.root_link_ang_vel_w
    assert float(w.norm(dim=-1).min()) >= 2.0 - 1e-3 and float(w.norm(dim=-1).max()) <= 6.0 + 1e-3
    assert float(w[:, 2].abs().max()) < 1e-4                                   # about a horizontal axis
    from mjlab.utils.lab_api.math import quat_apply
    r = quat_apply(robot.data.root_link_quat_w, torch.tensor([0.0, 0.0, STAND_Z], device="cuda:0").expand(n, 3))
    assert torch.allclose(robot.data.root_link_lin_vel_w, torch.cross(w, r, dim=-1), atol=1e-4)
    body_z = quat_apply(robot.data.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device="cuda:0").expand(n, 3))
    falling = (torch.cross(w, body_z, dim=-1)[:, :2] * body_z[:, :2]).sum(-1)  # body z swings further from vertical
    assert bool((falling > 0).all())
    hold = mdp.servo_action(env).applied_action.clone()
    for _ in range(3):
        env.step(hold)
    assert float((tilt_deg(env) > tilt0).float().mean()) > 0.9
    env.close()




def test_nonfinite_guard_sanitises_logs_and_resets(tmp_path):
    """A NaN injected into one observation term of one env never reaches the observation (that row
    is 0), is logged with the env id, the term and the env's qpos / qvel, and that env (only) is reset at the next
    step as a time-out, counted under Episode_Termination/nonfinite; a NaN in a sensor buffer resets in the same step."""
    import json
    from mp4_recovery.gpu.getup_task import NonfiniteGuard
    env = _env(4, start_joints="stand", fixed_iter=0)
    env.termination_manager.get_term_cfg("nonfinite").func.log_path = str(tmp_path / "nf.jsonl")
    guard = env.termination_manager.get_term_cfg("nonfinite").func
    assert isinstance(guard, NonfiniteGuard)
    obs, _ = env.reset()
    zero = torch.zeros(4, 12, device="cuda:0")
    for _ in range(5):
        obs, *_ = env.step(zero)
    assert env.nonfinite_resets == 0 and guard.log == []
    om = env.observation_manager
    k = om._group_obs_term_names["actor"].index("gravity")
    term = om._group_obs_term_cfgs["actor"][k].func
    inner = term._inner
    fire = {"on": True}

    def poisoned(env_, **kw):
        x = inner(env_, **kw).clone()
        if fire["on"]:
            x[2, 1] = float("nan")
        return x
    object.__setattr__(term, "_inner", poisoned)
    qpos_before = torch.as_tensor(env.sim.data.qpos)[2].clone()
    obs, _, term_, timeout, extras = env.step(zero)
    assert bool(torch.isfinite(obs["actor"]).all()) and float(obs["actor"][2, 3:6].abs().max()) == 0.0
    assert float(obs["actor"][1, 3:6].abs().max()) > 0.5                       # the others untouched
    fire["on"] = False
    ep = env.episode_length_buf.clone()
    obs, _, term_, timeout, extras = env.step(zero)
    assert timeout.tolist() == [False, False, True, False] and not bool(term_.any())
    assert int(env.episode_length_buf[2]) == 0 and bool((env.episode_length_buf[[0, 1, 3]] == ep[[0, 1, 3]] + 1).all())
    assert env.nonfinite_resets == 1 and len(guard.log) == 1
    rec = guard.log[0]
    assert rec["env_id"] == 2 and rec["term"] == "obs/actor/gravity"
    assert len(rec["qpos"]) == qpos_before.numel() and len(rec["qvel"]) == env.sim.data.qvel.shape[1]
    assert np.allclose(rec["qpos"][:3], qpos_before[:3].cpu().numpy(), atol=0.05)
    assert json.loads((tmp_path / "nf.jsonl").read_text().splitlines()[0])["env_id"] == 2
    assert any("nonfinite" in key for key in extras.get("log", {})), list(extras.get("log", {}))
    # a sensor buffer: reset in the same step
    sensor = env.scene["feet_contact"]
    orig = type(sensor)._compute_data

    def bad_data(self_):
        d = orig(self_)
        d.force = d.force.clone()
        d.force[1, 0, 0] = float("inf")
        return d
    type(sensor)._compute_data = bad_data
    try:
        obs, _, term_, timeout, _ = env.step(zero)
    finally:
        type(sensor)._compute_data = orig
    assert bool(timeout[1]) and int(env.episode_length_buf[1]) == 0 and env.nonfinite_resets == 2
    assert guard.log[1]["term"].startswith("sensor/feet_contact.")
    assert bool(torch.isfinite(obs["critic"]).all())
    env.close()


