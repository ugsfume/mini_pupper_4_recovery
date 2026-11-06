#!/usr/bin/env python
"""Train the get-up policy with PPO (``getup_task``). ``scripts/recipes/getup.sh`` is the training path: a behaviour
clone of the searched motions, then seven PPO steps of this script.

    python -m mp4_recovery.gpu.train --limits --getup-phase 4 --getup-band 90 180 180 --obs-norm --envs 2048 \
        --iters 1000 --out runs/getup --tag limits --resume runs/getup/<run>/model_1499.pt

``--init-actor`` starts from a cloned actor (``getup_bc``), ``--resume`` fine-tunes a checkpoint: actor, critic,
optimizer and iteration come from it. Either way the critic warms up first (``--critic-warmup``, value-only updates;
``algorithms.SmoothPPO``).

Network: ELU MLPs (``--actor-hidden``, ``--critic-hidden``). PPO: adaptive learning rate (KL 0.01; ``--lr0``,
``--lr-max``), gamma 0.99, lambda 0.95, clip 0.2, 5 epochs x 4 mini-batches, 24 steps per env per iteration.
"""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper

from mp4_recovery import paths
from mp4_recovery.gpu.env_cfg import STEPS_PER_ITER, describe, noise_profile
from mp4_recovery.gpu.getup_task import GetupOptions, install_filter_ramp, make_getup_env_cfg
from mp4_recovery.gpu.mdp import servo_action
from mp4_recovery.obs_norm import freeze_updates
from mp4_recovery.paths import sha256


def runner_cfg(a) -> dict:
    cfg = asdict(RslRlOnPolicyRunnerCfg(
        seed=a.seed, num_steps_per_env=STEPS_PER_ITER, max_iterations=a.iters, save_interval=50,
        experiment_name="getup", logger="tensorboard", clip_actions=1.0,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        actor=RslRlModelCfg(hidden_dims=tuple(a.actor_hidden), activation="elu", obs_normalization=a.obs_norm,
                            distribution_cfg={"class_name": "mp4_recovery.gpu.getup_task:BoundedGaussianDistribution",
                                              "init_std": a.init_std, "std_type": "scalar",
                                              "std_min": a.std_bounds[0], "std_max": a.std_bounds[1]}),
        critic=RslRlModelCfg(hidden_dims=tuple(a.critic_hidden), activation="elu", obs_normalization=a.obs_norm),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=a.entropy_coef,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=a.lr0, schedule="adaptive", gamma=0.99,
            lam=0.95, desired_kl=0.01, max_grad_norm=1.0, class_name="mp4_recovery.gpu.algorithms:SmoothPPO"),
    ))
    cfg["algorithm"].update(actor_warmup_iters=a.critic_warmup, caps_temporal=a.caps_temporal,
                            caps_spatial=a.caps_spatial, caps_obs_noise=a.caps_obs_noise, caps_seed=a.seed,
                            lr_min=1.0e-5, lr_max=a.lr_max)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--out", default=str(paths.RUNS / "getup"))
    ap.add_argument("--tag", default="", help="ends the run directory's name")
    ap.add_argument("--init-actor", help="a getup_bc model_bc.pt to start the actor from")
    ap.add_argument("--resume", help="a model_<k>.pt to fine-tune")
    # the task
    ap.add_argument("--limits", action="store_true", help="the measured joint limits, in the model and the decoder")
    ap.add_argument("--widen-limits-deg", type=float, metavar="DEG",
                    help="let hip and knee targets go DEG past their potentiometer track ends instead of stopping 20 "
                         "deg inside them. Training directly inside the limits never learns to flip from the back")
    ap.add_argument("--getup-band", type=float, nargs="+", metavar="DEG", default=(90.0, 90.0, 90.0),
                    help="the band about the stand (deg): abduction, hip, knee, or 12 values in servo order")
    ap.add_argument("--getup-phase", type=float, default=0.0, metavar="S",
                    help="the command floats carry (phi, sin 2 pi phi, cos 2 pi phi), phi = t / S")
    ap.add_argument("--getup-episode-s", type=float, default=6.0, help="the episode length (s)")
    ap.add_argument("--getup-mix", type=float, nargs=12, metavar="W",
                    help="the start mix (crouch partial side nose back random) at iteration 0, then from iteration "
                         "1000 on (normalised)")
    ap.add_argument("--getup-midfall", type=float, default=0.0, metavar="SHARE", help="the mid-fall starts' share")
    ap.add_argument("--getup-end-on-success", action="store_true",
                    help="the episode ends when the success criterion has held 0.5 s")
    ap.add_argument("--getup-success-gyro", type=float, default=0.8,
                    help="the success reward's gyro limit (rad/s); 0: no gyro condition")
    ap.add_argument("--getup-action-rate", type=float, default=-0.02, help="the action-rate cost's weight")
    ap.add_argument("--getup-action-rate-tilted", type=float, default=0.1,
                    help="the action-rate cost's factor while tilted above 40 deg")
    ap.add_argument("--getup-joint-vel", type=float, default=0.0, help="weight (< 0) of the squared joint velocities")
    ap.add_argument("--getup-filter", type=float, default=0.0, metavar="ALPHA",
                    help="the target low-pass filter's final alpha; > 0 also moves the action-rate cost to the "
                         "applied target's change")
    ap.add_argument("--getup-filter-start", type=float, metavar="A0", help="the alpha at this run's first iteration")
    ap.add_argument("--getup-filter-ramp", type=int, default=0, metavar="N",
                    help="A0 -> ALPHA linearly over this run's first N iterations")
    # the networks and PPO
    ap.add_argument("--actor-hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--critic-hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--obs-norm", action="store_true",
                    help="empirical observation normalisation in the actor and the critic (folded into the first "
                         "layer at export); an --init-actor checkpoint's normaliser is loaded")
    ap.add_argument("--obs-norm-freeze", type=int, default=0, metavar="N",
                    help="the actor's normaliser is not updated for this run's first N iterations")
    ap.add_argument("--init-std", type=float, default=0.1, help="the exploration std at the start")
    ap.add_argument("--std-bounds", type=float, nargs=2, default=(0.0, 1.0e9), metavar=("LO", "HI"),
                    help="clamp the action std to [LO, HI]")
    ap.add_argument("--entropy-coef", type=float, default=0.005, help="PPO's entropy bonus")
    ap.add_argument("--critic-warmup", type=int, default=0, help="value-only updates before the actor moves")
    ap.add_argument("--lr0", type=float, default=1.0e-3, help="the adaptive learning rate's start")
    ap.add_argument("--lr-max", type=float, default=1.0e-2, help="the adaptive learning rate's ceiling")
    ap.add_argument("--caps-temporal", type=float, default=0.0, metavar="LAMBDA", help="the temporal smoothness loss")
    ap.add_argument("--caps-spatial", type=float, default=0.0, metavar="LAMBDA", help="the spatial smoothness loss")
    ap.add_argument("--caps-obs-noise", type=float, default=0.0, metavar="SIGMA",
                    help="the spatial loss's perturbation, in the actor's normalised input units")
    a = ap.parse_args()
    if a.resume and a.init_actor:
        raise SystemExit("--resume and --init-actor exclude each other")

    mix = {}
    if a.getup_mix:
        start, end = a.getup_mix[:6], a.getup_mix[6:]
        mix = {"mix_start": tuple(w / sum(start) for w in start), "mix_end": tuple(w / sum(end) for w in end)}
    opt = GetupOptions(num_envs=a.envs, noisy=True, noise_ramp=True, randomise=True, noise_end=noise_profile(),
                       band_deg=tuple(a.getup_band), success_gyro=a.getup_success_gyro or None, phase_s=a.getup_phase,
                       end_on_success=a.getup_end_on_success, midfall=a.getup_midfall, limits=a.limits,
                       widen_limits_deg=a.widen_limits_deg, episode_s=a.getup_episode_s,
                       action_rate_weight=a.getup_action_rate, action_rate_tilted_scale=a.getup_action_rate_tilted,
                       joint_vel_weight=a.getup_joint_vel, filter_alpha=a.getup_filter,
                       filter_alpha_start=a.getup_filter_start, filter_ramp_iters=a.getup_filter_ramp, **mix)
    env_cfg = make_getup_env_cfg(opt)
    env_cfg.seed = a.seed
    log_dir = Path(a.out) / (f"{datetime.now():%Y%m%d_%H%M%S}_s{a.seed}" + (f"_{a.tag}" if a.tag else ""))
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.terminations["nonfinite"].params["log_path"] = str(log_dir / "nonfinite.jsonl")

    cfg = runner_cfg(a)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env, clip_actions=1.0), cfg, str(log_dir), "cuda:0")
    if a.init_actor:
        sd = torch.load(a.init_actor, map_location="cuda:0", weights_only=False)["actor_state_dict"]
        has_norm = any(k.startswith("obs_normalizer.") for k in sd)
        if has_norm != a.obs_norm:
            raise SystemExit(f"--init-actor {'has' if has_norm else 'has no'} observation normaliser: "
                             f"{'add' if has_norm else 'drop'} --obs-norm")
        sd = {k: v for k, v in sd.items() if k.startswith("mlp.") or k.startswith("obs_normalizer.")}
        missing, unexpected = runner.alg.actor.load_state_dict(sd, strict=False)
        assert not unexpected and all("std" in k for k in missing), (missing, unexpected)
        with torch.no_grad():
            runner.alg.actor.distribution.std_param.fill_(a.init_std)
    norm_freeze = freeze_updates(runner.alg.actor, a.obs_norm_freeze) if a.obs_norm and a.obs_norm_freeze else None
    if a.resume:
        runner.load(a.resume, map_location="cuda:0",
                    load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": True})

    commit = subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(paths.REPO), "status", "--porcelain"], capture_output=True, text=True)
    (log_dir / "config.json").write_text(json.dumps({
        "task": describe(opt), "runner": cfg, "noise": "measured", "init_std": a.init_std, "critic_warmup": a.critic_warmup,
        "obs_norm": a.obs_norm, "obs_norm_freeze_iters": a.obs_norm_freeze if a.obs_norm else None,
        "resume": a.resume, "resume_sha256": sha256(Path(a.resume)) if a.resume else None,
        "init_actor": a.init_actor, "init_actor_sha256": sha256(Path(a.init_actor)) if a.init_actor else None,
        "repo_commit": commit.stdout.strip(), "repo_dirty": bool(dirty.stdout.strip()), "argv": sys.argv},
        indent=1, default=str))
    if opt.filter_alpha or opt.filter_alpha_start:
        install_filter_ramp(runner, servo_action(env), opt.filter_alpha, opt.filter_alpha_start, opt.filter_ramp_iters)
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    if norm_freeze is not None:
        print("OBS_NORM_FREEZE", json.dumps(norm_freeze))
    print("RUN_DIR", log_dir)


if __name__ == "__main__":
    main()
