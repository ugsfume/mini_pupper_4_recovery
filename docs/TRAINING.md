# Training

The get-up policy trains from searched motions with `scripts/recipes/getup.sh`. The walker isn't trained here: its weights ship with the
repository, and `scripts/recipes/walker.sh` writes its stance into them. Training and evaluation need the `gpu` install
([INSTALL.md](INSTALL.md)).

## The get-up recipe

```sh
sh scripts/recipes/getup.sh [SEED]     # SEED defaults to 41
```

Run it from the repository root. Each step fine-tunes the last checkpoint of the step before it, and a finished step is skipped if the
script runs again. Output goes to `runs/getup/s<SEED>/`, and the script's last line names the final checkpoint.

| Step | Iterations | Last checkpoint | From | What changes |
|---|---|---|---|---|
| Clone | 300 epochs, 4 rounds | `clone/model_bc.pt` | searched motions | Supervised copy of the motions into one network |
| `wide` | 1500 | 1499 | the clone | PPO, hip and knee targets allowed 60° past their track ends |
| `limits` | 1000 | 2498 | `wide` | The real limits; action-rate and joint-velocity costs |
| `filter` | 253 | 2750 | `limits` | Target filter ramped from α 0.5 to 0.75 over 200 iterations; learning rate at most 5e-5 |
| `smooth1`, `smooth2` | 400, 200 | 3149, 3348 | `filter` | Filter fixed at 0.75; smoothness loss, temporal 1, spatial 20 |
| `smooth3`, `smooth4` | 600, 600 | 3947, 4546 | `smooth2` | Smoothness loss, temporal 2, spatial 40 |

The result depends on the seed, so compare runs with the sequence evaluation (see [Checking a result](#checking-a-result)).

An iteration takes 5 to 9 s at 2048 robots on an RTX 5080, and a run takes about 3.2 GB of GPU memory. The clone takes about 35 minutes, the whole recipe about 12 hours.

`policies/getup_sim` is this recipe's output with seed 41. It has been tested in simulation only. `policies/getup`, the policy that has run on the
robot, was made with the recipe's kinds of steps on other seeds (31, clone 21) with slightly different step boundaries; the script doesn't
reproduce it exactly.

`checkpoints/` holds that seed-41 run's states at the end of the clone (`clone.pt`), `wide` (`wide_1499.pt`) and `smooth3` (`smooth3_3947.pt`).
`START=<checkpoint> sh scripts/recipes/getup.sh` starts from one: `clone.pt` at `wide`, `wide_1499.pt` at `limits`, `smooth3_3947.pt` at `smooth4`.

### What every step shares

| Option | Value |
|---|---|
| Robots | 2048 |
| Episode | 8 s; ends early on success |
| Starts | crouch 0.1, partial roll 0.1, side 0.3, nose or tail down 0.2, back 0.2, random 0.1, with 20 % of all starts mid-fall |
| Band | 90° on abductions, 180° on hips and knees, about the stand |
| Phase clock | 4 s |
| Networks | actor and critic 512-256-128, with an observation normaliser, frozen for each run's first 100 iterations |
| Exploration | std starts at 0.03, held in 0.03 to 0.3, no entropy bonus |
| Learning rate | starts at 1e-5, then adaptive to KL 0.01 |
| Critic warm-up | value-only updates first: 50 iterations in `wide` and `limits`, 30 after |

**Why such a small std and first learning rate.** With std 0.1 (about 9° of target noise per step at 1.57 rad per unit) the clone never
succeeds. At std 0.03 with a first learning rate of 1e-3, the first actor update takes success to 0, because at that std a tiny change of the
mean is a huge KL.

### Clone

Plays each searched motion in the get-up environment with its orientation perturbed, keeps the trials that succeed, and fits the network to
(observation, action) by mean squared error, with every start weighted equally. On its own the clone rarely gets up: it only gives PPO
somewhere to start.

### `wide`: past the track ends

Hip and knee targets may go 60° past their potentiometer's track ends; they normally stop 20° inside. Trained inside the limits from the
start, side get-ups reach 2–16 % success and the back stays at 0 % (6 s episodes). With the track ends lifted, one policy rises from every
start at 99–100 %, flipping from its back.

### `limits`: the real limits, and costs

Targets stay inside the limits from here on, which is what the robot can do: positions in the dead zone beyond a track end can't be read.
Action-rate weight −0.1, kept whole while tilted (the default is −0.02, ten times weaker while tilted), and joint velocity −5e-4.

Moving to the limits costs success at first, then training brings it back up.

These costs don't remove the flailing. Every leg joint reverses its target 8–11 times per second, by at least 10° each side, with 41–55 % of the
leg velocity power above 6 Hz; the searched motions read 0–0.05 reversals and 12–21 %. The cause is the action scale: π rad per unit on hips
and knees, so 0.01 of output is 1.8°, with only the 20° per step limit and nothing in the return that cares while the robot is airborne.
Raising the action-rate cost 25–50 times and adding the joint velocity cost for 300 iterations leaves the leg speed at 5.6–5.9 rad/s rms while
flipped.

### `filter`: the target filter

The filter (see [POLICY_CONTRACT.md](POLICY_CONTRACT.md#target-filter)) has to be trained under. On a finished policy, without retraining:

| Filter | Back | Left | Nose | Mid-fall |
|---|---|---|---|---|
| none | 99 | 91 | 95 | 98 |
| α 0.5 | 91 | 91 | 93 | 86 |
| α 0.75 | 0 | 15 | 56 | 15 |

(% success, 128 robots per start.) α 0.75 brings the leg speed to 2.7–3.1 rad/s, the searched motions' 2.8–3.1, so it's the target, and α 0.5 costs
almost nothing, so it's the start. A rate limit doesn't work instead: 5° per step (250°/s) gives 0–45 %, because the flip needs fast deliberate
slews. The problem is repetition, not speed.

The learning rate stays at 5e-5 or below from here on.

### `smooth1` to `smooth4`: the smoothness loss

Added to the PPO update, on the actor's mean:

```
temporal  mean |mu(o_t+1) - mu(o_t)|^2      over consecutive steps of one robot, not across a reset
spatial   mean |mu(o) - mu(o + eps)|^2      eps ~ N(0, 0.05^2), in the normalised input
```

The filter alone keeps success but misses the motion limits. The weights rise in two stages because they are easy to overdo: 5 and 100 freeze
the policy (back 0 %, left 4 % after 150 iterations), since this calibration leaves little room inside the track ends. The recipe uses 1 and 20, then 2 and 40.

## The searched motions

`data/searched_motions/` holds six open-loop get-ups, each a sequence of key postures with hold times, found by the cross-entropy method.

| File | Start | Postures | Band (abduction, hip, knee) | Trial |
|---|---|---|---|---|
| `search_left.json` | left side | 5 | 90, 90, 90 | 6 s |
| `search_right_a.json` | right side | 5 | 90, 90, 90 | 6 s |
| `search_right_b.json` | right side | 6 | 90, 90, 90 | 6 s |
| `search_nose.json` | nose down | 5 | 90, 90, 90 | 6 s |
| `search_back_a.json` | back | 6 | 30, 180, 180 | 8 s |
| `search_back_b.json` | back | 6 | 30, 180, 180 | 8 s |

Each generation tries 64 candidates on 16 randomised robots, for 20 to 25 generations. The score is the share of robots that stand (tilt under 12°,
joint rms under 10° from the stand, base above 0.05 m, held 0.5 s), with 0.5 × the best uprightness added for the back. Every file's best
candidate succeeds on all 16.

`python -m mp4_recovery.gpu.getup_search` makes new ones. It doesn't regenerate these files exactly: the right side's search started from a
mirrored left and the back's from an earlier search, and the back ran with hip and knee track ends lifted 60°, which this tool has no option for.
The searched motions aren't part of the recipe's run; it reads them.

## The walker

The walker's weights come from the `flat_omni` policy of `mini_pupper_4_sim`, with further training on rough ground and stairs, not included.
Retraining it isn't part of this work.

```sh
sh scripts/recipes/walker.sh [OUT]            # OUT defaults to runs/walker
```

This writes two things into the policy's meta and copies the weights and fixtures unchanged: the measured limits, and an abduction stand 12° further
out. A wider stance needs no retraining, and it's what makes the walker ride out shoves. Falls from a 1.0 m/s sideways shove, in simulation
(seed 11):

| Stance offset | Standing | Walking |
|---|---|---|
| 8° | 58–64 % | 21–23 % |
| 10° | 28–45 % | 9–11 % |
| 12° | 0–11 % | 4 % |

With the plain stand, 92 % fall against that 4 %. At 1.25–1.5 m/s sideways the 12° stance still falls about 45–70 % of the time, which is where the
get-up takes over.

The wider stance lowers success on box obstacles. Inside the limits, its fastest spins reach −0.26 rad/s against the plain stand's −0.32, and diagonals run about 5 % slow.

## Checking a result

```sh
# Export a checkpoint
python -m mp4_recovery.export_policy runs/getup/s41/<run>/model_4546.pt runs/exports/mine

# Success and motion per start, against the fixed limits
python -m mp4_recovery.eval.getup_eval --policy runs/exports/mine --seed 12 --robots 256 --seconds 8 \
    --starts left,right,nose,back,midfall,crouch,random --motion --json runs/eval/mine_s12.json
python -m mp4_recovery.eval.thresholds runs/eval/mine_s12.json

# The walk, shove, fall, get-up, walk sequence
python -m mp4_recovery.eval.sequence --walker policies/walker --getup runs/exports/mine \
    --schedule 0.5 1.5 2 4 --robots 256 --seconds 60 --seed 12 --getup-phase-restart 1.0 --json runs/eval/mine_seq_s12.json

# A video: watch it, the numbers don't show everything
MUJOCO_GL=egl python -m mp4_recovery.eval.getup_eval --policy runs/exports/mine --seed 14 --json runs/eval/v.json \
    --video back.mp4 --video-start back
```

`thresholds` requires 90 % success from left, right and nose, 85 % from back, mid-fall and random, a median time to a held stand of at most 5 s,
and limits on target reversals, velocity power, leg speed and impact from `src/mp4_recovery/data/motion_thresholds.json`. It exits 0 on a pass.

`policies/getup` and `policies/getup_sim` each exceed a reversal limit slightly (back and left start respectively), so `thresholds` exits 1 on them. An export from the widened-limits step also fails `runner check`, as its targets reach past the real limits: it's only a training state.

The walker's own checks are `eval.push_eval`, `eval.omni_eval` and `eval.terrain_eval`; each one's docstring has its commands.
