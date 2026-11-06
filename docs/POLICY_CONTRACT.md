# Policy contract

What a policy observes and what its output does on a Mini Pupper v4 SKU2. `src/mp4_recovery/contract.py` is the reference
implementation (numpy only, imported by the runner on the Pi); the training environment implements the same arithmetic in
torch (`gpu/mdp.py`), and `tests/` checks that the two agree. If the contract changes, this document, `contract.py` and the
exporter change together.

The per-robot numbers (servo signs, stand pose, calibration offsets) live in `src/mp4_recovery/data/sku2_contract.json`.

`python -m mp4_recovery.deploy.runner check --policy policies/<name>` checks an exported policy against this document (see
[Checking a policy](#checking-a-policy)).

## The two policies

| | `policies/walker` | `policies/getup` |
|---|---|---|
| Meta `kind` | `walk` | `getup` |
| Inputs | 450 (10 frames, history 10) | 45 (1 frame, history 0) |
| Hidden layers (ELU) | 128, 128, 64 | 512, 256, 128 |
| Command slice | `[vx, vy, wz]` | the phase |
| `scale` per leg, rad (abduction, hip, knee) | 0.25, 0.5, 0.5 | 1.57, π, π |
| Band about the stand, deg | 20, 60, 60 | 90, 180, 180 |
| Abduction stand offset, deg outward | 12 | 0 |
| Target filter α | none | 0.75 |

| Decoder bounds mode | `walk` | `getup`, with a hip envelope |

`policies/getup_sim` has the same layout as `policies/getup`.

Both run at 50 Hz (`CONTROL_DT = 0.02`). The robot reports state every 5 ms, and one policy step is 4 STATE frames, locked to the
ESP32's clock.

## The file

`policy.npz`:

| Key | Content |
|---|---|
| `n_layers` | number of linear layers, N |
| `W0..W(N-1)`, `b0..b(N-1)` | weights and biases, float32 |
| `scale` | the 12 per-servo action scales, rad |
| `meta` | a JSON string, see [Meta](#meta) |
| `fixture_obs`, `fixture_act` | 256 observations (N(0, 0.5²)) and the actor's torch outputs for them |

The forward pass (`policy_file.NumpyPolicy`), in float32, with ELU between layers and none after the last:

```
x = obs
for each layer i:
    x = x @ W_i.T + b_i
    if i is not the last:  x = x if x > 0 else exp(x) - 1
```

The output is the action mean, unclipped. An observation normaliser, where training used one (the get-up does), is folded into `W0`
and `b0`: nothing is normalised at run time.

## Coordinates

Observations and actions are in each servo's own angle, in radians, relative to the stand pose:

```
rel_i = sign_i * radians(servo_deg_i)
```

`servo_deg` is the firmware's servo angle without the calibration offset (0 is the stand). In the MuJoCo model the abduction and
hip are the `*_hip` and `*_thigh` joints, and the knee is the `*_knee_drive` tendon (the shin's angle).

Servo order, for observations and actions (servo ids 1 to 12):

```
FR abduction, FR hip, FR knee, FL abduction, FL hip, FL knee,
RR abduction, RR hip, RR knee, RL abduction, RL hip, RL knee
```

The firmware's units, both in float32:

```
tick = 511 + (int)((servo_deg + offset) * 10240 / 3100)
dd   = (int)(3100 - tick * 3100 / 1024)
```

`dd` is the servo's position in 0.1°, 0 to 3100. A measured position converts back with `servo_deg = (1553.02734375 - dd) / 10 - offset`.

## Observation frame (45 values)

No scaling is applied to any slice.

| Slice | Term | Content |
|---|---|---|
| `0:3` | `ang_vel` | Body angular velocity (gyro, with the bias measured before the run removed), rad/s |
| `3:6` | `gravity` | Gravity direction in the body frame, `-acc / \|acc\|` from the accelerometer; `(0, 0, -1)` upright |
| `6:9` | `command` | Walker: `[vx, vy, wz]` (m/s, m/s, rad/s), body frame. Get-up: the phase |
| `9:21` | `pos` | Servo positions, `rel`, minus the policy's stand offset, rad |
| `21:33` | `vel` | Servo velocities, rad/s, from the 200 Hz positions (`VelocityFilter`: median of 3, then a least-squares slope over 4) |
| `33:45` | `prev_action` | Previous applied target, in action units: `prev_rel / scale`, not clipped to ±1 |

Every frame is built from the newest STATE frame. The velocity filter takes every STATE frame in order; a missing tick repeats the
previous sample.

## Walker: history and command

The walker observes its last 10 frames: `D = 45 × 10`. The layout is by term: for each of the six terms in order, that term's 10
frames from oldest to newest.

| Slice | Term | Newest frame |
|---|---|---|
| `0:30` | `ang_vel` | `27:30` |
| `30:60` | `gravity` | `57:60` |
| `60:90` | `command` | `87:90` |
| `90:210` | `pos` | `198:210` |
| `210:330` | `vel` | `318:330` |
| `330:450` | `prev_action` | `438:450` |

After a reset the first frame fills all 10 slots (`contract.HistoryStacker`).

The command's `wz` comes from the runner's heading loop while no turn is commanded (`heading.pi_step`: `kp 2.0`, `ki 0.5`, `wz_max 0.4`
rad/s, `i_max 0.5`, conditional integration). A commanded turn is sent alone. The heading is the gyro's rate about the low-passed
gravity direction, integrated over every STATE frame, and is re-latched at each hand-back.

## Get-up: the phase input

The get-up's command slice carries a clock instead of a velocity:

```
phi = clip(t / 4 s, 0, 1)                       # t: time since this attempt began; meta getup_phase_s = 4
command = (phi, sin(2 pi phi), cos(2 pi phi))
```

It stays at 1 once it reaches the end. The clock starts at 0 when the runner enters the get-up, and the runner restarts it when it has been at
its end for 1 s and the filtered tilt is still above 45°. That's a runner rule; the policy only sees the clock.

## Action (12 values)

The decoder (`contract.ActionDecoder`), in order:

```
a    = clip(output, -1, 1)
rel  = scale * a
rel  = rel + alpha * (prev_rel - rel)                 # the target filter; alpha = 0 skips it
rel  = clip(rel, -band, +band)                        # about the policy's stand
rel  = clip(rel, prev_rel - 20 deg, prev_rel + 20 deg) # the step limit
phys = rel + offset                                   # the stance offset
phys = bounds.clamp(phys)                             # the joint bounds
dd   = firmware ticks from phys, then dd              # quantised as the firmware does
prev_rel = rel from dd, minus offset                  # what the next step filters and limits against
```

- `prev_rel` is the previous **applied** target, after the clamps and the quantisation. Its value in action units is the next
  observation's `prev_action`.
- The step limit is the same in simulation and on the robot.

**Reset.** A decoder reset sets `prev_rel` to the stand, or, at a change of policy, to the measured pose (`reset(rel0)`) clipped to
the band and bounds, so the first target moves from where the joints are. The velocity filter restarts from that frame, and the
history clears.

### Target filter

A first-order low-pass on the target, with the previous applied target as its state, so it needs no input beyond `prev_action`. Meta
`action_filter_alpha`: 0.75 on the get-up (about 2.3 Hz at 50 Hz), absent on the walker.

### Stance offset

Meta `stand_offset_abd_deg` D shifts a policy's abduction stand outward by D degrees. Outward is + in `rel` on FR and RL, − on FL
and RR. The policy's rel 0 is the calibrated stand plus the offset: its position observation is the measured `rel` minus the offset,
and its targets are its `rel` plus the offset. The band and the step limit act in the policy's coordinates, the bounds in the
calibrated stand's.

### Joint bounds

Meta `bounds` holds per servo a lower and an upper target, in the model's coordinates (contract `stand_policy_rad` + `rel`). From the
stand, in degrees:

| Servo | Walker | Get-up |
|---|---|---|
| FR abduction | −18.6 to 20.0 | −19.6 to 21.3 |
| FL abduction | −20.0 to 10.8 | −23.9 to 11.8 |
| RR abduction | −20.0 to 10.8 | −21.6 to 11.8 |
| RL abduction | −13.9 to 20.0 | −14.9 to 26.2 |
| FR hip | −142.7 to 127.3 | same |
| FL hip | −98.7 to 171.3 | same |
| RR hip | −159.7 to 110.3 | same |
| RL hip | −141.7 to 128.3 | same |
| FR knee | −151.3 to 118.7 | same |
| FL knee | −92.7 to 177.3 | same |
| RR knee | −131.3 to 138.7 | same |
| RL knee | −153.7 to 116.3 | same |

- **Abductions:** one unit's measured range, less a margin (`limits.py`): 3° for the walker, capped at 20° outward, and 2° for the get-up.
- **Hips and knees:** 20° inside the potentiometer's track ends (0 and 3100 dd), since positions in the dead zone beyond them can't
  be read. These depend on the calibration: a recalibrated robot needs a re-export.
- **Knee coupling:** the knee servo is also clamped so that the knee-thigh angle (the `*_calf` joint) stays inside `calf_range` given the
  hip's target, through `tendon_coefs`. A track end wins over the coupling where they disagree.
- **Hip envelope** (`hip_envelope`, get-up only): per leg the hip target's range as a piecewise-linear function of the abduction target,
  since near either abduction extreme the chassis stops the thigh swinging towards the body's middle.

`bounds.info.contract_stand_dd` records the stand the bounds were made for.

## Meta

JSON in `policy.npz`. The runner reads:

| Key | Meaning |
|---|---|
| `kind` | `walk` or `getup`; absent means `walk` |
| `contract_sha256` | SHA-256 of the contract file the policy was exported for |
| `omni` | the policy observes the whole command; required on the walker |
| `history` | frames observed; 0 and 1 both mean one frame |
| `band_deg` | the 12 bands; absent means 20 on abductions, 60 on hips and knees |
| `bounds` | the decoder's clamps |
| `stand_offset_abd_deg` | the stance offset; absent means 0 |
| `action_filter_alpha` | the target filter; absent means 0 |
| `getup_phase_s` | the phase clock's length; required on the get-up |

`limits`, `activation`, `obs`, `hidden`, `obs_normalizer`, `privileged`, `tested`, `checkpoint_sha256` and `source_policy_sha256` describe where the
policy came from; the runner doesn't need them. No value is a path on anyone's machine.

## Checking a policy

```sh
python -m mp4_recovery.deploy.runner check --policy policies/getup
```

Needs no robot and no port. It checks:

- **Fixtures:** `fixture_obs` through the numpy forward pass against `fixture_act`, within 1e-5 × the largest output (at least 1e-5), under this machine's numpy. Run it on
  the Pi too: `tools/deploy.sh` does.
- **Contract:** `contract_sha256` is the contract file's hash, or that of a calibration the contract lists under
  `calibrations_superseded` (`tools/make_contract.py` adds the one it replaces).
- **Inputs:** the first layer's width is `45 × max(history, 1)`, and the policy observes no term the runner doesn't feed.
- **Band and bounds:** the band is at most 90° (180° for a get-up policy with bounds), a walker's at most the walking band, and
  `contract_stand_dd` equals the contract's stand.
- **Decoder self-test:** 400 steps of saturated and random actions, from extreme measured poses, give targets inside the bounds and at
  least 15° from the track ends.

`runner run` repeats the band, layout and self-test checks before arming, and checks each target before sending it.

`tools/make_contract.py` writes a contract for a recalibrated robot, and `export_policy --from-policy` writes a policy's
bounds for it: see [ROBOT.md](ROBOT.md#adapting-to-another-unit).
