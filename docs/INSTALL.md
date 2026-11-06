# Installing

## Requirements

- [uv](https://docs.astral.sh/uv/) 0.12.7 or newer (a release that knows the
  `cu130` torch backend). uv installs Python 3.12 if it is missing.
- GPU mode: an NVIDIA GPU with driver 580 or newer (torch is the CUDA 13.0
  build). The CUDA toolkit (`nvcc`) is unnecessary; torch ships its own CUDA
  runtime.
- CPU mode on macOS: macOS 14 or newer on Apple silicon. torch 2.14 has no
  wheel for Intel Macs.

## Modes

| Mode | For | Installs |
|---|---|---|
| `gpu` | Machines with an NVIDIA GPU | mjlab, MuJoCo Warp, warp-lang, rsl_rl, CUDA 13.0 torch, plus everything in `cpu` |
| `cpu` | Any machine, including macOS and laptops without CUDA | numpy, scipy, pyserial, CPU torch, pytest |

Training and evaluation need `gpu`. `cpu` runs the policy checks, the deployment
code and `tests/cpu`.

`auto` (the default) picks `gpu` when `nvidia-smi` is on `PATH`, and picks
`cpu` on macOS.

## Running the installer

```bash
bash scripts/install/install.sh [--mode auto|gpu|cpu] [--dry-run] [--skip-verify]
```

It:

1. Creates `.venv` in the project root with Python 3.12.
2. Installs `requirements/<mode>.txt` under `requirements/constraints.txt`,
   choosing the torch build with uv's `--torch-backend` (`cu130` or `cpu`;
   on macOS, PyPI's torch).
3. Installs this package in editable mode. `pyproject.toml` lists only the
   packages both modes share; backend packages and all pins come from
   `requirements/`.
4. In `gpu` mode, patches MuJoCo Warp (`python -m mp4_recovery.gpu.patch_warp`,
   see below).
5. Imports every key package and reports whether torch sees CUDA. In `gpu` mode
   the check fails if CUDA is unavailable or MuJoCo Warp is not patched.

Use `--venv` for a different location. Re-running is safe; the venv is reused.

## The MuJoCo Warp patch

When a geom touches more than 50 heightfield cells (a fallen body on fine
terrain), mujoco-warp 3.7.0.1 skips the extra contacts and prints a warning
from the GPU kernel, for every geom pair and every physics substep. With
thousands of robots on rough ground that slows a run several-fold and fills the
disk. `python -m mp4_recovery.gpu.patch_warp` removes only the warning; the
contacts are handled as before. The feet never reach the limit. Reinstalling
mujoco-warp undoes the patch: run the command again. `--check` only reports.

## Pins and why

All versions live in `requirements/constraints.txt`. The shipped policies were
trained and evaluated on this set, and the fitted servo models were identified
on the MuJoCo 3.7 line. Change a pin only together with a re-evaluation.

| Package | Version | Reason |
|---|---|---|
| Python | 3.12 | Required by mjlab and the actuator code |
| `mujoco` | 3.7.0 | The MuJoCo line the servo models were fitted on and the policies trained on |
| `mujoco-warp` | 3.7.0.1 | Matches `mujoco` 3.7.0; the version `patch_warp` patches |
| `warp-lang` | 1.12.1 | Version the policies were trained with; mjlab requires >= 1.12.0 |
| `mjlab` | 1.3.0 | Training framework |
| `rsl-rl-lib` | 5.0.1 | Pinned by mjlab 1.3.0; `gpu/algorithms.py` extends its PPO |
| `torch` | 2.14.0 | Version the policies were trained with; CUDA 13.0 wheels (`--torch-backend cu130`) |

The remaining pins (numpy, scipy, tensordict, tensorboard, protobuf) are the
versions the set resolves with.

## Known risks

- **Headless Linux** needs EGL for rendering (videos from `eval/getup_eval --video`):
  `export MUJOCO_GL=egl`.
- **Weights & Biases** is pulled in by mjlab. Training logs to TensorBoard; set
  `WANDB_MODE=disabled` when running offline.

## Tests

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/cpu   # no GPU, CPU requirements only
pytest tests/gpu                           # needs the GPU requirements and at least 2 GB of GPU memory
```
