#!/bin/sh
# Train the get-up policy from the searched open-loop motions: a supervised clone of them, then seven PPO steps.
#
#   sh scripts/recipes/getup.sh [SEED]            # SEED defaults to 41; output in runs/getup/s$SEED
#
# Run from the repository root. Each step fine-tunes the previous step's last checkpoint; a finished step is skipped when
# the script runs again. Environment: PYTHON (default .venv/bin/python), OUT (replaces the output folder), START (a shipped
# checkpoint, skipping the steps before it: checkpoints/clone.pt starts at `wide`, wide_1499.pt at `limits`, smooth3_3947.pt
# at `smooth4`).
# The outcome depends on the random seed; compare runs with the sequence evaluation (mp4_recovery.eval.sequence).
#
# Some steps have identical options (smooth1 and smooth2, smooth3 and smooth4): they are only restarts at those iteration
# counts, which reset the critic warm-up, the normaliser freeze and the optimiser. They stay separate because this is what
# made the policy.
set -eu
SEED=${1:-41}
PY="env -u PYTHONPATH WANDB_MODE=disabled ${PYTHON:-.venv/bin/python} -B"
OUT=${OUT:-runs/getup/s$SEED}
mkdir -p "$OUT"
COMMON="--limits --getup-phase 4 --getup-band 90 180 180 --getup-episode-s 8 --std-bounds 0.03 0.3 \
 --init-std 0.03 --entropy-coef 0 --getup-success-gyro 0 --getup-end-on-success --getup-midfall 0.2 \
 --getup-mix 0.10 0.10 0.30 0.20 0.20 0.10 0.10 0.10 0.30 0.20 0.20 0.10 --actor-hidden 512 256 128 \
 --critic-hidden 512 256 128 --obs-norm --obs-norm-freeze 100 --lr0 1e-5 --seed $SEED"
# Hip and knee targets 60 degrees past their track ends: trained inside the limits, it never learns to flip from the back.
WIDE="--widen-limits-deg 60"
COSTS="--getup-action-rate=-0.1 --getup-action-rate-tilted 1.0 --getup-joint-vel=-5e-4"
RAMP="--getup-filter 0.75 --getup-filter-start 0.5 --getup-filter-ramp 200"
FILTER="--getup-filter 0.75 --getup-filter-start 0.75 --getup-filter-ramp 0"
LR="--lr-max 5e-5"
SMOOTH_A="--caps-temporal 1 --caps-spatial 20 --caps-obs-noise 0.05"
SMOOTH_B="--caps-temporal 2 --caps-spatial 40 --caps-obs-noise 0.05"
S=data/searched_motions

last_ckpt() {  # the highest-numbered checkpoint of the newest run with this name
  run=$(ls -d "$OUT"/*_"$1" 2>/dev/null | tail -n 1)
  [ -n "$run" ] && ls -v "$run"/model_*.pt | tail -n 1
}
step() {  # name, iterations, critic warm-up, then the options
  name=$1 iters=$2 warm=$3; shift 3
  if [ -n "$SKIP_TO" ]; then
    [ "$name" = "$SKIP_TO" ] || return 0
    SKIP_TO=
  fi
  if [ -z "$(last_ckpt "$name")" ] || [ ! -f "$OUT/$name.done" ]; then
    $PY -m mp4_recovery.gpu.train $COMMON --critic-warmup "$warm" --envs 2048 --iters "$iters" --out "$OUT" --tag "$name" "$@"
    touch "$OUT/$name.done"
  fi
  PREV=$(last_ckpt "$name")
}

SKIP_TO=
case "${START:-}" in
  "") ;;
  *clone*.pt) SKIP_TO=wide ;;
  *wide*.pt) SKIP_TO=limits ;;
  *smooth3*.pt) SKIP_TO=smooth4 ;;
  *) echo "START: clone, wide or smooth3 checkpoint expected, got $START" >&2; exit 2 ;;
esac
PREV=${START:-}
if [ -z "${START:-}" ] && [ ! -f "$OUT/clone/model_bc.pt" ]; then
  $PY -m mp4_recovery.gpu.train $COMMON $WIDE --critic-warmup 50 --envs 64 --iters 1 --out "$OUT/cfg" --tag cfg
  $PY -m mp4_recovery.gpu.getup_bc --seq left=$S/search_left.json --seq nose=$S/search_nose.json \
    --seq right=$S/search_right_a.json --seq right=$S/search_right_b.json --seq back=$S/search_back_a.json \
    --seq back=$S/search_back_b.json --config "$(ls -d "$OUT"/cfg/*_cfg | tail -n 1)/config.json" \
    --hidden 512 256 128 --obs-norm --prev-action-noise 0.3 --robots 256 --rounds 4 --epochs 300 \
    --eval-extra midfall --eval-robots 64 --seed "$SEED" --out "$OUT/clone"
fi
step wide 1500 50 --init-actor "${START:-$OUT/clone/model_bc.pt}" $WIDE
step limits 1000 50 --resume "$PREV" $COSTS
step filter 253 30 --resume "$PREV" $COSTS $RAMP $LR
step smooth1 400 30 --resume "$PREV" $COSTS $FILTER $LR $SMOOTH_A
step smooth2 200 30 --resume "$PREV" $COSTS $FILTER $LR $SMOOTH_A
step smooth3 600 30 --resume "$PREV" $COSTS $FILTER $LR $SMOOTH_B
step smooth4 600 30 --resume "$PREV" $COSTS $FILTER $LR $SMOOTH_B
echo "LAST $PREV"
