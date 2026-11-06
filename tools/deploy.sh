#!/bin/sh
# Copy what the robot runs to its Raspberry Pi and check the policies there. The Pi needs python3, numpy and pyserial; nothing
# is installed.
#
#   tools/deploy.sh HOST FOLDER        HOST: an ssh host (user@pi); FOLDER: relative to the home directory, or absolute
#   tools/deploy.sh - FOLDER           copy into a local folder, to try the subset without a Pi
#
# Then, on the Pi:
#
#   cd FOLDER
#   R="env PYTHONPATH=src python3 -B -m mp4_recovery.deploy.runner"
#   $R run --vx 0 --seconds 40 --log logs/run1
#   $R run --vx 0 --seconds 40 --getup-policy policies/getup_sim --log logs/run2     # this get-up is tested in simulation only
set -eu
[ $# -eq 2 ] || { sed -n 2,13p "$0"; exit 2; }
HOST=$1
DIR=$2
cd "$(dirname "$0")/.."
FILES="./src/mp4_recovery/__init__.py ./src/mp4_recovery/contract.py ./src/mp4_recovery/policy_file.py
  ./src/mp4_recovery/heading.py ./src/mp4_recovery/paths.py ./src/mp4_recovery/deploy
  ./src/mp4_recovery/data/sku2_contract.json ./src/mp4_recovery/data/imu_mount.json
  ./policies/walker/policy.npz ./policies/getup/policy.npz ./policies/getup_sim/policy.npz
  ./tools/first_step_timing.py ./tests/cpu/fake_esp32.py"
[ -d runs/streams ] && FILES="$FILES ./runs/streams"       # what tools/getup_stream.py writes by default
CHECK='for P in policies/walker policies/getup policies/getup_sim; do env PYTHONPATH=src python3 -B -m mp4_recovery.deploy.runner check --policy $P || exit 1; done'
if [ "$HOST" = "-" ]; then
  mkdir -p "$DIR"
  rsync -a --relative --exclude __pycache__ $FILES "$DIR/"
  echo "copied to $DIR"
else
  ssh -o ConnectTimeout=10 "$HOST" mkdir -p "$DIR"
  rsync -a --relative --exclude __pycache__ $FILES "$HOST:$DIR/"
  ssh -o ConnectTimeout=10 "$HOST" "cd '$DIR' && $CHECK"
fi
