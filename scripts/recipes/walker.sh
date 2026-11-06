#!/bin/sh
# Make the walker from its base weights: the limits and the stance offset are written into the policy's metadata, and
# the weights and fixtures are copied unchanged.
#
#   sh scripts/recipes/walker.sh [OUT]     # OUT defaults to runs/walker
#
# Run from the repository root. Environment: PYTHON (default .venv/bin/python), BASE (the policy directory holding the base
# weights, default policies/walker). The shipped walker is the only copy of those weights, so the default rebuilds it from
# itself; the abductions' stand moves 12 degrees outward, which needs no retraining.
set -eu
OUT=${1:-runs/walker}
env -u PYTHONPATH "${PYTHON:-.venv/bin/python}" -B -m mp4_recovery.export_policy --from-policy "${BASE:-policies/walker}" \
  "$OUT" --limits --stand-offset-abd 12
