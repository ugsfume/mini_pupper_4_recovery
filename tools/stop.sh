#!/bin/sh
# Stop whatever is using the servo link on the Pi, from any terminal: SIGINT to the process holding the port, which goes limp
# (the body drops about 7 cm) and saves its log.
#
#   tools/stop.sh HOST [PORT]          PORT: default /dev/ttyAMA0
#
# Backstops: the ESP32 idles the servos 100 ms after the last frame, and the servo power switch cuts everything.
[ $# -ge 1 ] || { sed -n 2,6p "$0"; exit 2; }
PORT=${2:-/dev/ttyAMA0}
exec ssh -o ConnectTimeout=5 -o BatchMode=yes "$1" "fuser -s -k -INT '$PORT' && echo 'stop sent' || echo 'nothing running'"
