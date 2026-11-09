#!/bin/bash
# usage:  ./run_dashboard.sh        -> dashboard.py
#         ./run_dashboard.sh new    -> dashboard_new.py
cd ~/Documents/pi4
source .venv/bin/activate
export DISPLAY=:0
SCRIPT=scripts/dashboard.py
if [ "$1" = "new" ]; then SCRIPT=scripts/dashboard_new.py; shift; fi
pkill -f "scripts/dashboard" 2>/dev/null
pkill -f "scripts/live_pick" 2>/dev/null
pkill -f "scripts/basler_live" 2>/dev/null
sleep 1
sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
echo "starting $SCRIPT - start the RAPID program within 30 s"
python "$SCRIPT" --camera --send --host 192.168.125.205 "$@"
