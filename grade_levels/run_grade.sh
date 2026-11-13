#!/bin/bash
# usage:  ./grade_levels/run_grade.sh 1     -> Grade level I
#         ./grade_levels/run_grade.sh 2     -> Grade level II
#         ./grade_levels/run_grade.sh 3     -> Grade level III
# extra options are passed on, e.g.  ./grade_levels/run_grade.sh 3 --imgsz 512
GRADE="$1"
case "$GRADE" in 1|2|3) shift ;; *) echo "usage: $0 1|2|3 [options]"; exit 1 ;; esac
cd ~/Documents/pi4
source .venv/bin/activate
pkill -f "scripts/dashboard" 2>/dev/null
pkill -f "grade_levels/grade" 2>/dev/null
pkill -f "scripts/live_pick" 2>/dev/null
pkill -f "scripts/basler_live" 2>/dev/null
pkill -f "scripts/web_view" 2>/dev/null
sleep 1
sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
echo "starting Grade level $GRADE - start the RAPID program within 30 s"
python "grade_levels/grade${GRADE}_web.py" --camera --send --host 192.168.125.205 "$@"
