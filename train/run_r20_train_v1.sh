#!/bin/bash
# R20 training (2026-09-28): waits for the R20 B3-auto mining, then B2 x3 on the admitted pool (accept + hard_negative,
# = "random public true negatives", v6 B2) and B3-auto x3 on the R20 mined crops. GPUs 0,1,6,7; GPU0/7 after the
# frozen-rule sensitivity runs (B2 s0, B3_auto s1) finish.
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); R=$C; D=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}
S=$D/students_v1; LOG=$D/r20_chain.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
waitfile() { until [ -f "$1" ]; do sleep 60; done; }
train() { # outname arm seed gpu neglist
  local name=$1 arm=$2 seed=$3 gpu=$4 neg=$5
  if [ -f $S/${name}_s${seed}/meta.json ]; then say "skip $name s$seed"; return; fi
  say "start $name s$seed gpu$gpu"
  python -X utf8 $C/train/student_v1.py $C $D $arm $seed $S/${name}_s${seed} --gpu $gpu --neg-list $neg > $S/${name}_s${seed}.log 2>&1
  say "end $name s$seed rc=$?"
}
A=$D/negpool_r20/admitted.csv; M=$D/b3auto_r20/b3auto_crops.csv
say "r20 training scheduler start (B2 negatives = admitted pool)"
waitfile $D/b3auto_r20/summary.json
say "mining summary present"
( train B2_r20 B2 0 1 $A; train B3_auto_r20 B3_auto 0 1 $M ) &
( train B2_r20 B2 1 6 $A; train B3_auto_r20 B3_auto 1 6 $M ) &
( waitfile $S/B2_s0/meta.json; train B2_r20 B2 2 0 $A ) &
( waitfile $S/B3_auto_s1/meta.json; train B3_auto_r20 B3_auto 2 7 $M ) &
wait
say "r20 training done"
