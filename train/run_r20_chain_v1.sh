#!/bin/bash
# R20 chain (2026-09-28, registered deviation): E3 screen with lane-marking exclusion -> B3-auto re-mining ->
# B2 / B3-auto x3 seeds retrained on the R20 pools. Frozen-rule runs B2 s0 (GPU0) and B3_auto s1 (GPU7) are left
# running as the sensitivity result. GPUs 0,1,6,7 only.
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
say "r20 chain start"
say "E3 screen (R20)"
python -X utf8 $C/train/negpool_screen_r20.py $C $D $D/negpool_preds_v1 --workers 32 > $D/negpool_r20.log 2>&1
say "screen rc=$?"
say "B3-auto mining (R20)"
python -X utf8 $C/train/b3auto_mine_v1.py $C $D $D/b3auto_r20 --b0 $S/B0_s0/final.pt --gpu 1 --negpool-dir negpool_r20 --r20 > $D/b3auto_r20.log 2>&1
say "mining rc=$?"
A=$D/negpool_r20/accept.csv; M=$D/b3auto_r20/b3auto_crops.csv
( train B2_r20 B2 0 1 $A; train B3_auto_r20 B3_auto 0 1 $M ) &
( train B2_r20 B2 1 6 $A; train B3_auto_r20 B3_auto 1 6 $M ) &
( waitfile $S/B2_s0/meta.json; train B2_r20 B2 2 0 $A ) &
( waitfile $S/B3_auto_s1/meta.json; train B3_auto_r20 B3_auto 2 7 $M ) &
wait
say "r20 chain done"
