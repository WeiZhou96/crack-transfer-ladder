#!/bin/bash
# usage: run_eval_stage_v1.sh dev|test ; runs eval_students_v1 per GPU queue with bounded CPU threads.
STAGE=$1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); O=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}/students_eval_v1_20261028
cd $C
run() { g=$1; shift; for m in "$@"; do python -X utf8 train/eval_students_v1.py $STAGE --models train/eval_models_v1.json --out $O --gpu $g --only $m --workers 12 >> $O/${STAGE}_gpu$g.log 2>&1 || echo "FAIL $m" >> $O/${STAGE}_gpu$g.log; done; }
run 0 B0_s0 B0_s1 B0_s2 B1_s0 B1_s1 B1_s2 &
run 1 B2_s0 B2_s1 B2_s2 B3_auto_s0 B3_auto_s1 B3_auto_s2 &
run 2 B10_s0 B10_s1 B10_s2 B11_s0 B11_s1 B11_s2 &
run 3 B1legacy_s0 B1legacy_s1 B1legacy_s2 B2matched_s0 &
run 6 B2matched_s1 B2matched_s2 B3matched_s0 B3matched_s1 B3matched_s2 &
wait
echo done > $O/${STAGE^^}_DONE
