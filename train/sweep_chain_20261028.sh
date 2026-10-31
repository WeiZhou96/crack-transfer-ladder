#!/bin/bash
# addendum 4: threshold sweep for primary (after SDNET + held-out probe), ablation subset (after its chain) and
# addendum-3 models (after their chain). Uses each eval dir's sealed thresholds.
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); D=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}
cd $C
sw() { E=$1; M=$2; g=$3; shift 3; for m in "$@"; do python -X utf8 train/eval_sweep_v1.py sweep --models $M --out $E --gpu $g --only $m --workers 12 >> $E/sweep_gpu$g.log 2>&1 || echo "FAIL $m" >> $E/sweep_gpu$g.log; done; }
run_all() { E=$1; M=$2; mapfile -t L < <(python -c "import json;print('\n'.join(json.load(open('$M'))))"); n=${#L[@]}
  G=(0 1 2 3 6); for i in 0 1 2 3 4; do part=(); for ((j=i; j<n; j+=5)); do part+=("${L[$j]}"); done; sw $E $M ${G[$i]} "${part[@]}" & done; wait; }
P=$D/students_eval_v1_20261028; A=$D/students_eval_ablation_v1_20261028; X=$D/students_eval_add3_v1
until [ -f $P/HELDOUT_DONE ]; do sleep 120; done
run_all $P train/eval_models_primary_v1.json; echo done > $P/SWEEP_DONE
until [ -f $A/HELDOUT_DONE ]; do sleep 300; done
run_all $A train/eval_models_sweep_ablation_v1.json; echo done > $A/SWEEP_DONE
until [ -f $X/ALL_DONE ]; do sleep 300; done
run_all $X train/eval_models_add3_v1.json; echo done > $X/SWEEP_DONE
