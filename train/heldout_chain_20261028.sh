#!/bin/bash
# addendum 2 held-out synthetic probe for the primary (after SDNET) and ablation/post-hoc (after their chain) models.
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); D=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}; HQ=$D/quadruplets/v4_heldout_s20261029
cd $C
hq() { E=$1; M=$2; g=$3; shift 3; for m in "$@"; do python -X utf8 train/eval_strata_v3.py quads --quad-dir $HQ --models $M --eval-dir $E --out $E/strata --gpu $g --only $m >> $E/heldout_gpu$g.log 2>&1 || echo "FAIL $m" >> $E/heldout_gpu$g.log; done; }
run_all() { E=$1; M=$2; mapfile -t L < <(python -c "import json;print('\n'.join(json.load(open('$M'))))"); n=${#L[@]}
  G=(0 1 2 3 6); for i in 0 1 2 3 4; do part=(); for ((j=i; j<n; j+=5)); do part+=("${L[$j]}"); done; hq $E $M ${G[$i]} "${part[@]}" & done; wait; }
P=$D/students_eval_v1_20261028; A=$D/students_eval_ablation_v1_20261028
until [ -f $P/SDNET_DONE ]; do sleep 120; done
run_all $P train/eval_models_v1.json; echo done > $P/HELDOUT_DONE
until [ -f $A/ALL_DONE ]; do sleep 300; done
run_all $A train/eval_models_ablation_v1.json; echo done > $A/HELDOUT_DONE
