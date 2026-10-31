#!/bin/bash
# addendum 5 part A: scene-novel synthetic probe for primary (now), ablation/post-hoc and addendum-3 models (after their chains).
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); D=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}; NQ=$D/quadruplets/sdnet_scene_novel_s20261030
cd $C
nq() { E=$1; M=$2; g=$3; shift 3; for m in "$@"; do python -X utf8 train/eval_strata_v4.py quads --quad-dir $NQ --models $M --eval-dir $E --out $E/strata --gpu $g --only $m >> $E/novel_gpu$g.log 2>&1 || echo "FAIL $m" >> $E/novel_gpu$g.log; done; }
run_all() { E=$1; M=$2; mapfile -t L < <(python -c "import json;print('\n'.join(json.load(open('$M'))))"); n=${#L[@]}
  G=(0 1 2 3 6); for i in 0 1 2 3 4; do part=(); for ((j=i; j<n; j+=5)); do part+=("${L[$j]}"); done; nq $E $M ${G[$i]} "${part[@]}" & done; wait; }
P=$D/students_eval_v1_20261028; A=$D/students_eval_ablation_v1_20261028; X=$D/students_eval_add3_v1
[ -f $NQ/render_summary.json ] || { echo "probe not rendered"; exit 1; }
run_all $P train/eval_models_v1.json; echo done > $P/NOVEL_DONE
until [ -f $A/ALL_DONE ]; do sleep 300; done
run_all $A train/eval_models_ablation_v1.json; echo done > $A/NOVEL_DONE
until [ -f $X/ALL_DONE ]; do sleep 300; done
run_all $X train/eval_models_add3_v1.json; echo done > $X/NOVEL_DONE
