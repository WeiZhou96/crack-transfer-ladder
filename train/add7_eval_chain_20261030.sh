#!/bin/bash
# addendum 7: wait for the 6 student_v5 control runs (copy of add6_eval_chain_20261029.sh, names changed) -> dev thresholds -> seal -> test legs -> strata (v4: quads, held-out
# quads, hardproxy, dacl10k) -> SDNET (v2) -> scene-novel probe -> threshold sweep, in a separate eval dir.
# Same evaluation path as addendum 3 (add3_eval_chain_20261028.sh) plus the addendum-4/5 stages.
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); D=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}; O=$D/students_eval_add7_v1
M=train/eval_models_add7_v1.json; HQ=$D/quadruplets/v4_heldout_s20261029; NQ=$D/quadruplets/sdnet_scene_novel_s20261030
cd $C; mkdir -p $O
until [ -f $D/students_v5_add7/ADD7_DONE.json ]; do sleep 300; done
Q0="A_x00only_s0 B2_B1_s2"; Q1="B2_B1_s0"; Q2="A_x00only_s1"; Q3="B2_B1_s1"; Q6="A_x00only_s2"
stage() { st=$1; g=$2; shift 2; for m in "$@"; do python -X utf8 train/eval_students_v2.py $st --models $M --out $O --gpu $g --only $m --workers 12 >> $O/${st}_gpu$g.log 2>&1 || echo "FAIL $m" >> $O/${st}_gpu$g.log; done; }
stage dev 0 $Q0 & stage dev 1 $Q1 & stage dev 2 $Q2 & stage dev 3 $Q3 & stage dev 6 $Q6 & wait
python -X utf8 train/seal_thresholds_v1.py $O $M > $O/seal.log 2>&1 || { echo SEAL_FAIL > $O/CHAIN_FAIL; exit 1; }
stage test 0 $Q0 & stage test 1 $Q1 & stage test 2 $Q2 & stage test 3 $Q3 & stage test 6 $Q6 & wait
strata() { g=$1; shift; for m in "$@"; do for st in quads hardproxy dacl10k; do python -X utf8 train/eval_strata_v4.py $st --models $M --eval-dir $O --out $O/strata --gpu $g --only $m >> $O/strata_gpu$g.log 2>&1 || echo "FAIL $st $m" >> $O/strata_gpu$g.log; done; python -X utf8 train/eval_strata_v4.py quads --quad-dir $HQ --models $M --eval-dir $O --out $O/strata --gpu $g --only $m >> $O/strata_gpu$g.log 2>&1 || echo "FAIL heldout $m" >> $O/strata_gpu$g.log; python -X utf8 train/eval_sdnet_v2.py --models $M --sealed $O/dev_thresholds_sealed.json --out $O/sdnet --gpu $g --only $m >> $O/sdnet_gpu$g.log 2>&1 || echo "FAIL sdnet $m" >> $O/sdnet_gpu$g.log; python -X utf8 train/eval_strata_v4.py quads --quad-dir $NQ --models $M --eval-dir $O --out $O/strata --gpu $g --only $m >> $O/novel_gpu$g.log 2>&1 || echo "FAIL novel $m" >> $O/novel_gpu$g.log; done; }
strata 0 $Q0 & strata 1 $Q1 & strata 2 $Q2 & strata 3 $Q3 & strata 6 $Q6 & wait
echo done > $O/ALL_DONE
sw() { g=$1; shift; for m in "$@"; do python -X utf8 train/eval_sweep_v1.py sweep --models $M --out $O --gpu $g --only $m --workers 12 >> $O/sweep_gpu$g.log 2>&1 || echo "FAIL $m" >> $O/sweep_gpu$g.log; done; }
sw 0 $Q0 & sw 1 $Q1 & sw 2 $Q2 & sw 3 $Q3 & sw 6 $Q6 & wait
echo done > $O/SWEEP_DONE
