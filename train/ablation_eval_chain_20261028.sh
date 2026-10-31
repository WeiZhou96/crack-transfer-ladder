#!/bin/bash
# wait for E4 + post-hoc training -> dev thresholds -> seal -> test legs -> strata (v2) -> SDNET, for the 23 models.
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); D=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}; O=$D/students_eval_ablation_v1_20261028
M=train/eval_models_ablation_v1.json
cd $C; mkdir -p $O
until [ -f $D/students_v5/E4_DONE.json ] && [ -f $D/students_v5_posthoc/POSTHOC_DONE.json ]; do sleep 300; done
Q0="A_x01only_s0 A_x01only_s1 A_x01only_s2 A_paired_pair_s0 B12_auto_posthoc_s0"
Q1="A_x1only_s0 A_x1only_s1 A_x1only_s2 A_paired_marg_s0 B12_auto_posthoc_s1"
Q2="A_paired_noaux_s0 A_paired_noaux_s1 A_paired_noaux_s2 B11_taulow_s0 B12_auto_posthoc_s2"
Q3="B11_tauhigh_s0 B11_v3quads_s0 B2_g7_s0 B2_g7_s1"
Q6="B2_g7_s2 B2_g2_s0 B2_g2_s1 B2_g2_s2"
stage() { st=$1; g=$2; shift 2; for m in "$@"; do python -X utf8 train/eval_students_v1.py $st --models $M --out $O --gpu $g --only $m --workers 12 >> $O/${st}_gpu$g.log 2>&1 || echo "FAIL $m" >> $O/${st}_gpu$g.log; done; }
for st in dev; do stage $st 0 $Q0 & stage $st 1 $Q1 & stage $st 2 $Q2 & stage $st 3 $Q3 & stage $st 6 $Q6 & wait; done
python -X utf8 train/seal_thresholds_v1.py $O $M > $O/seal.log 2>&1 || { echo SEAL_FAIL > $O/CHAIN_FAIL; exit 1; }
stage test 0 $Q0 & stage test 1 $Q1 & stage test 2 $Q2 & stage test 3 $Q3 & stage test 6 $Q6 & wait
strata() { g=$1; shift; for m in "$@"; do for st in quads hardproxy dacl10k; do python -X utf8 train/eval_strata_v2.py $st --models $M --eval-dir $O --out $O/strata --gpu $g --only $m >> $O/strata_gpu$g.log 2>&1 || echo "FAIL $st $m" >> $O/strata_gpu$g.log; done; python -X utf8 train/eval_sdnet_v1.py --models $M --sealed $O/dev_thresholds_sealed.json --out $O/sdnet --gpu $g --only $m >> $O/sdnet_gpu$g.log 2>&1 || echo "FAIL sdnet $m" >> $O/sdnet_gpu$g.log; done; }
strata 0 $Q0 & strata 1 $Q1 & strata 2 $Q2 & strata 3 $Q3 & strata 6 $Q6 & wait
echo done > $O/ALL_DONE
