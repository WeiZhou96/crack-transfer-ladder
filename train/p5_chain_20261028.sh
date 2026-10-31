#!/bin/bash
# dev done -> seal -> P5 test (27 models) -> P5 report -> E2/E3 strata for the 18 primary models.
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); O=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}/students_eval_v1_20261028
cd $C
until [ -f $O/DEV_DONE ]; do sleep 60; done
python -X utf8 train/seal_thresholds_v1.py $O train/eval_models_v1.json > $O/seal.log 2>&1 || { echo SEAL_FAIL > $O/CHAIN_FAIL; exit 1; }
bash train/run_eval_stage_v1.sh test
python -X utf8 train/analyze_p5_v1.py $O train/eval_models_v1.json $O/p5_report > $O/p5_report.log 2>&1 || echo P5_REPORT_FAIL >> $O/CHAIN_FAIL
strata() { g=$1; shift; for m in "$@"; do for st in quads hardproxy dacl10k; do python -X utf8 train/eval_strata_v1.py $st --models train/eval_models_primary_v1.json --out $O/strata --gpu $g --only $m >> $O/strata_gpu$g.log 2>&1 || echo "FAIL $st $m" >> $O/strata_gpu$g.log; done; done; }
strata 0 B0_s0 B0_s1 B0_s2 B1_s0 &
strata 1 B1_s1 B1_s2 B2_s0 B2_s1 &
strata 2 B2_s2 B3_auto_s0 B3_auto_s1 B3_auto_s2 &
strata 3 B10_s0 B10_s1 B10_s2 &
strata 6 B11_s0 B11_s1 B11_s2 &
wait
echo done > $O/STRATA_DONE
