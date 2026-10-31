#!/bin/bash
# after strata: analyze strata (E2/E3) -> SDNET confirmation leg for the 27 sealed P5 models.
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); O=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}/students_eval_v1_20261028
cd $C
until [ -f $O/STRATA_DONE ]; do sleep 60; done
python -X utf8 train/analyze_strata_v1.py $O train/eval_models_primary_v1.json $O/strata_report > $O/strata_report.log 2>&1 || echo STRATA_REPORT_FAIL >> $O/CHAIN_FAIL
sd() { g=$1; shift; for m in "$@"; do python -X utf8 train/eval_sdnet_v1.py --models train/eval_models_v1.json --sealed $O/dev_thresholds_sealed.json --out $O/sdnet --gpu $g --only $m >> $O/sdnet_gpu$g.log 2>&1 || echo "FAIL $m" >> $O/sdnet_gpu$g.log; done; }
sd 0 B0_s0 B0_s1 B0_s2 B1_s0 B1_s1 B1_s2 &
sd 1 B2_s0 B2_s1 B2_s2 B3_auto_s0 B3_auto_s1 B3_auto_s2 &
sd 2 B10_s0 B10_s1 B10_s2 B11_s0 B11_s1 B11_s2 &
sd 3 B1legacy_s0 B1legacy_s1 B1legacy_s2 B2matched_s0 &
sd 6 B2matched_s1 B2matched_s2 B3matched_s0 B3matched_s1 B3matched_s2 &
wait
echo done > $O/SDNET_DONE
