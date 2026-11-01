#!/bin/bash
# In-domain recall at every sweep threshold for all arms with false-alarm sweeps (post hoc, pre-submission review).
export OMP_NUM_THREADS=4
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); D=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}
O=$D/anchor_recall_sweep; cd $C
SP="$D/students_eval_v1_20261028=train/eval_models_v1.json $D/students_eval_ablation_v1_20261028=train/eval_models_ablation_v1.json $D/students_eval_add3_v1=train/eval_models_add3_v1.json $D/students_eval_add6_v1=train/eval_models_add6_v1.json $D/students_eval_add7_v1=train/eval_models_add7_v1.json"
L=("" "" "" ""); i=0
for a in B0 B11 B2 B10 B1 B3_auto B0_v6 R50_B0 R50_B2 R50_B11 A_x01only A_x1only A_paired_noaux B2_g7 B12_auto_posthoc B1_v5 A_x01only_B1 A_x00only B2_B1; do for s in 0 1 2; do
  L[$((i%4))]="${L[$((i%4))]} ${a}_s$s"; i=$((i+1)); done; done
G=(5 5 6 7)
for k in 0 1 2 3; do
  python -X utf8 posthoc/anchor_recall_sweep_v1.py $O $SP --gpu ${G[$k]} --only ${L[$k]} > $O.p$k.log 2>&1 &
done
wait
python -X utf8 posthoc/analyze_indomain_matched_v1.py $O
