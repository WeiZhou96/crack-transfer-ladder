#!/bin/bash
# Addendum 7 analysis (post hoc), after students_eval_add7_v1/SWEEP_DONE. Same estimators as addenda 3 and 6.
export OMP_NUM_THREADS=8
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); T=$C/train; D=${CRACKCF_DATA:?set CRACKCF_DATA to the data root}
O=$D/students_eval_add7_v1; P=$D/supp_posthoc_v1
SP="$D/students_eval_v1_20261028=$T/eval_models_v1.json $D/students_eval_ablation_v1_20261028=$T/eval_models_ablation_v1.json $D/students_eval_add3_v1=$T/eval_models_add3_v1.json $D/students_eval_add6_v1=$T/eval_models_add6_v1.json $O=$T/eval_models_add7_v1.json"
cd $C

# 1. program report: frozen analyze_program_v2 with the addendum-7 contrasts appended (estimator unchanged)
sed 's/^           ("R50_B11", "R50_B0"), ("R50_B2", "R50_B0"), ("R50_B0", "B0")]$/&\
COMPARE += [("A_x00only", "B0"), ("A_x00only", "A_x01only"), ("A_x00only", "B2"), ("A_x00only", "B0_v6"),\
            ("B2_B1", "B1_v5"), ("B2_B1", "B0"), ("B2_B1", "B2"), ("B2_B1", "B1"), ("B2_B1", "B0_v6"),\
            ("B1_v5", "B0"), ("A_x01only_B1", "B1_v5")]   # addendum 7 (post hoc)/' $T/analyze_program_v2.py > $T/analyze_program_v2_add7.py
grep -c "addendum 7 (post hoc)" $T/analyze_program_v2_add7.py
diff $T/analyze_program_v2.py $T/analyze_program_v2_add7.py
sha256sum $T/analyze_program_v2_add7.py $T/analyze_add7_v1.py $0 >> $P/add7_registration_sha256.txt
python -X utf8 $T/analyze_program_v2_add7.py $O/program_report_all_add7 $SP > $O/program_report_all_add7.log 2>&1 || echo "FAIL program"
python -X utf8 $T/analyze_add7_v1.py $O/program_report_all_add7/program_report.json $O/add7_verdicts.json || echo "FAIL verdicts"

# 2. scene-novel probe (rung 3)
python -X utf8 $T/analyze_probe_v1.py $O/probe_report_scene_novel_all_add7 $D/quadruplets/sdnet_scene_novel_s20261030 \
  $D/students_eval_v1_20261028=$T/eval_models_primary_v1.json $D/students_eval_ablation_v1_20261028=$T/eval_models_ablation_v1.json \
  $D/students_eval_add6_v1=$T/eval_models_add6_v1.json $O=$T/eval_models_add7_v1.json > $O/probe_add7.log 2>&1 || echo "FAIL probe"

# 3. threshold sweep and matched recall (primary + add6 + add7)
python -X utf8 $T/analyze_sweep_v1.py $O/sweep_report_primary_add6_add7 $D/students_eval_v1_20261028=$T/eval_models_primary_v1.json \
  $D/students_eval_add6_v1=$T/eval_models_add6_v1.json $O=$T/eval_models_add7_v1.json > $O/sweep_add7.log 2>&1 || echo "FAIL sweep"
python -X utf8 $T/analyze_sweep_overlap_v1.py $O/sweep_report_primary_add6_add7/sweep_report.json \
  $T/eval_models_primary_v1.json,$T/eval_models_add6_v1.json,$T/eval_models_add7_v1.json \
  $O/sweep_report_primary_add6_add7/sweep_overlap.json >> $O/sweep_add7.log 2>&1 || echo "FAIL overlap"

# 4. distractor-specific ladder, alternative accounts, two-level bootstrap (independent-review script, unchanged)
cd $P
python -X utf8 review_reanalysis_v1.py review_reanalysis_add7.json $SP > reanalysis_add7.log 2>&1 || echo "FAIL reanalysis"

# 5. in-domain anchor recall for the six new models (calibration tiles only)
python -X utf8 anchor_recall_v1.py anchor_recall --gpu 0 $SP --only A_x00only_s0 A_x00only_s1 A_x00only_s2 > anchor_add7_gpu0.log 2>&1 &
python -X utf8 anchor_recall_v1.py anchor_recall --gpu 5 $SP --only B2_B1_s0 B2_B1_s1 B2_B1_s2 > anchor_add7_gpu5.log 2>&1 &
wait
echo done > $O/ANALYSIS_DONE
