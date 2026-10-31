"""Addendum 7 (post hoc): scoring of P18-P21 exactly as worded, from the analyze_program_v2_add7 report
(program_report.json over the primary, ablation, addendum-3, addendum-6 and addendum-7 eval dirs). native_r95, arm mean
of 3 seeds. Also lists the relative change of each addendum-7 arm vs B0, vs its controls, and recall differences
(point values and the report's paired cluster bootstrap CIs where the contrast exists; descriptive).
usage: python analyze_add7_v1.py <program_report.json> <out.json>
"""
import json
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
A = R["arm_A3"]
r = R["rates"]["positive_axis_recall"]["native_r95"]
OP = "native_r95"


def g(leg, a):
    return A.get(leg, {}).get(OP, {}).get(a)


def rel(leg, a, b):
    x, y = g(leg, a), g(leg, b)
    return None if x is None or not y else x / y - 1


pred = {}
v = {}
for leg in ("bcl_noise", "uav1388_background"):
    v[leg] = {"A_x00only_vs_B0": rel(leg, "A_x00only", "B0"), "A_x01only_vs_B0": rel(leg, "A_x01only", "B0")}
pred["P18_background_control"] = {"values": v, "verdict": bool(all(
    abs(x["A_x00only_vs_B0"]) < 0.5 * abs(x["A_x01only_vs_B0"]) for x in v.values()))}
x, y = g("pavedistress_background_test", "B2_B1"), g("pavedistress_background_test", "B1_v5")
pred["P19_stack_in_domain"] = {"values": [x, y], "ratio": x / y, "verdict": bool(x <= 0.2 * y)}
v = {leg: [g(leg, "B2_B1"), g(leg, "B1_v5")] for leg in ("bcl_noise", "uav1388_background")}
pred["P20_stack_cross_domain"] = {"values": v, "ratios": {k: a / b for k, (a, b) in v.items()},
                                  "verdict": bool(all(a <= 1.15 * b for a, b in v.values()))}
v = [r.get("B2_B1"), r.get("B1_v5")]
pred["P21_stack_recall"] = {"values": v, "diff_pp": 100 * (v[0] - v[1]), "verdict": bool(v[0] >= v[1] - 0.02)}

desc = {}
C = R["comparisons"]
RC = R["rate_comparisons"]["positive_axis_recall"][OP]
for pair in ("A_x00only_vs_B0", "A_x00only_vs_A_x01only", "A_x00only_vs_B2", "A_x00only_vs_B0_v6", "B2_B1_vs_B1_v5",
             "B2_B1_vs_B0", "B2_B1_vs_B2", "B2_B1_vs_B1", "B2_B1_vs_B0_v6", "B1_v5_vs_B0", "A_x01only_B1_vs_B1_v5"):
    desc[pair] = {leg: (None if pair not in C[leg][OP] else
                        {k: C[leg][OP][pair][k] for k in ("A3_a", "A3_b", "rel_change", "rel_change_ci95")})
                  for leg in C}
    desc[pair]["recall"] = RC.get(pair)
json.dump({"predictions": pred, "descriptive": desc, "method": __doc__}, open(sys.argv[2], "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(json.dumps({k: v["verdict"] for k, v in pred.items()}, indent=1))
