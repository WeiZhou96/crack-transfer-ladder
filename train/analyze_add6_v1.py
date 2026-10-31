"""Addendum 6: scoring of P15-P17 exactly as worded, from an analyze_program_v2 report (program_report.json) computed
over the primary eval dir (B1) and the addendum-6 eval dir (A_x01only_B1, B1_v5). native_r95, arm mean of 3 seeds.
Also lists the relative change of each addendum-6 arm vs B0 and vs B1_v5 on every A3 leg (point values; descriptive).
usage: python analyze_add6_v1.py <program_report.json> <out.json>
"""
import json
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
A = R["arm_A3"]
r = R["rates"]["positive_axis_recall"]["native_r95"]
LEGS = ["bcl_noise", "pavedistress_background_test", "uav1388_background"]


def g(leg, a):
    return A.get(leg, {}).get("native_r95", {}).get(a)


S, B5, B1 = "A_x01only_B1", "B1_v5", "B1"
pred = {}
v = {leg: [g(leg, S), g(leg, B5)] for leg in LEGS}
pred["P15_stack_false_alarms"] = {"values": v, "verdict": bool(all(x[0] <= x[1] for x in v.values()))}
v = [r.get(S), r.get(B5)]
pred["P16_stack_recall"] = {"values": v, "diff_pp": 100 * (v[0] - v[1]), "verdict": bool(v[0] >= v[1] - 0.02)}
v = {leg: [g(leg, B5), g(leg, B1)] for leg in LEGS}
pred["P17_B1_trainer_equivalence"] = {"values": v, "recall": [r.get(B5), r.get(B1)], "verdict": bool(
    all(abs(x[0] / x[1] - 1) <= 0.15 for x in v.values()) and abs(r[B5] - r[B1]) <= 0.02)}
desc = {}
for a in (S, B5):
    desc[a] = {}
    for leg in A:
        x, b0, b5 = g(leg, a), g(leg, "B0"), g(leg, B5)
        desc[a][leg] = {"A3": x, "vs_B0": None if not (x is not None and b0) else x / b0 - 1,
                        "vs_B1_v5": None if not (x is not None and b5) else x / b5 - 1}
    desc[a]["recall_pp_vs_B0"] = 100 * (r[a] - r["B0"])
json.dump({"predictions": pred, "descriptive": desc, "method": __doc__}, open(sys.argv[2], "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(json.dumps({k: v["verdict"] for k, v in pred.items()}, indent=1))
