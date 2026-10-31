"""Fit tau_delta from observer CSVs (v6: psychometric curve, threshold at 50% detection).
Per stratum, pooled over observers: P(yes | dC) = g + (1 - g) * F(log dC), F logistic; g = false-alarm rate on blank trials.
tau_delta = dC at F = 0.5 (guess-corrected 50%). 95% interval by bootstrap over observers and their trials (2,000).
usage: python visibility_calibration_fit_v1.py <manifest.csv> <out.json> obs1.csv [obs2.csv ...]
"""
import csv
import json
import sys

import numpy as np
from scipy.optimize import minimize

key = {r["trial_id"]: r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8"))}
responses = [r for f in sys.argv[3:] for r in csv.DictReader(open(f, encoding="utf-8-sig"))]


def fit(x, y, gamma):
    lx = np.log(x)

    def nll(p):
        pr = gamma + (1 - gamma) / (1 + np.exp(-(lx - p[0]) * np.exp(p[1])))
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    return minimize(nll, [np.median(lx), 0.0], method="Nelder-Mead").x


result = {}
for stratum in sorted({k["stratum"] for k in key.values()}):
    rows = [(key[r["trial_id"]], int(r["response_crack"]), r["observer"])
            for r in responses if key[r["trial_id"]]["stratum"] == stratum]
    blanks = [y for k, y, _ in rows if k["is_blank"] == "True"]
    gamma = float(np.mean(blanks)) if blanks else 0.0
    crack = [(max(float(k["delta_c_median"]), 1e-4), y, o) for k, y, o in rows if k["is_blank"] != "True"]
    x = np.array([c[0] for c in crack])
    y = np.array([c[1] for c in crack])
    obs = np.array([c[2] for c in crack])
    mu, _ = fit(x, y, gamma)
    rng = np.random.default_rng(0)
    observers = np.unique(obs)
    boots = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(np.where(obs == o)[0], int((obs == o).sum()))
                              for o in rng.choice(observers, len(observers))])
        boots.append(fit(x[idx], y[idx], gamma)[0])
    result[stratum] = {"observers": int(len(observers)), "crack_trials": int(len(x)),
                       "false_alarm_rate": round(gamma, 4), "tau_delta_50": round(float(np.exp(mu)), 5),
                       "ci95": [round(float(np.exp(np.percentile(boots, 2.5))), 5),
                                round(float(np.exp(np.percentile(boots, 97.5))), 5)]}
json.dump(result, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
print(json.dumps(result, indent=1))
