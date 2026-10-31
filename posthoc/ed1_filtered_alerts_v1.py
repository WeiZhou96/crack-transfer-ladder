"""Post hoc (independent review, item A13): image-level alert rates of the published models on the same footing as
the component account. For each of the seven main-run models and the three negative legs, at the model's sealed
calibration threshold (t_proxy_r95 in results.csv): unfiltered alert (any pixel >= t, as in the original account) and
filtered alert (at least one component counted by the prereg filter). Reads the stored probability PNGs only.
usage: python ed1_filtered_alerts_v1.py <out.json>
"""
import os
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C / "poc"))
from common import binarize_prediction, load_probability_png, negative_image_accounts  # noqa: E402

D = Path(os.environ.get("CRACKCF_DATA", "data"))
PRED = D / "ed1_pred_v2"
LEGS = {"bcl_noise": "neg_bcl_noise.csv", "pavedistress_test": "neg_pavedistress_test.csv",
        "uav1388_background": "neg_uav1388_background.csv"}
MODELS = ["deepcrack_cracktree260", "crackformer2_crack537", "ct_crackseg_deepcrack", "scsegamba_tut",
          "crackmamba_crackseg9k", "deepcrackliu_deepcrack537", "cracksam_adapter_khanhha"]


def thresholds(leg):
    with open(D / "ed1_legs_v3/main" / leg / "results.csv", encoding="utf-8-sig") as f:
        return {r["model"]: float(r["t_proxy_r95"]) for r in csv.DictReader(f)}


def job(a):
    model, image_id, t = a
    p = PRED / model / (image_id.replace("/", "__") + ".png")
    b = binarize_prediction(load_probability_png(p), t)
    acc = negative_image_accounts(b, min_area_px=64, min_skel_px=20)
    return int(acc["image_fired"]), int(acc["fp_components"] > 0)


if __name__ == "__main__":
    out = {}
    with Pool(24) as pool:
        for leg, fn in LEGS.items():
            with open(D / "ed1_inputs_v2" / fn, encoding="utf-8-sig") as f:
                ids = [r["image_id"] for r in csv.DictReader(f)]
            th = thresholds(leg)
            for m in MODELS:
                res = pool.map(job, [(m, i, th[m]) for i in ids], chunksize=64)
                n = len(res)
                out.setdefault(leg, {})[m] = {"threshold": th[m], "n_images": n,
                                              "alert_unfiltered": sum(r[0] for r in res) / n,
                                              "alert_filtered": sum(r[1] for r in res) / n}
                print(leg, m, out[leg][m], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
