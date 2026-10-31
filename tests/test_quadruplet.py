from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render import brush, quadruplet as Q  # noqa: E402
from render.operators import load_domains  # noqa: E402
from render.operators import acquisition  # noqa: E402

DOMAINS = load_domains(Path(__file__).resolve().parents[1] / "configs" / "operators.yaml")


def make_world(tmp: Path, material="concrete", side=128):
    """A source image with realistic texture and one dark crack + GT, and one crack-free background image; returns Resources."""
    rng = np.random.default_rng(0)
    h, w = 200, 240
    tex = 0.55 + 0.05 * rng.standard_normal((h, w))
    gt = np.zeros((h, w), np.uint8)
    for x in range(30, 210):
        y = int(100 + 25 * np.sin(x / 30.0))
        gt[y - 1:y + 2, x] = 1
    src = np.clip(tex * np.where(gt > 0, 0.45, 1.0), 0, 1)
    Image.fromarray((np.repeat(src[..., None], 3, 2) * 255).astype(np.uint8)).save(tmp / "src.png")
    Image.fromarray(gt * 255).save(tmp / "src_gt.png")
    ys, xs = np.nonzero(gt)
    y0, y1, x0, x1 = max(0, ys.min() - 8), min(h, ys.max() + 9), max(0, xs.min() - 8), min(w, xs.max() + 9)
    (tmp / "mask_library/v2/masks").mkdir(parents=True)
    Image.fromarray(gt[y0:y1, x0:x1] * 255).save(tmp / "mask_library/v2/masks/i1.png")
    inst = {"instance_id": "i1", "image_id": "src", "source_dataset": "synthetic", "material_pool": material,
            "source_path": str(tmp / "src.png"), "source_mask_path": str(tmp / "src_gt.png"),
            "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)], "skel_len_px": 180.0, "mask_crop": "masks/i1.png"}
    bg = np.clip(0.6 + 0.05 * rng.standard_normal((side, side)), 0, 1)
    Image.fromarray((np.repeat(bg[..., None], 3, 2) * 255).astype(np.uint8)).save(tmp / "bg.png")
    registry = {"bg": {"path": str(tmp / "bg.png")}}
    cand = {"crop_id": "bg#0", "image_id": "bg", "source_dataset": "synthetic", "group_id": "bg", "domain": "rgb_photo",
            "material": material, "surface": "", "x": "0", "y": "0", "side": str(side)}
    pools = {n: [] for n in Q.STRATA}
    pools["rgb_concrete" if material == "concrete" else "rgb_asphalt"].append(cand)
    lib = {"asphalt": [], "concrete": []}
    lib[material].append(inst)
    return Q.Resources(tmp, DOMAINS, registry, pools, lib, {"i1": inst})


class BrushTests(unittest.TestCase):
    def test_ratio_is_one_outside_band_and_dark_inside(self):
        with tempfile.TemporaryDirectory() as d:
            res = make_world(Path(d))
            f = res.brush_field("i1")
            band = f.ratio != 1.0
            self.assertTrue((f.mask.astype(bool) <= band).all() or (f.mask.astype(bool) & ~band).sum() == 0)
            self.assertLess(f.stats["ratio_median_in_mask"], 0.7)
            M, R = brush.place(f, 5, 3, -7, (128, 128))
            out, omega = brush.apply(np.full((128, 128, 3), 0.5, np.float32), R)
            self.assertTrue(np.array_equal(out[~omega], np.full((128, 128, 3), 0.5, np.float32)[~omega]))

    def test_bright_polarity_brightens_and_keeps_outside(self):
        R = np.ones((32, 32), np.float32)
        R[10:13, :] = 0.2
        bg = np.full((32, 32, 3), 0.3, np.float32)
        out, omega = brush.apply(bg, R, "bright")
        self.assertTrue((out[10:13] > 0.3 + 0.1).all())
        self.assertTrue(np.array_equal(out[~omega], bg[~omega]))

    def test_dihedral_covers_eight_distinct(self):
        a = np.arange(12).reshape(3, 4)
        self.assertEqual(len({brush.dihedral(a, k).tobytes() + bytes(brush.dihedral(a, k).shape) for k in range(8)}), 8)


class QuadrupletTests(unittest.TestCase):
    def run_op(self, op, material):
        with tempfile.TemporaryDirectory() as d:
            res = make_world(Path(d), material)
            for i in range(3):
                rng = np.random.default_rng(np.random.SeedSequence([7, i]))
                spec = Q.sample_spec(rng, op, i, res)
                r1 = Q.render_group(spec, res)
                # Q2 strict zero (before A_ν)
                for k, v in r1["measured"]["q2_pre_max_abs_outside"].items():
                    self.assertEqual(v, 0.0, f"{op} {k}")
                # pixel-exact reproducibility
                again = Q.rerender_hashes(spec, res)
                self.assertEqual(again["x11.png"], Q.sha256_bytes(Q.encode_png(r1["images"]["x11"])))
                # the crack is clearly visible (high ΔC) and labels are monotone in τ_Δ
                if spec["crack"].get("contrast_alpha", 1.0) == 1.0:
                    self.assertGreater(r1["measured"]["dc10_median"], 0.05)
                y_lo, ign_lo = Q.labels_from_dc(r1["M"], r1["skel_yx"], r1["dc10"], 0.0)
                y_hi, ign_hi = Q.labels_from_dc(r1["M"], r1["skel_yx"], r1["dc10"], 10.0)
                self.assertGreaterEqual(y_lo.sum(), y_hi.sum())
                self.assertEqual(int(y_hi.sum()), 0)
                self.assertEqual(int((y_lo | ign_lo).sum()), int(r1["M"].sum()))
                if spec["r_target"]:
                    self.assertEqual(r1["measured"]["r"]["d_bin"], spec["r_realized_bin"])

    def test_contrast_alpha_monotone(self):
        with tempfile.TemporaryDirectory() as d:
            res = make_world(Path(d), "concrete")
            spec = Q.sample_spec(np.random.default_rng(np.random.SeedSequence([7, 1])), "joint", 1, res)
            for k in ("exposure_on", "motion_on", "defocus_on", "noise_on", "jpeg_on"):
                spec["theta_nu"][k] = False
            dcs = []
            for a in (1.0, 0.3, 0.05):
                spec["crack"]["contrast_alpha"] = a
                dcs.append(Q.render_group(spec, res)["measured"]["dc10_median"])
            self.assertTrue(dcs[0] > dcs[1] > dcs[2] > 0, dcs)

    def test_joint_concrete(self):
        self.run_op("joint", "concrete")

    def test_shadow_asphalt(self):
        self.run_op("shadow", "asphalt")

    def test_repair_asphalt(self):
        self.run_op("repair_boundary", "asphalt")

    def test_acquisition_paired_noise_identical(self):
        theta = acquisition.sample_theta(np.random.default_rng(3), DOMAINS["acquisition"])
        theta.update(noise_on=True, shot_gain=0.005, read_sigma=0.005, jpeg_on=False, motion_on=False, defocus_on=False)
        a = np.full((64, 64, 3), 0.5, np.float32)
        b = a.copy()
        b[30:34, :] = 0.2
        xa, xb = acquisition.render(a, theta), acquisition.render(b, theta)
        diff = np.abs(xa - xb).max(axis=2) > 0
        self.assertFalse(diff[:30].any() or diff[34:].any())


if __name__ == "__main__":
    unittest.main()
