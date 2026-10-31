from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "render"))
from operators import OPERATORS, Context, PreconditionError, load_domains
from operators import relation, visibility
from operators.base import srgb_to_linear

DOMAINS = load_domains(Path(__file__).resolve().parents[1] / "configs" / "operators.yaml")


def textured_background(h=160, w=224, seed=0, level=0.55):
    rng = np.random.default_rng(seed)
    base = level + 0.06 * rng.standard_normal((h, w))
    return np.repeat(np.clip(base, 0, 1)[..., None], 3, axis=2).astype(np.float32)


def to_u8(x):
    return np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)


class OperatorContractTests(unittest.TestCase):
    def check_contract(self, name, material):
        op, domain = OPERATORS[name], DOMAINS[name]
        ctx = Context(material=material)
        image = textured_background()
        for seed in range(12):
            theta = op.sample_theta(np.random.default_rng(seed), ctx, domain, image.shape[:2])
            for key, spec in domain.items():
                if isinstance(spec, dict) and "range" in spec and key in theta:
                    lo, hi = spec["range"]
                    self.assertTrue(lo <= theta[key] <= hi, f"{name}.{key}={theta[key]} ∉ {spec['range']}")
            result = op.render(image, theta, ctx, domain)
            # Q2 strict zero: bit-exactly unchanged outside Ω_Z (also after 8-bit quantisation)
            changed = np.any(to_u8(result.image) != to_u8(image), axis=2)
            self.assertFalse(np.any(changed & ~result.omega), f"{name} seed={seed}: pixels changed outside Ω_Z")
            if result.omega.any():
                self.assertTrue(changed.any(), f"{name} seed={seed}: support declared but nothing changed")
            again = op.render(image, dict(theta), ctx, domain)
            np.testing.assert_array_equal(result.image, again.image)
            self.assertEqual(result.image.shape, image.shape)

    def test_shadow_contract(self):
        self.check_contract("shadow", "asphalt")

    def test_joint_contract(self):
        self.check_contract("joint", "concrete")

    def test_repair_boundary_contract(self):
        self.check_contract("repair_boundary", "asphalt")

    def test_joint_requires_concrete(self):
        op, domain = OPERATORS["joint"], DOMAINS["joint"]
        ctx = Context(material="asphalt")
        theta = op.sample_theta(np.random.default_rng(0), Context(material="concrete"), domain, (64, 64))
        with self.assertRaises(PreconditionError):
            op.render(textured_background(64, 64), theta, ctx, domain)

    def test_unknown_material_is_rejected(self):
        for name in OPERATORS:
            op, domain = OPERATORS[name], DOMAINS[name]
            theta = op.sample_theta(np.random.default_rng(1), Context(material="concrete"), domain, (64, 64))
            with self.assertRaises(PreconditionError):
                op.render(textured_background(64, 64), theta, Context(material="unassigned"), domain)

    def test_shadow_core_transmittance_matches_k(self):
        op, domain = OPERATORS["shadow"], DOMAINS["shadow"]
        ctx = Context(material="asphalt")
        image = textured_background(level=0.6)
        theta = {"operator": "shadow", "kind": "linear", "transmittance_k": 0.5, "penumbra_px": 2.0,
                 "sky_blue_shift": 0.0, "light_azimuth_deg": 0.0, "seed": 3,
                 "line_width_px": 15.0, "line_angle_deg": 0.0, "line_offset_px": 0.0}
        out = op.render(image, theta, ctx, domain).image
        h = image.shape[0] // 2
        ratio = srgb_to_linear(out[h - 3:h + 4]).mean() / srgb_to_linear(image[h - 3:h + 4]).mean()
        self.assertAlmostEqual(float(ratio), 0.5, delta=0.02)

    def test_joint_is_not_a_black_line(self):
        op, domain = OPERATORS["joint"], DOMAINS["joint"]
        ctx = Context(material="concrete")
        image = textured_background(level=0.6)
        theta = {"operator": "joint", "angle_deg": 0.0, "width_px": 8.0, "groove_darkness": 0.4,
                 "lip_contrast": 0.1, "edge_jitter_px": 0.0, "light_azimuth_deg": 90.0, "seed": 1,
                 "offsets_px": [0.0], "n_joints": 1}
        result = op.render(image, theta, ctx, domain)
        h = image.shape[0] // 2
        groove = result.image[h - 1:h + 2]
        self.assertGreater(float(groove.mean()), 0.2)              # keeps background texture and brightness, not a black line
        self.assertGreater(float(groove.std()), 0.01)
        above, below = result.image[h - 5].mean(), result.image[h + 5].mean()
        self.assertGreater(abs(float(above - below)), 0.02)        # one lip bright, the other dark

    def test_repair_measured_area_tracks_theta(self):
        op, domain = OPERATORS["repair_boundary"], DOMAINS["repair_boundary"]
        ctx = Context(material="asphalt")
        theta = op.sample_theta(np.random.default_rng(5), ctx, domain, (200, 200))
        theta.update({"center_frac": [0.5, 0.5], "edge_jitter_px": 0.0, "area_frac": 0.2})
        result = op.render(textured_background(200, 200), theta, ctx, domain)
        self.assertAlmostEqual(result.measured["area_frac"], 0.2, delta=0.02)
        self.assertGreater(result.measured["boundary_len_px"], 100)


class RelationAndVisibilityTests(unittest.TestCase):
    def crack(self, h=120, w=200, row=60, width=3):
        mask = np.zeros((h, w), np.uint8)
        mask[row - width // 2: row + width // 2 + 1, 20:180] = 1
        return mask

    def line(self, h, w, row=None, col=None, width=1):
        m = np.zeros((h, w), np.uint8)
        if row is not None:
            m[row - width // 2: row + width // 2 + 1, :] = 1
        else:
            m[:, col - width // 2: col + width // 2 + 1] = 1
        return m

    def test_crossing_line_is_overlap_perpendicular_one_crossing(self):
        crack = self.crack()
        center, support = self.line(120, 200, col=100), self.line(120, 200, col=100, width=5)
        r = relation.measure(center, support, crack)
        self.assertEqual(r["d_bin"], "overlap")
        self.assertEqual(r["dphi_bin"], ">60")
        self.assertEqual(r["n_bin"], ">=1")

    def test_parallel_far_line(self):
        crack = self.crack(row=20)
        center, support = self.line(120, 200, row=100), self.line(120, 200, row=100, width=3)
        r = relation.measure(center, support, crack)
        self.assertEqual(r["d_bin"], "far")
        self.assertEqual(r["dphi_bin"], "<15")
        self.assertEqual(r["n_cross"], 0)

    def test_exclusion_band_is_reported(self):
        crack = self.crack(row=40, width=3)
        center, support = self.line(120, 200, row=52), self.line(120, 200, row=52, width=1)
        r = relation.measure(center, support, crack)
        self.assertEqual(r["d_bin"], "excluded")

    def test_visibility_follows_contrast(self):
        # low-noise background: coarse texture alone can push C_w over τ_c=0.02 (τ_c must be calibrated against human visibility); only the logic is checked here
        crack = self.crack()
        rng = np.random.default_rng(0)
        image = np.repeat((0.6 + 0.002 * rng.standard_normal((120, 200)))[..., None], 3, axis=2).astype(np.float32)
        invisible = image.copy()
        visible = image.copy()
        visible[crack > 0] *= 0.5
        v_hi, s_hi = visibility.visibility_mask(visible, crack)
        v_lo, s_lo = visibility.visibility_mask(invisible, crack)
        self.assertGreater(s_hi["visible_frac"], 0.95)
        self.assertLess(s_lo["visible_frac"], 0.5)
        self.assertFalse(np.any(v_hi & (crack == 0)))

    def test_paired_visibility_ignores_texture_and_tracks_crack(self):
        crack = self.crack()
        rng = np.random.default_rng(1)
        rough = np.clip(0.5 + 0.15 * rng.standard_normal((120, 200)), 0.05, 1.0)   # coarse texture
        background = np.repeat(rough[..., None], 3, axis=2).astype(np.float32)
        # no crack drawn: texture alone exceeds the frozen τ_c on many points, but ΔC is exactly zero
        _, frozen = visibility.visibility_mask(background, crack)
        self.assertGreater(frozen["visible_frac"], 0.3)
        v0, s0 = visibility.visibility_mask_paired(background, background, crack, tau_delta=0.02)
        self.assertEqual(s0["visible_frac"], 0.0)
        drawn = background.copy()
        drawn[crack > 0] *= 0.5
        v1, s1 = visibility.visibility_mask_paired(drawn, background, crack, tau_delta=0.05)
        self.assertGreater(s1["visible_frac"], 0.9)
        self.assertFalse(np.any(v1 & (crack == 0)))


if __name__ == "__main__":
    unittest.main()
