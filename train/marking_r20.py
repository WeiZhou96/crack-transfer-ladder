"""R20 (post-freeze deviation, registered 2026-09-28): lane-marking exclusion for committee consensus on laser frames.

Why: on PaveDistress laser frames, retro-reflective lane markings are very bright; after the R18 inversion they become
dark wide bands, and the three teachers (same training data, never saw the laser domain) fire on them together.
Under the frozen s05 rule 1,031/1,455 train negatives were dropped; 77.6 % of them contain markings and the median
fraction of teacher-fired pixels inside markings is 98 %, while only 2.8 % of accepted frames contain markings.
The admission therefore removed the most valuable confuser negatives, and B3-auto kept only 62 crops.

Rule (deterministic, model-free, decided after seeing the frozen-rule result -> disclosed like R18):
  marking = connected regions of median-filtered (9 px) RAW laser gray >= 180 with area >= 20,000 px, dilated by 15 px.
  Committee members' components are intersected with NOT marking before the cross-family same-location consensus;
  baseline firing (hard_negative) is still evaluated on the full frame. Marking pixels stay in the training negatives
  (source label: non-crack; paint crazing inside markings is thereby treated as non-crack -- a task-definition choice).
  Applies to PaveDistress-sourced frames only; other sources are unchanged.
The frozen-rule outputs (negpool/, b3auto_v1/) are kept and reported as the sensitivity result.
"""
import cv2
import numpy as np
from PIL import Image

LASER_SOURCES = {"pavedistress", "pavedistress_anchor", "pavedistress_hard_proxy"}
GRAY_MIN, AREA_MIN, DILATE_PX, MEDIAN_K = 180, 20000, 15, 9


def marking_mask(raw_path: str) -> np.ndarray:
    g = np.asarray(Image.open(raw_path).convert("L"))
    bright = (cv2.medianBlur(g, MEDIAN_K) >= GRAY_MIN).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= AREA_MIN
    big = keep[lab].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))
    return cv2.dilate(big, k) > 0
