"""Background pool B candidates v3 = v2 + crop quality check (v2 kept): drop crops with >1% near-saturated pixels
(all channels >=250 or <=5; catches white/black padding and borders, e.g. CrackTree260 margins), near-constant content (gray std < 2),
or bright structure (>2% of pixels with gray >= max(178, median+77), i.e. lane paint / kerbs; added after kit-v3 preview showed markings).
Original v2 docstring follows.
Background pool B candidates v2 (supersedes v1 layout; v1 kept). Adds imaging-domain strata and a multi-scale tier.
domain: laser_gray (PaveDistress line-scan frames) | rgb_photo (DeepCrack / CrackTree260 / CFD).
Crop side: the largest of (512, 256, 192, 128) that yields ≥1 crop per image is NOT used (would bias toward crack-free edges);
instead a fixed tier per domain/material: laser_gray 512; rgb asphalt 192; rgb concrete 128 (interim until SDNET2018 concrete
arrives; tier recorded per crop). Rule unchanged: FREEZE.v3 split=train, crop ∩ dilate(GT,30px)=∅, watermark consensus 'no' only,
material from dataset or DeepCrack per-image table; structure-surface concrete tagged. admission=pending.
usage: python background_pool_v2.py <data_root> <datasets.yaml> <material_csv> <watermark_consensus_csv> <out_dir>"""
import csv, json, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np, yaml
from PIL import Image
root, cfgp, matp, wmp, out = map(Path, sys.argv[1:6]); out.mkdir(parents=True, exist_ok=False)
cfg = yaml.safe_load(cfgp.read_text(encoding='utf-8'))['datasets']
reg = {json.loads(l)['image_id']: json.loads(l) for l in open(root / 'registry/registry.jsonl', encoding='utf-8')}
asg = [r for r in csv.DictReader(open(root / 'splits/split_assignments.v3.csv', encoding='utf-8-sig')) if r['split'] == 'train']
mat = {r['image_id']: r for r in csv.DictReader(open(matp, encoding='utf-8'))}
wm = {r['image_id']: r['overlay_final'] for r in csv.DictReader(open(wmp, encoding='utf-8'))}
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
TIER = {('laser_gray', 'asphalt'): 512, ('rgb_photo', 'asphalt'): 192, ('rgb_photo', 'concrete'): 128}
rows, skipped = [], Counter()
for a in asg:
    r = reg[a['image_id']]; ds = r['source_dataset']
    if ds not in ('deepcrack', 'cracktree260', 'cfd', 'pavedistress'): continue
    if ds == 'pavedistress' and 'Background/' not in r['image_id']: continue
    if ds == 'deepcrack' and wm.get(r['image_id']) != 'no': skipped['watermark'] += 1; continue
    domain = 'laser_gray' if ds == 'pavedistress' else 'rgb_photo'
    material = cfg[ds].get('material', 'unknown'); surface = ''
    if material == 'mixed':
        m = mat.get(r['image_id'], {}); material = m.get('material', 'unassigned'); surface = m.get('surface', '')
    if (domain, material) not in TIER: skipped[f'no_tier:{material}'] += 1; continue
    side = TIER[(domain, material)]; W, H = r['width'], r['height']
    forbid = cv2.dilate((np.asarray(Image.open(r['mask_path']).convert('L')) > 0).astype(np.uint8), K) if r.get('mask_path') else np.zeros((H, W), np.uint8)
    img = None
    for y in range(0, H - side + 1, side):
        for x in range(0, W - side + 1, side):
            if forbid[y:y + side, x:x + side].any(): continue
            if img is None:
                img = np.asarray(Image.open(r['path']).convert('RGB'))
            crop = img[y:y + side, x:x + side]
            sat = ((crop >= 250).all(axis=2) | (crop <= 5).all(axis=2)).mean()
            g = crop.mean(axis=2)
            if sat > 0.01 or g.std() < 2:
                skipped['crop_qc'] += 1; continue
            if (g >= max(178.0, float(np.median(g)) + 77.0)).mean() > 0.02:
                skipped['crop_bright_structure'] += 1; continue
            rows.append({'crop_id': f"{r['image_id']}#y{y}_x{x}_s{side}", 'image_id': r['image_id'], 'source_dataset': ds, 'group_id': a['group_id'],
                         'domain': domain, 'material': material, 'surface': surface, 'x': x, 'y': y, 'side': side,
                         'tier': 'interim_small' if side < 192 else 'standard', 'admission': 'pending'})
with open(out / 'candidates.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
strata = Counter((r['domain'], r['material'], r['surface'] or '-', r['side']) for r in rows)
summ = {'candidates': len(rows), 'strata': {'|'.join(map(str, k)): v for k, v in sorted(strata.items())},
        'groups_per_stratum': {'|'.join(map(str, k)): len({r['group_id'] for r in rows if (r['domain'], r['material'], r['surface'] or '-', r['side']) == k}) for k in strata},
        'skipped_images': dict(skipped), 'rule': 'FREEZE.v3 train; crop ∩ dilate(GT,30px)=∅; watermark-clean; admission pending'}
(out / 'summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding='utf-8'); print(json.dumps(summ, ensure_ascii=False, indent=1))
