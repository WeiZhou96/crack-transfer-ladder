"""Hash-bound resumable prediction outputs; no model loading or image inference."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def begin_output(directory, contract):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'metadata.json'
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding='utf-8'))
        if prior.get('contract') != contract:
            raise ValueError('prediction contract differs (weight/input/preprocessing); use a new output directory')
    elif any(directory.iterdir()):
        raise ValueError('existing outputs lack a hash-bound metadata receipt; refuse reuse')
    else:
        prior = {'schema': 'student_prediction_receipt_v2', 'contract': contract, 'completed': {}}
    if not isinstance(prior.get('completed'), dict):
        raise ValueError('invalid prediction receipt')
    prior['complete'] = False
    write_json(manifest, prior)
    return prior


def validated_existing(path, receipt, image_id):
    path = Path(path)
    record = receipt['completed'].get(image_id)
    if not path.exists():
        if record:
            raise ValueError(f'receipted prediction is missing: {image_id}')
        return False
    if record is None:
        raise ValueError(f'prediction has no completion receipt: {image_id}')
    if record.get('sha256') != sha256(path):
        raise ValueError(f'prediction hash mismatch: {image_id}')
    return True
