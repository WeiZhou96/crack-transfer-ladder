"""Validate an immutable exhaustive construction audit before synthetic training."""
import json
from pathlib import Path
from prediction_receipt_v1 import sha256

EXPECTED_FILES={'x00.png','x01.png','x10.png','x11.png','M.png','omega_z.png','omega_m.png','dc.npz','manifest.json'}


def validate_quadruplets(quad, audit, is_bound):
    quad,audit=Path(quad).resolve(),Path(audit).resolve()
    for name in ['summary.json','groups.jsonl','inputs.json']:
        if not is_bound(audit/name):
            raise ValueError('quadruplet audit not registered in active freeze: '+name)
    summary=json.loads((audit/'summary.json').read_text())
    if summary.get('passed') is not True or summary.get('failed_groups') or summary.get('changed_inputs'):
        raise ValueError('quadruplet construction audit did not pass')
    if sha256(audit/'groups.jsonl')!=summary.get('groups_receipt_sha256'):
        raise ValueError('quadruplet group receipt hash mismatch')
    records=[json.loads(s) for s in (audit/'groups.jsonl').read_text().splitlines()]
    specs=[json.loads(s) for s in (quad/'specs.jsonl').read_text().splitlines()]
    indexes=[json.loads(s) for s in (quad/'index.jsonl').read_text().splitlines()]
    byid={r['group_id']:r for r in records}
    specid={s['group_id']:s for s in specs}
    indexid={s['group_id']:s for s in indexes}
    if len(byid)!=len(records) or len(specid)!=len(specs) or len(indexid)!=len(indexes):
        raise ValueError('duplicate quadruplet ids')
    if not byid or set(byid)!=set(specid) or set(byid)!=set(indexid) or summary.get('groups')!=len(byid):
        raise ValueError('quadruplet audit population mismatch')
    inputs={}
    for path,digest in json.loads((audit/'inputs.json').read_text())['files'].items():
        canonical=str(Path(path).resolve())
        if canonical in inputs and inputs[canonical]!=digest:
            raise ValueError('conflicting audit path aliases')
        inputs[canonical]=digest
    for name in ['specs.jsonl','index.jsonl']:
        if inputs.get(str(quad/name))!=sha256(quad/name) or not is_bound(quad/name):
            raise ValueError('quadruplet inputs differ from audited/frozen inputs')
    for gid,row in byid.items():
        if row.get('issues') or row['operator']!=specid[gid]['operator'] or row['stratum']!=specid[gid]['stratum']:
            raise ValueError('invalid group audit: '+gid)
        if indexid[gid]['operator']!=row['operator'] or indexid[gid]['stratum']!=row['stratum']:
            raise ValueError('index differs from audited group: '+gid)
        if set(row['files_sha256'])!=EXPECTED_FILES:
            raise ValueError('incomplete group file coverage: '+gid)
        directory=(quad/row['operator']/gid).resolve()
        if not directory.is_relative_to(quad):
            raise ValueError('unsafe quadruplet path')
        for name,expected in row['files_sha256'].items():
            if sha256(directory/name)!=expected:
                raise ValueError('quadruplet artifact changed: '+gid+'/'+name)
        if json.loads((directory/'manifest.json').read_text())['spec']!=specid[gid]:
            raise ValueError('manifest differs from frozen specification')
    return {'groups':len(byid),'audit_summary_sha256':sha256(audit/'summary.json'),
            'group_receipt_sha256':sha256(audit/'groups.jsonl')}
