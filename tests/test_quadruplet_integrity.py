import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'train'))
from prediction_receipt_v1 import sha256,write_json
from quadruplet_integrity_v1 import validate_quadruplets,EXPECTED_FILES


class TrainingIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name).resolve()
        self.quad,self.audit=root/'quad',root/'audit'
        self.group=self.quad/'joint'/'joint_00000'
        self.group.mkdir(parents=True);self.audit.mkdir()
        spec=dict(group_id='joint_00000',operator='joint',stratum='rgb_concrete')
        for name in EXPECTED_FILES:
            (self.group/name).write_bytes(b'fixture '+name.encode())
        write_json(self.group/'manifest.json',{'spec':spec})
        for name in ['specs.jsonl','index.jsonl']:
            (self.quad/name).write_text(json.dumps(spec)+'\n')
        row=dict(spec,issues=[],files_sha256={name:sha256(self.group/name) for name in EXPECTED_FILES})
        (self.audit/'groups.jsonl').write_text(json.dumps(row)+'\n')
        write_json(self.audit/'summary.json',dict(passed=True,groups=1,failed_groups=[],changed_inputs=[],
                                                  groups_receipt_sha256=sha256(self.audit/'groups.jsonl')))
        write_json(self.audit/'inputs.json',dict(files={str(self.quad/n):sha256(self.quad/n) for n in ['specs.jsonl','index.jsonl']}))

    def test_valid_receipt_and_tampered_dc(self):
        self.assertEqual(validate_quadruplets(self.quad,self.audit,lambda _:True)['groups'],1)
        (self.group/'dc.npz').write_bytes(b'changed contrast')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            validate_quadruplets(self.quad,self.audit,lambda _:True)

    def test_unfrozen_audit_refused(self):
        with self.assertRaisesRegex(ValueError,'not registered'):
            validate_quadruplets(self.quad,self.audit,lambda _:False)

    def test_data_volume_symlink_alias(self):
        alias=Path(self.tmp.name)/'data_alias'
        try:
            alias.symlink_to(self.quad,target_is_directory=True)
        except OSError as error:
            self.skipTest('symlink unavailable: '+str(error))
        inputs={str(alias/n):sha256(self.quad/n) for n in ['specs.jsonl','index.jsonl']}
        write_json(self.audit/'inputs.json',dict(files=inputs))
        self.assertEqual(validate_quadruplets(self.quad,self.audit,lambda _:True)['groups'],1)

    def test_group_receipt_modification_detected(self):
        with (self.audit/'groups.jsonl').open('a') as f:
            f.write('{}\n')
        with self.assertRaisesRegex(ValueError,'receipt hash'):
            validate_quadruplets(self.quad,self.audit,lambda _:True)

    def test_incomplete_file_coverage_refused_even_when_rehashed(self):
        row=json.loads((self.audit/'groups.jsonl').read_text())
        del row['files_sha256']['dc.npz']
        (self.audit/'groups.jsonl').write_text(json.dumps(row)+'\n')
        summary=json.loads((self.audit/'summary.json').read_text())
        summary['groups_receipt_sha256']=sha256(self.audit/'groups.jsonl')
        write_json(self.audit/'summary.json',summary)
        with self.assertRaisesRegex(ValueError,'coverage'):
            validate_quadruplets(self.quad,self.audit,lambda _:True)


if __name__=='__main__':
    unittest.main()
