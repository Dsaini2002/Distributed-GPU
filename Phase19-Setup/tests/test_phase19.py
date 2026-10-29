import base64
import csv
import io
import os
from pathlib import Path
import tempfile
import unittest
import uuid
import zipfile

import torch
from fastapi.testclient import TestClient
from studio_zoo.train import dataset_and_info
from studio_multi.core import (dataset_fingerprint, deserialize_state,
                               local_epoch, model_from_info, serialize_state,
                               weighted_average)
from studio_multi.coordinator import bundle_dataset, worker_url
from studio_multi import worker


class Phase19Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / 'regression.csv'
        with self.file.open('w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out)
            writer.writerow(['a', 'b', 'target'])
            writer.writerows((i / 10, (i % 5) / 10, (2*i - 5) / 10) for i in range(30))
        self.dataset, self.info = dataset_and_info('tabular_mlp_regression', self.file)
        torch.manual_seed(42)
        self.state = model_from_info('tabular_mlp_regression', self.info).state_dict()

    def test_state_roundtrip_tensor_only(self):
        encoded = serialize_state(self.state)
        recovered = deserialize_state(encoded)
        self.assertTrue(all(torch.equal(v, recovered[k]) for k, v in self.state.items()))
        with self.assertRaises(Exception):
            deserialize_state('not a state')

    def test_sharding_and_aggregation(self):
        first = local_epoch('tabular_mlp_regression', self.file, self.state,
                            rank=0, world_size=2, epoch=1, batch_size=8,
                            lr=0.001, device='cpu')
        second = local_epoch('tabular_mlp_regression', self.file, self.state,
                             rank=1, world_size=2, epoch=1, batch_size=8,
                             lr=0.001, device='cpu')
        self.assertEqual(first['samples'] + second['samples'], 30)
        avg = weighted_average([first, second])
        self.assertEqual(set(avg), set(self.state))
        self.assertFalse(torch.equal(avg['net.0.weight'], self.state['net.0.weight']))

    def test_url_validation(self):
        self.assertEqual(worker_url('192.168.50.2:45420'), 'http://192.168.50.2:45420')
        self.assertEqual(worker_url('127.0.0.1:45420'), 'http://127.0.0.1:45420')
        for bad in ['example.com:45420', '8.8.8.8:45420',
                    '192.168.50.2:45420/redirect', '192.168.50.2:45420@evil',
                    '192.168.50.2:notaport']:
            with self.assertRaises(ValueError, msg=bad):
                worker_url(bad)

    def test_dataset_sync_and_worker_auth(self):
        orig_storage = worker.STORAGE
        orig_token = os.environ.get('DGPU_TOKEN')
        orig_allow = os.environ.get('DGPU_P19_ALLOW_CPU')
        worker.STORAGE = self.root / 'worker'
        os.environ['DGPU_TOKEN'] = 'test-secret-only'
        os.environ['DGPU_P19_ALLOW_CPU'] = '1'
        self.addCleanup(lambda: setattr(worker, 'STORAGE', orig_storage))
        self.addCleanup(lambda: os.environ.pop('DGPU_TOKEN', None) if orig_token is None else os.environ.__setitem__('DGPU_TOKEN', orig_token))
        self.addCleanup(lambda: os.environ.pop('DGPU_P19_ALLOW_CPU', None) if orig_allow is None else os.environ.__setitem__('DGPU_P19_ALLOW_CPU', orig_allow))
        client = TestClient(worker.app)
        session_id = uuid.uuid4().hex[:16]
        entry, zipped = bundle_dataset(self.file)
        request = {'session_id': session_id, 'model': 'tabular_mlp_regression',
                   'dataset_entry': entry, 'dataset_b64': zipped,
                   'fingerprint': dataset_fingerprint(self.file)}
        self.assertEqual(client.post('/api/p19/session', json=request).status_code, 401)
        auth = {'Authorization': 'Bearer test-secret-only'}
        self.assertEqual(client.post('/api/p19/session', json=request, headers=auth).status_code, 200)
        payload = {'session_id': session_id, 'model': 'tabular_mlp_regression',
                   'state_b64': serialize_state(self.state),
                   'rank': 1, 'world_size': 2, 'epoch': 1,
                   'batch_size': 4, 'learning_rate': 0.001}
        answer = client.post('/api/p19/round', json=payload, headers=auth)
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()['samples'], 15)
        self.assertEqual(answer.json()['device'], 'cpu')
        self.assertEqual(set(deserialize_state(answer.json()['state_b64'])), set(self.state))

    def test_zip_slip_rejected(self):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, 'w') as z:
            z.writestr('../steal.txt', 'bad')
        with self.assertRaises(Exception):
            worker._safe_extract(raw.getvalue(), self.root / 'try')


if __name__ == '__main__':
    unittest.main()
