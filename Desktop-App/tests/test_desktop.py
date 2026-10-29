import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
from desktop_app import core

class DesktopCoreTests(TestCase):
    def test_private_ipv4(self):
        self.assertEqual(core.valid_private_ip('192.168.50.1'), '192.168.50.1')
        for ip in ['8.8.8.8','127.0.0.1','169.254.1.2','example.com','http://192.168.50.1','::1']:
            with self.assertRaises(ValueError):
                core.valid_private_ip(ip)

    def test_no_token_overwrite(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            cfg=root/'.dgpu'/'worker_config.json'
            cfg.parent.mkdir()
            cfg.write_text(json.dumps({'protected_token':'keep-me','custom':'preserve'}))
            with patch.object(core,'decrypt_existing_token',return_value='old-token'):
                token,created=core.create_new_token(root,False)
            self.assertEqual(token,'old-token')
            self.assertFalse(created)
            self.assertEqual(json.loads(cfg.read_text())['custom'],'preserve')

    def test_missing_existing_during_running_controller(self):
        with TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'already running'):
                core.create_new_token(Path(d),True)
            self.assertFalse((Path(d)/'.dgpu'/'worker_config.json').exists())

    def test_secure_new_token_length_and_persist(self):
        with TemporaryDirectory() as d:
            with patch.object(core,'_power_shell',return_value='protected-dpapi-data'):
                token,created=core.create_new_token(Path(d),False)
            self.assertTrue(created)
            self.assertGreaterEqual(len(token),40)
            self.assertNotIn(token,(Path(d)/'.dgpu'/'worker_config.json').read_text())
            self.assertEqual(core._json_read(Path(d)/'.dgpu'/'worker_config.json')['protected_token'],'protected-dpapi-data')

    def test_api_no_token_in_frontend(self):
        with patch.object(core.DesktopBridge,'_host_api',return_value={'models':[]}) as mock:
            bridge=core.DesktopBridge()
            r=bridge.catalog()
            self.assertTrue(r['ok'],r)
            self.assertEqual(mock.call_count,2)
            self.assertTrue(all(c.kwargs.get('authenticated') is True for c in mock.call_args_list))
            self.assertNotIn('DGPU_TOKEN',str(r))

    def test_training_spec_uses_real_api_and_approved_workers(self):
        b=core.DesktopBridge()
        b.role='host'; b.host='192.168.50.1'
        def fake_api(path,method='GET',payload=None,authenticated=False):
            if path.endswith('/catalog'):
                self.assertTrue(authenticated)
                return {'models':[{'key':'tabular_mlp_regression'}]}
            if path.endswith('/workers'): return {'workers':[{'ip':'192.168.50.2','port':45420}]}
            if path.endswith('/train'):
                self.assertEqual(payload['workers'],['192.168.50.2:45420'])
                self.assertEqual(payload['model'],'tabular_mlp_regression')
                return {'id':'1234567890abcdef','state':'running'}
            raise AssertionError(path)
        with patch.object(b,'_host_api',side_effect=fake_api),patch.object(core,'tcp_open',return_value=True):
            r=b.start_training({'model':'tabular_mlp_regression','dataset':'data/demo.csv','epochs':3,'batch_size':8,'learning_rate':.001})
        self.assertTrue(r['ok'],r)

    def test_reject_unsafe_dataset(self):
        b=core.DesktopBridge();b.role='host'
        with patch.object(b,'_host_api',return_value={'models':[{'key':'tabular_mlp_regression'}]}):
            for path in ['../secrets','C:/Users/a/file.csv','data/../private.csv']:
                self.assertFalse(b.start_training({'model':'tabular_mlp_regression','dataset':path})['ok'])

    def test_approval_matches_host_entry(self):
        b=core.DesktopBridge();b.role='host'
        with patch.object(b,'_host_api',return_value={'devices':[]}):
            self.assertFalse(b.approve('0'*32,'012345ABCDEF')['ok'])

    def test_agent_join_origin_contract(self):
        b=core.DesktopBridge();
        with patch.object(core,'api_json',return_value={'message':'Requested'}) as api:
            r=b._agent_api('/api/agent/join','POST',{'host':'192.168.50.1','code':'ABCD-EFGH'})
        self.assertEqual(r['message'],'Requested')
        self.assertEqual(api.call_args.kwargs['origin'],'http://127.0.0.1:45430')



class CatalogIntegrationTests(TestCase):
    def test_union_15_models_and_two_execution_modes(self):
        b = core.DesktopBridge()
        local = [{'key': f'model_{i}', 'family': 'Example', 'task': 'regression', 'dataset': 'CSV', 'description': 'Model', 'suggested_batch': 4} for i in range(13)]
        local += [{'key': 'distilgpt2_lora', 'family': 'Language'}, {'key': 'yolo11n_detection', 'family': 'Vision'}]
        remote = [dict(x) for x in local[:13]]
        def fake(path, **kw):
            self.assertTrue(kw['authenticated'])
            return {'models': local if path == '/api/model-zoo/catalog' else remote}
        with patch.object(b,'_host_api',side_effect=fake):
            result=b.catalog()
        self.assertTrue(result['ok'],result)
        models=result['data']['models']
        self.assertEqual(len(models),15)
        self.assertEqual(sum('multi_gpu' in m['modes'] for m in models),13)
        self.assertEqual(sum(m['experimental'] for m in models),2)

    def test_local_training_does_not_require_remote_worker(self):
        b=core.DesktopBridge(); b.role='host'; b.host='192.168.50.1'
        calls=[]
        def fake(path,method='GET',payload=None,authenticated=False):
            calls.append((path,method,payload,authenticated))
            if path.endswith('/catalog'): return {'models':[{'key':'tabular_mlp_regression'}]}
            if path=='/api/model-zoo/train': return {'id':'1234567890abcdef','state':'running'}
            raise AssertionError(path)
        with patch.object(b,'_host_api',side_effect=fake):
            result=b.start_training({'mode':'single_gpu','model':'tabular_mlp_regression','dataset':'data/demo.csv','epochs':3,'batch_size':8,'learning_rate':0.001})
        self.assertTrue(result['ok'],result)
        self.assertEqual(result['data']['mode'],'single_gpu')
        self.assertEqual(calls[-1][0],'/api/model-zoo/train')
        self.assertTrue(all(call[-1] for call in calls))

    def test_training_unsupported_mode_rejected(self):
        b=core.DesktopBridge();b.role='host'
        def fake(path,**kwargs):
            if path.endswith('/model-zoo/catalog'):return {'models':[{'key':'yolo11n_detection'}]}
            if path.endswith('/distributed-zoo/catalog'):return {'models':[]}
            raise AssertionError(path)
        with patch.object(b,'_host_api',side_effect=fake):
            r=b.start_training({'mode':'multi_gpu','model':'yolo11n_detection','dataset':'data/yolo.yaml'})
        self.assertFalse(r['ok'])
        self.assertIn('not supported',r['error'])

    def test_local_and_multigpu_job_status_urls(self):
        b=core.DesktopBridge()
        with patch.object(b,'_host_api',return_value={'state':'completed'}) as mock:
            self.assertTrue(b.training_status('0123456789abcdef','single_gpu')['ok'])
            self.assertEqual(mock.call_args.args[0],'/api/model-zoo/status/0123456789abcdef')
            self.assertTrue(mock.call_args.kwargs.get('authenticated'))
            self.assertTrue(b.training_status('0123456789abcdef','multi_gpu')['ok'])
            self.assertEqual(mock.call_args.args[0],'/api/distributed-zoo/status/0123456789abcdef')

    def test_partial_catalog_success_reports_unavailable_mode(self):
        b=core.DesktopBridge()
        def fake(path,**kwargs):
            if path=='/api/model-zoo/catalog':raise RuntimeError('500: service offline')
            return {'models':[{'key':'tabular_mlp_regression'}]}
        with patch.object(b,'_host_api',side_effect=fake):
            result=b.catalog()
        self.assertTrue(result['ok'])
        self.assertEqual(result['data']['models'][0]['modes'],['multi_gpu'])
        self.assertIn('single_gpu',result['data']['errors'])

    def test_combined_training_history(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            for folder,mode in [('model_zoo_runs','a'*16),('phase19_runs','b'*16)]:
                p=root/'.dgpu'/folder/mode/'status.json'
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps({'id':mode,'model':'tabular_mlp_regression','state':'completed','checkpoint':str(p.parent/'model.pt')}))
            b=core.DesktopBridge();b.root=root;b.role='host'
            r=b.training_history()
            self.assertTrue(r['ok'],r)
            self.assertEqual({j['mode'] for j in r['data']},{'single_gpu','multi_gpu'})
            self.assertTrue(all(j['checkpoint'] for j in r['data']))


if __name__ == "__main__": main()
