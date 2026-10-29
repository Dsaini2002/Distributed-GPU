import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
from desktop_app import core,gpu_metrics

class Tests(unittest.TestCase):
    def test_all_gpus_and_missing_values(self):
        response=subprocess.CompletedProcess([],0,'0, GPU A, 20, 100, 900, 1000, 40\n1, GPU B, N/A, 200, 800, 1000, N/A\n','')
        with patch.object(gpu_metrics.subprocess,'run',return_value=response):
            data=gpu_metrics.gpu_snapshot()
        self.assertEqual(len(data['gpus']),2)
        self.assertEqual(data['gpus'][0]['free'],900)
        self.assertIsNone(data['gpus'][1]['temperature'])
        self.assertIsNone(data['gpus'][1]['util'])

    def test_driver_failure(self):
        with patch.object(gpu_metrics.subprocess,'run',side_effect=FileNotFoundError):
            result=gpu_metrics.gpu_snapshot()
        self.assertEqual(result['gpus'],[])
        self.assertTrue(result['error'])

    def test_approved_worker_authentication_and_invalid_address(self):
        b=core.DesktopBridge();b.role='host'
        with patch.object(b,'_host_api',return_value={'workers':[{'ip':'192.168.50.2'},{'ip':'8.8.8.8'}]}),patch.object(b,'_get_token',return_value='test-token'),patch.object(core,'api_json',return_value={'gpus':[],'sampled_at':'now'}) as request,patch.object(gpu_metrics,'gpu_snapshot',return_value={'gpus':[]}):
            result=b.cluster_gpu_status()
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['data']['nodes']),3)
        self.assertEqual(request.call_count,1)
        self.assertEqual(request.call_args.kwargs['token'],'test-token')
        self.assertTrue(result['data']['nodes'][2]['error'])
        self.assertNotIn('test-token',str(result))

    def test_old_worker_or_offline_is_not_zero_utilization(self):
        b=core.DesktopBridge();b.role='host'
        with patch.object(b,'_host_api',return_value={'workers':[{'ip':'192.168.50.2'}]}),patch.object(b,'_get_token',return_value='test-token'),patch.object(core,'api_json',side_effect=RuntimeError('404')),patch.object(gpu_metrics,'gpu_snapshot',return_value={'gpus':[]}):
            node=b.cluster_gpu_status()['data']['nodes'][1]
        self.assertTrue(node['error'])
        self.assertEqual(node['gpus'],[])

    def test_native_terminal_arguments(self):
        b=core.DesktopBridge()
        with tempfile.TemporaryDirectory(prefix='gpu terminal ') as folder:
            b.root=Path(folder);(b.root/'studio').mkdir();(b.root/'studio/GPU-Terminal.ps1').touch()
            with patch.object(core.sys,'platform','win32'),patch.object(core.subprocess,'Popen') as launch:
                result=b.open_terminal()
            self.assertTrue(result['ok'],result)
            args=launch.call_args.args[0]
            self.assertEqual(args[args.index('-Root')+1],folder)
            self.assertNotIn('shell',launch.call_args.kwargs)
            self.assertEqual(launch.call_args.kwargs['creationflags'],subprocess.CREATE_NEW_CONSOLE)

    def test_http_auth_and_no_redirect(self):
        seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                seen.append(self.headers.get('Authorization'))
                if self.path=='/redirect':
                    self.send_response(302);self.send_header('Location','/target');self.end_headers();return
                self.send_response(200);self.end_headers();self.wfile.write(b'{"gpus": []}')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            self.assertEqual(core.api_json(url,token='test-token'),{'gpus':[]})
            with self.assertRaises(RuntimeError):core.api_json(url+'/redirect',token='test-token')
            self.assertEqual(seen,['Bearer test-token','Bearer test-token'])
        finally:server.shutdown();server.server_close();thread.join()

    def test_worker_endpoint_auth_gate(self):
        # Execute the actual endpoint body, without importing GPU training dependencies.
        worker=Path(__file__).resolve().parents[2]/'studio_multi/worker.py'
        tree=ast.parse(worker.read_text(encoding='utf-8-sig'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gpu_metrics')
        node.decorator_list=[]
        class Denied(Exception):pass
        def deny(token):raise Denied()
        namespace={'Header':lambda **kw:'','_auth':deny}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(worker),'exec'),namespace)
        with self.assertRaises(Denied):namespace['gpu_metrics']('bad-token')

if __name__=='__main__':unittest.main()
