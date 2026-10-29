import os
import time
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from studio_pairing import host
from studio_pairing.crypto import new_pairing_key, parse_public_key, wrap_token, unwrap_token
from studio_pairing.network import private_ipv4


class Phase20Tests(unittest.TestCase):
    def setUp(self):
        self.secret='TEST_ONLY_NO_REAL_TOKEN_1234567890'
        os.environ['DGPU_TOKEN']=self.secret
        host._PENDING.clear();host._APPROVED.clear();host._ATTEMPTS.clear()
        host._SESSION.update(code=None,expires=0,attempts=0)
        self.app=FastAPI()
        self.app.include_router(host.router)
        self.client=TestClient(self.app)
        self.auth={'Authorization':'Bearer '+self.secret}

    def tearDown(self):
        os.environ.pop('DGPU_TOKEN',None)

    def test_token_roundtrip(self):
        private, pub64, fingerprint=new_pairing_key()
        key, verified=parse_public_key(pub64)
        self.assertEqual(verified,fingerprint)
        self.assertEqual(unwrap_token(private,wrap_token(key,self.secret)),self.secret)

    def test_owner_required(self):
        self.assertEqual(self.client.post('/api/cluster/open').status_code,401)
        self.assertEqual(self.client.post('/api/cluster/open',headers=self.auth).status_code,200)

    def test_full_pairing(self):
        with patch.object(host,'is_valid_ip', return_value='192.168.50.2'):
            d=self.client.post('/api/cluster/open',headers=self.auth).json()
            private,pub,fp=new_pairing_key()
            req=self.client.post('/api/cluster/request',json={
                'code':d['code'],'name':'RTX2050','gpu':'RTX 2050','public_key':pub})
            self.assertEqual(req.status_code,200,req.text)
            request=req.json()
            self.assertEqual(request['fingerprint'],fp)
            p={'request_id':request['request_id'],'poll_secret':request['poll_secret']}
            before=self.client.post('/api/cluster/poll',json=p).json()
            self.assertEqual(before['status'],'pending')
            approved=self.client.post('/api/cluster/approve',json={'request_id':request['request_id'],'fingerprint':fp},headers=self.auth)
            self.assertEqual(approved.status_code,200,approved.text)
            done=self.client.post('/api/cluster/poll',json=p).json()
            self.assertEqual(done['status'],'approved')
            self.assertEqual(unwrap_token(private,done['encrypted_token']),self.secret)

    def test_wrong_code_and_wrong_poll_secret(self):
        with patch.object(host,'is_valid_ip',return_value='192.168.50.2'):
            self.client.post('/api/cluster/open',headers=self.auth)
            _,pub,_=new_pairing_key()
            r=self.client.post('/api/cluster/request',json={'code':'BAD-BAD1','name':'B','public_key':pub})
            self.assertEqual(r.status_code,403)

    def test_pairing_expires(self):
        with patch.object(host,'is_valid_ip',return_value='192.168.50.2'):
            d=self.client.post('/api/cluster/open',headers=self.auth).json()
            _,pub,_=new_pairing_key()
            host._SESSION['expires']=time.monotonic()-1
            req=self.client.post('/api/cluster/request',json={'code':d['code'],'name':'B','public_key':pub})
            self.assertEqual(req.status_code,403)

    def test_wrong_poll_secret_rejected(self):
        with patch.object(host,'is_valid_ip',return_value='192.168.50.2'):
            d=self.client.post('/api/cluster/open',headers=self.auth).json()
            _,pub,_=new_pairing_key()
            req=self.client.post('/api/cluster/request',json={'code':d['code'],'name':'B','public_key':pub}).json()
            bad=self.client.post('/api/cluster/poll',json={'request_id':req['request_id'],'poll_secret':'WRONG'})
            self.assertEqual(bad.status_code,403)

    def test_pairing_rate_limit(self):
        with patch.object(host,'is_valid_ip',return_value='192.168.50.2'):
            self.client.post('/api/cluster/open',headers=self.auth)
            _,pub,_=new_pairing_key()
            statuses=[]
            for _ in range(7):
                r=self.client.post('/api/cluster/request',json={'code':'WRONG123','name':'B','public_key':pub})
                statuses.append(r.status_code)
            self.assertEqual(statuses[:6],[403]*6)
            self.assertEqual(statuses[6],429)

    def test_private_ipv4(self):
        self.assertEqual(private_ipv4('192.168.50.2'),'192.168.50.2')
        with self.assertRaises(ValueError): private_ipv4('8.8.8.8')
        with self.assertRaises(ValueError): private_ipv4('example.com')


if __name__=='__main__': unittest.main()
