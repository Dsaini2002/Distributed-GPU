"""Read-only cluster metrics for the native project terminal."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desktop_app.core import DesktopBridge, valid_private_ip


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--host-ip',default='192.168.50.1')
    parser.add_argument('--role',choices=['host','worker'],default='host')
    args=parser.parse_args()
    bridge=DesktopBridge()
    bridge.root=Path(args.root).resolve()
    bridge.host=valid_private_ip(args.host_ip)
    bridge.role=args.role
    result=bridge.cluster_gpu_status()
    if not result['ok']:
        print(result['error']);return 1
    data=result['data']
    if data['error']: print(data['error'])
    for node in data['nodes']:
        print('\n'+node['name']+' | '+node['address'])
        print(node['error'] or 'Sample: '+str(node['sampled_at']))
        for gpu in node['gpus']:
            def value(key):
                v=gpu.get(key)
                return 'unavailable' if v is None else str(v)
            print('  GPU '+value('id')+': '+value('name'))
            print('  Load: '+value('util')+'% | Temperature: '+value('temperature')+' C')
            print('  VRAM used/free/total: '+value('used')+' / '+value('free')+' / '+value('total')+' MiB')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
