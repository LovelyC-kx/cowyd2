"""CPU correctness checks plus optional local throughput (not GPU predictions)."""
import argparse
import json
import tempfile
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from baseline_suite.models import FACTORIES,build_model
from baseline_suite.runtime import restore,atomic_save
from baseline_suite.data import StrictPairs
from baseline_suite.div2k import DIV2KDataset
from baseline_suite.protocol import Protocol
from baseline_suite.metrics import y_metrics
from baseline_suite.train import main as train


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(1234)
    counts={}
    for name in FACTORIES:
        m=build_model(name);x=torch.rand(2,3,25,29);out=m(x)
        assert out.shape==x.shape and torch.isfinite(out).all(),name
        out.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),name
        m.eval()
        with torch.no_grad():
            z=m(x);assert torch.equal(z,m(x)),name+' hidden-state leakage'
            lr=torch.rand(1,3,23,27)
            y1=restore(m,lr,2,24,8,1);y3=restore(m,lr,2,24,8,3)
            assert y1.shape==(1,3,46,54) and torch.allclose(y1,y3,atol=2e-5,rtol=2e-5),name+' tile batching'
        counts[name]=sum(p.numel() for p in m.parameters())
    ref=torch.rand(3,24,24);score=y_metrics(ref,ref)
    assert score['psnr_y']==float('inf') and abs(score['ssim_y']-1)<1e-10
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);root=tmp/'DIV2K'
        for split,ids in [('train',range(1,5)),('valid',range(801,803))]:
            d=root/f'DIV2K_{split}_HR';d.mkdir(parents=True)
            for i in ids:
                image=np.random.default_rng(i).integers(0,256,size=(48,48,3),dtype=np.uint8)
                Image.fromarray(image).save(d/f'{i:04d}.png')
        ds=DIV2KDataset(root,'train',Protocol(hr_patch_size=32),True)
        ds.set_epoch(2);x=ds[0];ds.set_epoch(3);y=ds[0];ds.set_epoch(2)
        assert torch.equal(x['lr'],ds[0]['lr']) and not torch.equal(x['lr'],y['lr'])
        val=DIV2KDataset(root,'valid',Protocol(hr_patch_size=32),True)
        v=val[0]['lr'];val.set_epoch(99);assert torch.equal(v,val[0]['lr'])
        common=['--data-root',str(root),'--device','cpu','--workers','0','--threads','2',
                '--patch-size','32','--batch-size','2','--allow-partial','--max-train-steps','1',
                '--max-val-images','1','--tile-size','32','--overlap','8','--val-every','2',
                '--save-every','1','--log-every','1']
        # Every model trains, validates, and produces a strictly reloadable checkpoint.
        for name in FACTORIES:
            dest=tmp/name
            train(['--model',name,'--epochs','1','--output',str(dest)]+common)
            ck=torch.load(dest/'model_last.pth',weights_only=True,map_location='cpu')
            assert ck['phase']=='complete' and ck['row']['train_images']==2
            build_model(name).load_state_dict(ck['state_dict'],strict=True)
        # Continuing a fixed protocol to a larger total epoch budget is supported.
        train(['--model','srcnn','--epochs','2','--output',str(tmp/'continuous')]+common)
        train(['--model','srcnn','--epochs','2','--output',str(tmp/'srcnn'),
               '--resume',str(tmp/'srcnn'/'model_last.pth')]+common)
        c=torch.load(tmp/'continuous'/'model_last.pth',weights_only=True,map_location='cpu')
        r=torch.load(tmp/'srcnn'/'model_last.pth',weights_only=True,map_location='cpu')
        assert all(torch.equal(v,r['state_dict'][k]) for k,v in c['state_dict'].items()),'Resume diverged'
        # Validation interruption must retain trained epoch and not train it twice.
        failed=tmp/'interrupted'
        with patch('baseline_suite.train.evaluate',side_effect=RuntimeError('simulated validation interruption')):
            try:train(['--model','srcnn','--epochs','2','--output',str(failed)]+common)
            except RuntimeError as e:assert 'simulated' in str(e)
            else:raise AssertionError('Expected interruption')
        ck=torch.load(failed/'model_last.pth',weights_only=True,map_location='cpu')
        assert ck['phase']=='validation_pending'
        before={k:v.clone() for k,v in ck['state_dict'].items()}
        train(['--model','srcnn','--epochs','2','--output',str(failed),
               '--resume',str(failed/'model_last.pth')]+common)
        after=torch.load(failed/'model_last.pth',weights_only=True,map_location='cpu')
        assert all(torch.equal(v,after['state_dict'][k]) for k,v in before.items())
        path=tmp/'atomic.pt';atomic_save({'x':torch.tensor([1])},path)
        with patch('baseline_suite.runtime.torch.save',side_effect=RuntimeError('disk failure')):
            try:atomic_save({'x':torch.tensor([2])},path)
            except RuntimeError:pass
        assert torch.load(path,weights_only=True)['x'].item()==1
        assert not list(tmp.glob('atomic.pt.*.tmp'))
        # Strict pairing and train/validation identity protection.
        d1=tmp/'input';d2=tmp/'target';d1.mkdir();d2.mkdir()
        Image.fromarray(np.zeros((24,24,3),dtype=np.uint8)).save(d1/'a.png')
        Image.fromarray(np.zeros((24,24,3),dtype=np.uint8)).save(d2/'b.png')
        try:StrictPairs(d1,d2)
        except ValueError:pass
        else:raise AssertionError('Mismatched pairs accepted')
    result={'status':'passed','parameter_counts':counts,'device':'CPU',
            'checks':['all six forward/backward/odd shapes','stateless recurrence','batched tile equivalence',
                      'fixed Y metrics','deterministic crops/noise','all six end-to-end training/reload',
                      'epoch extension and exact CPU resume','resume pending validation without retraining',
                      'atomic save failure preserves old checkpoint','strict pairing']}
    print(json.dumps(result,indent=2))
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
