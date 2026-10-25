"""SwinIR extraction, capacity and shared-pipeline CPU correctness checks."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import torch
from baseline_suite.swinir import SwinIRMatched
from baseline_suite._swinir_vendor import SwinIR,window_partition,window_reverse
from baseline_suite.runtime import prepare_model,forward_padded,restore
from baseline_suite.train import main as train


def equivalent_state(target,source):
    """Only deterministic masks are intentionally omitted by cache-enabled model."""
    expected=target.state_dict();actual=source.state_dict()
    for key in set(expected)-set(actual):
        assert key.endswith('attn_mask'),key
        actual[key]=expected[key]
    target.load_state_dict(actual,strict=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(1234)
    x=torch.randn(2,16,24,5)
    assert torch.equal(window_reverse(window_partition(x,8),8,16,24),x)
    counts={}
    for mode in ('restore','sr'):
        default=SwinIRMatched(mode=mode)
        counts[mode]=sum(p.numel() for p in default.parameters())
        tiny=torch.rand(1,3,12,16)
        out=forward_padded(default,prepare_model(default,tiny,2))
        assert out.shape==(1,3,24,32)
        out.mean().backward()
        unused=[n for n,p in default.named_parameters() if p.grad is None]
        assert not unused,unused
        assert all(torch.isfinite(p.grad).all() for p in default.parameters() if p.grad is not None)
        del default
        kw=dict(mode=mode,embed_dim=24,depths=(2,2),num_heads=3)
        cached=SwinIRMatched(**kw).eval();plain=SwinIRMatched(**kw,cache_masks=False).eval()
        equivalent_state(plain,cached)
        for h,w in [(32,40),(64,64),(13,17),(1,3)]:
            inp=torch.rand(1,3,h,w)
            with torch.no_grad():
                c=cached(inp);r=plain(inp)
            assert torch.allclose(c,r,atol=1e-6,rtol=1e-6),(mode,h,w,(c-r).abs().max())
        assert len(cached.mask_cache.values)<=2
        inp=torch.rand(1,3,23,29)
        with torch.no_grad():
            first=restore(cached,inp,2,32,8,1)
            batched=restore(cached,inp,2,32,8,3)
        assert first.shape==(1,3,46,58) and torch.allclose(first,batched,atol=2e-5,rtol=2e-5)
        if mode=='sr':
            try:restore(cached,inp,3,tile_size=0)
            except ValueError:pass
            else:raise AssertionError('Wrong native SR scale accepted')
        # Direct wrapper/reference equivalence on window-aligned inputs.
        reference=SwinIR(img_size=64,patch_size=1,in_chans=3,embed_dim=24,
                         depths=(2,2),num_heads=(3,3),window_size=8,mlp_ratio=2,
                         drop_path_rate=0.,upscale=2 if mode=='sr' else 1,
                         upsampler='pixelshuffledirect' if mode=='sr' else '',
                         img_range=1.,resi_connection='1conv').eval()
        equivalent_state(reference,cached.core)
        inp=torch.rand(1,3,32,40)
        with torch.no_grad():assert torch.allclose(reference(inp),cached(inp),atol=1e-6,rtol=1e-6)
    for scale in [2,3,4]:
        m=SwinIRMatched(mode='sr',scale=scale,embed_dim=24,depths=(2,),num_heads=3).eval()
        with torch.no_grad():
            y=restore(m,torch.rand(1,3,11,13),scale,tile_size=0)
        assert y.shape==(1,3,11*scale,13*scale)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)/'DIV2K'
        for split,ids in [('train',[1,2]),('valid',[801])]:
            d=root/f'DIV2K_{split}_HR';d.mkdir(parents=True)
            for i in ids:Image.fromarray(np.random.default_rng(i).integers(0,256,(48,48,3),dtype=np.uint8)).save(d/f'{i:04d}.png')
        for name in ['swinir','swinir_sr']:
            output=Path(temp)/name
            common=['--model',name,'--model-kwargs','{"embed_dim":24,"depths":[2,2],"num_heads":3}',
                    '--data-root',str(root),'--device','cpu','--threads','2','--workers','0',
                    '--batch-size','2','--patch-size','32','--scale','2','--allow-partial',
                    '--max-train-steps','1','--max-val-images','1','--tile-size','32','--overlap','8',
                    '--val-every','1','--save-every','1','--output',str(output)]
            train(common+['--epochs','1'])
            train(common+['--epochs','2','--resume',str(output/'model_last.pth')])
            checkpoint=torch.load(output/'model_last.pth',weights_only=True,map_location='cpu')
            assert checkpoint['epoch']==2 and checkpoint['phase']=='complete'
            reloaded=SwinIRMatched(mode='sr' if name.endswith('_sr') else 'restore',**checkpoint['config']['model_kwargs'])
            reloaded.load_state_dict(checkpoint['state_dict'],strict=True)
    result=dict(status='passed',device='CPU',parameters=counts,submission_parameters=2302901,
                checks=['window partition/reverse','default-capacity forward/backward both modes',
                        'cached vs uncached output equality','bounded mask cache','upstream branch output equality',
                        'odd and tiny dimensions','tile batching equivalence','native scales2/3/4 and wrong-scale rejection',
                        'both modes shared train/validation/strict reload/epoch-extension resume'],
                limitations=['No CUDA speed/memory/AMP measurements','No trained accuracy or official pretrained equivalence claim'])
    print(json.dumps(result,indent=2))
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
