"""Shared model boundary and batched tile evaluation; ANN-only dependencies."""
import os
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from .metrics import y_metrics,json_scores


def atomic_save(checkpoint,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(checkpoint,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def prepare(lr,scale):
    return lr if scale==1 else F.interpolate(lr,scale_factor=scale,mode='bicubic',align_corners=False,antialias=False)


def prepare_model(model,lr,scale):
    if getattr(model,'input_mode','bicubic')=='native':
        if getattr(model,'scale',None)!=scale:
            raise ValueError('Native SR head scale differs from requested reconstruction scale')
        return lr
    return prepare(lr,scale)


def forward_padded(model,x):
    h,w=x.shape[-2:]
    if not getattr(model,'handles_padding',False):
        x=F.pad(x,(0,-w%8,0,-h%8),mode='replicate')
    y=model(x)
    factor=getattr(model,'scale',1) if getattr(model,'input_mode','bicubic')=='native' else 1
    expected=(*x.shape[:2],x.shape[-2]*factor,x.shape[-1]*factor)
    if tuple(y.shape)!=expected:raise ValueError(f'Output {y.shape} must equal {expected}')
    return y[...,:h*factor,:w*factor]


@torch.no_grad()
def restore(model,lr,scale=2,tile_size=192,overlap=32,tile_batch=4):
    if tile_size<0 or overlap<0 or (tile_size and overlap>=tile_size) or tile_batch<1:
        raise ValueError('Invalid tiling settings')
    x=prepare_model(model,lr,scale);h,w=x.shape[-2:]
    factor=scale if getattr(model,'input_mode','bicubic')=='native' else 1
    if factor!=1 and tile_size:
        if tile_size%factor or overlap%factor:
            raise ValueError('For native SR, HR tile-size and overlap must be divisible by scale')
        tile_size//=factor;overlap//=factor
    if not tile_size or max(h,w)<=tile_size:return forward_padded(model,x)
    if x.shape[0]!=1:raise ValueError('Tiled evaluation expects one source image')
    def positions(length):
        return [0] if length<=tile_size else list(range(0,length-tile_size,tile_size-overlap))+[length-tile_size]
    coords=[(y,z) for y in positions(h) for z in positions(w)]
    total=x.new_zeros(x.shape[0],x.shape[1],h*factor,w*factor);count=x.new_zeros(1,1,h*factor,w*factor)
    for start in range(0,len(coords),tile_batch):
        block=coords[start:start+tile_batch]
        batch=torch.cat([x[...,y:y+tile_size,z:z+tile_size] for y,z in block],0)
        output=forward_padded(model,batch)
        for j,(y,z) in enumerate(block):
            y*=factor;z*=factor
            hh,ww=output.shape[-2:];total[...,y:y+hh,z:z+ww]+=output[j:j+1]
            count[...,y:y+hh,z:z+ww]+=1
    return total/count


@torch.no_grad()
def evaluate(model,loader,device,scale=2,tile_size=192,overlap=32,tile_batch=4,border=0,
             max_images=None,output=None,log_every=10):
    model.eval();rows=[]
    if output:Path(output).mkdir(parents=True,exist_ok=True)
    for i,batch in enumerate(loader):
        if max_images and i>=max_images:break
        lr=batch['lr'].to(device,non_blocking=True)
        restored=restore(model,lr,scale,tile_size,overlap,tile_batch).clamp(0,1)
        target=batch['hr'][0]
        row={'name':batch['name'][0],**y_metrics(restored[0],target,border)}
        base=prepare(lr,scale)[0]
        row.update({'bicubic_'+k:v for k,v in y_metrics(base,target,border).items()})
        rows.append(row)
        if output:
            a=restored[0].cpu().permute(1,2,0).numpy()
            Image.fromarray(np.rint(a*255).astype(np.uint8)).save(Path(output)/batch['name'][0])
        if log_every and ((i+1)%log_every==0):print(f'  validation {i+1}/{len(loader)}',flush=True)
    if not rows:raise ValueError('No validation images')
    means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='name'}
    return means,[dict(name=r['name'],**json_scores({k:v for k,v in r.items() if k!='name'})) for r in rows]
