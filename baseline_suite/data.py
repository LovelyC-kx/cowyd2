"""Deterministic paired-image crops, without silent pair-by-order fallback."""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .div2k import read_rgb,DIV2KDataset
from .protocol import file_sha256


class StrictPairs(Dataset):
    def __init__(self,inputs,targets,patch=None,scale=1,seed=1234):
        self.patch=patch;self.scale=scale;self.seed=seed;self.epoch=0
        def files(folder):
            result={}
            for p in sorted(Path(folder).iterdir()):
                if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
                    if p.stem in result:raise ValueError('Duplicate image stem: '+p.stem)
                    result[p.stem]=p.resolve()
            return result
        x,y=files(inputs),files(targets)
        if not x or x.keys()!=y.keys():raise ValueError('Input/target stems must match EXACTLY')
        self.pairs=[(name,x[name],y[name]) for name in sorted(x)]

    def __len__(self):return len(self.pairs)

    def set_epoch(self,epoch):self.epoch=epoch

    def __getitem__(self,index):
        name,inp,tar=self.pairs[index];x,y=read_rgb(inp),read_rgb(tar)
        if y.shape[-2:]!=tuple(v*self.scale for v in x.shape[-2:]):
            raise ValueError(f'{name}: target dimensions must equal scale times input')
        if self.patch:
            p=self.patch
            if min(y.shape[-2:])<p:raise ValueError(f'{name}: target smaller than patch')
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
            h,w=x.shape[-2:];lp=p//self.scale
            top,left=int(rng.integers(h-lp+1)),int(rng.integers(w-lp+1))
            x=x[:,top:top+lp,left:left+lp]
            y=y[:,top*self.scale:top*self.scale+p,left*self.scale:left*self.scale+p]
        return {'lr':x.contiguous(),'hr':y.contiguous(),'name':name+'.png'}

    def manifest(self):
        return {'files':[{'name':n,'input_sha256':file_sha256(x),'target_sha256':file_sha256(y)} for n,x,y in self.pairs]}


def datasets(config):
    from .protocol import Protocol
    if config['task']=='div2k':
        proto=Protocol(**config['protocol'])
        return (DIV2KDataset(config['data_root'],'train',proto,config['allow_partial']),
                DIV2KDataset(config['data_root'],'valid',proto,config['allow_partial']))
    kwargs=dict(scale=config['protocol']['scale'],seed=config['protocol']['seed'])
    train=StrictPairs(config['train_input'],config['train_target'],config['protocol']['hr_patch_size'],**kwargs)
    valid=StrictPairs(config['val_input'],config['val_target'],None,**kwargs)
    paths=lambda ds:{p for _,x,y in ds.pairs for p in (x,y)}
    if paths(train)&paths(valid):raise ValueError('Train and validation paths overlap')
    return train,valid
