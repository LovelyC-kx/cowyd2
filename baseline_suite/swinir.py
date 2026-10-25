"""Capacity-matched official SwinIR branches; no changes to attention equations.

Source: JingyunLiang/SwinIR, Apache-2.0; see SWINIR_LICENSE.txt and SWINIR.md.
Default capacity: C96, four RSTBs of six Swin blocks, six heads, MLP ratio2.
"""
from collections import OrderedDict
import types
import torch
from torch import nn
import torch.nn.functional as F
from ._swinir_vendor import SwinIR,SwinTransformerBlock


class _MaskCache:
    """Share immutable shifted-window masks across blocks; bounded to two shapes."""
    def __init__(self,limit=2):self.values=OrderedDict();self.limit=limit

    def get(self,block,size):
        key=(tuple(size),block.window_size,block.shift_size,str(block.norm1.weight.device))
        if key not in self.values:
            mask=block._original_calculate_mask(size).to(block.norm1.weight.device)
            self.values[key]=mask
            while len(self.values)>self.limit:self.values.popitem(last=False)
        self.values.move_to_end(key)
        return self.values[key]


def _cached_mask(block,size):
    return block._shared_mask_cache.get(block,size)


class SwinIRMatched(nn.Module):
    """mode='restore': same-size residual head; mode='sr': native-LR direct SR.

    Native SR is scale-specific. Restore mode uses external bicubic and has an
    internal scale of 1, irrespective of the external reconstruction scale.
    """
    def __init__(self,mode='restore',scale=2,embed_dim=96,depths=(6,6,6,6),
                 num_heads=6,window_size=8,mlp_ratio=2.,drop_path_rate=0.,
                 cache_masks=True):
        super().__init__()
        if mode not in ('restore','sr'):raise ValueError('mode must be restore or sr')
        if scale not in (1,2,3,4) or (mode=='sr' and scale==1):raise ValueError('Native SR requires scale2/3/4')
        depths=tuple(depths)
        heads=(num_heads,)*len(depths) if isinstance(num_heads,int) else tuple(num_heads)
        if not depths or min(depths)<1 or len(heads)!=len(depths):raise ValueError('Invalid depths/heads')
        if window_size<2 or any(h<1 or embed_dim%h for h in heads):raise ValueError('Width must be divisible by heads')
        self.input_mode='native' if mode=='sr' else 'bicubic'
        self.handles_padding=mode=='sr'
        self.scale=scale if mode=='sr' else 1
        self.window_size=window_size
        self.core=SwinIR(img_size=64,patch_size=1,in_chans=3,embed_dim=embed_dim,
                         depths=depths,num_heads=heads,window_size=window_size,
                         mlp_ratio=mlp_ratio,drop_path_rate=drop_path_rate,
                         upscale=self.scale,img_range=1.,
                         upsampler='pixelshuffledirect' if mode=='sr' else '',
                         resi_connection='1conv',use_checkpoint=False)
        self.configuration=dict(mode=mode,scale=self.scale,embed_dim=embed_dim,depths=list(depths),
                                num_heads=list(heads),window_size=window_size,mlp_ratio=mlp_ratio,
                                drop_path_rate=drop_path_rate,cache_masks=cache_masks)
        if cache_masks:
            self.mask_cache=_MaskCache()
            for block in self.core.modules():
                if isinstance(block,SwinTransformerBlock):
                    block._original_calculate_mask=block.calculate_mask
                    block._shared_mask_cache=self.mask_cache
                    block.calculate_mask=types.MethodType(_cached_mask,block)
                    # Deterministically derived, not a learned/training state.
                    block._non_persistent_buffers_set.add('attn_mask')
                    if block.shift_size:
                        block.attn_mask=self.mask_cache.get(block,block.input_resolution)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3:raise ValueError('Expected BCHW RGB')
        h,w=x.shape[-2:]
        # Original reflection padding fails on images smaller than its padding.
        ph,pw=-h%self.window_size,-w%self.window_size
        if ph or pw:
            mode='reflect' if ph<h and pw<w else 'replicate'
            x=F.pad(x,(0,pw,0,ph),mode=mode)
        y=self.core(x)
        return y[...,:h*self.scale,:w*self.scale]


def build_swinir(**kwargs):return SwinIRMatched(mode='restore',**kwargs)


def build_swinir_sr(**kwargs):return SwinIRMatched(mode='sr',**kwargs)
