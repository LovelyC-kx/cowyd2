"""Clean-room restoration adaptations; see BASELINES.md for provenance/deviations.

Every model accepts/returns BCHW RGB at the SAME resolution. The shared runtime
performs bicubic upsampling first. No pretrained weights or original-paper scores
are implied. In particular CRNN is a ConvGRU restoration baseline, NOT OCR CRNN.
"""
import torch
from torch import nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,64,9,padding=4), nn.ReLU(inplace=True),
                                 nn.Conv2d(64,32,1), nn.ReLU(inplace=True),
                                 nn.Conv2d(32,3,5,padding=2))

    def forward(self,x):
        return self.net(x)


class VDSR(nn.Module):
    def __init__(self,width=64,depth=20):
        super().__init__()
        if depth<3:raise ValueError('depth >= 3 required')
        layers=[nn.Conv2d(3,width,3,padding=1,bias=False),nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers.extend([nn.Conv2d(width,width,3,padding=1,bias=False),nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(width,3,3,padding=1,bias=False))
        self.net=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):return x+self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self,width,post_relu=False,res_scale=1.):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.ReLU(inplace=True),
                                nn.Conv2d(width,width,3,padding=1))
        self.post_relu=post_relu;self.res_scale=res_scale

    def forward(self,x):
        y=x+self.body(x)*self.res_scale
        return F.relu(y) if self.post_relu else y


class EDSR(nn.Module):
    """16x64 baseline trunk, adapted to common preupsampled RGB interface."""
    def __init__(self,width=64,blocks=16,res_scale=1.):
        super().__init__()
        self.head=nn.Conv2d(3,width,3,padding=1)
        self.body=nn.Sequential(*[ResidualBlock(width,res_scale=res_scale) for _ in range(blocks)],
                                nn.Conv2d(width,width,3,padding=1))
        self.tail=nn.Conv2d(width,3,3,padding=1)

    def forward(self,x):
        f=self.head(x)
        return x+self.tail(self.body(f)+f)


class ResNetRestore(nn.Module):
    """Eight same-resolution 64-channel residual blocks with post-add ReLU.

    Restoration adaptation, no classifier/pooling, no BN. Not ResNet-18.
    """
    def __init__(self,width=64,blocks=8):
        super().__init__()
        self.head=nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.ReLU(inplace=True))
        self.body=nn.Sequential(*[ResidualBlock(width,post_relu=True) for _ in range(blocks)])
        self.tail=nn.Conv2d(width,3,3,padding=1)

    def forward(self,x):return x+self.tail(self.body(self.head(x)))


class DoubleConv(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1),nn.ReLU(inplace=True),
                         nn.Conv2d(cout,cout,3,padding=1),nn.ReLU(inplace=True))


class UNetRestore(nn.Module):
    """Three downsampling stages, widths 32/64/128/256, residual RGB head."""
    def __init__(self,width=32):
        super().__init__()
        self.e1=DoubleConv(3,width);self.e2=DoubleConv(width,width*2)
        self.e3=DoubleConv(width*2,width*4);self.mid=DoubleConv(width*4,width*8)
        self.u3=nn.Conv2d(width*8,width*4,1);self.d3=DoubleConv(width*8,width*4)
        self.u2=nn.Conv2d(width*4,width*2,1);self.d2=DoubleConv(width*4,width*2)
        self.u1=nn.Conv2d(width*2,width,1);self.d1=DoubleConv(width*2,width)
        self.out=nn.Conv2d(width,3,1)

    @staticmethod
    def up(x,skip,conv):
        return torch.cat([conv(F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)),skip],1)

    def forward(self,x):
        h,w=x.shape[-2:]
        a=F.pad(x,(0,-w%8,0,-h%8),mode='replicate')
        e1=self.e1(a);e2=self.e2(F.max_pool2d(e1,2));e3=self.e3(F.max_pool2d(e2,2))
        z=self.mid(F.max_pool2d(e3,2));z=self.d3(self.up(z,e3,self.u3))
        z=self.d2(self.up(z,e2,self.u2));z=self.d1(self.up(z,e1,self.u1))
        return (a+self.out(z))[...,:h,:w]


class ConvGRUCell(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.gates=nn.Conv2d(2*width,2*width,3,padding=1)
        self.candidate=nn.Conv2d(2*width,width,3,padding=1)

    def forward(self,x,h):
        reset,update=torch.sigmoid(self.gates(torch.cat([x,h],1))).chunk(2,1)
        proposal=torch.tanh(self.candidate(torch.cat([x,reset*h],1)))
        return (1-update)*h+update*proposal


class CRNNRestore(nn.Module):
    """Custom ConvGRU reconstruction baseline, four shared iterations at half size.

    No OCR sequence decoder/CTC, no hidden state retained between images/batches.
    """
    def __init__(self,width=32,steps=4):
        super().__init__()
        if steps<1:raise ValueError('steps >= 1 required')
        self.steps=steps
        self.encoder=nn.Sequential(nn.Conv2d(3,width,3,stride=2,padding=1),nn.ReLU(inplace=True))
        self.cell=ConvGRUCell(width)
        self.decoder=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.ReLU(inplace=True),
                                   nn.Conv2d(width,3,3,padding=1))

    def forward(self,x):
        f=self.encoder(x);h=torch.zeros_like(f)
        for _ in range(self.steps):h=self.cell(f,h)
        return x+self.decoder(F.interpolate(h,size=x.shape[-2:],mode='bilinear',align_corners=False))


def _swinir(**kwargs):
    from .swinir import build_swinir
    return build_swinir(**kwargs)


def _swinir_sr(**kwargs):
    from .swinir import build_swinir_sr
    return build_swinir_sr(**kwargs)


FACTORIES={'srcnn':SRCNN,'vdsr':VDSR,'edsr':EDSR,'crnn':CRNNRestore,
           'resnet':ResNetRestore,'unet':UNetRestore,'swinir':_swinir,'swinir_sr':_swinir_sr}


def build_model(name,**kwargs):
    if name not in FACTORIES:raise ValueError(f'Unknown model {name}; choose {list(FACTORIES)}')
    return FACTORIES[name](**kwargs)


if __name__=='__main__':
    for name in FACTORIES:
        m=build_model(name)
        print(f'{name:8s} {sum(p.numel() for p in m.parameters()):,}')
