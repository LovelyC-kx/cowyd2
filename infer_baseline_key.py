"""Baseline checkpoint -> measured key matrix, matching existing plot file schema."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from baseline_suite.models import build_model
from baseline_suite.runtime import restore,prepare
from baseline_suite.train import source_hashes
from infer_key import read_matrix,export,preview,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--matrix',type=Path,default=Path(__file__).with_name('yaoshi.xls'))
    p.add_argument('--opju',type=Path,help='Optional provenance, no numeric extraction')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--scales',type=int,nargs='+',choices=[1,2,3,4],
                   help='Default: trained scale for native SR, otherwise 1 2 3 4')
    p.add_argument('--polarity',choices=['positive','negative'],default='positive')
    p.add_argument('--low',type=float);p.add_argument('--high',type=float)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Choose a fresh output')
    ck=torch.load(a.weights,map_location='cpu',weights_only=True)
    if ck['format']!='cowyd2-baseline-v1':raise ValueError('Wrong checkpoint format')
    if ck['source_hashes']!=source_hashes():raise ValueError('Baseline source differs from checkpoint')
    c=ck['config'];torch.set_num_threads(c['threads'])
    model=build_model(c['model'],**c['model_kwargs']).to(a.device).eval()
    model.load_state_dict(ck['state_dict'],strict=True)
    native=getattr(model,'input_mode','bicubic')=='native'
    scales=a.scales or ([model.scale] if native else [1,2,3,4])
    if native and any(s!=model.scale for s in scales):
        p.error('Native SR is scale-specific; use only its trained scale')
    raw=read_matrix(a.matrix)
    if (a.low is None)!=(a.high is None):p.error('Both bounds required')
    low,high=(float(raw.min()),float(raw.max())) if a.low is None else (a.low,a.high)
    if not np.isfinite([low,high]).all() or high<=low:p.error('Invalid bounds')
    original=(raw-low)/(high-low);clipped=int(np.count_nonzero((original<0)|(original>1)))
    x=np.clip(original,0,1)
    if a.polarity=='negative':x=1-x
    report=dict(matrix=str(a.matrix.resolve()),matrix_sha256=sha(a.matrix),input_shape=list(raw.shape),
                low=low,high=high,polarity=a.polarity,clipped_input_count=clipped,
                weights=str(a.weights.resolve()),weights_sha256=sha(a.weights),model=c['model'],
                trained_task=c['task'],trained_scale=c['scale'],training_data_fingerprint=ck['data_fingerprint'],
                reference_available=False,metrics=None,scales={},
                interpretation='Direct measured-input transfer. Non-training scales are exploratory. No physical resolution claim.',
                model_input_mode=getattr(model,'input_mode','bicubic'),
                tiling='full frame; may differ from validation tiles')
    if a.opju:report['opju']=dict(path=str(a.opju.resolve()),sha256=sha(a.opju))
    a.output.mkdir(parents=True)
    export(a.output,'input_signal',raw);export(a.output,'input_normalized',x);preview(a.output/'input_preview.png',x)
    inp=torch.from_numpy(x.astype(np.float32))[None,None].repeat(1,3,1,1).to(a.device)
    print(f'Strict load OK: {c["model"]}, {sum(p.numel() for p in model.parameters()):,} parameters',flush=True)
    with torch.no_grad():
        for scale in dict.fromkeys(scales):
            folder=a.output/f'x{scale}';folder.mkdir()
            rgb=restore(model,inp,scale,tile_size=0)[0].cpu().permute(1,2,0).numpy()
            if not np.isfinite(rgb).all():raise ValueError('Non-finite output')
            gray=rgb@np.array([.299,.587,.114]);base=prepare(inp,scale)[0,0].cpu().numpy()
            def inverse(v):return (1-v if a.polarity=='negative' else v)*(high-low)+low
            export(folder,'restored_normalized',gray);export(folder,'restored_signal',inverse(gray))
            export(folder,'bicubic_signal',inverse(base));np.save(folder/'restored_rgb.npy',rgb.transpose(2,0,1))
            preview(folder/'restored_preview.png',gray);preview(folder/'bicubic_preview.png',base)
            report['scales'][str(scale)]=dict(output_shape=list(gray.shape),
                outside_mapping_range=int(np.count_nonzero((gray<0)|(gray>1))))
            (a.output/'report.json').write_text(json.dumps(report,indent=2))
            print(f'x{scale} -> {gray.shape}: {folder}',flush=True)


if __name__=='__main__':main()
