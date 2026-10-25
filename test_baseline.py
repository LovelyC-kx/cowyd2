"""Re-evaluate a saved baseline on its recorded validation set; no training."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from baseline_suite.models import build_model
from baseline_suite.data import datasets
from baseline_suite.protocol import write_json,fingerprint
from baseline_suite.metrics import json_scores
from baseline_suite.runtime import evaluate
from baseline_suite.train import source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--data-root',type=Path,help='Relocate DIV2K only; contents are rechecked')
    p.add_argument('--save-images',action='store_true');a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Choose a fresh output')
    ck=torch.load(a.weights,map_location='cpu',weights_only=True)
    if ck['format']!='cowyd2-baseline-v1':raise ValueError('Wrong checkpoint format')
    if ck['source_hashes']!=source_hashes():raise ValueError('Baseline source differs from checkpoint')
    c=dict(ck['config'])
    if a.data_root:c['data_root']=str(a.data_root)
    torch.set_num_threads(c['threads']);device=torch.device(a.device)
    train,valid=datasets(c)
    data={'train':train.manifest(),'valid':valid.manifest()}
    proto={k:v for k,v in c['protocol'].items() if k not in ('epochs','batch_size','learning_rate')}
    if fingerprint({'task':c['task'],'protocol':proto,'files':data})!=ck['data_fingerprint']:
        raise ValueError('Dataset contents differ from training run')
    model=build_model(c['model'],**c['model_kwargs']).to(device).eval()
    model.load_state_dict(ck['state_dict'],strict=True)
    loader=DataLoader(valid,batch_size=1,num_workers=c['workers'],shuffle=False,
                      generator=torch.Generator().manual_seed(c['seed']))
    means,rows=evaluate(model,loader,device,c['scale'],c['tile_size'],c['overlap'],c['tile_batch'],
                        c['metric_border'],c['max_val_images'],a.output/'images' if a.save_images else None)
    write_json(a.output/'metrics.json',dict(epoch=ck['epoch'],validation_result=True,
               mean=json_scores(means),images=rows,data_fingerprint=ck['data_fingerprint']))
    print(json_scores(means))


if __name__=='__main__':main()
