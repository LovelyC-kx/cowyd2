"""Shared baseline training: fixed Adam/RGB-MSE; deterministic data; atomic resume."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import build_model,FACTORIES
from .data import datasets
from .protocol import Protocol,fingerprint,file_sha256,write_json
from .metrics import json_scores
from .runtime import atomic_save,forward_padded,prepare,evaluate


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def rng_state(device):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}


def restore_rng(rng,device):
    random.setstate(rng['python']);n=rng['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(rng['torch'])
    if device.type=='cuda' and rng['cuda'] is not None:torch.cuda.set_rng_state(rng['cuda'],device)


def source_hashes():
    return {p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')}


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=FACTORIES,required=True)
    p.add_argument('--model-kwargs',default='{}')
    p.add_argument('--task',choices=['div2k','paired'],default='div2k')
    p.add_argument('--data-root',type=Path)
    for k in ['train-input','train-target','val-input','val-target']:p.add_argument('--'+k,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--patch-size',type=int,default=192);p.add_argument('--scale',type=int,choices=[1,2,3,4])
    p.add_argument('--lr',type=float,default=5e-6);p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--blur-sigma',type=float,default=1.6);p.add_argument('--kernel-size',type=int,default=21)
    p.add_argument('--noise-std',type=float,default=.01);p.add_argument('--metric-border',type=int,default=0)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--val-every',type=int,default=5);p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--log-every',type=int,default=25)
    p.add_argument('--tile-size',type=int,default=192);p.add_argument('--overlap',type=int,default=32)
    p.add_argument('--tile-batch',type=int,default=4)
    p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    p.add_argument('--fast-cudnn',action='store_true',help='Explicit speed mode; may not be bitwise reproducible')
    p.add_argument('--clip-grad',type=float,default=0.)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--max-train-steps',type=int)
    p.add_argument('--max-val-images',type=int)
    a=p.parse_args(argv)
    if min(a.epochs,a.batch_size,a.val_every,a.save_every,a.log_every,a.threads,a.tile_batch)<1:p.error('Counts must be positive')
    if a.workers<0 or a.tile_size<0 or a.overlap<0 or (a.tile_size and a.overlap>=a.tile_size):p.error('Invalid worker/tiling settings')
    for n in ['max_train_steps','max_val_images']:
        if getattr(a,n) is not None and getattr(a,n)<1:p.error(n+' must be positive')
    if a.lr<=0 or not np.isfinite(a.lr) or a.clip_grad<0:p.error('Invalid LR/gradient clipping')
    if a.task=='div2k' and (a.data_root is None or a.scale==1):p.error('DIV2K requires --data-root and scale 2/3/4')
    if a.task=='paired' and any(getattr(a,k) is None for k in ['train_input','train_target','val_input','val_target']):p.error('Paired task requires four explicit disjoint input/target folders')
    return a


def main(argv=None):
    a=parse(argv);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    if device.type!='cuda' and a.precision!='fp32':raise ValueError('AMP options require CUDA')
    torch.set_num_threads(a.threads)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=a.fast_cudnn;torch.backends.cudnn.deterministic=not a.fast_cudnn
    scale=a.scale or (2 if a.task=='div2k' else 1)
    if a.patch_size<16 or a.patch_size%scale:raise ValueError('Patch must be >=16 and divisible by scale')
    if a.task=='div2k':
        proto=Protocol(scale=scale,hr_patch_size=a.patch_size,blur_sigma=a.blur_sigma,
                       kernel_size=a.kernel_size,noise_std=a.noise_std,seed=a.seed,
                       metric_border=a.metric_border,epochs=a.epochs,batch_size=a.batch_size,learning_rate=a.lr).to_dict()
    else:
        proto=dict(scale=scale,hr_patch_size=a.patch_size,seed=a.seed,metric_border=a.metric_border)
    config={k:(str(v.resolve()) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    config['protocol']=proto;config['scale']=scale;config['model_kwargs']=json.loads(a.model_kwargs)
    if not isinstance(config['model_kwargs'],dict):raise ValueError('--model-kwargs must be a JSON object')
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:raise FileExistsError('Output nonempty; use --resume or a fresh directory')
    train,valid=datasets(config)
    data={'train':train.manifest(),'valid':valid.manifest()}
    if a.task=='paired':
        pair_hash=lambda split:{(r['input_sha256'],r['target_sha256']) for r in data[split]['files']}
        if pair_hash('train')&pair_hash('valid'):raise ValueError('Identical image pairs appear in train and validation')
    data_proto={k:v for k,v in proto.items() if k not in ('epochs','batch_size','learning_rate')}
    data_hash=fingerprint({'task':a.task,'protocol':data_proto,'files':data})
    immutable={k:config[k] for k in ('model','model_kwargs','task','batch_size','lr','seed','precision','fast_cudnn','clip_grad','tile_size','overlap','tile_batch','max_train_steps','max_val_images','allow_partial')}
    signature=fingerprint({'settings':immutable,'data':data_hash})
    sources=source_hashes()
    model=build_model(a.model,**config['model_kwargs']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and a.precision=='fp16') if hasattr(torch,'amp') and hasattr(torch.amp,'GradScaler') else torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and a.precision=='fp16')
    start=1;best=-float('inf');pending=None
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=True)
        if ck['format']!='cowyd2-baseline-v1' or ck['signature']!=signature:raise ValueError('Resume data/model/training/evaluation signature mismatch')
        if ck['source_hashes']!=sources:raise ValueError('Resume source changed; use matching baseline_suite code')
        model.load_state_dict(ck['state_dict'],strict=True);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        best=ck['best_psnr_y'];start=ck['epoch']+(ck['phase']=='complete');pending=ck if ck['phase']=='validation_pending' else None
        restore_rng(ck['rng'],device)
        print(f'Resume epoch={ck["epoch"]}, phase={ck["phase"]}; next epoch={start}',flush=True)
    if start>a.epochs:raise ValueError('Checkpoint already completed requested total epochs')
    a.output.mkdir(parents=True,exist_ok=True)
    # Recover the checkpoint's completed row if interruption occurred before log append.
    if a.resume and ck['phase']=='complete':
        log=a.output/'train.jsonl';rows=[]
        if log.exists():
            for line in log.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
        if not any(r.get('epoch')==ck['epoch'] for r in rows):
            rows.append(ck['row'])
            temp=log.with_suffix('.jsonl.tmp')
            temp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));os.replace(temp,log)
    manifest=dict(config=config,parameters=sum(p.numel() for p in model.parameters()),data=data,
                  data_fingerprint=data_hash,signature=signature,source_hashes=sources,
                  author_pipeline_exact=False,smoke_test=bool(a.allow_partial or a.max_train_steps or a.max_val_images),
                  versions={n:__import__('importlib.metadata',fromlist=['version']).version(n) for n in ['torch','numpy','Pillow','scipy']})
    write_json(a.output/'manifest.json',manifest)
    with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps({'event':'start_or_resume','time':time.time(),'config':config})+'\n')
    print(f'Model={a.model}; parameters={manifest["parameters"]:,}; data={data_hash}',flush=True)
    print('Shared declared protocol, not a reproduction of unpublished THz-Net degradation or original-paper training recipes.',flush=True)
    def checkpoint(epoch,phase,row):
        return dict(format='cowyd2-baseline-v1',epoch=epoch,phase=phase,row=row,
                    state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    rng=rng_state(device),config=config,signature=signature,data_fingerprint=data_hash,
                    source_hashes=sources,best_psnr_y=best)
    for ep in range(start,a.epochs+1):
        if pending is not None:
            row=dict(pending['row']);pending=None
            print(f'Epoch {ep}/{a.epochs}: training already saved; redo pending validation',flush=True)
        else:
            train.set_epoch(ep)
            gen=torch.Generator().manual_seed(a.seed+ep)
            loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,
                              generator=gen,pin_memory=device.type=='cuda',drop_last=False)
            model.train();sync(device);tic=time.perf_counter();loss_sum=0.;seen=steps=0
            print(f'Epoch {ep}/{a.epochs}: training {len(loader)} batches',flush=True)
            for i,batch in enumerate(loader):
                if a.max_train_steps and i>=a.max_train_steps:break
                x=batch['lr'].to(device,non_blocking=True);y=batch['hr'].to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                ctx=torch.autocast('cuda',dtype=torch.float16 if a.precision=='fp16' else torch.bfloat16) if device.type=='cuda' and a.precision!='fp32' else nullcontext()
                with ctx:
                    pred=forward_padded(model,prepare(x,scale));loss=torch.nn.functional.mse_loss(pred,y)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
                scaler.scale(loss).backward()
                if a.clip_grad:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),a.clip_grad)
                scaler.step(optimizer);scaler.update()
                loss_sum+=float(loss.detach())*len(x);seen+=len(x);steps+=1
                if steps%a.log_every==0:print(f'  train batch {steps}/{len(loader)}; mean MSE={loss_sum/seen:.7f}',flush=True)
            if not seen:raise ValueError('No train batches')
            sync(device);row=dict(epoch=ep,mse_rgb=loss_sum/seen,train_images=seen,steps=steps,lr=a.lr,
                                 train_seconds=time.perf_counter()-tic,val_seconds=0.)
            print(f'Epoch {ep}: train finished {row["train_seconds"]:.1f}s',flush=True)
        is_best=False
        if ep%a.val_every==0 or ep==a.epochs:
            atomic_save(checkpoint(ep,'validation_pending',row),a.output/'model_last.pth')
            print(f'Epoch {ep}: full validation started',flush=True);sync(device);tic=time.perf_counter()
            vloader=DataLoader(valid,batch_size=1,shuffle=False,num_workers=a.workers,
                               generator=torch.Generator().manual_seed(a.seed),pin_memory=device.type=='cuda')
            means,rows=evaluate(model,vloader,device,scale,a.tile_size,a.overlap,a.tile_batch,
                                a.metric_border,a.max_val_images)
            sync(device);row['val_seconds']=time.perf_counter()-tic;row.update(json_scores(means))
            write_json(a.output/'validation_latest.json',dict(epoch=ep,mean=json_scores(means),images=rows))
            if means['psnr_y']>best:best=means['psnr_y'];is_best=True
            print(f'Epoch {ep}: validation finished {row["val_seconds"]:.1f}s',flush=True)
        row['seconds']=row['train_seconds']+row['val_seconds']
        ck=checkpoint(ep,'complete',row)
        atomic_save(ck,a.output/'model_last.pth')
        if is_best:atomic_save(ck,a.output/'model_best.pth')
        if ep%a.save_every==0:atomic_save(ck,a.output/f'model_epoch_{ep:04d}.pth')
        # Checkpoint before log. Resume repairs a missing final log entry below.
        with (a.output/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)


if __name__=='__main__':main()
