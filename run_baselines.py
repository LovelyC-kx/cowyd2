"""Run requested baselines SEQUENTIALLY on one GPU (never launch six at once)."""
import argparse
from pathlib import Path
import subprocess
import sys
import torch
from baseline_suite.models import FACTORIES


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=FACTORIES,default=list(FACTORIES))
    p.add_argument('--resume-existing',action='store_true')
    args,remaining=p.parse_known_args()
    if any(s.split('=')[0] in ('--model','--output','--resume') for s in remaining):
        p.error('Runner controls --model/--output/--resume')
    epoch=200
    for i,t in enumerate(remaining):
        if t=='--epochs':epoch=int(remaining[i+1])
        elif t.startswith('--epochs='):epoch=int(t.split('=',1)[1])
    root=Path(__file__).resolve().parent
    for name in args.models:
        output=args.output_root/name
        command=[sys.executable,str(root/'train_baseline.py'),'--model',name,'--output',str(output),*remaining]
        ck=output/'model_last.pth'
        if args.resume_existing and ck.exists():
            state=torch.load(ck,map_location='cpu',weights_only=True)
            if state['phase']=='complete' and state['epoch']>=epoch:
                print('Already complete, skip:',name,flush=True);continue
            command+=['--resume',str(ck)]
        print('Starting:',name,'->',output,flush=True)
        subprocess.run(command,check=True)


if __name__=='__main__':main()
