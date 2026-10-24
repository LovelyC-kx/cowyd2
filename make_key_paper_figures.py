"""Reproducible Fig4/5/6-style plots from existing key inference matrices."""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import importlib.metadata
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(fig, stem):
    fig.savefig(str(stem) + '.png', dpi=240, bbox_inches='tight')
    fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
    plt.close(fig)


def sample(a, x, y, shape):
    h, w = shape
    xx = (np.asarray(x) + .5) * a.shape[1] / w - .5
    yy = (np.asarray(y) + .5) * a.shape[0] / h - .5
    return map_coordinates(a, [yy, xx], order=1, mode='nearest')


def load_runs(root, explicit):
    reports = [p / 'report.json' for p in explicit] if explicit else sorted(root.rglob('report.json'))
    groups = {}
    for p in reports:
        if not (p.parent / 'input_normalized.npy').exists():
            continue
        r = json.loads(p.read_text())
        source = np.load(p.parent / 'input_normalized.npy')
        if not np.isfinite(source).all() or source.ndim != 2:
            raise ValueError(f'Invalid input: {p}')
        key = (r['matrix_sha256'], r['low'], r['high'], r['polarity'])
        for scale in range(1, 5):
            q = p.parent / f'x{scale}'
            if not (q / 'restored_normalized.npy').exists():
                continue
            out = np.load(q / 'restored_normalized.npy')
            base = (np.load(q / 'bicubic_signal.npy') - r['low']) / (r['high'] - r['low'])
            if r['polarity'] == 'negative':
                base = 1 - base
            expected = tuple(s * scale for s in source.shape)
            if out.shape != expected or base.shape != expected or not np.isfinite(out).all():
                raise ValueError(f'Invalid dimensions or values: {q}')
            g = groups.setdefault(key, {'input': source, 'scales': {}, 'sources': []})
            if not np.array_equal(source, g['input']):
                raise ValueError('Matching mapping but inconsistent input arrays')
            methods = g['scales'].setdefault(scale, {'Measured input': source, 'Bicubic': base})
            if not np.allclose(methods['Bicubic'], base, rtol=0, atol=1e-6):
                raise ValueError('Inconsistent bicubic baselines')
            methods[p.parent.name] = out
            g['sources'].append({'report': str(p.resolve()), 'report_sha256': digest(p),
                                 'matrix': str(q / 'restored_normalized.npy'),
                                 'matrix_sha256': digest(q / 'restored_normalized.npy')})
    return groups


def choose_roi(a, requested):
    h, w = a.shape
    if requested:
        x, y, rw, rh = requested
    else:
        gy, gx = np.gradient(a)
        margin = max(2, min(h, w) // 5)
        score = np.hypot(gx, gy)
        score[:margin] = score[-margin:] = 0
        score[:, :margin] = score[:, -margin:] = 0
        cy, cx = np.unravel_index(score.argmax(), score.shape)
        rw, rh = max(8, w // 4), max(8, h // 4)
        x, y = min(max(0, cx-rw//2), w-rw), min(max(0, cy-rh//2), h-rh)
    if min(x, y) < 0 or min(rw, rh) < 2 or x+rw > w or y+rh > h:
        raise ValueError('ROI outside input coordinates')
    return int(x), int(y), int(rw), int(rh)


def image(ax, a, shape, title, cmap):
    h, w = shape
    im = ax.imshow(a, origin='upper', extent=(-.5,w-.5,h-.5,-.5),
                   vmin=0, vmax=1, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=9, wrap=True)
    ax.set_xlabel('Input column index')
    ax.set_ylabel('Input row index')
    return im


def fig45(methods, source, folder, scale, roi, row, col, band, cmap):
    h, w = source.shape
    x, y, rw, rh = roi
    n = len(methods)
    fig, axes = plt.subplots(2,n,figsize=(3.2*n,6.5),squeeze=False,constrained_layout=True)
    for j,(name,a) in enumerate(methods.items()):
        im = image(axes[0,j],a,source.shape,name,cmap)
        axes[0,j].add_patch(Rectangle((x-.5,y-.5),rw,rh,fill=False,ec='white',lw=1.5))
        axes[0,j].axhline(row,color='cyan',ls='--',lw=1)
        axes[0,j].axvline(col,color='magenta',ls='--',lw=1)
        image(axes[1,j],a,source.shape,'Fixed ROI',cmap)
        axes[1,j].set_xlim(x-.5,x+rw-.5)
        axes[1,j].set_ylim(y+rh-.5,y-.5)
    fig.suptitle(f'Fig4a | x{scale} | measured input is not independent ground truth')
    fig.colorbar(im,ax=list(axes.flat),label='Normalized signal',shrink=.75)
    save(fig,folder/'Fig4a_images_ROI')

    offsets = np.linspace(-(band-1)/2,(band-1)/2,band)
    colors = plt.get_cmap('tab10')
    for direction,label,length,fixed,zoom in [('horizontal','b',w,row,(x,x+rw-1)),
                                              ('vertical','c',h,col,(y,y+rh-1))]:
        coord = np.linspace(0,length-1,(length-1)*scale+1)
        fig,axes = plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
        table = {'input_coordinate':coord}
        for j,(name,a) in enumerate(methods.items()):
            if direction == 'horizontal':
                p = np.mean([sample(a,coord,np.full_like(coord,np.clip(fixed+d,0,h-1)),source.shape) for d in offsets],axis=0)
            else:
                p = np.mean([sample(a,np.full_like(coord,np.clip(fixed+d,0,w-1)),coord,source.shape) for d in offsets],axis=0)
            table[name] = p
            for ax in axes:
                ax.plot(coord,p,label=name,color=colors(j%10),lw=1.4)
        axes[0].axvspan(*zoom,color='grey',alpha=.12)
        axes[1].set_xlim(*zoom)
        for ax in axes:
            ax.set_xlabel('Input '+('column' if direction=='horizontal' else 'row')+' index')
            ax.set_ylabel('Normalized signal')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'Fig4{label} | {direction} profile at {fixed}; band={band} input pixels')
        save(fig,folder/f'Fig4{label}_{direction}_profiles')
        with (folder/f'Fig4{label}_profiles.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(table);writer.writerows(zip(*table.values()))

    fig=plt.figure(figsize=(3.5*n,7),constrained_layout=True)
    # One shared height range includes overshoot, while all color limits stay [0,1].
    zlo=min(0.,min(float(a.min()) for a in methods.values()))
    zhi=max(1.,max(float(a.max()) for a in methods.values()))
    for j,(name,a) in enumerate(methods.items()):
        ax=fig.add_subplot(2,n,j+1);im=image(ax,a,source.shape,name,cmap)
        ax=fig.add_subplot(2,n,n+j+1,projection='3d')
        yy=(np.arange(a.shape[0])+.5)*h/a.shape[0]-.5
        xx=(np.arange(a.shape[1])+.5)*w/a.shape[1]-.5
        # At most 120 samples per axis for rendering, not for numerical analysis.
        iy=np.unique(np.linspace(0,a.shape[0]-1,min(120,a.shape[0])).astype(int))
        ix=np.unique(np.linspace(0,a.shape[1]-1,min(120,a.shape[1])).astype(int))
        xx,yy=np.meshgrid(xx[ix],yy[iy])
        ax.plot_surface(xx,yy,a[np.ix_(iy,ix)],cmap=cmap,vmin=0,vmax=1,
                        rcount=len(iy),ccount=len(ix),linewidth=0,antialiased=True)
        ax.set_zlim(zlo,zhi);ax.set_ylim(h-.5,-.5);ax.view_init(elev=30,azim=-60)
        ax.set_xlabel('Column');ax.set_ylabel('Row');ax.set_zlabel('Signal')
    fig.suptitle(f'Fig5 | x{scale} | signal surfaces, not physical object topography')
    fig.colorbar(im,ax=fig.axes,label='Normalized signal',shrink=.6)
    save(fig,folder/'Fig5_images_surfaces')


def fig6(methods, source, folder, args):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    h,w=source.shape;p=args.patch;stride=args.stride
    if p>min(h,w):raise ValueError('Patch exceeds input dimensions')
    positions=[(y,x) for y in range(0,h-p+1,stride) for x in range(0,w-p+1,stride)]
    if len(positions)>args.max_patches:
        keep=np.random.default_rng(args.seed).choice(len(positions),args.max_patches,replace=False)
        positions=[positions[i] for i in sorted(keep)]
    if len(positions)<5:raise ValueError('Need at least 5 patches per method')
    features=[];records=[]
    gy,gx=np.mgrid[0:h,0:w]
    for name,a in methods.items():
        aligned=sample(a,gx,gy,source.shape)
        for patch_id,(y,x) in enumerate(positions):
            features.append(aligned[y:y+p,x:x+p].ravel())
            records.append((name,patch_id,x,y,float(x+p/2),float(y+p/2)))
    features=np.asarray(features)
    standardized=StandardScaler().fit_transform(features)
    components=min(50,len(features)-1,features.shape[1])
    reduced=PCA(n_components=components,random_state=args.seed).fit_transform(standardized)
    perplexity=min(30.,max(2.,(len(features)-1)/3))
    settings=dict(n_components=2,perplexity=perplexity,learning_rate=200.,
                  init='pca',random_state=args.seed)
    settings['max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter']=1000
    ts=TSNE(**settings).fit_transform(reduced)
    neighbors=min(15,len(features)-1)
    um=umap.UMAP(n_neighbors=neighbors,min_dist=.1,n_components=2,
                 metric='euclidean',random_state=args.seed,n_jobs=1).fit_transform(reduced)
    fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    names=list(methods)
    for col,(title,embedding) in enumerate([('t-SNE',ts),('UMAP',um)]):
        for i,name in enumerate(names):
            mask=np.array([r[0]==name for r in records])
            axes[0,col].scatter(*embedding[mask].T,s=13,alpha=.65,label=name)
        axes[0,col].set_title(title+' | color = method');axes[0,col].legend(fontsize=7)
        im=axes[1,col].scatter(*embedding.T,c=[r[-1] for r in records],cmap='viridis',s=13,alpha=.7)
        axes[1,col].set_title(title+' | color = source patch row')
        fig.colorbar(im,ax=axes[1,col],label='Source patch center row')
        for ax in axes[:,col]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Fig6 | Single-image matched-patch exploration\nOverlapping patches are dependent; clusters do not establish domain generalization',fontsize=11)
    save(fig,folder/'Fig6_patch_embeddings')
    np.savez_compressed(folder/'Fig6_features_embeddings.npz',features=features,pca=reduced,tsne=ts,umap=um)
    with (folder/'Fig6_samples.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','patch_id','left','top','center_x','center_y','tsne_x','tsne_y','umap_x','umap_y'])
        writer.writerows([list(r)+list(t)+list(u) for r,t,u in zip(records,ts,um)])
    return dict(feature='flattened raw normalized spatial patches; no per-patch normalization',
                alignment='linear interpolation at input pixel centers',patch=p,stride=stride,
                patches_per_method=len(positions),pca_components=components,
                standardization='joint feature-wise StandardScaler',
                tsne=dict(settings),umap=dict(n_neighbors=neighbors,min_dist=.1,metric='euclidean',random_state=args.seed),
                inference='exploratory single-image visualization only; no independent samples or domain claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results'))
    p.add_argument('--runs',type=Path,nargs='+')
    p.add_argument('--output',type=Path,default=Path('paper_figures'))
    p.add_argument('--roi',type=int,nargs=4,metavar=('X','Y','WIDTH','HEIGHT'))
    p.add_argument('--row',type=float);p.add_argument('--col',type=float)
    p.add_argument('--band',type=int,default=3)
    p.add_argument('--cmap',default='turbo');p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--patch',type=int,default=16);p.add_argument('--stride',type=int,default=10)
    p.add_argument('--max-patches',type=int,default=81)
    p.add_argument('--fig6-scale',type=int,choices=[1,2,3,4],default=1)
    p.add_argument('--skip-fig6',action='store_true')
    args=p.parse_args()
    if args.band<1 or args.patch<2 or args.stride<1 or args.max_patches<5:p.error('Invalid patch/band settings')
    if args.output.exists() and any(args.output.iterdir()):p.error('Choose a fresh output directory')
    if not args.skip_fig6:
        try:import umap
        except ImportError:p.error('Install umap-learn, or explicitly use --skip-fig6')
    groups=load_runs(args.root,args.runs)
    if not groups:p.error('No completed inference results found')
    args.output.mkdir(parents=True)
    manifest={'args':{k:str(v) if isinstance(v,Path) else [str(x) for x in v] if k=='runs' and v else v for k,v in vars(args).items()},
              'script_sha256':digest(Path(__file__)),'versions':{},'groups':[]}
    for name in ['numpy','scipy','matplotlib','scikit-learn','umap-learn']:
        try:manifest['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    for i,(key,g) in enumerate(groups.items(),1):
        source=g['input'];roi=choose_roi(source,args.roi)
        x,y,rw,rh=roi
        row=args.row if args.row is not None else y+(rh-1)/2
        col=args.col if args.col is not None else x+(rw-1)/2
        if not 0<=row<source.shape[0] or not 0<=col<source.shape[1]:p.error('Profile coordinates outside image')
        group=args.output/f'group{i}';group.mkdir()
        record={'mapping':list(key),'roi_input_pixels':roi,'row':row,'col':col,
                'roi_rule':'manual' if args.roi else 'max input gradient away from boundary; not model-selected',
                'sources':g['sources'],'scales':sorted(g['scales'])}
        for scale,methods in sorted(g['scales'].items()):
            folder=group/f'x{scale}';folder.mkdir()
            fig45(methods,source,folder,scale,roi,row,col,args.band,args.cmap)
            print('Fig4/5 saved:',folder,flush=True)
        if not args.skip_fig6 and args.fig6_scale in g['scales']:
            record['fig6']=fig6(g['scales'][args.fig6_scale],source,group,args)
            print('Fig6 saved:',group,flush=True)
        elif not args.skip_fig6:
            record['fig6']={'skipped':f'No x{args.fig6_scale} results; choose --fig6-scale explicitly'}
        manifest['groups'].append(record)
        (args.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Done:',args.output,flush=True)


if __name__=='__main__':main()
