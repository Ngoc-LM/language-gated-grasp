"""Metric calls are imported from the dataset's unchanged eval_utils.py."""
import csv
import dataclasses
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import log, model_inputs
from data import make_loader
from infer import decode_top1
from eval_utils import is_success, rect_polygon

def evaluate(model, manifest, cfg, limit=None, *, out_csv=None, device=None):
    frame=manifest.copy() if isinstance(manifest,pd.DataFrame) else pd.read_csv(manifest,dtype={'scene':str,'stem':str},keep_default_na=False)
    if limit is not None and len(frame)>limit:frame=frame.sample(n=limit,random_state=0)
    frame=frame.reset_index(drop=True)
    if frame.empty:raise ValueError('Cannot evaluate an empty manifest')
    device=device or next(model.parameters()).device
    name=Path(manifest).stem if not isinstance(manifest,pd.DataFrame) else 'validation'
    output=Path(out_csv or Path(cfg.out)/(name+'_predictions.csv'));output.parent.mkdir(parents=True,exist_ok=True)
    loader=make_loader(frame,name,dataclasses.replace(cfg,num_workers=0))  # in-process: no worker spawn per evaluation
    text=np.load(Path(cfg.cache)/'text_emb.npy',mmap_mode='r',allow_pickle=False)
    was_training=model.training;model.eval();successes=total=0
    with output.open('w',newline='',encoding='utf-8') as stream,torch.inference_mode():
        writer=csv.writer(stream);writer.writerow(['stem','prompt','success','pred_cx','pred_cy','pred_length','pred_width','pred_angle','n_gt'])
        for step,batch in enumerate(loader,1):
            inputs=model_inputs(batch,text,cfg.variant,device)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type=='cuda',dtype=torch.float16):
                maps=model(*inputs)['maps']
            predictions=decode_top1(maps,cfg).cpu().numpy()
            for stem,prompt,pred,gts in zip(batch['stem'],batch['prompt'],predictions,batch['gts_416']):
                good=bool(is_success(pred,gts));successes+=int(good);total+=1
                writer.writerow([stem,prompt,int(good),*map(float,pred),len(gts)])
            if step%50==0 or total==len(frame):log(f'eval {name}: {total}/{len(frame)} success={successes/total:.4f}')
    model.train(was_training)
    return successes/total

def latency_ms(model,manifest,cfg,warmup=20,iterations=100):
    """Batch-1 cached-input forward + GPU blur/decode, excluding data loading and metric."""
    device=next(model.parameters()).device
    if device.type!='cuda':raise RuntimeError('Reported inference latency requires CUDA synchronization')
    batch=next(iter(make_loader(manifest,'latency',dataclasses.replace(cfg,num_workers=0),batch_size=1)))
    text=np.load(Path(cfg.cache)/'text_emb.npy',mmap_mode='r')
    inputs=model_inputs(batch,text,cfg.variant,device)
    model.eval();times=[]
    with torch.inference_mode():
        for i in range(warmup+iterations):
            torch.cuda.synchronize(device);started=time.perf_counter()
            with torch.cuda.amp.autocast(enabled=cfg.amp,dtype=torch.float16):maps=model(*inputs)['maps']
            decode_top1(maps,cfg)
            torch.cuda.synchronize(device)
            if i>=warmup:times.append((time.perf_counter()-started)*1000)
    return float(np.mean(times))

def qualitative(model,cfg,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import textwrap
    import packed_io
    frames=[]
    for split in ('test_seen','test_unseen'):
        frame=pd.read_csv(Path(cfg.cache)/(split+'.csv'),dtype={'scene':str,'stem':str},keep_default_na=False)
        chosen=frame.sample(n=4,replace=len(frame)<4,random_state=0).copy();chosen['display_split']=split;frames.append(chosen)
    samples=pd.concat(frames,ignore_index=True)
    loader=make_loader(samples,'qualitative',dataclasses.replace(cfg,num_workers=0),batch_size=8)
    batch=next(iter(loader));device=next(model.parameters()).device
    text=np.load(Path(cfg.cache)/'text_emb.npy',mmap_mode='r');model.eval()
    with torch.inference_mode(),torch.cuda.amp.autocast(enabled=cfg.amp and device.type=='cuda',dtype=torch.float16):
        predictions=decode_top1(model(*model_inputs(batch,text,cfg.variant,device))['maps'],cfg).cpu().numpy()
    figure,axes=plt.subplots(2,4,figsize=(16,9),layout='constrained')
    for index,(ax,pred,gts) in enumerate(zip(axes.flat,predictions,batch['gts_416'])):
        with packed_io.read_jpeg(cfg.pack_root,batch['scene'][index]) as image:ax.imshow(np.asarray(image))
        ax.add_collection(LineCollection([np.asarray(rect_polygon(*gt).exterior.coords) for gt in gts],colors='lime',linewidths=.6,alpha=.65))
        valid=np.isfinite(pred).all() and np.all(pred[2:4]>0)
        if valid:
            points=np.asarray(rect_polygon(*pred).exterior.coords);ax.plot(points[:,0],points[:,1],color='red',linewidth=2)
        status='success' if is_success(pred,gts) else 'failure'
        title=f'{samples.iloc[index]["display_split"]} | {status}\n'+textwrap.fill(batch['prompt'][index],42)
        if not valid:title+='\ninvalid predicted dimensions'
        ax.set_title(title,fontsize=9);ax.set_xlim(-.5,415.5);ax.set_ylim(415.5,-.5);ax.axis('off')
    figure.savefig(path,dpi=130);plt.close(figure)
    log(f'qualitative figure saved: {path}')
