import argparse
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import csv
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import Config
from common import log,read_json,save_json,seed_all,model_inputs,fingerprint,sha_file
from data import make_loader
from model import GraspNet,loss_components,parameter_counts
from evaluate import evaluate,latency_ms,qualitative
from eval_utils import harmonic_mean

def lr_multiplier(step,total,warmup):
    if warmup and step<warmup:return (step+1)/warmup
    progress=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
    return .5*(1+math.cos(math.pi*progress))

def finalize(model,cfg,out,device,counts,identity,best,best_epoch,step,stop_reason,started,metric_sha,val_path):
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model']);model.eval()
    limit=200 if cfg.smoke else None
    val=evaluate(model,val_path,cfg,limit,out_csv=out/'val_predictions.csv',device=device)
    seen=evaluate(model,Path(cfg.cache)/'test_seen.csv',cfg,limit,out_csv=out/'test_seen_predictions.csv',device=device)
    unseen=evaluate(model,Path(cfg.cache)/'test_unseen.csv',cfg,limit,out_csv=out/'test_unseen_predictions.csv',device=device)
    latency=latency_ms(model,Path(cfg.cache)/'test_seen.csv',cfg)
    qualitative(model,cfg,out/'qualitative.png')
    eval_counts={name:min(200,len(pd.read_csv(Path(cfg.cache)/(name+'.csv')))) if cfg.smoke
                 else len(pd.read_csv(Path(cfg.cache)/(name+'.csv'))) for name in ('val','test_seen','test_unseen')}
    metrics={'run_id':identity,'variant':cfg.variant,'smoke':cfg.smoke,'params':sum(counts.values()),**counts,
        'val':val,'seen':seen,'unseen':unseen,'H':harmonic_mean(seen,unseen),'latency_ms':latency,
        'latency_scope':'batch1 cached-input forward + GPU blur/decode; warmup20, timed100, per-iteration CUDA sync',
        'best_epoch':best_epoch,'best_selection_val':best,'steps':step,'wall_seconds':time.perf_counter()-started,
        'stop_reason':stop_reason,'eval_counts':eval_counts,'metric_sha256':metric_sha,'config':cfg.to_dict()}
    save_json(out/'metrics.json',metrics);log(f'{cfg.variant} DONE: val={val:.4f} seen={seen:.4f} unseen={unseen:.4f} '
                                               f'H={metrics["H"]:.4f} latency={latency:.2f}ms best_epoch={best_epoch}')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=list('ABCDE'),required=True);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--max-hours',type=float,default=3.5);parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--cache',default='/tmp/ga_cache');parser.add_argument('--pack-root',default='/tmp/ga_pack')
    parser.add_argument('--num-workers',type=int,default=2);parser.add_argument('--force',action='store_true')
    parser.add_argument('--eval-only',action='store_true',help='skip training; evaluate out/best.pt (used by the watchdog)')
    args=parser.parse_args()
    cfg=Config(variant=args.variant,seed=args.seed,epochs=args.epochs,max_hours=args.max_hours,batch=args.batch,smoke=args.smoke,
               out=args.out,cache=args.cache,pack_root=args.pack_root,num_workers=args.num_workers).validate()
    if not torch.cuda.is_available() or args.gpu>=torch.cuda.device_count():raise RuntimeError('Requested CUDA GPU is unavailable')
    torch.cuda.set_device(args.gpu);device=torch.device(f'cuda:{args.gpu}');torch.set_num_threads(2);seed_all(cfg.seed)
    torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=False  # speed; unit tests stay deterministic
    out=Path(cfg.out);out.mkdir(parents=True,exist_ok=True)
    prepared=read_json(Path(cfg.cache)/'prepared.json')
    roundtrip=read_json(Path(cfg.cache)/'target_roundtrip.json')
    if roundtrip.get('rate',0)<.80 or roundtrip.get('manifest_key')!=prepared['manifest_key']:
        raise RuntimeError('Run tests.py successfully on the current prepared validation cache before training')
    metric_sha=sha_file(Path(__file__).parent/'eval_utils.py')
    if roundtrip['metric_sha256']!=metric_sha:raise RuntimeError('Metric changed after target round-trip validation')
    source_sha={path.name:sha_file(path) for path in Path(__file__).parent.glob('*.py')}
    identity=fingerprint([{k:v for k,v in cfg.to_dict().items() if k!='out'},
                          prepared['image_key'],prepared['manifest_key'],source_sha])
    if (out/'metrics.json').is_file() and not args.force:
        if read_json(out/'metrics.json')['run_id']!=identity:raise RuntimeError('Output contains a different run; choose --out or use --force')
        log(f'{cfg.variant}: completed matching run already exists; reusing results');return
    if args.eval_only:
        if not (out/'best.pt').is_file():raise FileNotFoundError(out/'best.pt')
        started=time.perf_counter();model=GraspNet(cfg.variant).to(device);counts=parameter_counts(model)
        ck=torch.load(out/'best.pt',map_location=device,weights_only=True)
        log(f'{cfg.variant}: EVAL-ONLY on best.pt (epoch {ck.get("best_epoch")}, step {ck.get("step")})')
        finalize(model,cfg,out,device,counts,identity,float(ck.get('val_success',float('nan'))),int(ck.get('best_epoch',0)),
                 int(ck.get('step',0)),'watchdog_eval_only',started,metric_sha,Path(cfg.cache)/'val.csv')
        return
    started=time.perf_counter();model=GraspNet(cfg.variant).to(device)
    counts=parameter_counts(model)
    log(f'variant={cfg.variant}; GPU={args.gpu} {torch.cuda.get_device_name(args.gpu)}; trainable={counts["trainable"]:,}; '
        f'frozen={counts["frozen"]:,}; CLIP remains external/frozen; smoke={cfg.smoke}')
    save_json(out/'config.json',cfg.to_dict())
    train_path=Path(cfg.cache)/'train.csv';val_path=Path(cfg.cache)/'val.csv'
    loader=make_loader(train_path,'train',cfg,shuffle=True)
    if not len(loader):raise RuntimeError('Empty training loader')
    text=np.load(Path(cfg.cache)/'text_emb.npy',mmap_mode='r')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
    max_steps=300 if cfg.smoke else cfg.epochs*len(loader)
    epochs=max(cfg.epochs,math.ceil(max_steps/len(loader))) if cfg.smoke else cfg.epochs
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_multiplier(step,max_steps,cfg.warmup_steps))
    scaler=torch.cuda.amp.GradScaler(enabled=cfg.amp)
    val_frame=pd.read_csv(val_path,dtype={'scene':str,'stem':str},keep_default_na=False)
    fixed_n=min(200 if cfg.smoke else cfg.val_subset,len(val_frame))
    fixed_val=val_frame.sample(n=fixed_n,random_state=0).reset_index(drop=True)  # same selection subset for every seed
    save_json(out/'fixed_val_stems.json',fixed_val['stem'].tolist())
    best=-1.;best_epoch=0;step=0;stop_reason='epochs';window_images=0;window_start=time.perf_counter()
    fields=['step','epoch','event','loss','q','cos','sin','width','gate','lr','val_success','imgs_s','elapsed_s']
    with (out/'train_log.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for epoch in range(1,epochs+1):
            model.train();loss_values={};clock_stop=False
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                targets=batch['targets'].to(device,non_blocking=True)
                with torch.cuda.amp.autocast(enabled=cfg.amp,dtype=torch.float16):
                    output=model(*model_inputs(batch,text,cfg.variant,device));losses=loss_components(output,targets,cfg)
                if not torch.isfinite(losses['total']):raise FloatingPointError('Nonfinite training loss')
                old_scale=scaler.get_scale();scaler.scale(losses['total']).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=old_scale:scheduler.step()
                step+=1;window_images+=len(batch['img'])
                loss_values={name:float(value.detach().float().item()) for name,value in losses.items()}
                elapsed=time.perf_counter()-started;rate=window_images/max(time.perf_counter()-window_start,1e-6)
                writer.writerow({'step':step,'epoch':epoch,'event':'train','loss':loss_values['total'],
                    **{k:loss_values[k] for k in ('q','cos','sin','width','gate')},'lr':optimizer.param_groups[0]['lr'],
                    'val_success':'','imgs_s':rate,'elapsed_s':elapsed})
                if step%cfg.log_every==0:
                    log(f'{cfg.variant} step={step} epoch={epoch} losses={loss_values} lr={optimizer.param_groups[0]["lr"]:.6g} imgs/s={rate:.1f}')
                    stream.flush();window_images=0;window_start=time.perf_counter()
                clock_stop=elapsed>cfg.max_hours*3600
                if step>=max_steps or clock_stop:break
            # Complete this evaluation even if the wall-time budget elapsed mid-epoch/evaluation.
            val_success=evaluate(model,fixed_val,cfg,out_csv=out/f'val_epoch_{epoch:02d}.csv',device=device)
            writer.writerow({'step':step,'epoch':epoch,'event':'val','lr':optimizer.param_groups[0]['lr'],
                             'val_success':val_success,'elapsed_s':time.perf_counter()-started})
            stream.flush();log(f'{cfg.variant} epoch={epoch} step={step} fixed_val_success={val_success:.4f}')
            if val_success>best:
                best,best_epoch=val_success,epoch
                pending=out/'best.pt.partial'
                torch.save({'model':model.state_dict(),'config':cfg.to_dict(),'best_epoch':epoch,
                            'val_success':best,'step':step,'metric_sha256':metric_sha},pending)
                pending.replace(out/'best.pt')
            if clock_stop or time.perf_counter()-started>cfg.max_hours*3600:stop_reason='wall_time';break
            if step>=max_steps:stop_reason='smoke_steps' if cfg.smoke else 'epochs';break
            window_images=0;window_start=time.perf_counter()
    finalize(model,cfg,out,device,counts,identity,best,best_epoch,step,stop_reason,started,metric_sha,val_path)

if __name__=='__main__':main()
