"""Run variants in two-GPU waves with a hard watchdog so the notebook always finishes and saves outputs."""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from common import log,run_variant,run_seed

def last_lines(path,n=4):
    with Path(path).open('rb') as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-16384))
        lines=stream.read().decode('utf-8',errors='replace').splitlines()
    return [line for line in lines if 'Warning' not in line and 'autocast' not in line and 'GradScaler' not in line][-n:]

def spawn(variant,gpu,out,args,extra=()):
    stream=(out/'stdout.log').open('a',encoding='utf-8',buffering=1)
    stream.write(f'\nLAUNCH {datetime.datetime.now().isoformat()} variant={variant} GPU={gpu} extra={list(extra)}\n')
    command=[sys.executable,'-u',str(Path(__file__).parent/'train.py'),'--variant',run_variant(variant),
             '--seed',str(run_seed(variant)),'--gpu',str(gpu),
             '--out',str(out),'--cache',args.cache,'--pack-root',args.pack_root,'--max-hours',str(args.max_hours),*extra]
    if args.smoke:command.append('--smoke')
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONWARNINGS='ignore::FutureWarning')
    try:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=Path(__file__).parent)
    except BaseException:
        stream.close();raise
    return process,stream

def stop(process):
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait()

def run_wave(variants,args,base):
    status={};children=[]
    hard=(args.max_hours+args.grace_hours)*3600;started=time.monotonic()
    try:
        for gpu,variant in enumerate(variants):
            out=base/variant;out.mkdir(parents=True,exist_ok=True)
            process,stream=spawn(variant,gpu,out,args);children.append([variant,gpu,process,stream,out])
            log(f'Launched {variant} on GPU {gpu}; hard watchdog at {hard/3600:.2f} h')
        next_tail=time.monotonic()+args.interval;timed_out=False
        while any(child[2].poll() is None for child in children):
            if time.monotonic()-started>hard:
                timed_out=True;log('WATCHDOG: wall-time limit reached; terminating training processes')
                for child in children:stop(child[2])
                break
            if time.monotonic()>=next_tail:
                for variant,gpu,process,stream,out in children:
                    stream.flush();log(f'--- {variant} GPU{gpu} status={process.poll()} elapsed={(time.monotonic()-started)/3600:.2f}h ---')
                    for line in last_lines(out/'stdout.log'):log(f'[{variant}] {line}')
                next_tail=time.monotonic()+args.interval
            time.sleep(5)
        for variant,gpu,process,stream,out in children:
            stream.flush();code=process.returncode
            status[variant]={'returncode':code,'watchdog':timed_out and code!=0,'metrics':(out/'metrics.json').is_file()}
            log(f'[{variant}] exit={code}; metrics.json={status[variant]["metrics"]}')
            for line in last_lines(out/'stdout.log',8):log(f'[{variant}] {line}')
        # Final evaluation for any variant that has a checkpoint but no metrics (watchdog kill or late crash).
        evals=[]
        for variant,gpu,process,stream,out in children:
            if not (out/'metrics.json').is_file() and (out/'best.pt').is_file():
                log(f'[{variant}] running eval-only on best.pt (GPU {gpu})')
                proc,s=spawn(variant,gpu,out,args,('--eval-only',));evals.append((variant,proc,s,out))
        deadline=time.monotonic()+args.eval_minutes*60
        for variant,proc,s,out in evals:
            try:proc.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:log(f'[{variant}] eval-only timed out');stop(proc)
            s.close();status[variant]['eval_only_returncode']=proc.returncode;status[variant]['metrics']=(out/'metrics.json').is_file()
            for line in last_lines(out/'stdout.log',6):log(f'[{variant}] {line}')
    finally:
        for variant,gpu,process,stream,out in children:
            stop(process)
            if not stream.closed:stream.close()
    return status

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variants',nargs='+',default=['A','B'])
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--base',default='/kaggle/working/runs')
    parser.add_argument('--cache',default='/tmp/ga_cache');parser.add_argument('--pack-root',default='/tmp/ga_pack')
    parser.add_argument('--max-hours',type=float,default=3.5);parser.add_argument('--grace-hours',type=float,default=0.5)
    parser.add_argument('--eval-minutes',type=float,default=30);parser.add_argument('--interval',type=float,default=600)
    args=parser.parse_args()
    if len(set(args.variants))!=len(args.variants):raise ValueError('Variants must be unique')
    for spec in args.variants:run_variant(spec)
    import torch
    if torch.cuda.device_count()<2:raise RuntimeError('Select Kaggle GPU T4 x2 before launching parallel runs')
    base=Path(args.base);base.mkdir(parents=True,exist_ok=True);status={}
    for begin in range(0,len(args.variants),2):status.update(run_wave(args.variants[begin:begin+2],args,base))
    (base/'launch_status.json').write_text(json.dumps(status,indent=2))
    missing=[v for v,s in status.items() if not s['metrics']]
    log(f'Launch finished. status={status}' + (f' | WITHOUT metrics: {missing}' if missing else ''))

if __name__=='__main__':main()
