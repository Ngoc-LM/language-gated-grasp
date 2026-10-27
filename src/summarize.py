import argparse
import shutil
from pathlib import Path
import pandas as pd
from common import log,read_json,working_budget

def summarize(variants,base='/kaggle/working/runs',destination='/kaggle/working/results',cache='/tmp/ga_cache'):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);rows=[]
    for variant in variants:
        source=Path(base)/variant;out=destination/variant;out.mkdir(parents=True,exist_ok=True)
        if (source/'metrics.json').is_file():
            metric=read_json(source/'metrics.json')
            row={key:metric.get(key) for key in ('variant','params','trainable','frozen','val','seen','unseen','H','latency_ms','best_epoch','smoke','stop_reason')}
            row['run']=variant;row['seed']=metric.get('config',{}).get('seed')
            row['wall_hours']=metric['wall_seconds']/3600;row['status']='complete'
        else:
            row={'variant':variant,'status':'NO metrics.json'}
            log_path=source/'train_log.csv'
            if log_path.is_file():
                frame=pd.read_csv(log_path);vals=frame.loc[frame['event']=='val']
                if len(vals):row.update(best_fixed_val=float(vals['val_success'].max()),epochs_done=int(vals['epoch'].max()))
        rows.append(row)
        for name in ['metrics.json','train_log.csv','qualitative.png','best.pt','config.json','fixed_val_stems.json','stdout.log',
                     'val_predictions.csv','test_seen_predictions.csv','test_unseen_predictions.csv']:
            path=source/name
            if path.is_file():working_budget(path.stat().st_size);shutil.copyfile(path,out/name)
        log(f'Collected {variant}: {row}')
    frame=pd.DataFrame(rows);frame.to_csv(destination/'summary.csv',index=False)
    for name in ('prepared.json','manifest_fix.json','target_roundtrip.json','target_roundtrip.csv'):
        path=Path(cache)/name
        if path.is_file():working_budget(path.stat().st_size);shutil.copyfile(path,destination/name)
    if (Path(base)/'launch_status.json').is_file():shutil.copyfile(Path(base)/'launch_status.json',destination/'launch_status.json')
    working_budget();log(frame.to_string(index=False));return frame

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variants',nargs='+',default=['A','B'])
    parser.add_argument('--base',default='/kaggle/working/runs');parser.add_argument('--destination',default='/kaggle/working/results')
    parser.add_argument('--cache',default='/tmp/ga_cache')
    args=parser.parse_args();summarize(args.variants,args.base,args.destination,args.cache)

if __name__=='__main__':main()
