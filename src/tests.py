"""Run before training: exact architecture/FiLM checks and a real-data target round trip."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import log,read_json,save_json,sha_file,seed_all
from config import Config
from model import GraspNet,FiLM,loss_components,parameter_counts
from data import make_loader
from infer import decode_top1
from eval_utils import is_success

def test_shapes_and_film(device):
    torch.set_num_threads(2)
    image=torch.randint(0,256,(2,3,224,224),dtype=torch.uint8,device=device)
    text=torch.randn(2,512,device=device);tokens=torch.randn(2,196,768,device=device)
    expected={'conv1':(2,32,224,224),'conv2':(2,64,112,112),'conv3':(2,128,56,56),
        'pre_residual':(2,128,56,56),'up1':(2,64,112,112),'up2':(2,32,224,224),
        'final':(2,32,224,224),'maps':(2,4,224,224)}
    for i in range(5):expected[f'residual_{i}']=(2,128,56,56)
    expected['dropout']=(2,32,224,224)
    for name in ('q','cos','sin','width'):expected['head_'+name]=(2,1,224,224)
    base_output=None
    for variant in 'ABCDE':
        seed_all(0);model=GraspNet(variant).to(device).eval()
        with torch.inference_mode():output=model(image,text,tokens,return_shapes=True)
        for name,shape in expected.items():assert output['shapes'][name]==shape,(variant,name,output['shapes'][name])
        if variant!='A':
            for i in range(5):assert output['shapes'][f'film_{i}']==(2,128,56,56)
        if variant in 'CDE':
            assert output['gate_logits'].shape==(2,1,14,14)
            assert abs(model.gate.log_tau.exp().item()-10)<1e-4
        if variant=='A':base_output=output['maps'].clone()
        if variant=='B':assert torch.equal(base_output,output['maps']),'B must equal A at zero-FiLM initialization'
        log(f'PASS shapes {variant}: {parameter_counts(model)}')
    layer=FiLM().to(device);features=torch.randn(2,128,56,56,device=device)
    assert torch.equal(layer(features,text),features)
    log('PASS FiLM identity at initialization')

def target_roundtrip(cfg,device):
    frame=pd.read_csv(Path(cfg.cache)/'val.csv',dtype={'scene':str,'stem':str},keep_default_na=False)
    if len(frame)<200:raise RuntimeError(f'Target round-trip requires 200 validation rows, found {len(frame)}')
    chosen=frame.sample(n=200,random_state=0).reset_index(drop=True)
    loader=make_loader(chosen,'target_roundtrip',cfg)
    records=[]
    with torch.inference_mode():
        for step,batch in enumerate(loader,1):
            predictions=decode_top1(batch['targets'].to(device),cfg).cpu().numpy()
            for stem,pred,gts in zip(batch['stem'],predictions,batch['gts_416']):
                records.append({'stem':stem,'success':int(is_success(pred,gts)),'n_gt':len(gts),
                    **{name:float(value) for name,value in zip(('cx','cy','length','width','angle'),pred)}})
            if step%10==0 or len(records)==200:log(f'target round-trip {len(records)}/200')
    rate=sum(row['success'] for row in records)/200
    pd.DataFrame(records).to_csv(Path(cfg.cache)/'target_roundtrip.csv',index=False)
    prepared=read_json(Path(cfg.cache)/'prepared.json')
    save_json(Path(cfg.cache)/'target_roundtrip.json',{'rate':rate,'n':200,'threshold':.80,'seed':0,
        'manifest_key':prepared['manifest_key'],'metric_sha256':sha_file(Path(__file__).parent/'eval_utils.py')})
    log(f'target round-trip (oracle decode upper bound): {rate:.3%}; required >=80%; diagnostics: {cfg.cache}/target_roundtrip.csv')
    assert rate>=.80,'Target round-trip failed: inspect target_roundtrip.csv before training; metric/target rules were not relaxed'
    return rate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--cache',default='/tmp/ga_cache');parser.add_argument('--pack-root',default='/tmp/ga_pack')
    args=parser.parse_args();device=torch.device(args.device)
    cfg=Config(cache=args.cache,pack_root=args.pack_root,num_workers=0)
    test_shapes_and_film(device);target_roundtrip(cfg,device);log('ALL UNIT CHECKS PASSED')

if __name__=='__main__':main()
