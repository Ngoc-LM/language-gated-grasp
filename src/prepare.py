"""One-time LOCAL copy, CRC-checked image decode, leakage fix and frozen CLIP caches."""
import argparse
import gc
import io
import multiprocessing as mp
import os
import shutil
import string
import time
import unicodedata
import zlib
from pathlib import Path
from common import run_variant
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import log, read_json, save_json, sha_file, fingerprint, require_free

CLIP_ID = 'openai/clip-vit-base-patch16'
VERBS = sorted(['take hold of','get hold of','grab hold of','pick up','grasp','grab','grip','hold','take',
    'lift','fetch','seize','catch','clutch','give me','bring me','hand me','pass me','get'],key=len,reverse=True)
SEPARATORS = [' by its ',' at its ',' on its ',' by the ',' at the ',' on the ',' from its ',' with its ']

def normalize_name(prompt):
    text = ''.join(' ' if char in string.punctuation or unicodedata.category(char).startswith('P') else char
                   for char in str(prompt).lower())
    text = ' '.join(text.split())
    for verb in VERBS:
        if text == verb or text.startswith(verb+' '): text = text[len(verb):].lstrip(); break
    while text.split(' ',1)[0] in ('the','a','an'): text = text.split(' ',1)[1] if ' ' in text else ''
    positions = [position for sep in SEPARATORS if (position := text.find(sep)) >= 0]
    name = (text[:min(positions)] if positions else text).strip()
    if len(name)>3 and name.endswith('s') and not name.endswith('ss'): name=name[:-1]
    if not name: raise ValueError(f'Empty normalized object name for prompt {prompt!r}')
    return name

def copy_pack(root, destination):
    destination = Path(destination); (destination/'packed').mkdir(parents=True,exist_ok=True)
    old = read_json(destination/'copy.json') if (destination/'copy.json').is_file() else {}
    inventory = {}
    for source in sorted((root/'packed').iterdir()):
        if not source.is_file(): continue
        stat = source.stat()
        identity = [str(root), stat.st_size, stat.st_mtime_ns]
        target = destination/'packed'/source.name
        if target.is_file() and target.stat().st_size == stat.st_size and old.get(source.name)==identity:
            log(f'copy {source.name}: cached ({stat.st_size/1e6:.1f} MB)')
        else:
            require_free(destination,stat.st_size+200_000_000,'Copy packed source')
            started=time.perf_counter(); pending=target.with_name(target.name+'.partial')
            shutil.copyfile(source,pending)
            if pending.stat().st_size != stat.st_size: raise IOError(f'Incomplete copy: {source.name}')
            pending.replace(target)
            elapsed=time.perf_counter()-started
            log(f'copy {source.name}: {stat.st_size/1e6:.1f} MB in {elapsed:.2f}s ({stat.st_size/max(elapsed,1e-6)/1e6:.1f} MB/s)')
        inventory[source.name]=identity
    save_json(destination/'copy.json',inventory)
    return fingerprint(inventory)

def decode_init(blob, path, n):
    global _blob, _image_array
    _blob = open(blob,'rb')
    _image_array = np.memmap(path,mode='r+',dtype=np.uint8,shape=(n,224,224,3))

def decode_one(item):
    index, offset, nbytes, crc = item
    _blob.seek(offset); data=_blob.read(nbytes)
    if len(data)!=nbytes or (zlib.crc32(data)&0xffffffff)!=crc: raise IOError(f'JPEG CRC/size error at image {index}')
    with Image.open(io.BytesIO(data)) as image:
        if image.size!=(416,416): raise ValueError(f'Image {index} is {image.size}, expected 416x416')
        _image_array[index]=np.asarray(image.convert('RGB').resize((224,224),resample=Image.Resampling.BILINEAR))
    return index

def decode_images(pack, cache, key):
    table=pd.read_parquet(pack/'packed/images.parquet').sort_values('scene').reset_index(drop=True)
    if table['scene'].duplicated().any(): raise ValueError('Duplicate scenes in image index')
    n=len(table); path=cache/'img224.u8'; meta=cache/'images.json'
    decode_key=fingerprint([key,'PIL-RGB-BILINEAR-224-v1',sha_file(pack/'packed/images.parquet')])
    if path.is_file() and meta.is_file() and path.stat().st_size==n*224*224*3 and read_json(meta).get('key')==decode_key:
        save_json(cache/'scene_to_index.json',{scene:i for i,scene in enumerate(table['scene'])})
        log(f'image cache: reused N={n}'); return n,decode_key
    path.unlink(missing_ok=True); pending=path.with_name(path.name+'.partial'); pending.unlink(missing_ok=True)
    require_free(cache,n*224*224*3+500_000_000,'224 uint8 image cache')
    initial=np.memmap(pending,mode='w+',dtype=np.uint8,shape=(n,224,224,3)); initial.flush(); del initial
    jobs=[(i,int(row.offset),int(row.nbytes),int(row.crc32)) for i,row in enumerate(table.itertuples(index=False))]
    started=time.perf_counter();done=0
    with mp.get_context('spawn').Pool(4,initializer=decode_init,
            initargs=(str(pack/'packed/images.bin'),str(pending),n)) as pool:
        for _ in pool.imap_unordered(decode_one,jobs,chunksize=32):
            done+=1
            if done%1000==0 or done==n:
                log(f'decode images {done}/{n}; {done/max(time.perf_counter()-started,1e-6):.1f} images/s')
    flushed=np.memmap(pending,mode='r+',dtype=np.uint8,shape=(n,224,224,3));flushed.flush();del flushed
    pending.replace(path)
    save_json(cache/'scene_to_index.json',{scene:i for i,scene in enumerate(table['scene'])})
    save_json(meta,{'key':decode_key,'N':n,'shape':[n,224,224,3],'dtype':'uint8'})
    return n,decode_key

def fix_manifests(root,cache):
    from collections import Counter
    frames={};source_hashes={}
    for name in ('train','val','test_seen','test_unseen'):
        payload=(root/(name+'.csv')).read_bytes()
        import hashlib
        source_hashes[name]=hashlib.sha256(payload).hexdigest()
        frames[name]=pd.read_csv(io.BytesIO(payload),dtype={'stem':str,'scene':str,'obj':str,'part':str},keep_default_na=False)
        frames[name]['obj_name']=frames[name]['prompt'].map(normalize_name)
    unseen=set(frames['test_unseen']['obj_name']);removals={}
    for name in ('train','val'):
        frame=frames[name];bad=frame['obj_name'].isin(unseen)
        top=Counter(frame.loc[bad,'obj_name']).most_common(20)
        removals[name]={'before':len(frame),'removed':int(bad.sum()),'top20':top}
        frames[name]=frame.loc[~bad].reset_index(drop=True)
        log(f'leakage fix {name}: removed {int(bad.sum())}/{len(frame)}; remaining={len(frames[name])}; top20={top}')
        if frames[name].empty: raise RuntimeError(f'No {name} rows remain after the grip/name leakage fix')
    test_scenes=set(frames['test_seen']['scene'])|set(frames['test_unseen']['scene'])
    assert not set(frames['train']['scene'])&set(frames['val']['scene'])
    for name in ('train','val'):
        assert not set(frames[name]['scene'])&test_scenes,'Source manifests leak official test scenes'
        assert not set(frames[name]['obj_name'])&unseen
    for name,frame in frames.items():frame.to_csv(cache/(name+'.csv'),index=False)
    save_json(cache/'manifest_fix.json',{'source_sha256':source_hashes,'unseen_names':sorted(unseen),'removals':removals,
                                      'counts':{name:len(frame) for name,frame in frames.items()},'version':'grip-v1'})
    return frames,fingerprint([source_hashes,'grip-v1'])

def load_clip(device):
    from transformers import CLIPModel
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    model=CLIPModel.from_pretrained(CLIP_ID,torch_dtype=torch.float16,cache_dir='/tmp/ga_hf',
                                   token=os.environ.get('HF_TOKEN') or False).to(device)
    model.eval().requires_grad_(False)
    log(f'CLIP cache encoder: trainable=0; frozen={sum(p.numel() for p in model.parameters()):,}; device={device}; dtype=fp16')
    return model

def text_embeddings(frames,cache,key,batch_size=256):
    prompts=sorted(set().union(*(set(frame['prompt']) for frame in frames.values())))
    text_key=fingerprint([key,CLIP_ID,prompts,'projected-text-v1'])
    path,meta=cache/'text_emb.npy',cache/'text.json'
    if path.is_file() and meta.is_file() and read_json(meta).get('key')==text_key:
        array=np.load(path,mmap_mode='r');
        if array.shape==(len(prompts),512) and array.dtype==np.float16:
            save_json(cache/'prompt_to_row.json',{prompt:i for i,prompt in enumerate(prompts)})
            log(f'text cache reused P={len(prompts)}');return len(prompts)
    require_free(cache,len(prompts)*512*2+200_000_000,'CLIP text embeddings')
    from transformers import AutoTokenizer
    device=torch.device('cuda:0');model=load_clip(device)
    tokenizer=AutoTokenizer.from_pretrained(CLIP_ID,cache_dir='/tmp/ga_hf',token=os.environ.get('HF_TOKEN') or False)
    pending=path.with_name(path.name+'.partial')
    array=np.lib.format.open_memmap(pending,mode='w+',dtype=np.float16,shape=(len(prompts),512))
    with torch.inference_mode():
        for step,start in enumerate(range(0,len(prompts),batch_size),1):
            tokens=tokenizer(prompts[start:start+batch_size],padding=True,truncation=True,max_length=77,return_tensors='pt')
            output=model.get_text_features(**{k:v.to(device) for k,v in tokens.items()})
            # Transformers 4.x returns a Tensor; newer versions return projected pooler_output.
            features=output if isinstance(output,torch.Tensor) else output.pooler_output
            if features.shape!=(min(batch_size,len(prompts)-start),512):raise ValueError('Unexpected CLIP text feature shape')
            if not torch.isfinite(features).all():raise FloatingPointError('Nonfinite frozen CLIP text features')
            array[start:start+len(features)]=features.float().cpu().numpy().astype(np.float16)
            if step%25==0 or start+len(features)==len(prompts):log(f'CLIP text {start+len(features)}/{len(prompts)} prompts')
    array.flush();del array,model;gc.collect();torch.cuda.empty_cache()
    pending.replace(path)
    save_json(cache/'prompt_to_row.json',{prompt:i for i,prompt in enumerate(prompts)})
    save_json(meta,{'key':text_key,'P':len(prompts),'shape':[len(prompts),512],'dtype':'float16','model':CLIP_ID})
    return len(prompts)

def token_worker(gpu,start,stop,n,cache_string,batch_size):
    cache=Path(cache_string);torch.cuda.set_device(gpu);torch.set_num_threads(2)
    device=torch.device(f'cuda:{gpu}');model=load_clip(device)
    images=np.memmap(cache/'img224.u8',mode='r',dtype=np.uint8,shape=(n,224,224,3))
    tokens=np.memmap(cache/'clip_tokens.f16.partial',mode='r+',dtype=np.float16,shape=(n,196,768))
    mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
    with torch.inference_mode():
        for step,begin in enumerate(range(start,stop,batch_size),1):
            end=min(begin+batch_size,stop)
            image=torch.from_numpy(np.array(images[begin:end],copy=True)).to(device).permute(0,3,1,2).float()/255
            pixel_values=((image-mean)/std).half()
            hidden=model.vision_model(pixel_values=pixel_values).last_hidden_state[:,1:,:]
            features=model.vision_model.post_layernorm(hidden)
            if features.shape!=(end-begin,196,768):raise ValueError('Expected CLIP (B,196,768) patch tokens')
            tokens[begin:end]=features.cpu().numpy().astype(np.float16,copy=False)
            if step%50==0 or end==stop:log(f'GPU {gpu} CLIP patches {end-start}/{stop-start}')
    tokens.flush();del tokens,images,model

def patch_tokens(cache,n,key,batch_size=64):
    path,meta=cache/'clip_tokens.f16',cache/'tokens.json'
    token_key=fingerprint([key,CLIP_ID,'post-layernorm-patches-v1'])
    size=n*196*768*2
    if path.is_file() and path.stat().st_size==size and meta.is_file() and read_json(meta).get('key')==token_key:
        log(f'CLIP patch cache reused: {size/1e9:.2f} GB');return
    if torch.cuda.device_count()<2:raise RuntimeError('C/D token preparation requires both GPUs (T4 x2)')
    path.unlink(missing_ok=True);pending=path.with_name(path.name+'.partial');pending.unlink(missing_ok=True)
    require_free(cache,25_000_000_000,'C/D patch tokens require at least 25 GB FREE on /tmp')
    initial=np.memmap(pending,mode='w+',dtype=np.float16,shape=(n,196,768));initial.flush();del initial
    context=mp.get_context('spawn');middle=(n+1)//2
    processes=[context.Process(target=token_worker,args=(gpu,start,end,n,str(cache),batch_size))
               for gpu,start,end in ((0,0,middle),(1,middle,n))]
    for process in processes:process.start()
    for process in processes:process.join()
    if any(process.exitcode!=0 for process in processes):raise RuntimeError('A CLIP token-cache GPU worker failed; cache is incomplete')
    pending.replace(path)
    save_json(meta,{'key':token_key,'N':n,'shape':[n,196,768],'dtype':'float16','bytes':size,'model':CLIP_ID})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--variants',nargs='+',default=['A','B'])
    parser.add_argument('--pack-root',default='/tmp/ga_pack');parser.add_argument('--cache',default='/tmp/ga_cache')
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Preparation needs GPU access for frozen fp16 CLIP; select T4 x2')
    root,pack,cache=Path(args.root),Path(args.pack_root),Path(args.cache)
    cache.mkdir(parents=True,exist_ok=True);pack.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();copy_key=copy_pack(root,pack)
    n,image_key=decode_images(pack,cache,copy_key)
    frames,manifest_key=fix_manifests(root,cache)
    p=text_embeddings(frames,cache,manifest_key)
    if any(run_variant(spec) in ('C','D','E') for spec in args.variants):patch_tokens(cache,n,image_key)
    else:log('A/B only: no CLIP patch-token cache allocated')
    save_json(cache/'prepared.json',{'N':n,'P':p,'variants':args.variants,'image_key':image_key,
        'manifest_key':manifest_key,'metric_sha256':sha_file(Path(__file__).parent/'eval_utils.py'),
        'seconds':time.perf_counter()-started})
    log(f'PREPARE complete: N={n}, P={p}, elapsed={time.perf_counter()-started:.1f}s')

if __name__=='__main__':main()
