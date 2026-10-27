import hashlib
import json
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch

def log(message):
    print(message, flush=True)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.partial')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    pending.replace(path)

def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_seed(worker_id):
    import cv2
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)

def require_free(path, needed, context):
    free = shutil.disk_usage(path).free
    if free < needed:
        raise RuntimeError(f'{context}: need {needed/1e9:.2f} GB free at {path}; available {free/1e9:.2f} GB')

def working_budget(additional=0):
    root = Path('/kaggle/working')
    if root.exists():
        used = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
        if used+additional >= 15_000_000_000:
            raise RuntimeError('The /kaggle/working 15 GB output budget would be exceeded')

def run_variant(spec):
    """Run specs: 'A'..'E' (seed 0) or e.g. 'C_s1' (variant C, seed 1). E is C0, the text-free CLIP gate control."""
    import re
    match = re.fullmatch(r'([ABCDE])(?:_s(\d+))?', str(spec))
    if match is None: raise ValueError(f'Invalid run spec {spec!r}; use A-E or e.g. C_s1')
    return match.group(1)

def run_seed(spec):
    run_variant(spec)
    return int(spec.split('_s')[1]) if '_s' in spec else 0

def text_batch(cache, indices, device):
    rows = indices.detach().cpu().numpy() if isinstance(indices, torch.Tensor) else np.asarray(indices)
    return torch.from_numpy(np.asarray(cache[rows], dtype=np.float32)).to(device, non_blocking=True)

def model_inputs(batch, text_cache, variant, device):
    image = batch['img'].to(device, non_blocking=True)
    text = None if variant == 'A' else text_batch(text_cache, batch['prompt_idx'], device)
    tokens = batch.get('clip_tokens')
    tokens = tokens.to(device, non_blocking=True) if tokens is not None else None
    return image, text, tokens
