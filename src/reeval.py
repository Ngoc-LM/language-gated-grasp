"""Re-evaluate saved checkpoints on the test split with the plateau-aware decoder (no retraining).

The original runs decoded the top-1 grasp with a plain argmax (infer.decode_top1, decode='argmax'). On flat
plateaus that picks the first maximum. This script rebuilds the inputs the checkpoints need for the test split
(224x224 images, frozen CLIP patch tokens and prompt embeddings, exactly as prepare.py does), then evaluates every
checkpoint twice: with decode='argmax' (must reproduce the original numbers) and with decode='plateau'.
It also recomputes the ground-truth-map decoding ceiling on the same 200 validation pairs as tests.py.

    python reeval.py                      # Kaggle: ga-pp-packed and the finished run outputs attached
    python reeval.py --runs A B           # only some runs
Writes OUT/<decode>/<run>/{test_seen,test_unseen}_predictions.csv, OUT/reeval_summary.csv, OUT/target_roundtrip.json.
"""
import argparse
import dataclasses
import glob
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import log, run_variant, save_json
from config import Config
from eval_utils import harmonic_mean, is_success
from evaluate import evaluate
from infer import decode_top1
from data import make_loader
from model import GraspNet
from packed_io import read_jpeg
from prepare import CLIP_ID, normalize_name

RUNS = ['A', 'B', 'B_s1', 'E', 'E_s1', 'C', 'C_s1', 'D']

def find_root(arg):
    if arg: return Path(arg)
    hits = sorted(glob.glob('/kaggle/input/**/ga_eval/test_seen.csv', recursive=True))
    if not hits: raise FileNotFoundError('ga_eval/test_seen.csv not found; pass --data-root')
    return Path(hits[0]).parent

def find_checkpoints(runs, runs_dir):
    found = {}
    for run in runs:
        pattern = f'{runs_dir}/**/results/{run}/best.pt' if runs_dir else f'/kaggle/input/**/results/{run}/best.pt'
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits: found[run] = Path(hits[0]).parent
    log('checkpoints: ' + ', '.join(f'{k} <- {v}' for k, v in found.items()))
    return found

def read(path):
    return pd.read_csv(path, dtype={'stem': str, 'scene': str, 'obj': str, 'part': str}, keep_default_na=False)

def build_cache(root, cache, device):
    """Mini cache in the prepare.py format, restricted to test scenes and the 200 round-trip validation pairs."""
    cache.mkdir(parents=True, exist_ok=True)
    tests = {name: read(root / f'{name}.csv') for name in ('test_seen', 'test_unseen')}
    val = read(root / 'val.csv'); unseen = set(tests['test_unseen']['prompt'].map(normalize_name))
    val = val[~val['prompt'].map(normalize_name).isin(unseen)].reset_index(drop=True)   # same leakage fix as prepare.py
    val200 = val.sample(n=200, random_state=0).reset_index(drop=True)                    # same sample as tests.py
    for name, frame in {**tests, 'val200': val200}.items(): frame.to_csv(cache / f'{name}.csv', index=False)
    scenes = sorted(set(tests['test_seen']['scene']) | set(tests['test_unseen']['scene']) | set(val200['scene']))
    n = len(scenes); log(f'building mini cache in {cache}: decoding {n} scenes')
    images = np.memmap(cache / 'img224.u8', mode='w+', dtype=np.uint8, shape=(n, 224, 224, 3))
    for i, scene in enumerate(scenes):
        image = read_jpeg(root, scene)
        if image.size != (416, 416): raise ValueError(f'{scene}: {image.size}')
        images[i] = np.asarray(image.resize((224, 224), resample=Image.Resampling.BILINEAR))
        if (i + 1) % 100 == 0 or i + 1 == n: log(f'decoded {i + 1}/{n} images')
    images.flush(); del images
    save_json(cache / 'images.json', {'N': n}); save_json(cache / 'scene_to_index.json', {s: i for i, s in enumerate(scenes)})
    log('loading frozen CLIP ViT-B/16'); from transformers import AutoTokenizer, CLIPModel
    clip = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(CLIP_ID)
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device)[None, :, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711], device=device)[None, :, None, None]
    pixels = np.fromfile(cache / 'img224.u8', dtype=np.uint8).reshape(n, 224, 224, 3)
    tokens = np.memmap(cache / 'clip_tokens.f16', mode='w+', dtype=np.float16, shape=(n, 196, 768))
    prompts = sorted(set().union(*(set(f['prompt']) for f in (*tests.values(), val200))))
    text = np.zeros((len(prompts), 512), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, n, 64):
            x = torch.from_numpy(pixels[start:start + 64].copy()).to(device).permute(0, 3, 1, 2).float() / 255
            hidden = clip.vision_model(pixel_values=((x - mean) / std).half()).last_hidden_state[:, 1:, :]
            tokens[start:start + 64] = clip.vision_model.post_layernorm(hidden).cpu().numpy().astype(np.float16)
            log(f'CLIP patch tokens {min(start + 64, n)}/{n}')
        for start in range(0, len(prompts), 256):
            t = tokenizer(prompts[start:start + 256], padding=True, truncation=True, max_length=77, return_tensors='pt')
            out = clip.get_text_features(**{k: v.to(device) for k, v in t.items()})
            out = out if isinstance(out, torch.Tensor) else out.pooler_output
            text[start:start + len(out)] = out.float().cpu().numpy().astype(np.float16)
    tokens.flush(); del tokens, clip; torch.cuda.empty_cache()
    np.save(cache / 'text_emb.npy', text); save_json(cache / 'prompt_to_row.json', {p: i for i, p in enumerate(prompts)})
    log(f'mini cache: {n} scenes, {len(prompts)} prompts')

def roundtrip(cache, root, out):
    result = {}
    for mode in ('argmax', 'plateau'):
        cfg = Config(cache=str(cache), pack_root=str(root), num_workers=0, decode=mode)
        loader = make_loader(read(cache / 'val200.csv'), 'roundtrip', cfg); ok = []
        with torch.inference_mode():
            for batch in loader:
                preds = decode_top1(batch['targets'], cfg).numpy()
                ok += [bool(is_success(p, g)) for p, g in zip(preds, batch['gts_416'])]
        result[mode] = float(np.mean(ok)); log(f'GT-map decoding ceiling ({mode}): {result[mode]:.4f} on {len(ok)} pairs')
    save_json(out / 'target_roundtrip.json', {'n': 200, 'seed': 0, **result})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root'); parser.add_argument('--runs-dir', help='folder containing results/<run>/best.pt')
    parser.add_argument('--runs', nargs='+', default=RUNS); parser.add_argument('--cache', default='/tmp/reeval_cache')
    parser.add_argument('--out', default='/kaggle/working/reeval'); parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    log('reeval: start'); device = torch.device(f'cuda:{args.gpu}'); torch.cuda.set_device(device)
    root = find_root(args.data_root); cache = Path(args.cache); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if not (cache / 'prompt_to_row.json').is_file(): build_cache(root, cache, device)
    if not (out / 'target_roundtrip.json').is_file(): roundtrip(cache, root, out)
    found = find_checkpoints(args.runs, args.runs_dir)
    summary_path = out / 'reeval_summary.csv'
    rows = pd.read_csv(summary_path).to_dict('records') if summary_path.is_file() else []
    for run, folder in found.items():
        variant = run_variant(run)
        model = GraspNet(variant).to(device)
        model.load_state_dict(torch.load(folder / 'best.pt', map_location=device, weights_only=True)['model']); model.eval()
        original = json.loads((folder / 'metrics.json').read_text()) if (folder / 'metrics.json').is_file() else {}
        for mode in ('argmax', 'plateau'):
            cfg = Config(variant=variant, cache=str(cache), pack_root=str(root), decode=mode, out=str(out / mode / run))
            score = {split: evaluate(model, cache / f'test_{split}.csv', cfg, device=device,
                                     out_csv=out / mode / run / f'test_{split}_predictions.csv') for split in ('seen', 'unseen')}
            row = {'run': run, 'decode': mode, 'seen': score['seen'], 'unseen': score['unseen'],
                   'H': harmonic_mean(score['seen'], score['unseen']),
                   'original_seen': original.get('seen'), 'original_unseen': original.get('unseen')}
            rows = [r for r in rows if not (r['run'] == run and r['decode'] == mode)] + [row]
            log(f"REEVAL {run} {mode}: seen={row['seen']:.4f} unseen={row['unseen']:.4f} H={row['H']:.4f} "
                f"(original seen={row['original_seen']} unseen={row['original_unseen']})")
        del model; torch.cuda.empty_cache()
    frame = pd.DataFrame(rows); frame.to_csv(summary_path, index=False)
    log('REEVAL SUMMARY\n' + frame.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == '__main__':
    main()
