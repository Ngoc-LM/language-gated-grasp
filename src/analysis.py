"""Paired significance tests and a prompt-swap test on finished runs.

1) McNemar (exact, two-sided) on per-pair test predictions of two runs + scene-clustered bootstrap CI.
2) Prompt swap: give each test image the prompt of a DIFFERENT object in the same scene and count how often
   the top-1 grasp still lands on the ORIGINAL object's part ("stays on original"). A prompt-blind model keeps
   its prediction, so its stay rate equals its normal success; a model that follows the prompt moves away.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from common import log, run_variant
from config import Config
from model import GraspNet
from evaluate import evaluate

RUNS = ['A', 'B', 'E', 'C', 'D', 'B_s1', 'E_s1', 'C_s1']
PAIRS = [('C', 'E'), ('C_s1', 'E_s1'), ('C_s1', 'E'), ('E', 'B'), ('E_s1', 'B_s1'), ('C', 'B'), ('C_s1', 'B_s1'),
         ('D', 'C'), ('B', 'A')]

def find_runs():
    found = {}
    for run in RUNS:
        hits = sorted(glob.glob(f'/kaggle/input/**/results/{run}/metrics.json', recursive=True))
        if hits: found[run] = Path(hits[0]).parent
    log(f'found runs: { {k: str(v) for k, v in found.items()} }')
    return found

def load_preds(folder, split):
    frame = pd.read_csv(folder / f'test_{split}_predictions.csv', dtype={'stem': str}, keep_default_na=False)
    return frame[['stem', 'prompt', 'success']]

def mcnemar(found, out):
    rows = []
    rng = np.random.default_rng(0)
    for a, b in PAIRS:
        if a not in found or b not in found: continue
        for split in ('seen', 'unseen'):
            m = load_preds(found[a], split).merge(load_preds(found[b], split), on=['stem', 'prompt'], suffixes=('_a', '_b'))
            sa, sb = m['success_a'].to_numpy(), m['success_b'].to_numpy()
            n10, n01 = int(((sa == 1) & (sb == 0)).sum()), int(((sa == 0) & (sb == 1)).sum())
            p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            scenes = m['stem'].str[:64].to_numpy(); uniq = np.unique(scenes)
            idx = {s: np.where(scenes == s)[0] for s in uniq}; diffs = []
            for _ in range(2000):
                pick = np.concatenate([idx[s] for s in rng.choice(uniq, size=len(uniq), replace=True)])
                diffs.append(sa[pick].mean() - sb[pick].mean())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append({'a': a, 'b': b, 'split': split, 'n': len(m), 'acc_a': sa.mean(), 'acc_b': sb.mean(),
                         'diff': sa.mean() - sb.mean(), 'a_only': n10, 'b_only': n01, 'mcnemar_p': p,
                         'boot_lo': lo, 'boot_hi': hi, 'scenes': len(uniq)})
    frame = pd.DataFrame(rows); frame.to_csv(out / 'mcnemar.csv', index=False)
    log('\n' + frame.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return frame

def swap_frames(cache):
    frames = []
    for split in ('seen', 'unseen'):
        f = pd.read_csv(Path(cache) / f'test_{split}.csv', dtype={'scene': str, 'stem': str, 'obj': str, 'part': str},
                        keep_default_na=False); f['split'] = split; frames.append(f)
    test = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(0); rows = []
    for _, r in test.iterrows():
        cand = test[(test['scene'] == r['scene']) & (test['obj'] != r['obj']) & (test['prompt'] != r['prompt'])]
        if len(cand) == 0: continue
        c = cand.iloc[int(rng.integers(len(cand)))]
        new = r.copy(); new['prompt'] = c['prompt']; rows.append(new)
    swapped = pd.DataFrame(rows).reset_index(drop=True)
    original = test.loc[test['stem'].isin(swapped['stem'])].drop_duplicates('stem').reset_index(drop=True)
    swapped = swapped.drop_duplicates('stem').reset_index(drop=True)
    log(f'prompt swap: {len(swapped)} of {len(test)} test pairs have another object in the same scene')
    return original, swapped

def swap_test(found, cache, pack, out, device):
    original, swapped = swap_frames(cache)
    text_rows = json.loads((Path(cache) / 'prompt_to_row.json').read_text())
    missing = set(swapped['prompt']) - set(text_rows)
    if missing: raise RuntimeError(f'{len(missing)} swapped prompts lack cached text embeddings')
    rows = []
    for run, folder in found.items():
        variant = run_variant(run)
        cfg = Config(variant=variant, cache=cache, pack_root=pack, out=str(out / 'swap' / run)).validate()
        model = GraspNet(variant).to(device)
        state = torch.load(folder / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(state['model']); model.eval()
        normal = evaluate(model, original, cfg, out_csv=out / 'swap' / run / 'original.csv', device=device)
        stay = evaluate(model, swapped, cfg, out_csv=out / 'swap' / run / 'swapped.csv', device=device)
        rows.append({'run': run, 'n': len(swapped), 'success_original_prompt': normal, 'stay_on_original_after_swap': stay,
                     'relative_stay': stay / max(normal, 1e-9)})
        log(f'SWAP {run}: original-prompt success={normal:.4f}; after swap still on original part={stay:.4f}')
        del model; torch.cuda.empty_cache()
    frame = pd.DataFrame(rows); frame.to_csv(out / 'prompt_swap.csv', index=False)
    log('\n' + frame.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='/tmp/ga_cache'); parser.add_argument('--pack-root', default='/tmp/ga_pack')
    parser.add_argument('--out', default='/kaggle/working/results/analysis'); parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    found = find_runs()
    mcnemar(found, out)
    device = torch.device(f'cuda:{args.gpu}'); torch.cuda.set_device(device)
    swap_test(found, args.cache, args.pack_root, out, device)

if __name__ == '__main__':
    main()
