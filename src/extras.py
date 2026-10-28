"""Post-hoc analyses on finished runs (no retraining).

1) Paired McNemar tests (exact, two-sided) and scene-clustered bootstrap CIs on per-pair test predictions.
2) Coverage of the 3.6% training subset: which seen test object names occur in training, and success split by it.
3) Multi-object scenes: share of test pairs whose scene has >1 annotated object, and success restricted to them.
4) Gate maps: the language gate A of C and the text-free gate of C0 for two prompts on the same image.

Usage (Kaggle, GPU, Internet on, finished run outputs attached as inputs):
    python extras.py            # writes /kaggle/working/extras/
"""
import argparse
import base64
import glob
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

RUNS = ['A', 'B', 'B_s1', 'E', 'E_s1', 'C', 'C_s1', 'D']
PAIRS = [('C', 'E'), ('C_s1', 'E_s1'), ('E', 'B'), ('E_s1', 'B_s1'), ('C', 'B'), ('C_s1', 'B_s1'), ('D', 'C'), ('B', 'A')]

def log(message):
    print(message, flush=True)

def find_root(arg=None):
    if arg: return Path(arg)
    hits = sorted(glob.glob('/kaggle/input/**/ga_eval/test_seen.csv', recursive=True))
    if not hits: raise FileNotFoundError('Attach the ga-pp-packed dataset (ga_eval/test_seen.csv not found)')
    return Path(hits[0]).parent

def find_checkpoints():
    found = {}
    for run in RUNS:
        hits = sorted(glob.glob(f'/kaggle/input/**/results/{run}/best.pt', recursive=True))
        if hits: found[run] = Path(hits[0]).parent
    return found

def find_runs(runs_dir=None):
    """Per-pair test predictions: runs_dir/<run>/ (e.g. reeval.py output) or results/<run>/ of attached Kaggle inputs."""
    found = {}
    for run in RUNS:
        pattern = f'{runs_dir}/{run}/test_seen_predictions.csv' if runs_dir else \
            f'/kaggle/input/**/results/{run}/test_seen_predictions.csv'
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits: found[run] = Path(hits[0]).parent
    log('runs with per-pair predictions: ' + ', '.join(f'{k} <- {v}' for k, v in found.items()))
    return found

def read_csv(path):
    return pd.read_csv(path, dtype={'stem': str, 'scene': str, 'obj': str, 'part': str}, keep_default_na=False)

def test_frame(root):
    frames = []
    for split in ('seen', 'unseen'):
        frame = read_csv(root / f'test_{split}.csv'); frame['split'] = split; frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def predictions(found, run, test):
    frames = []
    for split in ('seen', 'unseen'):
        frame = read_csv(found[run] / f'test_{split}_predictions.csv')[['stem', 'prompt', 'success']]
        frame['split'] = split; frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    merged = frame.merge(test[['stem', 'prompt', 'scene', 'obj', 'split']], on=['stem', 'prompt', 'split'], how='left')
    if merged['scene'].isna().any(): raise RuntimeError(f'{run}: predictions do not match the test manifest')
    return merged

def cluster_bootstrap(diff, scenes, rng, reps=2000):
    codes, uniq = pd.factorize(scenes)
    sums = np.bincount(codes, weights=diff); counts = np.bincount(codes)
    draws = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    stats = sums[draws].sum(1) / counts[draws].sum(1)
    return np.percentile(stats, [2.5, 97.5])

def mcnemar(found, test, out):
    rows = []; rng = np.random.default_rng(0)
    for a, b in PAIRS:
        if a not in found or b not in found: log(f'skip {a} vs {b}: predictions missing'); continue
        pa, pb = predictions(found, a, test), predictions(found, b, test)
        m = pa.merge(pb[['stem', 'prompt', 'split', 'success']], on=['stem', 'prompt', 'split'], suffixes=('_a', '_b'))
        for split in ('seen', 'unseen'):
            s = m[m['split'] == split]; sa, sb = s['success_a'].to_numpy(), s['success_b'].to_numpy()
            n10, n01 = int(((sa == 1) & (sb == 0)).sum()), int(((sa == 0) & (sb == 1)).sum())
            p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            lo, hi = cluster_bootstrap((sa - sb).astype(float), s['scene'].to_numpy(), rng)
            rows.append({'a': a, 'b': b, 'split': split, 'pairs': len(s), 'scenes': s['scene'].nunique(),
                         'acc_a': sa.mean(), 'acc_b': sb.mean(), 'diff': sa.mean() - sb.mean(),
                         'a_only': n10, 'b_only': n01, 'mcnemar_p': p, 'boot95_lo': lo, 'boot95_hi': hi})
    frame = pd.DataFrame(rows); frame.to_csv(out / 'mcnemar.csv', index=False)
    log('MCNEMAR\n' + frame.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

def coverage(root, found, test, out):
    from prepare import normalize_name
    train = read_csv(root / 'train.csv'); train['obj_name'] = train['prompt'].map(normalize_name)
    test = test.copy(); test['obj_name'] = test['prompt'].map(normalize_name)
    unseen_names = set(test.loc[test['split'] == 'unseen', 'obj_name'])
    train = train[~train['obj_name'].isin(unseen_names)]
    names = set(train['obj_name']); counts = train['obj_name'].value_counts()
    test['in_train'] = test['obj_name'].isin(names)
    test['train_pairs_of_name'] = test['obj_name'].map(counts).fillna(0).astype(int)
    scene_objs = test.groupby('scene')['obj'].nunique(); test['multi_object'] = test['scene'].map(scene_objs) > 1
    summary = {'train_pairs': len(train), 'train_scenes': int(train['scene'].nunique()), 'train_names': len(names)}
    # Independent leakage check: word-boundary search of every unseen test name (and its plural) in the raw text of
    # the filtered training prompts, without the prompt-template parser used by the filter itself.
    import re
    lowered = train['prompt'].str.lower()
    hits = {}
    for name in sorted(unseen_names):
        pattern = r'\b' + re.escape(name) + r'(?:e?s)?\b'
        found = lowered.str.contains(pattern, regex=True)
        if found.any(): hits[name] = {'pairs': int(found.sum()), 'examples': train.loc[found, 'prompt'].head(5).tolist()}
    summary['unseen_name_mentions_in_train'] = {'names': len(hits), 'pairs': int(sum(h['pairs'] for h in hits.values())),
                                                'by_name': hits}
    for split in ('seen', 'unseen'):
        s = test[test['split'] == split]
        summary[split] = {'pairs': len(s), 'scenes': int(s['scene'].nunique()), 'names': int(s['obj_name'].nunique()),
                          'names_in_train': int(s.loc[s['in_train'], 'obj_name'].nunique()),
                          'pairs_in_train': int(s['in_train'].sum()),
                          'median_train_pairs_per_name': float(s['train_pairs_of_name'].median()),
                          'pairs_in_multi_object_scenes': int(s['multi_object'].sum())}
    rows = []
    for run in found:
        p = predictions(found, run, test).merge(test[['stem', 'prompt', 'split', 'in_train', 'multi_object']],
                                                on=['stem', 'prompt', 'split'])
        for split in ('seen', 'unseen'):
            s = p[p['split'] == split]
            row = {'run': run, 'split': split, 'all': s['success'].mean(),
                   'multi_object': s.loc[s['multi_object'], 'success'].mean(),
                   'single_object': s.loc[~s['multi_object'], 'success'].mean()}
            if split == 'seen':
                row['name_in_train'] = s.loc[s['in_train'], 'success'].mean()
                row['name_not_in_train'] = s.loc[~s['in_train'], 'success'].mean()
            rows.append(row)
    frame = pd.DataFrame(rows); frame.to_csv(out / 'coverage_success.csv', index=False)
    (out / 'coverage.json').write_text(json.dumps(summary, indent=2))
    log('COVERAGE ' + json.dumps(summary))
    log('SUCCESS BY SUBSET\n' + frame.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return test

def run_logs(out):
    rows = []
    for run in RUNS:
        hits = sorted(glob.glob(f'/kaggle/input/**/results/{run}/train_log.csv', recursive=True))
        if not hits: continue
        path = Path(hits[0])
        f = pd.read_csv(path); v = f.loc[f['event'] == 'val']
        rows.append({'run': run, **{f'epoch{int(e)}': round(float(x), 4) for e, x in zip(v['epoch'], v['val_success'])}})
    for name in ('manifest_fix.json', 'prepared.json', 'target_roundtrip.json'):
        hits = sorted(glob.glob(f'/kaggle/input/**/results/{name}', recursive=True))
        if hits:
            data = json.loads(Path(hits[0]).read_text())
            data.pop('unseen_names', None)
            log(f'{name}: ' + json.dumps(data)[:1500])
    frame = pd.DataFrame(rows); frame.to_csv(out / 'fixed_val_curves.csv', index=False)
    log('FIXED-VAL CURVES\n' + frame.to_string(index=False))

def gate_maps(root, found, test, out, max_scenes=400):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoTokenizer, CLIPModel
    from config import Config
    from eval_utils import load_gt_array, is_success
    from infer import decode_top1
    from model import GraspNet
    from packed_io import read_jpeg
    device = torch.device('cuda:0'); cfg = Config(variant='C')
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch16', torch_dtype=torch.float16).to(device).eval()
    tok = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    models = {}
    for run in ('C', 'E'):
        if run not in found: raise RuntimeError(f'{run} checkpoint missing')
        net = GraspNet(run).to(device)
        net.load_state_dict(torch.load(found[run] / 'best.pt', map_location=device, weights_only=True)['model']); net.eval()
        models[run] = net
    labels = np.load(root / 'packed' / 'labels.npy', mmap_mode='r', allow_pickle=False)
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device)[None, :, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711], device=device)[None, :, None, None]
    multi = test.groupby('scene').filter(lambda g: g['obj'].nunique() > 1)
    scenes = list(dict.fromkeys(multi['scene']))[:max_scenes]
    log(f'gate maps: {len(scenes)} multi-object scenes')
    records = []
    with torch.inference_mode():
        for scene in scenes:
            rows = multi[multi['scene'] == scene].drop_duplicates('obj')
            image416 = read_jpeg(root, scene)
            img = torch.from_numpy(np.asarray(image416.resize((224, 224), Image.Resampling.BILINEAR))).to(device)
            img = img.permute(2, 0, 1)[None]
            pix = ((img.float() / 255 - mean) / std).half()
            hidden = clip.vision_model(pixel_values=pix).last_hidden_state[:, 1:, :]
            tokens = clip.vision_model.post_layernorm(hidden).float()
            t = tok(list(rows['prompt']), padding=True, truncation=True, max_length=77, return_tensors='pt').to(device)
            text = clip.get_text_features(**t)
            text = (text if isinstance(text, torch.Tensor) else text.pooler_output).float()
            entry = {'scene': scene, 'split': rows['split'].iloc[0], 'prompts': []}
            for i, row in enumerate(rows.itertuples(index=False)):
                gts = load_gt_array(labels[int(row.label_start):int(row.label_start) + int(row.label_count)])
                item = {'prompt': row.prompt, 'obj': row.obj, 'gts': np.asarray(gts, dtype=float).round(1).tolist()}
                for run, net in models.items():
                    o = net(img, text[i:i + 1], tokens.half())
                    pred = decode_top1(o['maps'], cfg)[0].cpu().numpy()
                    item[run] = {'pred': pred.round(2).tolist(), 'ok': bool(is_success(pred, gts)),
                                 'gate': torch.sigmoid(o['gate_logits'][0, 0]).cpu().numpy().round(3).tolist()}
                entry['prompts'].append(item)
            records.append(entry)
    def score(e):
        p = e['prompts'][:2]
        if len(p) < 2: return -1
        c_ok = p[0]['C']['ok'] and p[1]['C']['ok']; e_bad = not (p[0]['E']['ok'] and p[1]['E']['ok'])
        moved = float(np.abs(np.array(p[0]['C']['gate']) - np.array(p[1]['C']['gate'])).mean())
        return (2 if c_ok else 0) + (1 if e_bad else 0) + moved
    stats = {'scenes': len(records), 'both_prompts_C_ok': 0, 'both_prompts_E_ok': 0}
    for e in records:
        p = e['prompts'][:2]
        stats['both_prompts_C_ok'] += int(p[0]['C']['ok'] and p[1]['C']['ok'])
        stats['both_prompts_E_ok'] += int(p[0]['E']['ok'] and p[1]['E']['ok'])
        diff = lambda r: float(np.abs(np.array(p[0][r]['gate']) - np.array(p[1][r]['gate'])).mean())
        e['gate_change'] = {'C': diff('C'), 'E': diff('E')}
    stats['mean_gate_change_C'] = float(np.mean([e['gate_change']['C'] for e in records]))
    stats['mean_gate_change_E'] = float(np.mean([e['gate_change']['E'] for e in records]))
    log('GATE STATS ' + json.dumps(stats))
    (out / 'gate_stats.json').write_text(json.dumps(stats, indent=2))
    records.sort(key=score, reverse=True)
    (out / 'gate_examples.json').write_text(json.dumps(records[:20]))
    for rank, e in enumerate(records[:6]):
        p = e['prompts'][:2]
        log(f"CAND {rank} {e['split']} {e['scene'][:12]} | {p[0]['prompt']} C={p[0]['C']['ok']} E={p[0]['E']['ok']} "
            f"| {p[1]['prompt']} C={p[1]['C']['ok']} E={p[1]['E']['ok']} | change C={e['gate_change']['C']:.3f}")
    best = records[:3]
    for rank, e in enumerate(best):
        buffer = io.BytesIO(); read_jpeg(root, e['scene']).resize((320, 320), Image.Resampling.BICUBIC).save(buffer, 'JPEG', quality=90)
        e['jpeg320_b64'] = base64.b64encode(buffer.getvalue()).decode()
    (out / 'gate_best.json').write_text(json.dumps(best))
    log(f'wrote gate_best.json ({(out / "gate_best.json").stat().st_size} bytes)')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='/kaggle/working/extras')
    parser.add_argument('--skip-gpu', action='store_true')
    parser.add_argument('--data-root', help='ga_eval folder (default: search /kaggle/input)')
    parser.add_argument('--runs-dir', help='folder with <run>/test_{seen,unseen}_predictions.csv (default: attached inputs)')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = find_root(args.data_root); log(f'data root: {root}')
    found = find_runs(args.runs_dir); test = test_frame(root)
    log(f'test pairs: {len(test)}; scenes: {test["scene"].nunique()}')
    mcnemar(found, test, out)
    test = coverage(root, found, test, out)
    run_logs(out)
    if not args.skip_gpu:
        gate_maps(root, find_checkpoints(), test, out)
    log('EXTRAS DONE')

if __name__ == '__main__':
    main()
