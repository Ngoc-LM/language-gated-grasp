"""Copy the small per-run artifacts of the attached Kaggle run outputs (no checkpoints) into one folder.

    python collect_artifacts.py --out /kaggle/working/reeval/original   # add --zip to pack /kaggle/working/reeval
"""
import argparse
import glob
import shutil
from pathlib import Path

RUNS = ['A', 'B', 'B_s1', 'E', 'E_s1', 'C', 'C_s1', 'D']
FILES = ['metrics.json', 'config.json', 'train_log.csv', 'test_seen_predictions.csv', 'test_unseen_predictions.csv']
SHARED = ['manifest_fix.json', 'prepared.json', 'target_roundtrip.json', 'target_roundtrip.csv']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='/kaggle/working/reeval/original'); parser.add_argument('--zip')
    args = parser.parse_args(); out = Path(args.out)
    for run in RUNS:
        hits = sorted(glob.glob(f'/kaggle/input/**/results/{run}/metrics.json', recursive=True))
        if not hits: continue
        source = Path(hits[0]).parent; target = out / run; target.mkdir(parents=True, exist_ok=True)
        copied = [name for name in FILES if (source / name).is_file() and shutil.copy2(source / name, target / name)]
        print(f'{run}: {source} -> {target} ({", ".join(copied)})', flush=True)
    for name in SHARED:
        hits = sorted(glob.glob(f'/kaggle/input/**/results/{name}', recursive=True))
        if hits: shutil.copy2(hits[0], out / name); print(f'{name}: {hits[0]}', flush=True)
    if args.zip:
        archive = shutil.make_archive(args.zip, 'zip', root_dir=Path(args.out).parent)
        print(f'archive: {archive} ({Path(archive).stat().st_size / 1e6:.1f} MB)', flush=True)

if __name__ == '__main__':
    main()
