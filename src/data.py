"""All per-sample reads use local memory maps; no geometric/image augmentation."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from eval_utils import load_gt_array, rect_polygon
from common import read_json, worker_seed

def draw_targets(gts_416, cfg):
    targets = np.zeros((4, cfg.img, cfg.img), dtype=np.float32)
    scale = cfg.img/416.0
    for gt in gts_416:
        cx, cy, length, width, angle = map(float, gt)
        cx, cy, length, width = cx*scale, cy*scale, length*scale, width*scale
        polygon = rect_polygon(cx, cy, max(length/3, cfg.min_rect_px), max(width, cfg.min_rect_px), angle)
        corners = np.rint(np.asarray(polygon.exterior.coords)[:4]).astype(np.int32)
        mask = np.zeros((cfg.img, cfg.img), dtype=np.uint8)
        cv2.fillPoly(mask, [corners], 1)
        inside = mask.astype(bool)
        targets[0, inside] = 1.0
        targets[1, inside] = np.cos(2*angle)
        targets[2, inside] = np.sin(2*angle)
        targets[3, inside] = np.clip(length/cfg.width_norm, 0, 1)
    return targets

class GraspDataset(Dataset):
    def __init__(self, manifest_csv, split, cfg):
        self.cfg, self.split = cfg, split
        self.frame = manifest_csv.reset_index(drop=True).copy() if isinstance(manifest_csv, pd.DataFrame) else pd.read_csv(
            manifest_csv, dtype={'stem': str, 'scene': str, 'obj': str, 'part': str}, keep_default_na=False)
        self.rows = self.frame.to_dict('records')
        self.scene_to_idx = read_json(Path(cfg.cache)/'scene_to_index.json')
        self.prompt_to_idx = read_json(Path(cfg.cache)/'prompt_to_row.json')
        self.n_images = int(read_json(Path(cfg.cache)/'images.json')['N'])
        self._pid = None

    def _open(self):
        if self._pid != os.getpid():
            self._labels = np.load(Path(self.cfg.pack_root)/'packed/labels.npy', mmap_mode='r', allow_pickle=False)
            self._images = np.memmap(Path(self.cfg.cache)/'img224.u8', mode='r', dtype=np.uint8,
                                     shape=(self.n_images, self.cfg.img, self.cfg.img, 3))
            self._tokens = None
            if self.cfg.variant in ('C', 'D', 'E'):
                self._tokens = np.memmap(Path(self.cfg.cache)/'clip_tokens.f16', mode='r', dtype=np.float16,
                                        shape=(self.n_images, 196, 768))
            self._pid = os.getpid()

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        self._open()
        row = self.rows[index]
        scene_idx = int(self.scene_to_idx[row['scene']])
        start, count = int(row['label_start']), int(row['label_count'])
        gts = load_gt_array(self._labels[start:start+count]).astype(np.float32)
        result = {'img': torch.from_numpy(np.array(self._images[scene_idx].transpose(2, 0, 1), copy=True)),
                  'prompt_idx': int(self.prompt_to_idx[row['prompt']]), 'scene_idx': scene_idx,
                  'targets': torch.from_numpy(draw_targets(gts, self.cfg)), 'gts_416': gts,
                  'stem': row['stem'], 'scene': row['scene'], 'prompt': row['prompt']}
        if self._tokens is not None:
            result['clip_tokens'] = torch.from_numpy(np.array(self._tokens[scene_idx], copy=True))
        return result

def collate(samples):
    result = {key: torch.stack([item[key] for item in samples]) for key in ('img', 'targets')}
    for key in ('prompt_idx', 'scene_idx'): result[key] = torch.tensor([item[key] for item in samples], dtype=torch.long)
    for key in ('gts_416', 'stem', 'scene', 'prompt'): result[key] = [item[key] for item in samples]
    if 'clip_tokens' in samples[0]: result['clip_tokens'] = torch.stack([item['clip_tokens'] for item in samples])
    return result

def make_loader(manifest, split, cfg, shuffle=False, batch_size=None):
    dataset = GraspDataset(manifest, split, cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(dataset, batch_size=batch_size or cfg.batch, shuffle=shuffle, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), collate_fn=collate, worker_init_fn=worker_seed,
        persistent_workers=cfg.num_workers > 0, generator=generator, drop_last=False,
        **({'multiprocessing_context':'spawn'} if cfg.num_workers > 0 else {}))
