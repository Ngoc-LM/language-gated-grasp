"""End-to-end batch-1 latency for every variant, measured sequentially in ONE process on ONE idle GPU.

Stages (CUDA-synchronised, warmup 20, timed 100, fp16):
  head   = GraspNet forward + GPU blur/decode on cached inputs (what metrics.json reports as latency_ms)
  text   = CLIP tokenizer (CPU) + CLIP text encoder        (variants B-E; once per instruction)
  vision = CLIP ViT-B/16 image encoder -> 196 patch tokens  (variants C/D/E; once per image)
  e2e    = text + vision + head in one loop (uint8 224x224 image and raw prompt string -> grasp)
GraspNet weights are randomly initialised: run time does not depend on weight values.
"""
import argparse
import json
import time
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import log, read_json, seed_all
from config import Config
from model import GraspNet, parameter_counts
from infer import decode_top1
from prepare import load_clip, CLIP_ID

def timed(fn, device, warmup=20, iterations=100):
    times = []
    for i in range(warmup + iterations):
        torch.cuda.synchronize(device); start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        if i >= warmup: times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.std(times))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='/tmp/ga_cache'); parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--variants', nargs='+', default=list('ABCDE'))
    parser.add_argument('--out', default='/kaggle/working/results')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}'); torch.cuda.set_device(device); seed_all(0)
    torch.backends.cudnn.benchmark = True
    cache = Path(args.cache); cfg = Config()
    from transformers import AutoTokenizer
    import os
    tokenizer = AutoTokenizer.from_pretrained(CLIP_ID, cache_dir='/tmp/ga_hf', token=os.environ.get('HF_TOKEN') or False)
    clip = load_clip(device)
    n = int(read_json(cache / 'images.json')['N'])
    images = np.memmap(cache / 'img224.u8', mode='r', dtype=np.uint8, shape=(n, 224, 224, 3))
    row = pd.read_csv(cache / 'test_seen.csv', dtype={'scene': str, 'stem': str}, keep_default_na=False).iloc[0]
    scene_idx = int(read_json(cache / 'scene_to_index.json')[row['scene']]); prompt = str(row['prompt'])
    image_hwc = torch.from_numpy(np.array(images[scene_idx], copy=True)).to(device)          # uint8 (224,224,3)
    image = image_hwc.permute(2, 0, 1)[None].contiguous()                                   # uint8 (1,3,224,224)
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device)[None, :, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711], device=device)[None, :, None, None]

    def encode_text():
        tokens = tokenizer([prompt], padding=True, truncation=True, max_length=77, return_tensors='pt')
        output = clip.get_text_features(**{k: v.to(device) for k, v in tokens.items()})
        features = output if isinstance(output, torch.Tensor) else output.pooler_output
        return features.float()

    def encode_image():
        pixel_values = ((image.float() / 255 - mean) / std).half()
        hidden = clip.vision_model(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        return clip.vision_model.post_layernorm(hidden)

    rows = []
    with torch.inference_mode():
        text_cached = encode_text(); tokens_cached = encode_image()
        clip_text_params = sum(p.numel() for p in clip.text_model.parameters()) + sum(p.numel() for p in clip.text_projection.parameters())
        clip_vision_params = sum(p.numel() for p in clip.vision_model.parameters())
        for variant in args.variants:
            model = GraspNet(variant).to(device).eval()
            uses_text, uses_vision = variant != 'A', variant in ('C', 'D', 'E')
            t_in = text_cached if uses_text else None; v_in = tokens_cached if uses_vision else None

            def head():
                with torch.autocast('cuda', dtype=torch.float16):
                    maps = model(image, t_in, v_in)['maps']
                decode_top1(maps, cfg)

            def e2e():
                t = encode_text() if uses_text else None
                v = encode_image() if uses_vision else None
                with torch.autocast('cuda', dtype=torch.float16):
                    maps = model(image, t, v)['maps']
                decode_top1(maps, cfg)

            head_ms, head_sd = timed(head, device)
            text_ms, _ = timed(encode_text, device) if uses_text else (0.0, 0.0)
            vision_ms, _ = timed(encode_image, device) if uses_vision else (0.0, 0.0)
            e2e_ms, e2e_sd = timed(e2e, device)
            counts = parameter_counts(model)
            frozen = (clip_text_params if uses_text else 0) + (clip_vision_params if uses_vision else 0)
            rows.append({'variant': variant, 'trainable_params': counts['trainable'], 'frozen_clip_params_used': frozen,
                         'head_ms': head_ms, 'head_sd': head_sd, 'clip_text_ms': text_ms, 'clip_vision_ms': vision_ms,
                         'e2e_ms': e2e_ms, 'e2e_sd': e2e_sd})
            log(f'LATENCY {variant}: head={head_ms:.2f}+-{head_sd:.2f} text={text_ms:.2f} vision={vision_ms:.2f} '
                f'e2e={e2e_ms:.2f}+-{e2e_sd:.2f} ms; trainable={counts["trainable"]:,} frozen_used={frozen:,}')
            del model; torch.cuda.empty_cache()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows); frame.to_csv(out / 'latency_e2e.csv', index=False)
    (out / 'latency_e2e.json').write_text(json.dumps({'gpu': torch.cuda.get_device_name(device), 'batch': 1, 'dtype': 'fp16',
        'warmup': 20, 'iterations': 100, 'prompt': prompt, 'note': __doc__, 'rows': rows}, indent=2))
    log('\n' + frame.to_string(index=False))

if __name__ == '__main__':
    main()
