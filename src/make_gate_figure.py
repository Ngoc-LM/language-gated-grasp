"""Render the gate-map figure of the paper from extras.py output (gate_best.json).

    python make_gate_figure.py gate_best.json --index 0 --out ../paper/fig/gate.png

Panels: C0 and C, each for two prompts on the same image. Heat map: gate sigma(tau*S) on the 14x14 patch grid
(bilinear upsampling for display). Green: ground-truth grasps of the prompted part. Red: top-1 prediction.
"""
import argparse
import base64
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from eval_utils import rect_polygon

def corners(rect, scale):
    cx, cy, length, width, angle = rect
    poly = rect_polygon(cx * scale, cy * scale, length * scale, width * scale, angle)
    return np.asarray(poly.exterior.coords)[:4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json'); parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--out', default='gate.png')
    args = parser.parse_args()
    with open(args.json) as stream: entry = json.load(stream)[args.index]
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(entry['jpeg320_b64']))).convert('RGB'))
    size = image.shape[0]; scale = size / 416.0
    prompts = entry['prompts'][:2]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    panels = [('E', 0, 'C0, P1'), ('E', 1, 'C0, P2'), ('C', 0, 'C, P1'), ('C', 1, 'C, P2')]
    for ax, (run, k, label) in zip(axes, panels):
        item = prompts[k]
        gate = np.asarray(item[run]['gate'], dtype=np.float32)
        heat = np.asarray(Image.fromarray(gate).resize((size, size), Image.Resampling.BILINEAR))
        ax.imshow(image); ax.imshow(heat, cmap='inferno', alpha=0.55, vmin=0, vmax=1)
        for gt in item['gts']:
            ax.add_patch(Polygon(corners(gt, scale), closed=True, fill=False, ec='#2ecc40', lw=1.2))
        ax.add_patch(Polygon(corners(item[run]['pred'], scale), closed=True, fill=False,
                             ec='#ff2d2d', lw=1.6))
        mark = '✓' if item[run]['ok'] else '✗'
        ax.text(6, 22, f'{label} {mark}', color='white', fontsize=8, weight='bold',
                bbox=dict(fc='black', alpha=0.55, pad=1.2, lw=0))
        ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.03)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print('P1:', prompts[0]['prompt']); print('P2:', prompts[1]['prompt'])

if __name__ == '__main__':
    main()
