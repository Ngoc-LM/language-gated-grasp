"""CPU-only checks that need no dataset or GPU:  python -m pytest tests -q"""
import dataclasses
import sys
from math import pi
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import Config  # noqa: E402
from data import draw_targets  # noqa: E402
from eval_utils import harmonic_mean, is_success  # noqa: E402
from infer import decode_top1  # noqa: E402
from model import GraspNet, parameter_counts  # noqa: E402
import tests as unit_checks  # noqa: E402  (src/tests.py)

TRAINABLE = {'A': 1_897_988, 'B': 2_554_628, 'C': 2_882_821, 'D': 2_882_821, 'E': 2_751_749}

def test_shapes_and_film_identity():
    unit_checks.test_shapes_and_film(torch.device('cpu'))

@pytest.mark.parametrize('variant', list(TRAINABLE))
def test_parameter_counts_match_reported(variant):
    assert parameter_counts(GraspNet(variant))['trainable'] == TRAINABLE[variant]

def test_metric_sanity():
    gt = np.array([[208., 208., 120., 60., 0.3]])
    assert is_success(gt[0], gt)
    assert is_success(gt[0] + [0, 0, 0, 0, np.deg2rad(29)], gt)
    assert not is_success(gt[0] + [0, 0, 0, 0, np.deg2rad(31)], gt)
    assert is_success(gt[0] + [0, 0, 0, 0, pi], gt)          # a grasp is symmetric under 180 degrees
    assert not is_success(gt[0] + [80, 0, 0, 0, 0], gt)       # IoU below 0.25
    assert harmonic_mean(0.5, 0.5) == pytest.approx(0.5)

@pytest.mark.parametrize('mode', ['plateau', 'argmax'])
def test_target_roundtrip(mode):
    """Decoding drawn ground-truth target maps must recover a successful grasp."""
    rng = np.random.default_rng(0); cfg = dataclasses.replace(Config(), decode=mode); offsets = []
    for _ in range(50):
        length = rng.uniform(30, 140)
        gts = np.array([[rng.uniform(80, 336), rng.uniform(80, 336), length, length / 2,
                         rng.uniform(-pi / 2, pi / 2)]], dtype=np.float32)
        pred = decode_top1(torch.from_numpy(draw_targets(gts, cfg))[None], cfg)[0].numpy()
        assert is_success(pred, gts)
        offsets.append(np.hypot(*(pred[:2] - gts[0, :2])))
    if mode == 'plateau':
        assert np.median(offsets) < 3.0          # the plateau decoder returns the rectangle centre

def test_plateau_equals_argmax_for_strict_maximum():
    maps = torch.rand(3, 4, 224, 224); cfg = Config()
    assert torch.allclose(decode_top1(maps, cfg), decode_top1(maps, dataclasses.replace(cfg, decode='argmax')))
