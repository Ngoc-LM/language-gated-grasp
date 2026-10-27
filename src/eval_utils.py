"""CPU grasp geometry in LGD's image-coordinate convention; no model or training code."""
from math import pi
import numpy as np
import torch
from shapely.geometry import Polygon

def load_gt_array(raw, scale=1.0):
    """Convert a raw float32 (N,6) array/view to [cx,cy,length,width,angle_rad].

    Column 0 is ignored. Source row (_,x,y,w,h,theta_deg) maps to center
    (row=y,col=x), angle=-theta_deg*pi/180, length=w, width=h. Scale affects
    x,y,w,h only. Input memmap views are never mutated.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError('Expected raw array with shape (N,6)')
    if raw.dtype != np.float32:
        raise TypeError('Expected float32 raw labels')
    result = raw[:, 1:6].astype(np.float64, copy=True)
    result[:, :4] *= scale
    result[:, 4] = -np.deg2rad(raw[:, 5].astype(np.float64))
    if not np.isfinite(result).all() or np.any(result[:, 2:4] <= 0):
        raise ValueError('GT has nonfinite values or nonpositive dimensions')
    return result

def load_gt(path, scale=1.0):
    """Compatibility wrapper: load one CPU tensor file, then delegate to load_gt_array."""
    tensor = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'Expected a torch tensor: {path}')
    return load_gt_array(tensor.detach().cpu().numpy(), scale=scale)

def rect_polygon(cx, cy, length, width, angle):
    """Four-corner Polygon in (x=column,y=row) coordinates, with image y pointing down.

    Positive angle rotates counterclockwise on the displayed image, matching LGD.
    The length-axis vector is (cos(angle), -sin(angle)); the width-axis vector is
    (sin(angle), cos(angle)). Thus a stored positive theta_deg becomes a negative
    angle and its length axis slopes down to the right. Polygons are not clipped.
    """
    values = np.asarray([cx, cy, length, width, angle], dtype=float)
    if not np.isfinite(values).all() or length <= 0 or width <= 0:
        raise ValueError('Rectangle parameters must be finite with positive length/width')
    u = np.array([np.cos(angle), -np.sin(angle)]) * (length/2)
    v = np.array([np.sin(angle), np.cos(angle)]) * (width/2)
    center = np.array([cx, cy])
    return Polygon([center-u-v, center+u-v, center+u+v, center-u+v])

def angle_diff(a, b):
    """Symmetric angular distance modulo pi (parallel/opposite axes are equivalent)."""
    return abs(((a-b+pi/2) % pi)-pi/2)

def polygon_iou(a, b):
    """Continuous, unclipped polygon IoU, without the separate angle gate."""
    first, second = rect_polygon(*a), rect_polygon(*b)
    intersection = first.intersection(second).area
    union = first.area+second.area-intersection
    return float(intersection/union) if union > 0 else 0.0

def is_success(pred, gts, iou_thr=0.25, ang_thr=pi/6):
    """Strict gates: ANY GT with angle_diff < ang_thr AND IoU > iou_thr succeeds."""
    pred, gts = np.asarray(pred, dtype=float), np.asarray(gts, dtype=float)
    if pred.shape != (5,) or gts.ndim != 2 or gts.shape[1] != 5:
        raise ValueError('Expected pred shape (5,) and gts shape (N,5)')
    if not np.isfinite(pred).all() or np.any(pred[2:4] <= 0):
        return False
    if not np.isfinite(gts).all() or np.any(gts[:, 2:4] <= 0):
        raise ValueError('GT rectangles must be finite with positive length/width')
    for gt in gts:
        if angle_diff(pred[4], gt[4]) < ang_thr and polygon_iou(pred, gt) > iou_thr:
            return True
    return False

def success_rate(pairs):
    """Fraction of successful (prediction, GT-array) pairs; an empty input is an error."""
    count, successes = 0, 0
    for prediction, gts in pairs:
        successes += int(is_success(prediction, gts))
        count += 1
    if count == 0:
        raise ValueError('success_rate requires at least one pair')
    return successes/count

def harmonic_mean(s, u):
    """2*s*u/(s+u), or 0 when both rates are zero."""
    if not (np.isfinite(s) and np.isfinite(u) and 0 <= s <= 1 and 0 <= u <= 1):
        raise ValueError('s and u must be finite success rates in [0,1]')
    return 2*s*u/(s+u) if s+u else 0.0
