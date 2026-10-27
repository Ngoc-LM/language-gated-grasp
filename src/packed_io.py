"""Portable access to ga_eval/packed; cached metadata, memmapped labels, cached JPEG handles.

root is ga_eval/ (or its packed/ directory). Metadata contains no absolute paths.
Treat returned DataFrames and the labels memmap as read-only. read_jpeg returns a
fully decoded RGB PIL.Image, independent of its compressed input buffer.
"""
import io
import os
import threading
import zlib
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

_local = threading.local()

def _directory(root):
    path = Path(os.path.abspath(os.fspath(root)))
    return path if path.name == 'packed' else path / 'packed'

@lru_cache(maxsize=8)
def _open(directory, pid):
    path = Path(directory)
    annotations = pd.read_parquet(path / 'annotations.parquet')
    labels = np.load(path / 'labels.npy', mmap_mode='r', allow_pickle=False)
    images = pd.read_parquet(path / 'images.parquet')
    if labels.dtype != np.float32 or labels.ndim != 2 or labels.shape[1] != 6:
        raise ValueError('Expected float32 labels.npy with shape (total_rows,6)')
    if annotations['stem'].duplicated().any() or images['scene'].duplicated().any():
        raise ValueError('Packed annotation stems and image scenes must be unique')
    starts = annotations['label_start'].to_numpy(dtype=np.int64)
    counts = annotations['label_count'].to_numpy(dtype=np.int64)
    if np.any(starts < 0) or np.any(counts < 0) or np.any(starts+counts > len(labels)):
        raise ValueError('Annotation label slices exceed labels.npy')
    index = {row.scene: {'offset': int(row.offset), 'nbytes': int(row.nbytes), 'crc32': int(row.crc32)}
             for row in images.itertuples(index=False)}
    return annotations, labels, index, path / 'images.bin'

def open_packed(root):
    """Return (annotations DataFrame, read-only labels memmap, image index dict, blob Path).

    Labels for an annotation row are labels[row.label_start:row.label_start+row.label_count].
    The PID cache key prevents inherited memmaps/metadata from being reused after fork.
    """
    return _open(str(_directory(root)), os.getpid())

def _handle(path):
    pid = os.getpid()
    if getattr(_local, 'pid', None) != pid:
        for handle in getattr(_local, 'handles', {}).values():
            handle.close()
        _local.pid, _local.handles = pid, {}
    key = str(path)
    handle = _local.handles.get(key)
    if handle is None or handle.closed:
        handle = open(path, 'rb')
        _local.handles[key] = handle
    return handle

def read_jpeg(root, scene):
    """Seek/read one CRC-checked JPEG using a cached handle per process and thread.

    Independent handles avoid seek races between threads and shared file offsets
    between forked training DataLoader workers. No source image file is opened.
    """
    annotations, labels, index, path = open_packed(root)
    record = index[str(scene)]
    handle = _handle(path)
    handle.seek(record['offset'])
    data = handle.read(record['nbytes'])
    if len(data) != record['nbytes'] or (zlib.crc32(data) & 0xffffffff) != record['crc32']:
        raise IOError(f'Packed JPEG is truncated or has a CRC mismatch: {scene}')
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.convert('RGB')

def close_packed(root=None):
    """Close this thread's cached JPEG handles and cached label maps before rebuilding."""
    directory = _directory(root) if root is not None else None
    for key, handle in list(getattr(_local, 'handles', {}).items()):
        if directory is None or Path(key).parent == directory:
            handle.close()
            del _local.handles[key]
    # Cache clearing releases metadata. Existing memmap references remain valid;
    # a notebook rebuilding files must first drop its references (cell 4 does).
    _open.cache_clear()
