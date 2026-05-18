"""Ring buffer of self-play samples and a padded batch collator.

A sample is ``(planes, idxs, pi, z)``:
    planes  float16 (16,10,9)   canonical input
    idxs    int64   (L,)        canonical policy indices of the *legal* moves
    pi      float32 (L,)        MCTS visit distribution over those moves
    z       float32 scalar      game outcome from this state's mover view

Because ``idxs`` already enumerates exactly the legal moves, the training
loss masks to that subset by gathering logits at ``idxs`` -- no dense 8100
mask is ever materialised.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def add(self, planes, idxs, pi, z):
        self.buf.append((
            planes.astype(np.float16),
            idxs.astype(np.int64),
            pi.astype(np.float32),
            np.float32(z),
        ))

    def add_many(self, samples):
        for s in samples:
            self.add(*s)

    def sample_batch(self, batch_size: int, device: str):
        items = random.sample(self.buf, min(batch_size, len(self.buf)))
        planes = np.stack([it[0] for it in items]).astype(np.float32)
        z = np.array([it[3] for it in items], dtype=np.float32)

        max_l = max(len(it[1]) for it in items)
        b = len(items)
        idx_pad = np.zeros((b, max_l), dtype=np.int64)
        pi_pad = np.zeros((b, max_l), dtype=np.float32)
        mask = np.zeros((b, max_l), dtype=np.float32)
        for i, it in enumerate(items):
            L = len(it[1])
            idx_pad[i, :L] = it[1]
            pi_pad[i, :L] = it[2]
            mask[i, :L] = 1.0

        t = lambda a: torch.from_numpy(a).to(device)
        return (t(planes), t(idx_pad), t(pi_pad), t(mask), t(z))
