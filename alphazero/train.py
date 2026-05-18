"""Network optimisation.

Loss = policy cross-entropy (over the legal subset) + value MSE.
L2 regularisation is applied via the optimiser's ``weight_decay``.

The policy term mirrors AlphaZero's ``-pi . log p``, but the softmax is
taken over only the legal moves of each position (gathered via the stored
indices) so probability mass is never spent on illegal actions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, net, cfg):
        self.net = net
        self.cfg = cfg
        # AdamW: decoupled weight decay (true L2 regularisation), unlike
        # Adam where `weight_decay` is folded into the adaptive update.
        self.opt = torch.optim.AdamW(
            net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=cfg.amp and cfg.device == "cuda"
        )

    def _loss(self, planes, idx_pad, pi_pad, mask, z):
        logits, value = self.net(planes)

        # Gather logits at each sample's legal indices, mask the pad slots
        # with a large finite negative (an exact -inf would make the 0*-inf
        # pad terms NaN), then log-softmax over the legal subset only.
        # Done in fp32: under AMP the logits are fp16, where -1e9 overflows
        # and the softmax is numerically fragile.
        gathered = logits.gather(1, idx_pad).float()
        gathered = gathered.masked_fill(mask == 0, -1e9)
        logp = F.log_softmax(gathered, dim=1)
        policy_loss = -(pi_pad * logp * mask).sum(dim=1).mean()

        value_loss = F.mse_loss(value.float(), z)
        total = policy_loss + self.cfg.value_loss_weight * value_loss
        return total, policy_loss.detach(), value_loss.detach()

    def step(self, batch):
        planes, idx_pad, pi_pad, mask, z = batch
        self.net.train()
        self.opt.zero_grad(set_to_none=True)

        use_amp = self.cfg.amp and self.cfg.device == "cuda"
        with torch.autocast("cuda", enabled=use_amp):
            total, pl, vl = self._loss(planes, idx_pad, pi_pad, mask, z)

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(total.detach()), float(pl), float(vl)

    def train_epoch(self, buffer, steps: int):
        if len(buffer) < self.cfg.batch_size:
            return None
        agg = [0.0, 0.0, 0.0]
        for _ in range(steps):
            batch = buffer.sample_batch(self.cfg.batch_size, self.cfg.device)
            t, p, v = self.step(batch)
            agg[0] += t
            agg[1] += p
            agg[2] += v
        return tuple(a / steps for a in agg)
