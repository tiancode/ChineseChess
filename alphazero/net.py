"""ResNet policy/value network (AlphaZero-style dual head)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import N_PLANES, POLICY_SIZE


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class XiangqiNet(nn.Module):
    """Input ``(B, 16, 10, 9)`` -> (policy logits ``(B, 8100)``,
    value ``(B,)`` in ``[-1, 1]``)."""

    def __init__(self, channels: int = 128, res_blocks: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(res_blocks)])

        # Policy head: 1x1 conv -> flatten -> linear over the 8100 actions.
        self.p_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(4)
        self.p_fc = nn.Linear(4 * 10 * 9, POLICY_SIZE)

        # Value head.
        self.v_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(2)
        self.v_fc1 = nn.Linear(2 * 10 * 9, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.tower(self.stem(x))

        p = F.relu(self.p_bn(self.p_conv(x)))
        p = self.p_fc(p.flatten(1))  # raw logits; masking happens at use site

        v = F.relu(self.v_bn(self.v_conv(x)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v)).squeeze(1)
        return p, v


def save_ckpt(path: str, net: XiangqiNet, cfg, extra: dict | None = None):
    torch.save(
        {
            "model": net.state_dict(),
            "channels": cfg.channels,
            "res_blocks": cfg.res_blocks,
            "extra": extra or {},
        },
        path,
    )


def load_net(path: str, device: str) -> "XiangqiNet":
    ck = torch.load(path, map_location=device, weights_only=True)
    net = XiangqiNet(ck["channels"], ck["res_blocks"]).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net
