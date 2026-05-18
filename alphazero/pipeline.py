"""The AlphaZero loop: self-play -> train -> (optional) evaluation gate.

Each iteration:
  1. play ``games_per_iter`` self-play games with the current network,
  2. push the samples into the replay buffer,
  3. take ``train_steps_per_iter`` optimisation steps,
  4. if ``arena_games > 0``, play the candidate vs the frozen best; promote
     (and checkpoint) the candidate only if it scores >= ``arena_win_rate``.

Usage:
  python -m alphazero.pipeline                 # full GPU run
  python -m alphazero.pipeline --smoke         # tiny end-to-end sanity run
  python -m alphazero.pipeline --iterations 5  # override any Config field
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time

import numpy as np
import torch

from .config import Config
from .net import XiangqiNet, save_ckpt
from .train import Trainer
from .replay_buffer import ReplayBuffer
from .mcts import Evaluator
from .selfplay import play_game
from .arena import play_match


def _build_cfg() -> Config:
    p = argparse.ArgumentParser(description="AlphaZero Xiangqi training")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU profile to validate the pipeline")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint to resume the network weights from")
    # Allow overriding any scalar Config field from the CLI.
    base = Config()
    for k, v in base.as_dict().items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() == "true",
                           default=None)
        else:
            p.add_argument(f"--{k}", type=type(v), default=None)
    args = p.parse_args()

    cfg = Config.smoke() if args.smoke else Config()
    for k in base.as_dict():
        val = getattr(args, k)
        if val is not None:
            setattr(cfg, k, val)
    cfg._resume = args.resume
    return cfg


def main():
    cfg = _build_cfg()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)  # replay_buffer.sample_batch uses random.sample
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    print(f"device={cfg.device} amp={cfg.amp} "
          f"net=({cfg.channels}x{cfg.res_blocks}) sims={cfg.sims}")

    net = XiangqiNet(cfg.channels, cfg.res_blocks).to(cfg.device)
    if getattr(cfg, "_resume", None):
        ck = torch.load(cfg._resume, map_location=cfg.device, weights_only=True)
        net.load_state_dict(ck["model"])
        print(f"resumed weights from {cfg._resume}")

    trainer = Trainer(net, cfg)
    buffer = ReplayBuffer(cfg.buffer_size)
    best_net = copy.deepcopy(net)

    for it in range(1, cfg.iterations + 1):
        t0 = time.time()

        # --- self-play -------------------------------------------------
        net.eval()
        ev = Evaluator(net, cfg)
        results = {0: 0, 1: 0, None: 0}
        n_samples = 0
        n_disabled = n_would = n_fp = 0
        for _ in range(cfg.games_per_iter):
            samples, winner, _ply, info = play_game(ev, cfg)
            buffer.add_many(samples)
            results[winner] += 1
            n_samples += len(samples)
            if info["resign_disabled"]:
                n_disabled += 1
                if info["would_resign"]:
                    n_would += 1
                    n_fp += int(info["false_positive"])
        t_sp = time.time() - t0

        # Resignation false-positive control: hold the FP rate (over the
        # resign-disabled games where the call fired) near the target by
        # nudging the threshold -- more negative => resign less eagerly.
        fp_rate = n_fp / n_would if n_would else 0.0
        if cfg.resign_auto_tune and n_would >= 4:
            if fp_rate > cfg.resign_target_fp:
                cfg.resign_value = max(-0.99, cfg.resign_value - 0.02)
            elif fp_rate < 0.5 * cfg.resign_target_fp:
                cfg.resign_value = min(-0.80, cfg.resign_value + 0.01)

        # --- train -----------------------------------------------------
        t1 = time.time()
        stats = trainer.train_epoch(buffer, cfg.train_steps_per_iter)
        t_tr = time.time() - t1

        loss_str = ("buffer warming up" if stats is None else
                    f"loss={stats[0]:.3f} (p={stats[1]:.3f} v={stats[2]:.3f})")
        print(f"[iter {it:>3}] R/B/D={results[0]}/{results[1]}/{results[None]} "
              f"samples={n_samples} buf={len(buffer)} | {loss_str} "
              f"| sp={t_sp:.0f}s tr={t_tr:.0f}s")
        if n_disabled:
            print(f"           resign: no-resign games={n_disabled} "
                  f"would-resign={n_would} fp={n_fp} "
                  f"rate={fp_rate:.2f} thr={cfg.resign_value:.2f}")

        # --- evaluation gate ------------------------------------------
        if cfg.arena_games > 0 and stats is not None:
            rate = play_match(net, best_net, cfg, cfg.arena_games)
            if rate >= cfg.arena_win_rate:
                best_net = copy.deepcopy(net)
                path = os.path.join(cfg.ckpt_dir, f"best_iter{it:03d}.pt")
                save_ckpt(path, net, cfg, {"iter": it, "arena": rate})
                save_ckpt(os.path.join(cfg.ckpt_dir, "best.pt"), net, cfg,
                          {"iter": it, "arena": rate})
                print(f"           candidate PROMOTED (score {rate:.2f}) "
                      f"-> {path}")
            else:
                print(f"           candidate rejected (score {rate:.2f})")
        else:
            save_ckpt(os.path.join(cfg.ckpt_dir, "latest.pt"), net, cfg,
                      {"iter": it})

    print("done.")


if __name__ == "__main__":
    main()
