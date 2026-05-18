"""Central configuration. GPU-oriented defaults (NVIDIA / CUDA).

Tune ``RES_BLOCKS`` / ``CHANNELS`` to your VRAM. The defaults
(channels=128, blocks=10) train comfortably on ~6-8 GB; a smoke-test
profile (``Config.smoke()``) shrinks everything for a quick end-to-end run
on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import torch


@dataclass
class Config:
    # -- device ------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True  # mixed precision on CUDA

    # -- network -----------------------------------------------------------
    in_planes: int = 16          # see encoding.py (14 piece + 2 scalar)
    channels: int = 128
    res_blocks: int = 10
    policy_size: int = 90 * 90   # flat from*to action space (8100)

    # -- MCTS --------------------------------------------------------------
    sims: int = 160              # simulations per move in self-play
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.2
    dirichlet_eps: float = 0.25
    # Plies for which moves are sampled (temperature=1) for exploration;
    # afterwards the visit-count argmax is played.
    temp_moves: int = 30

    # -- self-play ---------------------------------------------------------
    games_per_iter: int = 40
    max_game_len: int = 220      # cap; reaching it scores the game a draw
    resign_value: float = -0.92  # root value below which a side resigns
    resign_after: int = 25       # ...only after this many plies
    # Fraction of self-play games in which resignation is *disabled* and the
    # game is played out, so the would-resign call can be checked against the
    # real result (AlphaZero's resignation false-positive control).
    resign_disable_frac: float = 0.1
    resign_target_fp: float = 0.05  # keep the false-positive rate below this
    resign_auto_tune: bool = True   # adjust resign_value to hold the target

    # -- training ----------------------------------------------------------
    buffer_size: int = 120_000   # samples (positions)
    batch_size: int = 512
    train_steps_per_iter: int = 800
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    value_loss_weight: float = 1.0

    # -- pipeline ----------------------------------------------------------
    iterations: int = 100
    arena_games: int = 20        # candidate vs best; 0 disables the gate
    arena_sims: int = 120
    arena_win_rate: float = 0.55 # promote candidate if it scores >= this
    ckpt_dir: str = "alphazero/checkpoints"
    seed: int = 0

    @staticmethod
    def smoke() -> "Config":
        """Tiny, fast settings to validate the whole loop end to end."""
        c = Config()
        c.device = "cpu"
        c.amp = False
        c.channels = 32
        c.res_blocks = 2
        c.sims = 16
        c.games_per_iter = 2
        c.max_game_len = 40
        c.temp_moves = 8
        c.buffer_size = 4_000
        c.batch_size = 64
        c.train_steps_per_iter = 20
        c.iterations = 2
        c.arena_games = 2
        c.arena_sims = 8
        return c

    def as_dict(self) -> dict:
        return asdict(self)
