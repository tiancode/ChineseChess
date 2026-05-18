"""Self-play game generation.

One game yields a list of ``(planes, idxs, pi, z)`` samples. ``pi`` is the
root MCTS visit distribution; ``z`` is filled in once the game ends, as the
final result seen from each stored position's mover.
"""

from __future__ import annotations

import random

import numpy as np

from .xiangqi.board import opposite
from .xiangqi.game import GameState
from .encoding import encode_state, move_to_index
from .mcts import Evaluator, run_mcts, select_move


def play_game(ev: Evaluator, cfg, *, verbose: bool = False):
    """Play one self-play game.

    Returns ``(samples, winner, ply, info)``. With probability
    ``cfg.resign_disable_frac`` resignation is *disabled* and the game is
    played to its real end; ``info`` then lets the caller check whether the
    would-resign call was a false positive (the side that would have resigned
    did not actually lose), so the threshold can be tuned to hold
    ``cfg.resign_target_fp``.
    """
    game = GameState()
    records = []  # (planes, idxs, pi, side_to_move)
    ply = 0
    winner = None  # RED / BLACK / None(draw)

    allow_resign = random.random() >= cfg.resign_disable_frac
    would_resign_side = None  # first side whose best line went hopeless

    while True:
        tv = game.terminal_value()
        if tv is not None:
            # Side to move has no reply (-1 -> it lost) or a draw (0).
            winner = None if tv == 0.0 else opposite(game.side_to_move)
            break
        if ply >= cfg.max_game_len:
            winner = None  # capped: scored a draw
            break

        ev.clear()
        root, visits = run_mcts(game, ev, cfg, add_noise=True)

        side = game.side_to_move
        moves = list(visits.keys())
        counts = np.array([visits[m] for m in moves], dtype=np.float32)

        # Resignation: the best line is hopeless after enough plies. Record
        # the first occurrence either way; only actually resign when allowed.
        if ply >= cfg.resign_after and would_resign_side is None:
            best = max(root.children.values(), key=lambda c: c.N)
            if best.Q < cfg.resign_value:
                would_resign_side = side
                if allow_resign:
                    winner = opposite(side)
                    break
                # else: play on so the true outcome is observed.

        planes = encode_state(game)
        idxs = np.fromiter((move_to_index(side, m) for m in moves),
                           dtype=np.int64, count=len(moves))
        pi = counts / counts.sum()
        records.append((planes, idxs, pi, side))

        temp = 1.0 if ply < cfg.temp_moves else 0.0
        mv = select_move(visits, temp)
        game.apply(mv)
        ply += 1

    samples = []
    for planes, idxs, pi, side in records:
        if winner is None:
            z = 0.0
        else:
            z = 1.0 if winner == side else -1.0
        samples.append((planes, idxs, pi, z))

    # A false positive: resignation was disabled, the would-resign call
    # fired, yet that side did not actually lose (it drew or won).
    fp = (not allow_resign and would_resign_side is not None
          and winner != opposite(would_resign_side))
    info = {
        "resign_disabled": not allow_resign,
        "would_resign": would_resign_side is not None,
        "false_positive": fp,
    }

    if verbose:
        res = {None: "draw"}.get(winner, "Red" if winner == 0 else "Black")
        print(f"  game: {ply} plies, result={res}, {len(samples)} samples")
    return samples, winner, ply, info
