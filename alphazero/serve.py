"""Long-lived MCTS inference sidecar for the Rust GUI.

Speaks line-delimited JSON over stdin/stdout. Reuses the trained network and
the exact play-time recipe (see :func:`alphazero.play.ai_move`): PUCT MCTS
with no Dirichlet noise, then the visit-count argmax -- deterministic.

Protocol
--------
On startup, after the model is loaded, emit exactly one line::

    {"ready": true, "device": "cpu"}

Then, per request line::

    {"moves": [[from, to], ...], "sims": 200}

reply with one line::

    {"move": [from, to]}      # chosen move
    {"move": null}            # terminal position / no legal move
    {"move": null, "error": "..."}   # request failed; server keeps running

``moves`` is the whole game history from the start position, as flat 0..89
board indices identical to the Rust ``Move {from, to}`` encoding
(``index = rank * 9 + file``, rank 0 = top). Replaying it on a fresh
``GameState`` rebuilds the path-dependent repetition / no-capture counters
that MCTS needs; the Python rules are perft-validated against the Rust engine,
so the reconstruction is exact.

Usage::

    python -m alphazero.serve --ckpt alphazero/checkpoints/best.pt
    python -m alphazero.serve --selftest      # one move from start, then exit
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .net import load_net
from .mcts import Evaluator, run_mcts, select_move
from .xiangqi.game import GameState


def _choose_move(req: dict, ev: Evaluator, cfg: Config) -> dict:
    """Reconstruct the game from the move history and return the AI's move."""
    moves = req.get("moves") or []
    sims = int(req.get("sims", cfg.sims))

    game = GameState()
    for m in moves:
        game.apply((int(m[0]), int(m[1])))

    # Nothing to play: mate/stalemate or a scored draw.
    if game.terminal_value() is not None or not game.legal_moves():
        return {"move": None}

    cfg.sims = max(1, sims)
    ev.clear()  # bound the leaf cache; matches play.py's per-move reset
    _root, visits = run_mcts(game, ev, cfg, add_noise=False)
    mv = select_move(visits, 0.0)
    return {"move": [int(mv[0]), int(mv[1])]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="alphazero/checkpoints/best.pt")
    ap.add_argument("--sims", type=int, default=200,
                    help="default simulations if a request omits 'sims'")
    ap.add_argument("--selftest", action="store_true",
                    help="play one move from the start position, print it, exit")
    args = ap.parse_args()

    cfg = Config()
    cfg.sims = args.sims
    net = load_net(args.ckpt, cfg.device)
    ev = Evaluator(net, cfg)

    if args.selftest:
        out = _choose_move({"moves": [], "sims": args.sims}, ev, cfg)
        print(json.dumps(out), flush=True)
        return

    # Handshake: the first stdout line the Rust client waits for.
    print(json.dumps({"ready": True, "device": cfg.device}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            out = _choose_move(req, ev, cfg)
        except Exception as e:  # never die on a single bad request
            out = {"move": None, "error": repr(e)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
