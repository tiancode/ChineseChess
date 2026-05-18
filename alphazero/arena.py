"""Evaluation gate: pit a candidate network against the current best.

Games are deterministic-ish (no Dirichlet noise, temperature 0) so the
result reflects strength rather than exploration. Colours alternate to
cancel first-move advantage.
"""

from __future__ import annotations

from .xiangqi.board import RED, opposite
from .xiangqi.game import GameState
from .mcts import Evaluator, run_mcts, select_move


def _play(ev_red: Evaluator, ev_black: Evaluator, cfg):
    """Return winner: RED, BLACK, or None for a draw."""
    game = GameState()
    ply = 0
    while True:
        tv = game.terminal_value()
        if tv is not None:
            return None if tv == 0.0 else opposite(game.side_to_move)
        if ply >= cfg.max_game_len:
            return None
        ev = ev_red if game.side_to_move == RED else ev_black
        ev.clear()
        _root, visits = run_mcts(game, ev, cfg, add_noise=False)
        game.apply(select_move(visits, 0.0))
        ply += 1


def play_match(cand_net, best_net, cfg, n_games: int):
    """Candidate vs best over ``n_games``. Returns candidate's score rate in
    ``[0, 1]`` (win=1, draw=0.5)."""
    arena_cfg = _ArenaCfg(cfg)
    cand_ev = Evaluator(cand_net, arena_cfg)
    best_ev = Evaluator(best_net, arena_cfg)

    score = 0.0
    for g in range(n_games):
        if g % 2 == 0:  # candidate plays Red
            w = _play(cand_ev, best_ev, arena_cfg)
            cand_is = RED
        else:           # candidate plays Black
            w = _play(best_ev, cand_ev, arena_cfg)
            cand_is = 1
        if w is None:
            score += 0.5
        elif w == cand_is:
            score += 1.0
    return score / n_games


class _ArenaCfg:
    """A thin view of ``Config`` with arena sim count and no noise."""

    def __init__(self, cfg):
        self.__dict__.update(cfg.as_dict())
        self.sims = cfg.arena_sims
