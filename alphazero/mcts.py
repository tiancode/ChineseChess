"""PUCT Monte-Carlo Tree Search guided by the network.

A simulation descends the tree while applying moves to one cloned
``GameState`` (``apply`` / ``undo``), so repetition and no-capture counters
stay correct inside the search -- the same do/undo discipline the Rust
searcher uses. Leaves are evaluated by the network instead of rollouts;
terminal nodes use the true Xiangqi result.

Values are negamax: ``simulate`` returns the value from the perspective of
the side to move at that node, and the caller negates across the edge.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from .encoding import encode_state, move_to_index


class _Node:
    __slots__ = ("P", "N", "W", "Q", "children", "expanded", "terminal", "tv")

    def __init__(self, prior: float = 0.0):
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = None      # dict: move -> _Node
        self.expanded = False
        self.terminal = False
        self.tv = 0.0             # cached terminal value (valid iff terminal)


class Evaluator:
    """Wraps a network with a per-search leaf cache (positions repeat a lot
    inside a single tree, especially in repetition-prone Xiangqi lines)."""

    def __init__(self, net, cfg):
        self.net = net
        self.cfg = cfg
        self.device = cfg.device
        self.cache: dict = {}

    def clear(self):
        self.cache.clear()

    def evaluate(self, game):
        """Return ``(priors, value)`` where ``priors`` is a list of
        ``(move, prob)`` over the legal moves and ``value`` is in ``[-1, 1]``
        from the side-to-move's perspective."""
        # The key must include the path-dependent scalar planes (repetition
        # and no-progress counts, see encoding.py): the same board+side can
        # recur with different draw-proximity, which the network sees.
        key = (
            game.board.tobytes(),
            game.side_to_move,
            min(game._repetition_count(), 3),
            min(game._plies_since_capture(), 120),
        )
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        moves = game.legal_moves()
        planes = encode_state(game)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        use_amp = self.cfg.amp and self.device == "cuda"
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    logits, value = self.net(x)
            else:
                logits, value = self.net(x)
        logits = logits[0].float().cpu().numpy()
        value = float(value[0])

        side = game.side_to_move
        idxs = np.fromiter((move_to_index(side, m) for m in moves),
                           dtype=np.int64, count=len(moves))
        ml = logits[idxs]
        ml -= ml.max()
        pr = np.exp(ml)
        pr /= pr.sum()
        priors = list(zip(moves, pr.tolist()))

        self.cache[key] = (priors, value)
        return priors, value


def _select(node: _Node, c_puct: float):
    """PUCT child selection: argmax Q + c_puct * P * sqrt(N_parent)/(1+N)."""
    sqrt_n = math.sqrt(max(node.N, 1))
    best, best_mv, best_child = -1e30, None, None
    for mv, ch in node.children.items():
        u = c_puct * ch.P * sqrt_n / (1 + ch.N)
        score = ch.Q + u
        if score > best:
            best, best_mv, best_child = score, mv, ch
    return best_mv, best_child


def _expand(node: _Node, priors):
    node.children = {mv: _Node(p) for mv, p in priors}
    node.expanded = True


def _simulate(node: _Node, game, ev: Evaluator, c_puct: float) -> float:
    if node.terminal:
        return node.tv  # path-fixed for this node, computed once
    if not node.expanded:
        tv = game.terminal_value()
        if tv is not None:
            node.terminal = True
            node.tv = tv
            return tv
        priors, value = ev.evaluate(game)
        _expand(node, priors)
        return value

    mv, child = _select(node, c_puct)
    game.apply(mv)
    v = -_simulate(child, game, ev, c_puct)
    game.undo()

    child.N += 1
    child.W += v
    child.Q = child.W / child.N
    node.N += 1
    return v


def run_mcts(game, ev: Evaluator, cfg, *, add_noise: bool = True):
    """Run ``cfg.sims`` simulations from ``game`` (not mutated).

    Returns ``(root, visit_counts)`` where ``visit_counts`` is a dict
    ``move -> N`` over the root's legal moves.
    """
    root = _Node()
    work = game.clone()

    priors, _ = ev.evaluate(work)
    assert priors, "run_mcts called on a terminal position (no legal moves)"
    if add_noise and len(priors) > 0:
        noise = np.random.dirichlet([cfg.dirichlet_alpha] * len(priors))
        eps = cfg.dirichlet_eps
        priors = [(mv, (1 - eps) * p + eps * float(n))
                  for (mv, p), n in zip(priors, noise)]
    _expand(root, priors)

    for _ in range(cfg.sims):
        _simulate(root, work, ev, cfg.c_puct)

    visits = {mv: ch.N for mv, ch in root.children.items()}
    return root, visits


def select_move(visits: dict, temperature: float):
    """Pick a move from the root visit counts.

    ``temperature == 0`` -> the most-visited move (deterministic).
    Otherwise sample with probabilities proportional to ``N**(1/temp)``.
    """
    moves = list(visits.keys())
    counts = np.array([visits[m] for m in moves], dtype=np.float64)
    if temperature <= 1e-6 or counts.sum() == 0:
        return moves[int(counts.argmax())]
    logits = np.log(np.maximum(counts, 1e-9)) / temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return moves[int(np.random.choice(len(moves), p=probs))]
