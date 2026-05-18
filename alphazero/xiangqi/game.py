"""Game state: turn tracking, reversible apply/undo, repetition + no-capture
detection, and Xiangqi terminal scoring.

Terminal rule (identical to ``game.rs``): a side with no legal move *loses*
whether it is checkmated or stalemated. Threefold repetition is judged by the
CCA / Asian rules (see ``_repetition_judgment``, a mirror of
``game.rs::repetition_judgment``): perpetual check / chase is a loss for the
offending side; only a mutual or idle repetition draws. The no-capture limit
is still a draw. Documented approximations match the Rust side (chase = a
1-ply static-exchange test; pawn/general as the chaser is idle; ambiguous
sub-cases follow the common CCA interpretation).
"""

from __future__ import annotations

import numpy as np

from .board import RED, BLACK, CELLS, initial_board
from .rules import (
    legal_moves, in_check, make, unmake,
    tag_move, TAG_CHECK, TAG_CHASE,
)

# Plies without a capture after which the game is a draw (60 full moves).
NO_CAPTURE_PLY_LIMIT = 120


class GameState:
    __slots__ = ("board", "side_to_move", "history", "hashes",
                 "last_move", "_legal_cache")

    def __init__(self):
        self.board = initial_board()
        self.side_to_move = RED
        # (move, captured_code) pairs, oldest first, for undo.
        self.history: list = []
        # Position hashes (board bytes + side), including the start.
        self.hashes: list = [self._position_hash()]
        self.last_move = None
        self._legal_cache = None

    # -- copying -----------------------------------------------------------
    def clone(self) -> "GameState":
        g = GameState.__new__(GameState)
        g.board = self.board.copy()
        g.side_to_move = self.side_to_move
        g.history = list(self.history)
        g.hashes = list(self.hashes)
        g.last_move = self.last_move
        g._legal_cache = None
        return g

    # -- hashing -----------------------------------------------------------
    def _position_hash(self) -> int:
        # Board contents + side to move. Distinct from the engine Zobrist key
        # in the Rust code; used only for repetition detection here.
        return hash((self.board.tobytes(), self.side_to_move))

    # -- moves -------------------------------------------------------------
    def legal_moves(self) -> list:
        if self._legal_cache is None:
            self._legal_cache = legal_moves(self.board, self.side_to_move)
        return self._legal_cache

    def apply(self, mv) -> None:
        captured = make(self.board, mv)
        self.history.append((mv, captured))
        self.last_move = mv
        self.side_to_move ^= 1
        self.hashes.append(self._position_hash())
        self._legal_cache = None

    def undo(self) -> bool:
        if not self.history:
            return False
        mv, captured = self.history.pop()
        unmake(self.board, mv, captured)
        self.hashes.pop()
        self.side_to_move ^= 1
        self.last_move = self.history[-1][0] if self.history else None
        self._legal_cache = None
        return True

    # -- end-game ----------------------------------------------------------
    def _plies_since_capture(self) -> int:
        n = 0
        for _mv, cap in reversed(self.history):
            if cap != 0:
                break
            n += 1
        return n

    def _repetition_count(self) -> int:
        h = self.hashes[-1]
        return self.hashes.count(h)

    def _repetition_judgment(self) -> float:
        """CCA / Asian ruling for the just-repeated position; a mirror of
        ``game.rs::repetition_judgment``. Returns the value *for the side to
        move*: ``-1.0`` if it is the offending (perpetual check/chase) side,
        ``+1.0`` if the opponent is, ``0.0`` for a mutual or plain idle
        repetition.
        """
        last = len(self.history)            # hashes[last] == current position
        cur = self.hashes[last]
        j = -1
        for k in range(last - 1, -1, -1):
            if self.hashes[k] == cur:
                j = k
                break
        if j < 0:
            return 0.0

        # Rewind a copy to the cycle's start, then replay it tagging moves.
        b = self.board.copy()
        for mv, cap in reversed(self.history[j:last]):
            unmake(b, mv, cap)

        # agg[color] = [moves, checks, aggressive, chases]
        agg = [[0, 0, 0, 0], [0, 0, 0, 0]]
        for i in range(j, last):
            # Mover of move i, derived from the current side to move.
            mover = self.side_to_move if (last - i) % 2 == 0 else (self.side_to_move ^ 1)
            mv = self.history[i][0]
            a = agg[mover]
            a[0] += 1
            tag = tag_move(b, mv, mover)
            if tag == TAG_CHECK:
                a[1] += 1
                a[2] += 1
            elif tag == TAG_CHASE:
                a[2] += 1
                a[3] += 1
            make(b, mv)

        def level(a):
            if a[0] > 0 and a[1] == a[0]:
                return 2  # 长将: every move a check
            if a[0] > 0 and a[2] == a[0] and a[3] > 0:
                return 1  # 长捉: every move aggressive, at least one chase
            return 0

        rl, bl = level(agg[RED]), level(agg[BLACK])
        if (rl, bl) in ((0, 0), (2, 2), (1, 1)):
            loser = None
        elif bl == 0:
            loser = RED
        elif rl == 0:
            loser = BLACK
        elif (rl, bl) == (2, 1):       # 一将一捉: perpetual-check side loses
            loser = RED
        elif (rl, bl) == (1, 2):
            loser = BLACK
        else:
            loser = None
        if loser is None:
            return 0.0
        return -1.0 if loser == self.side_to_move else 1.0

    def terminal_value(self):
        """Value for the side to move if the game is over, else ``None``.

        A side with no legal reply loses (-1). A threefold repetition is ruled
        by ``_repetition_judgment`` (CCA): perpetual check/chase loses for the
        offender, so this may return -1.0, 0.0 *or* +1.0 (the victim of a
        perpetual already wins here). The no-capture limit is a draw (0).
        """
        if not self.legal_moves():
            return -1.0  # no reply: side to move loses (mate or stalemate)
        if self._repetition_count() >= 3:
            return self._repetition_judgment()
        if self._plies_since_capture() >= NO_CAPTURE_PLY_LIMIT:
            return 0.0
        return None

    def is_terminal(self) -> bool:
        return self.terminal_value() is not None

    def in_check(self) -> bool:
        return in_check(self.board, self.side_to_move)
