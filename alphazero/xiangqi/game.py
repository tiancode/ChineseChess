"""Game state: turn tracking, reversible apply/undo, repetition + no-capture
detection, and Xiangqi terminal scoring.

Terminal rule (identical to ``game.rs``): a side with no legal move *loses*
whether it is checkmated or stalemated. Threefold repetition and the
no-capture limit are scored as draws (the same deliberate simplification the
Rust engine makes -- perpetual check/chase is not specially punished).
"""

from __future__ import annotations

import numpy as np

from .board import RED, CELLS, initial_board
from .rules import legal_moves, in_check, make, unmake

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

    def terminal_value(self):
        """Value for the side to move if the game is over, else ``None``.

        A side with no legal reply loses (-1). Repetition / no-capture are
        draws (0). The winning side's +1 is produced by negamax backup in the
        searcher, so a positive terminal value never needs representing here.
        """
        if not self.legal_moves():
            return -1.0  # no reply: side to move loses (mate or stalemate)
        if self._repetition_count() >= 3:
            return 0.0
        if self._plies_since_capture() >= NO_CAPTURE_PLY_LIMIT:
            return 0.0
        return None

    def is_terminal(self) -> bool:
        return self.terminal_value() is not None

    def in_check(self) -> bool:
        return in_check(self.board, self.side_to_move)
