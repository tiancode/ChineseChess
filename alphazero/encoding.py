"""Board <-> tensor encoding and move <-> policy-index mapping.

Everything is **canonicalised to the side to move**: when Black is to move
the board is rotated 180 degrees (cell ``i`` -> ``89 - i``) and colours are
swapped, so the network always sees the mover sitting at the bottom playing
"up". The value head therefore always predicts from the mover's perspective,
and no explicit colour plane is needed.

Plane layout (16 channels, each 10x9):
    0..6   mover's pieces            (General..Soldier)
    7..13  opponent's pieces
    14     repetition scalar  = min(rep,3)/3   (broadcast)
    15     no-progress scalar = plies_since_capture / 120 (broadcast)

Action space: flat ``from*90 + to`` in canonical coordinates (8100). It is
sparse-but-simple and unambiguous for Xiangqi (no promotion / castling); MCTS
and training always mask to the legal subset.
"""

from __future__ import annotations

import numpy as np

from .xiangqi.board import RED, RANKS, FILES, CELLS, decode_piece
from .xiangqi.game import NO_CAPTURE_PLY_LIMIT

N_PLANES = 16
POLICY_SIZE = CELLS * CELLS  # 90 * 90 = 8100


def _flip_index(i: int) -> int:
    """180-degree board rotation: (file,rank) -> (8-file, 9-rank)."""
    return CELLS - 1 - i


def encode_state(game) -> np.ndarray:
    """Canonical ``(16, 10, 9)`` float32 planes for ``game``."""
    planes = np.zeros((N_PLANES, RANKS, FILES), dtype=np.float32)
    side = game.side_to_move
    board = game.board

    occ = np.nonzero(board)[0]
    for i in occ:
        i = int(i)
        kind, color = decode_piece(int(board[i]))
        ci = i if side == RED else _flip_index(i)
        channel = kind if color == side else 7 + kind
        planes[channel, ci // FILES, ci % FILES] = 1.0

    rep = min(game._repetition_count(), 3) / 3.0
    nocap = game._plies_since_capture() / float(NO_CAPTURE_PLY_LIMIT)
    planes[14, :, :] = rep
    planes[15, :, :] = nocap
    return planes


def move_to_index(side: int, mv) -> int:
    """Canonical policy index for a real-coordinate move of ``side``."""
    frm, to = mv
    if side != RED:
        frm = _flip_index(frm)
        to = _flip_index(to)
    return frm * CELLS + to


def index_to_move(side: int, index: int):
    """Inverse of :func:`move_to_index`."""
    cf, ct = divmod(index, CELLS)
    if side != RED:
        cf = _flip_index(cf)
        ct = _flip_index(ct)
    return (cf, ct)


def legal_index_array(game) -> tuple:
    """``(moves, indices)`` -- the legal moves and their canonical policy
    indices, in a fixed order, for the current side to move."""
    side = game.side_to_move
    moves = game.legal_moves()
    idxs = np.fromiter((move_to_index(side, m) for m in moves),
                       dtype=np.int64, count=len(moves))
    return moves, idxs
