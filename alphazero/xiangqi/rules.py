"""Move generation and rule enforcement.

A move is a ``(from_index, to_index)`` tuple. ``pseudo_moves`` doubles as the
attack generator (``is_attacked`` / ``in_check`` run it for the opponent),
exactly as in ``moves.rs``. ``legal_moves`` filters pseudo-moves by make ->
self-check / flying-general check -> unmake.
"""

from __future__ import annotations

import numpy as np

from .board import (
    RED, GENERAL, FILES, RANKS, CELLS, idx, file_of, rank_of, opposite,
    on_board, in_palace, crossed_river, own_half, forward, piece_color,
    piece_kind, encode_piece, find_general,
)
from .board import GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER


def make(board: np.ndarray, mv) -> int:
    """Apply ``mv``, returning the captured piece code (0 if none)."""
    frm, to = mv
    captured = int(board[to])
    board[to] = board[frm]
    board[frm] = 0
    return captured


def unmake(board: np.ndarray, mv, captured: int) -> None:
    frm, to = mv
    board[frm] = board[to]
    board[to] = captured


def _target_ok(board, color, f, r):
    """None=off-board/own piece (blocked); True=empty; False=enemy."""
    if not on_board(f, r):
        return None
    code = int(board[idx(f, r)])
    if code == 0:
        return True
    if piece_color(code) != color:
        return False
    return None


def _push_if(board, color, frm, f, r, out):
    if _target_ok(board, color, f, r) is not None:
        out.append((frm, idx(f, r)))


def _gen_general(board, color, f, r, frm, out):
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nf, nr = f + df, r + dr
        if in_palace(color, nf, nr):
            _push_if(board, color, frm, nf, nr, out)


def _gen_advisor(board, color, f, r, frm, out):
    for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        nf, nr = f + df, r + dr
        if in_palace(color, nf, nr):
            _push_if(board, color, frm, nf, nr, out)


def _gen_elephant(board, color, f, r, frm, out):
    for df, dr in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
        nf, nr = f + df, r + dr
        if not on_board(nf, nr) or not own_half(color, nr):
            continue
        # "Blocking the elephant's eye": the midpoint must be empty.
        if board[idx(f + df // 2, r + dr // 2)] != 0:
            continue
        _push_if(board, color, frm, nf, nr, out)


_HORSE_MOVES = (
    (1, 0, 2, 1), (1, 0, 2, -1), (-1, 0, -2, 1), (-1, 0, -2, -1),
    (0, 1, 1, 2), (0, 1, -1, 2), (0, -1, 1, -2), (0, -1, -1, -2),
)


def _gen_horse(board, color, f, r, frm, out):
    for lf, lr, tf, tr in _HORSE_MOVES:
        if not on_board(f + lf, r + lr) or board[idx(f + lf, r + lr)] != 0:
            continue  # "Hobbling the horse's leg".
        _push_if(board, color, frm, f + tf, r + tr, out)


def _slide(board, color, f, r, frm, out):
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nf, nr = f + df, r + dr
        while on_board(nf, nr):
            code = int(board[idx(nf, nr)])
            if code == 0:
                out.append((frm, idx(nf, nr)))
            else:
                if piece_color(code) != color:
                    out.append((frm, idx(nf, nr)))
                break
            nf += df
            nr += dr


def _gen_cannon(board, color, f, r, frm, out):
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nf, nr = f + df, r + dr
        # Phase 1: quiet moves over empty squares up to the screen.
        while on_board(nf, nr) and board[idx(nf, nr)] == 0:
            out.append((frm, idx(nf, nr)))
            nf += df
            nr += dr
        # Phase 2: hop the screen, capture the first enemy beyond it.
        nf += df
        nr += dr
        while on_board(nf, nr):
            code = int(board[idx(nf, nr)])
            if code != 0:
                if piece_color(code) != color:
                    out.append((frm, idx(nf, nr)))
                break
            nf += df
            nr += dr


def _gen_soldier(board, color, f, r, frm, out):
    fwd = forward(color)
    _push_if(board, color, frm, f, r + fwd, out)
    if crossed_river(color, r):
        _push_if(board, color, frm, f + 1, r, out)
        _push_if(board, color, frm, f - 1, r, out)


_DISPATCH = {
    GENERAL: _gen_general,
    ADVISOR: _gen_advisor,
    ELEPHANT: _gen_elephant,
    HORSE: _gen_horse,
    CHARIOT: _slide,
    CANNON: _gen_cannon,
    SOLDIER: _gen_soldier,
}


def pseudo_moves(board: np.ndarray, color: int) -> list:
    """All pseudo-legal moves for ``color`` (no self-check / flying-general
    filtering). Also the attack generator for check detection."""
    out: list = []
    occ = np.nonzero(board)[0]
    for i in occ:
        i = int(i)
        code = int(board[i])
        if piece_color(code) != color:
            continue
        _DISPATCH[piece_kind(code)](board, color, file_of(i), rank_of(i), i, out)
    return out


def generals_face(board: np.ndarray) -> bool:
    """True if the two generals share a clear file (the illegal
    'flying general' position)."""
    rg = find_general(board, RED)
    bg = find_general(board, 1)
    if rg < 0 or bg < 0:
        return False
    if file_of(rg) != file_of(bg):
        return False
    f = file_of(rg)
    lo = min(rank_of(rg), rank_of(bg))
    hi = max(rank_of(rg), rank_of(bg))
    for r in range(lo + 1, hi):
        if board[idx(f, r)] != 0:
            return False
    return True


def is_attacked(board: np.ndarray, sq: int, by: int) -> bool:
    return any(m[1] == sq for m in pseudo_moves(board, by))


def in_check(board: np.ndarray, color: int) -> bool:
    g = find_general(board, color)
    if g < 0:
        return True  # general already captured
    return is_attacked(board, g, opposite(color))


def legal_moves(board: np.ndarray, color: int) -> list:
    """Pseudo-moves that leave neither the flying-general rule exposed nor the
    mover's own general in check."""
    legal = []
    for mv in pseudo_moves(board, color):
        captured = make(board, mv)
        if not in_check(board, color) and not generals_face(board):
            legal.append(mv)
        unmake(board, mv, captured)
    return legal
