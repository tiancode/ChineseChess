"""Board representation and coordinate helpers.

The board is a flat ``numpy.int8`` array of length 90. Each cell holds a
piece *code*:

    0                      empty
    1 + kind*2 + color     occupied  (kind 0..6, color 0=Red 1=Black)

This mirrors the encoding used by ``game.rs::position_hash`` so a board can
be hashed by its raw bytes for repetition detection.
"""

from __future__ import annotations

import numpy as np

# Colors.
RED = 0
BLACK = 1

# Piece kinds (same order as the Rust ``PieceKind`` enum).
GENERAL = 0
ADVISOR = 1
ELEPHANT = 2
HORSE = 3
CHARIOT = 4
CANNON = 5
SOLDIER = 6

FILES = 9
RANKS = 10
CELLS = FILES * RANKS  # 90


def opposite(color: int) -> int:
    return color ^ 1


def idx(file: int, rank: int) -> int:
    return rank * FILES + file


def file_of(i: int) -> int:
    return i % FILES


def rank_of(i: int) -> int:
    return i // FILES


def on_board(file: int, rank: int) -> bool:
    return 0 <= file < FILES and 0 <= rank < RANKS


def in_palace(color: int, file: int, rank: int) -> bool:
    if not (3 <= file <= 5):
        return False
    if color == BLACK:
        return 0 <= rank <= 2
    return 7 <= rank <= 9  # RED


def crossed_river(color: int, rank: int) -> bool:
    return rank <= 4 if color == RED else rank >= 5


def own_half(color: int, rank: int) -> bool:
    return rank >= 5 if color == RED else rank <= 4


def forward(color: int) -> int:
    """Forward rank direction for a soldier (Red advances toward rank 0)."""
    return -1 if color == RED else 1


def encode_piece(kind: int, color: int) -> int:
    return 1 + kind * 2 + color


def decode_piece(code: int):
    """Return ``(kind, color)`` for a non-empty code, else ``None``."""
    if code == 0:
        return None
    c = code - 1
    return (c >> 1, c & 1)


def piece_color(code: int) -> int:
    return (code - 1) & 1


def piece_kind(code: int) -> int:
    return (code - 1) >> 1


def empty_board() -> np.ndarray:
    return np.zeros(CELLS, dtype=np.int8)


def initial_board() -> np.ndarray:
    """Standard opening position (Black on ranks 0..3, Red on ranks 6..9)."""
    b = empty_board()
    back = [CHARIOT, HORSE, ELEPHANT, ADVISOR, GENERAL,
            ADVISOR, ELEPHANT, HORSE, CHARIOT]

    for f, kind in enumerate(back):
        b[idx(f, 0)] = encode_piece(kind, BLACK)
    b[idx(1, 2)] = encode_piece(CANNON, BLACK)
    b[idx(7, 2)] = encode_piece(CANNON, BLACK)
    for f in range(0, 9, 2):
        b[idx(f, 3)] = encode_piece(SOLDIER, BLACK)

    for f, kind in enumerate(back):
        b[idx(f, 9)] = encode_piece(kind, RED)
    b[idx(1, 7)] = encode_piece(CANNON, RED)
    b[idx(7, 7)] = encode_piece(CANNON, RED)
    for f in range(0, 9, 2):
        b[idx(f, 6)] = encode_piece(SOLDIER, RED)

    return b


def find_general(board: np.ndarray, color: int) -> int:
    """Cell index of ``color``'s general, or -1 if it is gone."""
    target = encode_piece(GENERAL, color)
    hits = np.nonzero(board == target)[0]
    return int(hits[0]) if hits.size else -1
