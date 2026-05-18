"""Pure-Python Xiangqi rules, ported faithfully from the Rust engine.

Coordinate system (identical to the Rust code, see project CLAUDE.md):
  9 files (x: 0..=8) by 10 ranks (y: 0..=9); index = rank * 9 + file.
  Rank 0 is the top (Black's home); rank 9 is the bottom (Red's home).
  Red moves first and sits at the bottom (forward(Red) == -1).
"""

from .board import (
    RED, BLACK, GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER,
    FILES, RANKS, CELLS, idx, file_of, rank_of, opposite,
    initial_board, encode_piece, decode_piece, find_general,
)
from .rules import pseudo_moves, legal_moves, in_check, generals_face, is_attacked
from .game import GameState

__all__ = [
    "RED", "BLACK", "GENERAL", "ADVISOR", "ELEPHANT", "HORSE", "CHARIOT",
    "CANNON", "SOLDIER", "FILES", "RANKS", "CELLS", "idx", "file_of",
    "rank_of", "opposite", "initial_board", "encode_piece", "decode_piece",
    "find_general", "pseudo_moves", "legal_moves", "in_check",
    "generals_face", "is_attacked", "GameState",
]
