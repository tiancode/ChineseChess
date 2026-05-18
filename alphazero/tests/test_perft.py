"""Validate the Python rules port against the Rust engine's known values.

These are the same widely-published Xiangqi perft numbers asserted by
``src/tests.rs::perft_matches_known_values``. If the port is correct these
must match exactly.

Run directly:   python -m alphazero.tests.test_perft
Or with pytest: pytest alphazero/tests/test_perft.py
"""

from __future__ import annotations

import time

from alphazero.xiangqi.board import RED, initial_board, opposite
from alphazero.xiangqi.rules import legal_moves, make, unmake


def perft(board, side: int, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for mv in legal_moves(board, side):
        cap = make(board, mv)
        nodes += perft(board, opposite(side), depth - 1)
        unmake(board, mv, cap)
    return nodes


KNOWN = {1: 44, 2: 1_920, 3: 79_666}


def test_opening_move_count():
    b = initial_board()
    assert len(legal_moves(b, RED)) == 44


def test_perft_matches_known_values():
    b = initial_board()
    for depth, expected in KNOWN.items():
        assert perft(b, RED, depth) == expected, f"perft depth {depth}"


if __name__ == "__main__":
    b = initial_board()
    assert len(legal_moves(b, RED)) == 44, "opening move count"
    print("opening move count: 44  OK")
    for depth, expected in KNOWN.items():
        t = time.time()
        got = perft(initial_board(), RED, depth)
        dt = time.time() - t
        status = "OK" if got == expected else "FAIL"
        print(f"perft({depth}) = {got:>7}  (expected {expected:>7})  "
              f"{dt:6.2f}s  {status}")
        assert got == expected
    print("\nRules port validated against the Rust engine.")
