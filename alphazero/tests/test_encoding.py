"""Encoding invariants: the move <-> policy-index map must round-trip for
both sides, and the canonical planes must be self-consistent.

Run: python -m alphazero.tests.test_encoding
"""

from __future__ import annotations

import numpy as np

from alphazero.xiangqi.board import RED, BLACK
from alphazero.xiangqi.game import GameState
from alphazero.encoding import (
    encode_state, move_to_index, index_to_move, N_PLANES, POLICY_SIZE,
)


def test_move_index_roundtrip_both_sides():
    for side in (RED, BLACK):
        for frm in range(90):
            for to in range(90):
                i = move_to_index(side, (frm, to))
                assert 0 <= i < POLICY_SIZE
                assert index_to_move(side, i) == (frm, to)


def test_legal_moves_map_into_range():
    g = GameState()
    for mv in g.legal_moves():
        i = move_to_index(g.side_to_move, mv)
        assert 0 <= i < POLICY_SIZE
        assert index_to_move(g.side_to_move, i) == mv


def test_planes_shape_and_canonical_self_count():
    g = GameState()
    p = encode_state(g)
    assert p.shape == (N_PLANES, 10, 9)
    # Opening: 16 pieces a side. Channels 0..6 are the mover's pieces.
    assert p[0:7].sum() == 16
    assert p[7:14].sum() == 16

    # After a Red move it is Black to move; canonicalisation must still put
    # the mover's 16 pieces in the "self" channels.
    g.apply(g.legal_moves()[0])
    p = encode_state(g)
    assert g.side_to_move == BLACK
    assert p[0:7].sum() == 16 and p[7:14].sum() == 16


if __name__ == "__main__":
    test_move_index_roundtrip_both_sides()
    print("move<->index round-trip (both sides): OK")
    test_legal_moves_map_into_range()
    print("legal moves map into [0,8100): OK")
    test_planes_shape_and_canonical_self_count()
    print("canonical planes shape + self/opp counts: OK")
    print("\nEncoding invariants validated.")
