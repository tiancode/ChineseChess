"""CCA / Asian repetition rules for the Python port.

Mirrors ``src/tests.rs``'s repetition tests so the AlphaZero MCTS scores
perpetual check / chase the same way the Rust game does.

Run directly:   python -m alphazero.tests.test_repetition
Or with pytest: pytest alphazero/tests/test_repetition.py
"""

from __future__ import annotations

from alphazero.xiangqi.board import (
    RED, BLACK, GENERAL, CHARIOT, HORSE, SOLDIER, empty_board, encode_piece, idx,
)
from alphazero.xiangqi.game import GameState
from alphazero.xiangqi.rules import tag_move, TAG_CHECK, TAG_CHASE, TAG_IDLE


def _gs(board, side: int) -> GameState:
    """A GameState over a hand-placed board with the repetition hashes seeded
    to this position (mirrors ``GameState::with_position`` in Rust)."""
    g = GameState.__new__(GameState)
    g.board = board
    g.side_to_move = side
    g.history = []
    g.last_move = None
    g._legal_cache = None
    g.hashes = [g._position_hash()]
    return g


def _put(b, f, r, kind, color):
    b[idx(f, r)] = encode_piece(kind, color)


def test_tag_move_check_chase_idle():
    # Check: a chariot move that gives check.
    b = empty_board()
    _put(b, 0, 9, GENERAL, RED)
    _put(b, 4, 0, GENERAL, BLACK)
    _put(b, 4, 5, CHARIOT, RED)
    assert tag_move(b, (idx(4, 5), idx(4, 1)), RED) == TAG_CHECK

    # Chase: rook swings to attack an undefended horse, no check.
    b = empty_board()
    _put(b, 0, 9, GENERAL, RED)
    _put(b, 8, 0, GENERAL, BLACK)
    _put(b, 0, 4, CHARIOT, RED)
    _put(b, 4, 2, HORSE, BLACK)
    chase = (idx(0, 4), idx(4, 4))
    assert tag_move(b, chase, RED) == TAG_CHASE

    # Defended by a pawn: an unfavourable trade is not a chase -> idle.
    _put(b, 4, 1, SOLDIER, BLACK)
    assert tag_move(b, chase, RED) == TAG_IDLE

    # A pawn doing the threatening is idle (兵之捉算闲).
    b = empty_board()
    _put(b, 0, 9, GENERAL, RED)
    _put(b, 8, 0, GENERAL, BLACK)
    _put(b, 5, 4, SOLDIER, RED)   # crossed-river pawn
    _put(b, 4, 3, HORSE, BLACK)   # undefended
    assert tag_move(b, (idx(5, 4), idx(5, 3)), RED) == TAG_IDLE


def _run_cycle(g: GameState, cycle, times: int = 2):
    for _ in range(times):
        for mv in cycle:
            assert mv in g.legal_moves(), f"cycle move should be legal: {mv}"
            g.apply(mv)


def test_perpetual_check_loses():
    # Red chariot perpetually checks a bare Black king: Red is 长将 and loses,
    # so for the side to move at the repeat (Black) the value is +1.
    b = empty_board()
    _put(b, 3, 9, GENERAL, RED)
    _put(b, 4, 0, GENERAL, BLACK)
    _put(b, 0, 0, CHARIOT, RED)
    g = _gs(b, BLACK)
    _run_cycle(g, [
        (idx(4, 0), idx(4, 1)),  # Black king escapes
        (idx(0, 0), idx(0, 1)),  # Red re-checks on rank 1
        (idx(4, 1), idx(4, 0)),  # Black king back
        (idx(0, 1), idx(0, 0)),  # Red re-checks on rank 0
    ])
    assert g.terminal_value() == 1.0


def test_perpetual_chase_loses():
    # Red chariot perpetually chases an undefended horse (长捉); Red is the
    # side to move at the repeat, so it loses: value -1.
    b = empty_board()
    _put(b, 0, 9, GENERAL, RED)
    _put(b, 4, 0, GENERAL, BLACK)
    _put(b, 4, 9, CHARIOT, RED)
    _put(b, 4, 4, HORSE, BLACK)   # undefended target
    _put(b, 0, 0, HORSE, BLACK)   # idle shuffler
    g = _gs(b, RED)
    _run_cycle(g, [
        (idx(4, 9), idx(4, 8)),
        (idx(0, 0), idx(2, 1)),
        (idx(4, 8), idx(4, 9)),
        (idx(2, 1), idx(0, 0)),
    ])
    assert g.terminal_value() == -1.0


def test_defended_chase_is_a_draw():
    # Same shape, horse defended: not a chase -> both idle -> draw.
    b = empty_board()
    _put(b, 0, 9, GENERAL, RED)
    _put(b, 4, 0, GENERAL, BLACK)
    _put(b, 4, 9, CHARIOT, RED)
    _put(b, 4, 4, HORSE, BLACK)
    _put(b, 4, 3, SOLDIER, BLACK)  # guards the horse
    _put(b, 0, 0, HORSE, BLACK)
    g = _gs(b, RED)
    _run_cycle(g, [
        (idx(4, 9), idx(4, 8)),
        (idx(0, 0), idx(2, 1)),
        (idx(4, 8), idx(4, 9)),
        (idx(2, 1), idx(0, 0)),
    ])
    assert g.terminal_value() == 0.0


def test_pure_repetition_is_a_draw():
    # Both sides shuffle a horse: a plain positional repetition still draws.
    g = GameState()
    cycle = [
        (idx(1, 9), idx(2, 7)),  # Red horse out
        (idx(1, 0), idx(2, 2)),  # Black horse out
        (idx(2, 7), idx(1, 9)),  # Red horse back
        (idx(2, 2), idx(1, 0)),  # Black horse back
    ]
    _run_cycle(g, cycle)
    assert g.is_terminal()
    assert g.terminal_value() == 0.0


if __name__ == "__main__":
    test_tag_move_check_chase_idle()
    test_perpetual_check_loses()
    test_perpetual_chase_loses()
    test_defended_chase_is_a_draw()
    test_pure_repetition_is_a_draw()
    print("CCA repetition rules (Python port) OK")
