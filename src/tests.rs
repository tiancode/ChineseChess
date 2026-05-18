//! Rule-engine unit tests.
//!
//! Note: in any hand-built position the two generals must NOT sit on the same
//! file with a clear path between them — that is the (illegal) flying-general
//! position and the engine will correctly reject every move from it.

use crate::ai::search::SearchEngine;
use crate::ai::{make_engine, Engine, EngineKind};
use crate::board::*;
use crate::game::{DrawReason, GameState, GameStatus};
use crate::moves::{generals_face, in_check, legal_moves};

fn place(b: &mut Board, f: i32, r: i32, kind: PieceKind, color: Color) {
    b.cells[idx(f, r)] = Some(Piece { kind, color });
}

/// Bare-king skeleton with the generals on different files (no accidental
/// flying-general), so a test can add just the pieces it cares about.
fn skeleton() -> Board {
    let mut b = Board::empty();
    place(&mut b, 3, 9, PieceKind::General, Color::Red);
    place(&mut b, 5, 0, PieceKind::General, Color::Black);
    b
}

/// perft: count leaf nodes of the legal-move tree to `depth`.
fn perft(board: &mut Board, side: Color, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0;
    for mv in legal_moves(board, side) {
        let cap = board.make(mv);
        nodes += perft(board, side.opposite(), depth - 1);
        board.unmake(mv, cap);
    }
    nodes
}

#[test]
fn opening_move_count() {
    let mut b = Board::initial();
    assert_eq!(legal_moves(&mut b, Color::Red).len(), 44);
}

#[test]
fn perft_matches_known_values() {
    // Widely published Xiangqi perft from the initial position.
    let mut b = Board::initial();
    assert_eq!(perft(&mut b, Color::Red, 1), 44);
    assert_eq!(perft(&mut b, Color::Red, 2), 1_920);
    assert_eq!(perft(&mut b, Color::Red, 3), 79_666);
}

#[test]
fn cannon_needs_screen_to_capture() {
    let mut b = skeleton();
    place(&mut b, 0, 5, PieceKind::Cannon, Color::Red);
    place(&mut b, 0, 1, PieceKind::Soldier, Color::Black); // would-be victim

    // No screen between the cannon and the soldier: capture is illegal.
    let ms = legal_moves(&mut b, Color::Red);
    assert!(!ms.iter().any(|m| m.from == idx(0, 5) && m.to == idx(0, 1)));

    // Add a screen; the jump-capture becomes legal.
    place(&mut b, 0, 3, PieceKind::Soldier, Color::Red);
    let ms = legal_moves(&mut b, Color::Red);
    assert!(ms.iter().any(|m| m.from == idx(0, 5) && m.to == idx(0, 1)));
}

#[test]
fn horse_leg_is_blocked() {
    let mut b = skeleton();
    place(&mut b, 4, 4, PieceKind::Horse, Color::Red);
    place(&mut b, 4, 3, PieceKind::Soldier, Color::Red); // blocks the upward leg

    let ms = legal_moves(&mut b, Color::Red);
    assert!(!ms.iter().any(|m| m.from == idx(4, 4) && m.to == idx(3, 2)));
    assert!(!ms.iter().any(|m| m.from == idx(4, 4) && m.to == idx(5, 2)));
    // A sideways jump is unaffected.
    assert!(ms.iter().any(|m| m.from == idx(4, 4) && m.to == idx(2, 3)));
}

#[test]
fn elephant_cannot_cross_river() {
    let mut b = skeleton();
    place(&mut b, 2, 5, PieceKind::Elephant, Color::Red);

    let ms = legal_moves(&mut b, Color::Red);
    assert!(!ms.iter().any(|m| m.to == idx(4, 3))); // would cross the river
    assert!(ms.iter().any(|m| m.from == idx(2, 5) && m.to == idx(4, 7)));
}

#[test]
fn flying_general_is_illegal() {
    let mut b = Board::empty();
    place(&mut b, 4, 9, PieceKind::General, Color::Red);
    place(&mut b, 4, 2, PieceKind::General, Color::Black);
    assert!(generals_face(&b)); // clear file 4 between generals

    // A Red chariot on file 4 currently shields the generals.
    place(&mut b, 4, 5, PieceKind::Chariot, Color::Red);
    assert!(!generals_face(&b));
    let ms = legal_moves(&mut b, Color::Red);
    // It may not step off file 4 (that re-exposes the generals)...
    assert!(!ms.iter().any(|m| m.from == idx(4, 5) && rank_of(m.to) == 5));
    // ...but sliding along file 4 is fine.
    assert!(ms.iter().any(|m| m.from == idx(4, 5) && m.to == idx(4, 6)));
}

#[test]
fn detects_checkmate() {
    // Classic two-chariot mate against a bare Black general.
    let mut g = GameState::new();
    g.board = Board::empty();
    place(&mut g.board, 4, 0, PieceKind::General, Color::Black);
    place(&mut g.board, 0, 9, PieceKind::General, Color::Red); // off to the side
    place(&mut g.board, 0, 0, PieceKind::Chariot, Color::Red); // checks along rank 0
    place(&mut g.board, 5, 1, PieceKind::Chariot, Color::Red); // covers (4,1) and (5,0)
    g.side_to_move = Color::Black;

    assert!(in_check(&g.board, Color::Black));
    assert_eq!(g.status(), GameStatus::Win(Color::Red));
}

#[test]
fn detects_check_but_not_mate() {
    // Same idea, but the escape square (4,1) is not covered: only check.
    let mut g = GameState::new();
    g.board = Board::empty();
    place(&mut g.board, 4, 0, PieceKind::General, Color::Black);
    place(&mut g.board, 0, 9, PieceKind::General, Color::Red);
    place(&mut g.board, 0, 0, PieceKind::Chariot, Color::Red);
    g.side_to_move = Color::Black;

    assert!(in_check(&g.board, Color::Black));
    assert_eq!(g.status(), GameStatus::Check(Color::Black));
}

#[test]
fn threefold_repetition_is_a_draw() {
    // Both sides shuffle a knight back and forth. After two full cycles the
    // start position has occurred three times -> draw.
    let mut g = GameState::new();
    let cycle = [
        Move { from: idx(1, 9), to: idx(2, 7) }, // Red horse out
        Move { from: idx(1, 0), to: idx(2, 2) }, // Black horse out
        Move { from: idx(2, 7), to: idx(1, 9) }, // Red horse back
        Move { from: idx(2, 2), to: idx(1, 0) }, // Black horse back
    ];
    for _ in 0..2 {
        for mv in cycle {
            assert!(g.is_legal(mv), "shuffle move should be legal: {mv:?}");
            g.apply(mv);
        }
    }
    assert_eq!(g.status(), GameStatus::Draw(DrawReason::Repetition));
}

#[test]
fn ai_returns_a_legal_move() {
    let mut g = GameState::new();
    let mut engine = make_engine(EngineKind::AlphaBeta, 2, Color::Red);
    let mv = engine.best_move(&g).expect("engine should find a move");
    assert!(g.is_legal(mv));
}

/// End-to-end check of the Python AlphaZero sidecar (`alphazero/serve.py`)
/// through the Rust subprocess client. Needs Python + torch + the checkpoint,
/// so it is not run by default:
/// `cargo test alphazero_sidecar -- --ignored --nocapture`
#[test]
#[ignore]
fn alphazero_sidecar_returns_legal_move() {
    use crate::ai::alphazero::AlphaZeroEngine;
    let mut g = GameState::new();
    let mut e = AlphaZeroEngine::new(16); // few sims -> fast
    let mv = e
        .best_move(&g)
        .expect("sidecar should return a move from the start position");
    assert!(g.is_legal(mv), "sidecar move must be legal: {mv:?}");
    g.apply(mv);
    // A second call exercises history replay (Black to move now).
    let mv2 = e.best_move(&g).expect("sidecar should return a reply");
    assert!(g.is_legal(mv2), "second sidecar move must be legal: {mv2:?}");
}

/// Not run by default. `cargo test engine_benchmark -- --ignored --nocapture`
#[test]
#[ignore]
fn engine_benchmark() {
    use crate::ai::search::SearchEngine;
    use std::time::{Duration, Instant};

    // Mirrors make_engine()'s difficulty -> (max depth, budget ms) mapping.
    let levels: [(u8, u8, u64); 5] = [
        (1, 4, 200),
        (2, 6, 800),
        (3, 10, 3_000),
        (4, 18, 15_000),
        (5, 28, 90_000),
    ];

    let opening = GameState::new();
    let mut midgame = GameState::new();
    for _ in 0..10 {
        let m = midgame.legal_moves()[0];
        midgame.apply(m);
    }

    for (label, pos) in [("opening", &opening), ("midgame(10 plies)", &midgame)] {
        println!("\n=== {label} ===");
        for (diff, depth, ms) in levels {
            let mut e = SearchEngine::new(depth, Duration::from_millis(ms));
            let t = Instant::now();
            let mv = e.best_move(pos);
            let dt = t.elapsed();
            let n = e.nodes_searched();
            println!(
                "diff {diff} (depth<= {depth:>2}, {ms:>5}ms): {:>6} ms, {:>9} nodes, {:>6.0} knps, mv={:?}",
                dt.as_millis(),
                n,
                n as f64 / dt.as_secs_f64() / 1000.0,
                mv
            );
        }
    }
}

#[test]
fn engine_finds_mate_in_one() {
    // Red to move; Cd-e1 style: sliding the chariot to (5,1) mates the bare
    // Black general (covered by the rank-0 chariot + this one).
    let mut g = GameState::new();
    g.board = Board::empty();
    place(&mut g.board, 4, 0, PieceKind::General, Color::Black);
    place(&mut g.board, 0, 9, PieceKind::General, Color::Red);
    place(&mut g.board, 0, 0, PieceKind::Chariot, Color::Red); // checks along rank 0
    place(&mut g.board, 8, 1, PieceKind::Chariot, Color::Red); // one step from the mate
    g.side_to_move = Color::Red;

    let mut engine = SearchEngine::fixed_depth(3);
    let mv = engine.best_move(&g).expect("a move");
    g.apply(mv);
    assert_eq!(g.status(), GameStatus::Win(Color::Red));
}

#[test]
fn engine_grabs_free_material() {
    // Red chariot can capture an undefended Black chariot down the file.
    let mut g = GameState::new();
    g.board = Board::empty();
    place(&mut g.board, 3, 9, PieceKind::General, Color::Red);
    place(&mut g.board, 5, 0, PieceKind::General, Color::Black);
    place(&mut g.board, 4, 4, PieceKind::Chariot, Color::Red);
    place(&mut g.board, 4, 7, PieceKind::Chariot, Color::Black);
    g.side_to_move = Color::Red;

    let mut engine = SearchEngine::fixed_depth(4);
    let mv = engine.best_move(&g).expect("a move");
    assert_eq!(mv, Move { from: idx(4, 4), to: idx(4, 7) });
}

#[test]
fn from_save_rejects_corrupt_history() {
    // An out-of-range move index must be rejected, not panic on replay.
    let mut g = GameState::new();
    g.apply(Move { from: idx(1, 9), to: idx(2, 7) });
    let mut save = g.to_save();
    save.history.push((Move { from: 999, to: 0 }, None));
    assert!(GameState::from_save(save).is_none());
}

#[test]
fn save_load_roundtrip() {
    let mut g = GameState::new();
    let mv = *g.legal_moves().first().unwrap();
    g.apply(mv);
    let json = serde_json::to_string(&g.to_save()).unwrap();
    let restored =
        GameState::from_save(serde_json::from_str(&json).unwrap()).unwrap();
    assert_eq!(restored.side_to_move, g.side_to_move);
    assert_eq!(restored.log, g.log);
    assert_eq!(restored.board.cells, g.board.cells);
}
