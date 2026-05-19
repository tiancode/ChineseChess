//! Rule-engine unit tests.
//!
//! Note: in any hand-built position the two generals must NOT sit on the same
//! file with a clear path between them — that is the (illegal) flying-general
//! position and the engine will correctly reject every move from it.

use crate::ai::search::SearchEngine;
use crate::ai::{make_engine, Engine, EngineKind};
use crate::board::*;
use crate::game::{repetition_status, DrawReason, GameState, GameStatus, RepKind};
use crate::moves::{generals_face, in_check, legal_moves, tag_move, MoveTag};

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
    // Also the regression for the CCA change: both sides idle every move, so
    // a plain positional repetition must still be a draw.
    assert_eq!(g.status(), GameStatus::Draw(DrawReason::Repetition));
}

// --- CCA / Asian repetition rules ------------------------------------------

#[test]
fn tag_move_check_chase_idle() {
    // Check: a chariot move that gives check.
    let mut b = Board::empty();
    place(&mut b, 0, 9, PieceKind::General, Color::Red);
    place(&mut b, 4, 0, PieceKind::General, Color::Black);
    place(&mut b, 4, 5, PieceKind::Chariot, Color::Red);
    let mv = Move { from: idx(4, 5), to: idx(4, 1) }; // file 4 -> checks Bgen
    assert_eq!(tag_move(&b, mv, Color::Red), MoveTag::Check);

    // Chase: rook swings to attack an *undefended* horse, no check.
    let mut b = Board::empty();
    place(&mut b, 0, 9, PieceKind::General, Color::Red);
    place(&mut b, 8, 0, PieceKind::General, Color::Black);
    place(&mut b, 0, 4, PieceKind::Chariot, Color::Red);
    place(&mut b, 4, 2, PieceKind::Horse, Color::Black);
    let chase = Move { from: idx(0, 4), to: idx(4, 4) }; // then threatens (4,2)
    assert_eq!(tag_move(&b, chase, Color::Red), MoveTag::Chase);

    // Same, but the horse is defended by a soldier: an unfavourable trade is
    // not a chase -> idle.
    place(&mut b, 4, 1, PieceKind::Soldier, Color::Black); // Black pawn guards (4,2)
    assert_eq!(tag_move(&b, chase, Color::Red), MoveTag::Idle);

    // A pawn doing the threatening is idle (兵之捉算闲), never a chase.
    let mut b = Board::empty();
    place(&mut b, 0, 9, PieceKind::General, Color::Red);
    place(&mut b, 8, 0, PieceKind::General, Color::Black);
    place(&mut b, 5, 4, PieceKind::Soldier, Color::Red); // crossed-river pawn
    place(&mut b, 4, 3, PieceKind::Horse, Color::Black); // undefended
    let pawn = Move { from: idx(5, 4), to: idx(5, 3) }; // then pawn eyes (4,3)
    assert_eq!(tag_move(&b, pawn, Color::Red), MoveTag::Idle);
}

#[test]
fn perpetual_check_loses() {
    // Red chariot perpetually checks a bare Black king; Black only shuffles.
    // Red is 长将 -> Red loses, Black wins.
    let mut b = Board::empty();
    place(&mut b, 3, 9, PieceKind::General, Color::Red);
    place(&mut b, 4, 0, PieceKind::General, Color::Black);
    place(&mut b, 0, 0, PieceKind::Chariot, Color::Red); // checks Bgen on rank 0
    let mut g = GameState::with_position(b, Color::Black); // Black in check

    let cycle = [
        Move { from: idx(4, 0), to: idx(4, 1) }, // Black king escapes
        Move { from: idx(0, 0), to: idx(0, 1) }, // Red re-checks on rank 1
        Move { from: idx(4, 1), to: idx(4, 0) }, // Black king back
        Move { from: idx(0, 1), to: idx(0, 0) }, // Red re-checks on rank 0
    ];
    for _ in 0..2 {
        for mv in cycle {
            assert!(g.is_legal(mv), "cycle move should be legal: {mv:?}");
            g.apply(mv);
        }
    }
    assert_eq!(
        g.status(),
        GameStatus::PerpetualLoss { winner: Color::Black, kind: RepKind::Check }
    );
}

#[test]
fn perpetual_chase_loses() {
    // Red chariot oscillates on file 4, attacking a fixed undefended Black
    // horse every move (长捉); Black shuffles an idle horse far away.
    let mut b = Board::empty();
    place(&mut b, 0, 9, PieceKind::General, Color::Red);
    place(&mut b, 4, 0, PieceKind::General, Color::Black);
    place(&mut b, 4, 9, PieceKind::Chariot, Color::Red); // eyes horse down file 4
    place(&mut b, 4, 4, PieceKind::Horse, Color::Black); // undefended target
    place(&mut b, 0, 0, PieceKind::Horse, Color::Black); // idle shuffler
    let mut g = GameState::with_position(b, Color::Red);

    let cycle = [
        Move { from: idx(4, 9), to: idx(4, 8) }, // Red keeps chasing the horse
        Move { from: idx(0, 0), to: idx(2, 1) }, // Black idles
        Move { from: idx(4, 8), to: idx(4, 9) }, // Red keeps chasing
        Move { from: idx(2, 1), to: idx(0, 0) }, // Black idles back
    ];
    for _ in 0..2 {
        for mv in cycle {
            assert!(g.is_legal(mv), "cycle move should be legal: {mv:?}");
            g.apply(mv);
        }
    }
    assert_eq!(
        g.status(),
        GameStatus::PerpetualLoss { winner: Color::Black, kind: RepKind::Chase }
    );
}

#[test]
fn defended_chase_is_a_draw() {
    // Same shape, but the horse is defended so taking it loses material: the
    // rook's moves are not a chase -> both sides idle -> draw.
    let mut b = Board::empty();
    place(&mut b, 0, 9, PieceKind::General, Color::Red);
    place(&mut b, 4, 0, PieceKind::General, Color::Black);
    place(&mut b, 4, 9, PieceKind::Chariot, Color::Red);
    place(&mut b, 4, 4, PieceKind::Horse, Color::Black);
    place(&mut b, 4, 3, PieceKind::Soldier, Color::Black); // guards the horse
    place(&mut b, 0, 0, PieceKind::Horse, Color::Black);
    let mut g = GameState::with_position(b, Color::Red);

    let cycle = [
        Move { from: idx(4, 9), to: idx(4, 8) },
        Move { from: idx(0, 0), to: idx(2, 1) },
        Move { from: idx(4, 8), to: idx(4, 9) },
        Move { from: idx(2, 1), to: idx(0, 0) },
    ];
    for _ in 0..2 {
        for mv in cycle {
            assert!(g.is_legal(mv), "cycle move should be legal: {mv:?}");
            g.apply(mv);
        }
    }
    assert_eq!(g.status(), GameStatus::Draw(DrawReason::Repetition));
}

#[test]
fn repetition_decision_table_all_arms() {
    use Color::{Black, Red};
    let draw = GameStatus::Draw(DrawReason::Repetition);
    let chk = RepKind::Check;
    let cha = RepKind::Chase;
    let loss = |w, k| GameStatus::PerpetualLoss { winner: w, kind: k };

    // Both idle, or both committing the same offence -> draw.
    assert_eq!(repetition_status(0, cha, 0, cha), draw);
    assert_eq!(repetition_status(2, chk, 2, chk), draw);
    assert_eq!(repetition_status(1, cha, 1, cha), draw);
    // Exactly one side offends, the other idle -> offender loses.
    assert_eq!(repetition_status(2, chk, 0, cha), loss(Black, chk)); // Red 长将
    assert_eq!(repetition_status(1, cha, 0, cha), loss(Black, cha)); // Red 长捉
    assert_eq!(repetition_status(0, cha, 2, chk), loss(Red, chk)); // Black 长将
    assert_eq!(repetition_status(0, cha, 1, cha), loss(Red, cha)); // Black 长捉
    // 一将一捉: the perpetual-checking side loses (长将 is heavier).
    assert_eq!(repetition_status(2, chk, 1, cha), loss(Black, chk)); // Red checks
    assert_eq!(repetition_status(1, cha, 2, chk), loss(Red, chk)); // Black checks
}

#[test]
fn repetition_score_parity() {
    // Synthetic 4-ply cycle: path[0] is the repeating key K. The cycle moves
    // (by parity) split into the side-to-move's moves and the opponent's;
    // `repetition_score` must rule perpetual check for whichever side checks
    // on *all* of its cycle moves.
    let k = 42u64;
    let path = [k, 7, 8, 9];
    let mut e = SearchEngine::fixed_depth(1);
    let ply = 5;

    // self (side to move) checks every one of its cycle moves, opponent not.
    let s = e.repetition_score_probe(&path, &[false, true, false, true], k, ply, true);
    assert!(s < -20_000, "side perpetually checks -> it loses: {s}");

    // opponent checks every one of its moves, self not.
    let s = e.repetition_score_probe(&path, &[false, false, true, false], k, ply, true);
    assert!(s > 20_000, "opponent perpetually checks -> we win: {s}");

    // mutual perpetual check -> draw.
    let s = e.repetition_score_probe(&path, &[false, true, true, true], k, ply, true);
    assert_eq!(s, 0, "mutual perpetual check is a draw");

    // no side checks throughout -> not perpetual check -> draw (0).
    let s = e.repetition_score_probe(&path, &[false, false, false, false], k, ply, false);
    assert_eq!(s, 0, "no perpetual check -> draw");

    // Repetition formed off-path (key absent from path) -> draw fallback.
    let s = e.repetition_score_probe(&[1, 2, 3], &[false, false, false], 99, ply, true);
    assert_eq!(s, 0, "off-path repetition falls back to draw");
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

/// A/B strength gate. Pits the upgraded engine (`Tuning::full`) against the
/// pre-upgrade reference (`Tuning::baseline`) over a small book of distinct
/// openings, colours swapped, at a fixed per-move time budget. Prints the
/// score and an Elo estimate; the acceptance rule per phase is "non-negative,
/// and clearly positive for Phase 2 / 4".
///
/// Not run by default (timing-dependent, ~1-2 min):
/// `cargo test --release engine_ab_selfplay -- --ignored --nocapture`
#[test]
#[ignore]
fn engine_ab_selfplay() {
    use crate::ai::search::{SearchEngine, Tuning};
    use std::time::Duration;

    const MOVE_MS: u64 = 50; // per-move wall clock for both engines
    const DEPTH_CAP: u8 = 64; // time is the real limiter
    const MAX_PLIES: usize = 200; // adjudicate marathon games as a draw

    // Distinct, legal opening stems: indices into the legal-move list at each
    // ply (mod len), so positions diverge while staying sound. Empty = the
    // standard start.
    let openings: &[&[usize]] = &[
        &[],
        &[0],
        &[5],
        &[12],
        &[20],
        &[0, 0],
        &[7, 3],
        &[15, 9],
    ];

    fn winner_of(st: GameStatus) -> Option<Option<Color>> {
        // Some(Some(c)) = c won; Some(None) = draw; None = not terminal.
        match st {
            GameStatus::Win(c) | GameStatus::Stalemate(c) => Some(Some(c)),
            GameStatus::PerpetualLoss { winner, .. } => Some(Some(winner)),
            GameStatus::Draw(_) => Some(None),
            GameStatus::Ongoing | GameStatus::Check(_) => None,
        }
    }

    // One game. Returns A's points ×2 (win=2, draw=1, loss=0).
    fn play(opening: &[usize], a_is_red: bool, seed: u64) -> i32 {
        let mut g = GameState::new();
        for &pick in opening {
            let lm = g.legal_moves();
            if lm.is_empty() {
                break;
            }
            g.apply(lm[pick % lm.len()]);
        }

        let mut ta = Tuning::full();
        ta.variety = false; // deterministic: measure true-best play
        let mut tb = Tuning::baseline();
        tb.variety = false;
        let bud = Duration::from_millis(MOVE_MS);
        // Same TT size for both so the match isolates heuristic quality.
        let mut ea = SearchEngine::new_tuned(DEPTH_CAP, bud, ta, 20);
        let mut eb = SearchEngine::new_tuned(DEPTH_CAP, bud, tb, 20);
        ea.set_seed(seed);
        eb.set_seed(seed ^ 0x9E37_79B9);

        loop {
            if let Some(res) = winner_of(g.status()) {
                return match res {
                    None => 1,
                    Some(c) => {
                        if (c == Color::Red) == a_is_red {
                            2
                        } else {
                            0
                        }
                    }
                };
            }
            if g.history.len() >= MAX_PLIES {
                return 1; // drawn by adjudication
            }
            let a_to_move = (g.side_to_move == Color::Red) == a_is_red;
            let mv = if a_to_move {
                ea.best_move(&g)
            } else {
                eb.best_move(&g)
            };
            match mv {
                Some(m) => g.apply(m),
                None => {
                    // No move = side to move loses (status() agrees next loop).
                    return if a_to_move { 0 } else { 2 };
                }
            }
        }
    }

    let mut pts2 = 0i32; // A points ×2
    let mut games = 0i32;
    let (mut w, mut d, mut l) = (0i32, 0i32, 0i32);
    for (gi, op) in openings.iter().enumerate() {
        for (ci, &a_is_red) in [true, false].iter().enumerate() {
            let seed = 0x1234_5678 ^ ((gi as u64) << 8) ^ (ci as u64);
            let r = play(op, a_is_red, seed);
            pts2 += r;
            games += 1;
            match r {
                2 => w += 1,
                1 => d += 1,
                _ => l += 1,
            }
            println!(
                "game {games:>2}: opening {op:?} A={} -> {}",
                if a_is_red { "Red" } else { "Black" },
                match r {
                    2 => "A win",
                    1 => "draw",
                    _ => "B win",
                }
            );
        }
    }

    let score = pts2 as f64 / (2.0 * games as f64); // A's score fraction
    let elo = if score <= 0.0 {
        f64::NEG_INFINITY
    } else if score >= 1.0 {
        f64::INFINITY
    } else {
        -400.0 * (1.0 / score - 1.0).log10()
    };
    println!(
        "\nA(full) vs B(baseline): +{w} ={d} -{l} of {games}  score={score:.3}  Elo≈{elo:+.0}"
    );
    assert!(
        pts2 >= games,
        "upgraded engine must not regress vs baseline (score {score:.3} < 0.500)"
    );
}

/// Lazy-SMP correctness: with several worker threads sharing the atomic TT,
/// every returned move must still be legal and the game must progress without
/// a panic or hang (a data race / torn TT read would surface here).
#[test]
fn smp_engine_plays_legal_moves() {
    use crate::ai::search::{SearchEngine, Tuning};
    use std::time::Duration;

    let mut g = GameState::new();
    let mut e = SearchEngine::new_tuned(64, Duration::from_millis(40), Tuning::full(), 18)
        .with_threads(4);
    for _ in 0..12 {
        if !matches!(g.status(), GameStatus::Ongoing | GameStatus::Check(_)) {
            break;
        }
        let mv = e.best_move(&g).expect("a legal move");
        assert!(g.is_legal(mv), "SMP returned an illegal move: {mv:?}");
        g.apply(mv);
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
