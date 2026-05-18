//! Move generation and rule enforcement (check, flying general, legality).

use crate::board::*;

/// All pseudo-legal moves for `color` (does not filter self-check / flying
/// general). This doubles as an attack generator for check detection.
pub fn pseudo_moves(board: &Board, color: Color) -> Vec<Move> {
    let mut out = Vec::with_capacity(48);
    for i in 0..CELLS {
        let Some(p) = board.cells[i] else { continue };
        if p.color != color {
            continue;
        }
        let f = file_of(i);
        let r = rank_of(i);
        match p.kind {
            PieceKind::General => gen_general(board, color, f, r, i, &mut out),
            PieceKind::Advisor => gen_advisor(board, color, f, r, i, &mut out),
            PieceKind::Elephant => gen_elephant(board, color, f, r, i, &mut out),
            PieceKind::Horse => gen_horse(board, color, f, r, i, &mut out),
            PieceKind::Chariot => gen_chariot(board, color, f, r, i, &mut out),
            PieceKind::Cannon => gen_cannon(board, color, f, r, i, &mut out),
            PieceKind::Soldier => gen_soldier(board, color, f, r, i, &mut out),
        }
    }
    out
}

#[inline]
fn target_ok(board: &Board, color: Color, f: i32, r: i32) -> Option<bool> {
    // Some(true) => empty (quiet), Some(false) => enemy (capture), None => blocked/own
    if !on_board(f, r) {
        return None;
    }
    match board.cells[idx(f, r)] {
        None => Some(true),
        Some(p) if p.color != color => Some(false),
        Some(_) => None,
    }
}

fn push_if(board: &Board, color: Color, from: usize, f: i32, r: i32, out: &mut Vec<Move>) {
    if target_ok(board, color, f, r).is_some() {
        out.push(Move { from, to: idx(f, r) });
    }
}

fn gen_general(board: &Board, color: Color, f: i32, r: i32, from: usize, out: &mut Vec<Move>) {
    for (df, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let (nf, nr) = (f + df, r + dr);
        if in_palace(color, nf, nr) {
            push_if(board, color, from, nf, nr, out);
        }
    }
}

fn gen_advisor(board: &Board, color: Color, f: i32, r: i32, from: usize, out: &mut Vec<Move>) {
    for (df, dr) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let (nf, nr) = (f + df, r + dr);
        if in_palace(color, nf, nr) {
            push_if(board, color, from, nf, nr, out);
        }
    }
}

fn gen_elephant(board: &Board, color: Color, f: i32, r: i32, from: usize, out: &mut Vec<Move>) {
    for (df, dr) in [(2, 2), (2, -2), (-2, 2), (-2, -2)] {
        let (nf, nr) = (f + df, r + dr);
        if !on_board(nf, nr) || !own_half(color, nr) {
            continue;
        }
        // "Blocking the elephant's eye": midpoint must be empty.
        if board.cells[idx(f + df / 2, r + dr / 2)].is_some() {
            continue;
        }
        push_if(board, color, from, nf, nr, out);
    }
}

fn gen_horse(board: &Board, color: Color, f: i32, r: i32, from: usize, out: &mut Vec<Move>) {
    // (leg df, leg dr, target df, target dr)
    let moves = [
        (1, 0, 2, 1),
        (1, 0, 2, -1),
        (-1, 0, -2, 1),
        (-1, 0, -2, -1),
        (0, 1, 1, 2),
        (0, 1, -1, 2),
        (0, -1, 1, -2),
        (0, -1, -1, -2),
    ];
    for (lf, lr, tf, tr) in moves {
        if !on_board(f + lf, r + lr) || board.cells[idx(f + lf, r + lr)].is_some() {
            continue; // "Hobbling the horse's leg".
        }
        push_if(board, color, from, f + tf, r + tr, out);
    }
}

fn slide(board: &Board, color: Color, from: usize, f: i32, r: i32, out: &mut Vec<Move>) {
    for (df, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let (mut nf, mut nr) = (f + df, r + dr);
        while on_board(nf, nr) {
            match board.cells[idx(nf, nr)] {
                None => out.push(Move { from, to: idx(nf, nr) }),
                Some(p) => {
                    if p.color != color {
                        out.push(Move { from, to: idx(nf, nr) });
                    }
                    break;
                }
            }
            nf += df;
            nr += dr;
        }
    }
}

fn gen_chariot(board: &Board, color: Color, f: i32, r: i32, from: usize, out: &mut Vec<Move>) {
    slide(board, color, from, f, r, out);
}

fn gen_cannon(board: &Board, color: Color, f: i32, r: i32, from: usize, out: &mut Vec<Move>) {
    for (df, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let (mut nf, mut nr) = (f + df, r + dr);
        // Phase 1: quiet moves over empty squares until the screen.
        while on_board(nf, nr) && board.cells[idx(nf, nr)].is_none() {
            out.push(Move { from, to: idx(nf, nr) });
            nf += df;
            nr += dr;
        }
        // Phase 2: hop the screen, capture the first piece beyond it if enemy.
        nf += df;
        nr += dr;
        while on_board(nf, nr) {
            if let Some(p) = board.cells[idx(nf, nr)] {
                if p.color != color {
                    out.push(Move { from, to: idx(nf, nr) });
                }
                break;
            }
            nf += df;
            nr += dr;
        }
    }
}

fn gen_soldier(board: &Board, color: Color, f: i32, r: i32, from: usize, out: &mut Vec<Move>) {
    let fwd = forward(color);
    push_if(board, color, from, f, r + fwd, out);
    if crossed_river(color, r) {
        push_if(board, color, from, f + 1, r, out);
        push_if(board, color, from, f - 1, r, out);
    }
}

/// Are the two generals facing each other on a clear file? Such a position is
/// illegal (the "flying general" rule).
pub fn generals_face(board: &Board) -> bool {
    let (Some(rg), Some(bg)) = (
        board.find_general(Color::Red),
        board.find_general(Color::Black),
    ) else {
        return false;
    };
    if file_of(rg) != file_of(bg) {
        return false;
    }
    let f = file_of(rg);
    let (lo, hi) = (rank_of(rg).min(rank_of(bg)), rank_of(rg).max(rank_of(bg)));
    for r in (lo + 1)..hi {
        if board.cells[idx(f, r)].is_some() {
            return false;
        }
    }
    true
}

/// Is `sq` attacked by any piece of `by`?
pub fn is_attacked(board: &Board, sq: usize, by: Color) -> bool {
    pseudo_moves(board, by).iter().any(|m| m.to == sq)
}

/// Is `color`'s general currently in check?
pub fn in_check(board: &Board, color: Color) -> bool {
    match board.find_general(color) {
        Some(g) => is_attacked(board, g, color.opposite()),
        None => true, // general already captured
    }
}

/// Fully legal moves for `color`: pseudo-legal moves that do not leave the
/// mover's general in check and do not expose the flying-general rule.
pub fn legal_moves(board: &mut Board, color: Color) -> Vec<Move> {
    let mut legal = Vec::with_capacity(48);
    for mv in pseudo_moves(board, color) {
        let captured = board.make(mv);
        if !in_check(board, color) && !generals_face(board) {
            legal.push(mv);
        }
        board.unmake(mv, captured);
    }
    legal
}

// ---------------------------------------------------------------------------
// Repetition-rule support: classifying a move as check / chase / idle so the
// CCA (Asian) perpetual-check / perpetual-chase rules can be applied.
// ---------------------------------------------------------------------------

/// Nominal piece values (centipawns). Shared with the engine's material term
/// so the chase "wins material?" test uses the same scale. `General` is 0:
/// it is never a chase target (threatening it is *check*, handled separately).
pub fn piece_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::General => 0,
        PieceKind::Chariot => 1000,
        PieceKind::Cannon => 500,
        PieceKind::Horse => 450,
        PieceKind::Advisor => 200,
        PieceKind::Elephant => 200,
        PieceKind::Soldier => 100,
    }
}

/// Does the position (already advanced past `mover`'s move, so it is the
/// opponent to move) leave the opponent's general in check?
pub fn gives_check(board: &Board, mover: Color) -> bool {
    in_check(board, mover.opposite())
}

/// Is `mover` "chasing" (捉) in this position — i.e. threatening to capture an
/// opponent piece next move at a net material profit? CCA exclusions applied:
///
/// - The general is never a chase victim (that threat is *check*).
/// - A pawn or the general doing the threatening counts as idle, not chase
///   (兵/帅 之捉算闲).
/// - An adequately-defended victim is only a chase if taking it still wins
///   material (value(victim) > value(attacker)); an equal/again-defended
///   trade is 兑/idle, not 捉.
///
/// `board` is the position *after* `mover`'s move (opponent to move). Only
/// legal captures for `mover` are considered, so the threat is real.
pub fn threatens_win(board: &Board, mover: Color) -> bool {
    let mut b = *board;
    for m in legal_moves(&mut b, mover) {
        let Some(victim) = board.get(m.to) else {
            continue; // not a capture
        };
        if victim.kind == PieceKind::General {
            continue; // that is check, classified elsewhere
        }
        let attacker = match board.get(m.from) {
            Some(p) => p,
            None => continue,
        };
        if matches!(attacker.kind, PieceKind::Soldier | PieceKind::General) {
            continue; // a pawn/general chase is idle under CCA
        }
        // 1-ply static exchange: undefended victim is pure profit; a defended
        // one is only a chase if we still come out ahead after the recapture.
        let mut after = *board;
        let cap = after.make(m);
        let defended = is_attacked(&after, m.to, mover.opposite());
        after.unmake(m, cap);
        if !defended || piece_value(victim.kind) > piece_value(attacker.kind) {
            return true;
        }
    }
    false
}

/// What a single move "does" for the side to move, used by the repetition
/// rules. Check takes precedence over chase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveTag {
    Check,
    Chase,
    Idle,
}

/// Classify `mv` (played by `mover` from `board_before`).
pub fn tag_move(board_before: &Board, mv: Move, mover: Color) -> MoveTag {
    let mut b = *board_before;
    let cap = b.make(mv);
    let tag = if gives_check(&b, mover) {
        MoveTag::Check
    } else if threatens_win(&b, mover) {
        MoveTag::Chase
    } else {
        MoveTag::Idle
    };
    b.unmake(mv, cap);
    tag
}
