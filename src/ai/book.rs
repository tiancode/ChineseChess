//! Tiny opening "book": sound, recognised first moves so the engine does not
//! burn clock (and does not always start identically) in the opening, without
//! needing an external theory database. It only covers each side's **first**
//! move (history length 0 or 1) — every move it can return is a textbook
//! opening, so there is no deep-theory correctness risk; afterwards the search
//! takes over.

use crate::board::{file_of, forward, rank_of, Color, Move, PieceKind};
use crate::game::GameState;
use crate::moves::legal_moves;

/// A weighted-random principled opening move, or `None` (not in book / no
/// candidate matched → fall back to search). `rnd` is a random word supplied
/// by the caller so the engine keeps sole ownership of its RNG.
pub fn book_move(state: &GameState, rnd: u64) -> Option<Move> {
    if state.history.len() >= 2 {
        return None; // first move per side only
    }
    let side = state.side_to_move;
    let mut board = state.board;
    let back = if side == Color::Red { 9 } else { 0 };
    let fwd = forward(side);

    let mut cands: Vec<(Move, u32)> = Vec::new();
    for m in legal_moves(&mut board, side) {
        let Some(p) = state.board.get(m.from) else {
            continue;
        };
        let (ff, fr) = (file_of(m.from), rank_of(m.from));
        let (tf, tr) = (file_of(m.to), rank_of(m.to));
        let w = match p.kind {
            // 中炮 / 当头炮: cannon slides onto the central file. The single
            // most popular and strongest Xiangqi opening.
            PieceKind::Cannon if fr == tr && tf == 4 => 50,
            // 起马 / 屏风马 component: a back-rank knight steps toward the
            // centre (马二进三 / 马八进七).
            PieceKind::Horse if fr == back && (ff == 1 || ff == 7) && (tf == 2 || tf == 6) => 25,
            // 飞相 / 飞象: elephant to the central point.
            PieceKind::Elephant if tf == 4 => 20,
            // 兵三/七进一: a flank soldier advances one rank.
            PieceKind::Soldier if (ff == 2 || ff == 6) && tr - fr == fwd && tf == ff => 10,
            _ => 0,
        };
        if w > 0 {
            cands.push((m, w));
        }
    }
    if cands.is_empty() {
        return None;
    }
    let total: u32 = cands.iter().map(|(_, w)| *w).sum();
    let mut pick = (rnd % total as u64) as u32;
    for (m, w) in &cands {
        if pick < *w {
            return Some(*m);
        }
        pick -= *w;
    }
    cands.last().map(|(m, _)| *m)
}
