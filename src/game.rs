//! Game state: turn tracking, move application/undo, end-game detection,
//! human-readable notation, and JSON save/load.

use crate::board::*;
use crate::moves::{in_check, legal_moves, tag_move, MoveTag};
use serde::{Deserialize, Serialize};

/// Plies without a capture after which the game is declared a draw
/// (60 full moves, a common Xiangqi "natural draw" bound).
const NO_CAPTURE_PLY_LIMIT: usize = 120;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawReason {
    /// The same position (board + side to move) occurred three times.
    Repetition,
    /// 60 full moves passed with no capture.
    NoCapture,
}

/// Why a repeated position is a loss under CCA / Asian rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepKind {
    /// 长将: the losing side gave check on every move of the repeating cycle.
    Check,
    /// 长捉: the losing side chased (threatened to win material) every move.
    Chase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatus {
    Ongoing,
    Check(Color),       // `Color` is the side in check
    Win(Color),         // checkmate: `Color` wins
    Stalemate(Color),   // no legal move (not in check) -> `Color` wins (Xiangqi rule)
    Draw(DrawReason),
    /// Perpetual check / chase: the offending side loses; `winner` is the
    /// other side. CCA / Asian repetition rule.
    PerpetualLoss { winner: Color, kind: RepKind },
}

/// Apply the CCA repetition decision table to each side's offence level
/// (`2` = 长将 / perpetual check, `1` = 长捉 / perpetual chase, `0` = idle)
/// and its kind. Pure, so every arm is unit-testable without constructing a
/// position. The winner colours are fixed (Red is side 0, Black side 1).
pub(crate) fn repetition_status(rl: u8, rk: RepKind, bl: u8, bk: RepKind) -> GameStatus {
    let draw = GameStatus::Draw(DrawReason::Repetition);
    match (rl, bl) {
        // Both idle (plain positional repetition), or both committing the
        // same offence -> a legitimate draw.
        (0, 0) | (2, 2) | (1, 1) => draw,
        // Exactly one side offends, the other is idle -> offender loses.
        (_, 0) => GameStatus::PerpetualLoss { winner: Color::Black, kind: rk },
        (0, _) => GameStatus::PerpetualLoss { winner: Color::Red, kind: bk },
        // 一将一捉: the perpetual-checking side (level 2) loses.
        (2, 1) => GameStatus::PerpetualLoss { winner: Color::Black, kind: RepKind::Check },
        (1, 2) => GameStatus::PerpetualLoss { winner: Color::Red, kind: RepKind::Check },
        _ => draw,
    }
}

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub side_to_move: Color,
    /// (move, captured-piece) pairs, oldest first, for undo.
    pub history: Vec<(Move, Option<Piece>)>,
    /// Human-readable notation for each ply, parallel to `history`.
    pub log: Vec<String>,
    pub last_move: Option<Move>,
    /// Position hashes (board + side to move) for every reached position,
    /// including the start, for threefold-repetition detection.
    hashes: Vec<u64>,
}

/// FNV-1a hash of the position (cell contents + side to move).
fn position_hash(board: &Board, side: Color) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mix = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01B3);
    };
    for c in board.cells.iter() {
        let code = match c {
            None => 0u8,
            Some(p) => {
                let k = p.kind as u8; // PieceKind is a fieldless enum
                let col = if p.color == Color::Red { 0 } else { 1 };
                1 + k * 2 + col
            }
        };
        mix(&mut h, code);
    }
    mix(&mut h, if side == Color::Red { 0xA1 } else { 0xB2 });
    h
}

impl GameState {
    pub fn new() -> Self {
        let board = Board::initial();
        GameState {
            board,
            side_to_move: Color::Red,
            history: Vec::new(),
            log: Vec::new(),
            last_move: None,
            hashes: vec![position_hash(&board, Color::Red)],
        }
    }

    /// Build a state from a hand-placed board with the repetition `hashes`
    /// seeded to this position (so threefold detection works). Test-only.
    #[cfg(test)]
    pub fn with_position(board: Board, side: Color) -> Self {
        GameState {
            board,
            side_to_move: side,
            history: Vec::new(),
            log: Vec::new(),
            last_move: None,
            hashes: vec![position_hash(&board, side)],
        }
    }

    pub fn legal_moves(&mut self) -> Vec<Move> {
        legal_moves(&mut self.board, self.side_to_move)
    }

    pub fn is_legal(&mut self, mv: Move) -> bool {
        self.legal_moves().contains(&mv)
    }

    pub fn apply(&mut self, mv: Move) {
        let note = self.notation(mv);
        let captured = self.board.make(mv);
        self.history.push((mv, captured));
        self.log.push(note);
        self.last_move = Some(mv);
        self.side_to_move = self.side_to_move.opposite();
        self.hashes
            .push(position_hash(&self.board, self.side_to_move));
    }

    /// Undo one ply. Returns false if there is nothing to undo.
    pub fn undo(&mut self) -> bool {
        let Some((mv, captured)) = self.history.pop() else {
            return false;
        };
        self.board.unmake(mv, captured);
        self.log.pop();
        self.hashes.pop();
        self.side_to_move = self.side_to_move.opposite();
        self.last_move = self.history.last().map(|(m, _)| *m);
        true
    }

    /// Plies since the last capture (counts trailing non-capturing moves).
    fn plies_since_capture(&self) -> usize {
        self.history
            .iter()
            .rev()
            .take_while(|(_, captured)| captured.is_none())
            .count()
    }

    /// How many times the current position has occurred so far.
    fn repetition_count(&self) -> usize {
        match self.hashes.last() {
            Some(&h) => self.hashes.iter().filter(|&&x| x == h).count(),
            None => 0,
        }
    }

    /// CCA / Asian repetition judgment for the just-repeated position. Looks
    /// at the most recent identical position, treats the moves in between as
    /// one repeating cycle, classifies each side's contribution to it
    /// (perpetual check / perpetual chase / idle) via [`tag_move`], and
    /// applies the rule table:
    ///
    /// - exactly one side offends (长将 or 长捉), the other idle -> offender loses;
    /// - 一将一捉 (one perpetual-checks, the other perpetual-chases) -> the
    ///   perpetual-checking side loses (长将 is the heavier offence);
    /// - both 长将, or both 长捉, or both idle (a plain positional repetition)
    ///   -> draw.
    ///
    /// The cycle is capture-free (otherwise the position could not recur), so
    /// the chase test's material reasoning is well-defined.
    fn repetition_judgment(&self) -> GameStatus {
        let draw = GameStatus::Draw(DrawReason::Repetition);
        let last = self.history.len(); // hashes[last] == current position
        let cur = self.hashes[last];
        let Some(j) = (0..last).rev().find(|&k| self.hashes[k] == cur) else {
            return draw;
        };

        // Rewind a copy to the cycle's start, then replay it tagging moves.
        let mut board = self.board;
        for (mv, cap) in self.history[j..last].iter().rev() {
            board.unmake(*mv, *cap);
        }

        // Per side: total moves, all-check?, all-aggressive?, any chase?
        let mut agg = [[0u32; 4]; 2]; // [side][moves, checks, aggr, chases]
        for i in j..last {
            // Derive the mover of move `i` from the current side to move
            // (robust to who started): same side as now when `last - i` is
            // even, the other side when odd.
            let mover = if (last - i).is_multiple_of(2) {
                self.side_to_move
            } else {
                self.side_to_move.opposite()
            };
            let mv = self.history[i].0;
            let a = &mut agg[(mover == Color::Black) as usize];
            a[0] += 1;
            match tag_move(&board, mv, mover) {
                MoveTag::Check => {
                    a[1] += 1;
                    a[2] += 1;
                }
                MoveTag::Chase => {
                    a[2] += 1;
                    a[3] += 1;
                }
                MoveTag::Idle => {}
            }
            board.make(mv);
        }

        // Offence level: 2 = 长将 (every move a check), 1 = 长捉 (every move
        // aggressive with at least one chase), 0 = idle.
        let level = |a: &[u32; 4]| -> (u8, RepKind) {
            if a[0] > 0 && a[1] == a[0] {
                (2, RepKind::Check)
            } else if a[0] > 0 && a[2] == a[0] && a[3] > 0 {
                (1, RepKind::Chase)
            } else {
                (0, RepKind::Chase)
            }
        };
        let (rl, rk) = level(&agg[0]); // Red
        let (bl, bk) = level(&agg[1]); // Black
        repetition_status(rl, rk, bl, bk)
    }

    /// Current game status. Threefold repetition is judged by the CCA / Asian
    /// rules (see [`Self::repetition_judgment`]): perpetual check / chase is a
    /// loss for the offending side; only a mutual or idle repetition draws.
    pub fn status(&mut self) -> GameStatus {
        let side = self.side_to_move;
        let has_move = !self.legal_moves().is_empty();
        let checked = in_check(&self.board, side);

        if !has_move {
            // No legal reply: opponent wins whether it is mate or stalemate.
            return if checked {
                GameStatus::Win(side.opposite())
            } else {
                GameStatus::Stalemate(side.opposite())
            };
        }
        if self.repetition_count() >= 3 {
            return self.repetition_judgment();
        }
        if self.plies_since_capture() >= NO_CAPTURE_PLY_LIMIT {
            return GameStatus::Draw(DrawReason::NoCapture);
        }
        if checked {
            GameStatus::Check(side)
        } else {
            GameStatus::Ongoing
        }
    }

    /// Notation such as `炮 H3→E3`. Files are letters A..I from Red's left;
    /// ranks are 1..10 counting up from Red's home rank.
    fn notation(&self, mv: Move) -> String {
        let glyph = self
            .board
            .get(mv.from)
            .map(|p| p.glyph())
            .unwrap_or("?");
        format!(
            "{} {}{}→{}{}",
            glyph,
            file_letter(mv.from),
            rank_label(mv.from),
            file_letter(mv.to),
            rank_label(mv.to),
        )
    }

    pub fn to_save(&self) -> SaveGame {
        SaveGame {
            format: SAVE_FORMAT,
            cells: self.board.cells.to_vec(),
            side_to_move: self.side_to_move,
            history: self.history.clone(),
            log: self.log.clone(),
            last_move: self.last_move,
        }
    }

    pub fn from_save(s: SaveGame) -> Option<Self> {
        // Refuse formats newer than we understand (forward-incompatible).
        if s.format > SAVE_FORMAT {
            return None;
        }
        if s.cells.len() != CELLS {
            return None;
        }
        // Reject a corrupt save rather than panicking when the history is
        // replayed below (make/unmake index by these coordinates).
        if s.history
            .iter()
            .any(|(m, _)| m.from >= CELLS || m.to >= CELLS)
        {
            return None;
        }
        let mut cells = [None; CELLS];
        cells.copy_from_slice(&s.cells);
        let board = Board { cells };

        // Rebuild the repetition history by rewinding to the start and
        // replaying, so the save format stays unchanged but draw detection
        // keeps working after a load.
        let n = s.history.len();
        let start_side = if n.is_multiple_of(2) {
            s.side_to_move
        } else {
            s.side_to_move.opposite()
        };
        let mut b = board;
        for (mv, captured) in s.history.iter().rev() {
            b.unmake(*mv, *captured);
        }
        let mut side = start_side;
        let mut hashes = vec![position_hash(&b, side)];
        for (mv, _) in s.history.iter() {
            b.make(*mv);
            side = side.opposite();
            hashes.push(position_hash(&b, side));
        }

        Some(GameState {
            board,
            side_to_move: s.side_to_move,
            history: s.history,
            log: s.log,
            last_move: s.last_move,
            hashes,
        })
    }
}

pub fn file_letter(i: usize) -> char {
    (b'A' + file_of(i) as u8) as char
}

pub fn rank_label(i: usize) -> i32 {
    RANKS as i32 - rank_of(i) // rank 9 (Red home) -> 1, rank 0 -> 10
}

/// Current on-disk save format version. Bump when the structure changes.
pub const SAVE_FORMAT: u32 = 1;

#[derive(Serialize, Deserialize)]
pub struct SaveGame {
    /// Absent in pre-versioning saves -> defaults to 0, still loadable.
    #[serde(default)]
    pub format: u32,
    pub cells: Vec<Option<Piece>>,
    pub side_to_move: Color,
    pub history: Vec<(Move, Option<Piece>)>,
    pub log: Vec<String>,
    pub last_move: Option<Move>,
}
