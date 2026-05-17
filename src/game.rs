//! Game state: turn tracking, move application/undo, end-game detection,
//! human-readable notation, and JSON save/load.

use crate::board::*;
use crate::moves::{in_check, legal_moves};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatus {
    Ongoing,
    Check(Color),       // `Color` is the side in check
    Win(Color),         // checkmate: `Color` wins
    Stalemate(Color),   // no legal move (not in check) -> `Color` wins (Xiangqi rule)
    Draw(DrawReason),
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

    /// Current game status.
    ///
    /// Simplification: threefold repetition is scored as a draw. Full Xiangqi
    /// rules instead punish *perpetual check / chase* as a loss for the
    /// offending side; that nuance is intentionally not enforced here.
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
            return GameStatus::Draw(DrawReason::Repetition);
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
