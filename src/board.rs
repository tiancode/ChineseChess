//! Board representation, pieces, and coordinate helpers.
//!
//! Coordinate system: 9 files (x: 0..=8) by 10 ranks (y: 0..=9).
//! `index = rank * 9 + file`. Rank 0 is the top (Black's home rank);
//! rank 9 is the bottom (Red's home rank). Red moves first and sits at the
//! bottom, so Red soldiers advance toward smaller ranks.

use serde::{Deserialize, Serialize};

pub const FILES: usize = 9;
pub const RANKS: usize = 10;
pub const CELLS: usize = FILES * RANKS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Color {
    Red,
    Black,
}

impl Color {
    pub fn opposite(self) -> Color {
        match self {
            Color::Red => Color::Black,
            Color::Black => Color::Red,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PieceKind {
    General,
    Advisor,
    Elephant,
    Horse,
    Chariot,
    Cannon,
    Soldier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}

impl Piece {
    /// Traditional glyph used to draw the piece.
    pub fn glyph(self) -> &'static str {
        use Color::*;
        use PieceKind::*;
        match (self.kind, self.color) {
            (General, Red) => "帥",
            (General, Black) => "將",
            (Advisor, Red) => "仕",
            (Advisor, Black) => "士",
            (Elephant, Red) => "相",
            (Elephant, Black) => "象",
            (Horse, _) => "馬",
            (Chariot, _) => "車",
            (Cannon, Red) => "炮",
            (Cannon, Black) => "砲",
            (Soldier, Red) => "兵",
            (Soldier, Black) => "卒",
        }
    }

    /// Latin one-letter label, used when no CJK font is available so the
    /// board stays playable. Colour is conveyed by the disc/ink colour.
    pub fn ascii(self) -> &'static str {
        match self.kind {
            PieceKind::General => "K",
            PieceKind::Advisor => "A",
            PieceKind::Elephant => "E",
            PieceKind::Horse => "H",
            PieceKind::Chariot => "R",
            PieceKind::Cannon => "C",
            PieceKind::Soldier => "P",
        }
    }
}

#[inline]
pub fn idx(file: i32, rank: i32) -> usize {
    (rank as usize) * FILES + (file as usize)
}

#[inline]
pub fn file_of(i: usize) -> i32 {
    (i % FILES) as i32
}

#[inline]
pub fn rank_of(i: usize) -> i32 {
    (i / FILES) as i32
}

#[inline]
pub fn on_board(file: i32, rank: i32) -> bool {
    file >= 0 && file < FILES as i32 && rank >= 0 && rank < RANKS as i32
}

/// Is `(file, rank)` inside the 3x3 palace of `color`?
pub fn in_palace(color: Color, file: i32, rank: i32) -> bool {
    if !(3..=5).contains(&file) {
        return false;
    }
    match color {
        Color::Black => (0..=2).contains(&rank),
        Color::Red => (7..=9).contains(&rank),
    }
}

/// Has the piece at `rank` crossed the river, for the given color?
pub fn crossed_river(color: Color, rank: i32) -> bool {
    match color {
        Color::Red => rank <= 4,
        Color::Black => rank >= 5,
    }
}

/// Is `rank` on `color`'s own half (used to keep elephants from crossing)?
pub fn own_half(color: Color, rank: i32) -> bool {
    match color {
        Color::Red => rank >= 5,
        Color::Black => rank <= 4,
    }
}

/// Forward rank direction for a soldier of `color` (Red advances up = -1).
pub fn forward(color: Color) -> i32 {
    match color {
        Color::Red => -1,
        Color::Black => 1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Move {
    pub from: usize,
    pub to: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Board {
    pub cells: [Option<Piece>; CELLS],
}

impl Board {
    pub fn empty() -> Self {
        Board {
            cells: [None; CELLS],
        }
    }

    /// Standard opening position.
    pub fn initial() -> Self {
        let mut b = Board::empty();
        let back = [
            PieceKind::Chariot,
            PieceKind::Horse,
            PieceKind::Elephant,
            PieceKind::Advisor,
            PieceKind::General,
            PieceKind::Advisor,
            PieceKind::Elephant,
            PieceKind::Horse,
            PieceKind::Chariot,
        ];

        // Black at the top (ranks 0..=3).
        for (f, &kind) in back.iter().enumerate() {
            b.cells[idx(f as i32, 0)] = Some(Piece {
                kind,
                color: Color::Black,
            });
        }
        b.cells[idx(1, 2)] = Some(Piece { kind: PieceKind::Cannon, color: Color::Black });
        b.cells[idx(7, 2)] = Some(Piece { kind: PieceKind::Cannon, color: Color::Black });
        for f in (0..9).step_by(2) {
            b.cells[idx(f, 3)] = Some(Piece { kind: PieceKind::Soldier, color: Color::Black });
        }

        // Red at the bottom (ranks 6..=9).
        for (f, &kind) in back.iter().enumerate() {
            b.cells[idx(f as i32, 9)] = Some(Piece {
                kind,
                color: Color::Red,
            });
        }
        b.cells[idx(1, 7)] = Some(Piece { kind: PieceKind::Cannon, color: Color::Red });
        b.cells[idx(7, 7)] = Some(Piece { kind: PieceKind::Cannon, color: Color::Red });
        for f in (0..9).step_by(2) {
            b.cells[idx(f, 6)] = Some(Piece { kind: PieceKind::Soldier, color: Color::Red });
        }

        b
    }

    #[inline]
    pub fn get(&self, i: usize) -> Option<Piece> {
        self.cells[i]
    }

    /// Apply a move, returning the captured piece (if any) for later undo.
    pub fn make(&mut self, mv: Move) -> Option<Piece> {
        let captured = self.cells[mv.to];
        self.cells[mv.to] = self.cells[mv.from];
        self.cells[mv.from] = None;
        captured
    }

    /// Reverse `make`.
    pub fn unmake(&mut self, mv: Move, captured: Option<Piece>) {
        self.cells[mv.from] = self.cells[mv.to];
        self.cells[mv.to] = captured;
    }

    pub fn find_general(&self, color: Color) -> Option<usize> {
        self.cells.iter().position(|c| {
            matches!(c, Some(p) if p.kind == PieceKind::General && p.color == color)
        })
    }
}
