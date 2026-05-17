//! Pluggable AI interface.
//!
//! To swap in a stronger engine later, implement [`Engine`] and return it from
//! [`make_engine`]; nothing else in the program needs to change.

use std::time::Duration;

use crate::board::{Color, Move};
use crate::game::GameState;

pub mod search;

pub trait Engine: Send {
    /// Engine label (shown for debugging / future UI; part of the public
    /// engine API even though the bundled UI does not display it yet).
    #[allow(dead_code)]
    fn name(&self) -> String;
    /// Best move for `state.side_to_move`, or `None` if there is no legal move.
    fn best_move(&mut self, state: &GameState) -> Option<Move>;
}

/// Build the engine for a given difficulty (1 = easiest .. 5 = hardest).
/// Difficulty maps to a max search depth and a per-move time budget; the
/// iterative-deepening search stops at whichever limit is hit first.
/// This is the single place to change to use a different engine.
pub fn make_engine(difficulty: u8, _perspective: Color) -> Box<dyn Engine + Send> {
    let (depth, ms) = match difficulty.clamp(1, 5) {
        1 => (3, 120),
        2 => (5, 350),
        3 => (7, 900),
        4 => (10, 2_000),
        _ => (14, 4_500),
    };
    Box::new(search::SearchEngine::new(depth, Duration::from_millis(ms)))
}
