//! Pluggable AI interface.
//!
//! To swap in a stronger engine later, implement [`Engine`] and return it from
//! [`make_engine`]; nothing else in the program needs to change.

use std::time::Duration;

use crate::board::{Color, Move};
use crate::game::GameState;

pub mod alphazero;
pub mod search;

pub trait Engine: Send {
    /// Engine label (shown for debugging / future UI; part of the public
    /// engine API even though the bundled UI does not display it yet).
    #[allow(dead_code)]
    fn name(&self) -> String;
    /// Best move for `state.side_to_move`, or `None` if there is no legal move.
    fn best_move(&mut self, state: &GameState) -> Option<Move>;
}

/// Which engine to build. `AlphaBeta` is the bundled in-process searcher;
/// `AlphaZero` drives the trained network via the Python MCTS sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineKind {
    #[default]
    AlphaBeta,
    AlphaZero,
}

impl EngineKind {
    /// Short Chinese label for the engine-selection UI.
    pub fn label(self) -> &'static str {
        match self {
            EngineKind::AlphaBeta => "内置 AlphaBeta",
            EngineKind::AlphaZero => "AlphaZero",
        }
    }
}

/// Build an engine of `kind` for a given difficulty (1 = easiest .. 5 =
/// hardest). For `AlphaBeta`, difficulty maps to a max search depth and a
/// per-move time budget (iterative deepening stops at whichever limit is hit
/// first). For `AlphaZero`, difficulty maps to the MCTS simulation count.
/// This is the single place to change to use a different engine.
pub fn make_engine(
    kind: EngineKind,
    difficulty: u8,
    _perspective: Color,
) -> Box<dyn Engine + Send> {
    let d = difficulty.clamp(1, 5);
    match kind {
        EngineKind::AlphaBeta => {
            let (depth, ms) = match d {
                1 => (3, 120),
                2 => (5, 350),
                3 => (7, 900),
                4 => (10, 2_000),
                _ => (14, 4_500),
            };
            Box::new(search::SearchEngine::new(depth, Duration::from_millis(ms)))
        }
        EngineKind::AlphaZero => {
            let sims = match d {
                1 => 40,
                2 => 100,
                3 => 200,
                4 => 400,
                _ => 800,
            };
            Box::new(alphazero::AlphaZeroEngine::new(sims))
        }
    }
}
