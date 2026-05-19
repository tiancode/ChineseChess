//! Pluggable AI interface.
//!
//! To swap in a stronger engine later, implement [`Engine`] and return it from
//! [`make_engine`]; nothing else in the program needs to change.

use std::time::Duration;

use crate::board::{Color, Move};
use crate::game::GameState;

pub mod alphazero;
pub mod book;
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
            // The depth cap is set well above what the clock allows at the
            // higher levels, so the *time budget* is the real limiter there:
            // iterative deepening keeps going deeper until the deadline. This
            // is what makes level 5 actually think ~1.5 min instead of
            // finishing a shallow depth in a few seconds.
            // (depth cap, time ms, TT size 2^bits). Bigger budgets get a
            // bigger table (~48 B/entry: 2^18≈12 MB … 2^23≈400 MB) so deep
            // thinking is not throttled by TT thrash.
            let (depth, ms, tt_bits) = match d {
                1 => (4, 200, 18),
                2 => (6, 800, 20),
                3 => (10, 3_000, 21),
                4 => (18, 15_000, 22),
                _ => (28, 90_000, 23),
            };
            // Lazy-SMP thread count: harder levels (which actually think long
            // enough to benefit) get more cores, capped to what's available.
            let par = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let threads = match d {
                1 | 2 => 1,
                3 => 2,
                4 => 4,
                _ => 8,
            }
            .min(par)
            .max(1);
            Box::new(
                search::SearchEngine::new_tuned(
                    depth,
                    Duration::from_millis(ms),
                    search::Tuning::full(),
                    tt_bits,
                )
                .with_threads(threads),
            )
        }
        EngineKind::AlphaZero => {
            // AlphaZero strength scales with the MCTS simulation count. Unlike
            // AlphaBeta there is no time budget: wall-clock per move is sims ×
            // sidecar inference speed (CPU/GPU, model size), so the higher
            // levels are "stronger", not pinned to a fixed think time.
            let sims = match d {
                1 => 100,
                2 => 300,
                3 => 800,
                4 => 2_000,
                _ => 4_000,
            };
            Box::new(alphazero::AlphaZeroEngine::new(sims))
        }
    }
}
