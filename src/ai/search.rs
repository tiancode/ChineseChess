//! Stronger engine: iterative deepening + alpha-beta (PVS) with a
//! transposition table, quiescence search, MVV-LVA + killer move ordering,
//! Xiangqi piece-square evaluation, and repetition awareness.
//!
//! It is built behind the same [`Engine`] trait as before, so the UI,
//! threading, and difficulty wiring are unchanged.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::Engine;
use crate::board::*;
use crate::game::GameState;
use crate::moves::{in_check, legal_moves, pseudo_moves};
// Single source of truth for piece values: the engine's material term and the
// repetition rules' chase test must agree on the scale.
use crate::moves::piece_value as base_value;

const INF: i32 = 32_001;
const MATE: i32 = 30_000;
const MATE_THRESHOLD: i32 = MATE - 256;
const MAX_PLY: usize = 64;
/// Butterfly-history saturation bound. The gravity update keeps every entry
/// strictly inside (-HISTORY_MAX, HISTORY_MAX), so quiet ordering scores never
/// collide with the killer / countermove / capture bands.
const HISTORY_MAX: i32 = 1 << 20;

// --- Phase 2 selective-search tuning constants ---
/// Reverse-futility: prune when `eval - RFP_MARGIN*depth >= beta`.
const RFP_MARGIN: i32 = 80;
const RFP_MAX_DEPTH: i16 = 6;
/// Razoring: at depth ≤ 2, if `eval + RAZOR_MARGIN < alpha`, verify by qsearch.
const RAZOR_MARGIN: i32 = 320;
const RAZOR_MAX_DEPTH: i16 = 2;
/// Late move pruning applies at `depth ≤ LMP_MAX_DEPTH`.
const LMP_MAX_DEPTH: i16 = 3;

/// Plies (half-moves) into the game that still count as "opening". While the
/// game history is shorter than this, the root widens its equal-best pool so
/// games do not always start with the identical line.
const OPENING_PLIES: usize = 8;
/// In the opening, any root move scoring within this many centipawns of the
/// best joins the random-pick pool (a soldier is worth 100). Picked moves are
/// near-best, so play stays sound; afterwards an exact tie is required.
const OPENING_MARGIN: i32 = 80;

/// Feature toggles for the search heuristics. Real play uses [`Tuning::full`];
/// the A/B self-play harness pits it against [`Tuning::baseline`] (the
/// pre-upgrade behaviour) to measure each phase's Elo before it is accepted.
#[derive(Clone, Copy)]
pub struct Tuning {
    /// Order quiet moves by butterfly history (else a flat 0, original order).
    pub history: bool,
    /// Use the countermove reply band in ordering.
    pub countermove: bool,
    /// Classify captures winning/losing by SEE (else all captures rank ahead
    /// of quiets by MVV-LVA, as the original engine did).
    pub see_order: bool,
    /// Drop SEE-losing captures in quiescence.
    pub see_qprune: bool,
    /// Reverse futility / static null-move pruning at shallow non-PV nodes.
    pub rfp: bool,
    /// Razoring: shallow nodes far below alpha drop straight to quiescence.
    pub razor: bool,
    /// Null-move pruning.
    pub nmp: bool,
    /// Late move reductions.
    pub lmr: bool,
    /// Late move pruning (skip the ordered-last quiets at shallow depth).
    pub lmp: bool,
    /// Extend the search one ply on checking moves.
    pub check_ext: bool,
    /// Widen the opening root pool for move variety. Off in the harness so a
    /// match is deterministic and measures true-best play.
    pub variety: bool,
}

impl Tuning {
    /// Everything the upgraded engine knows (production default).
    pub fn full() -> Self {
        Tuning {
            history: true,
            countermove: true,
            see_order: true,
            see_qprune: true,
            rfp: true,
            razor: true,
            nmp: true,
            lmr: true,
            lmp: true,
            check_ext: true,
            variety: true,
        }
    }

    /// The pre-upgrade engine: TT move + MVV-LVA captures + killers only, all
    /// captures searched in quiescence. The fixed reference opponent (used
    /// only by the `#[ignore]` A/B harness, hence allowed-dead in normal builds).
    #[allow(dead_code)]
    pub fn baseline() -> Self {
        Tuning {
            history: false,
            countermove: false,
            see_order: false,
            see_qprune: false,
            rfp: false,
            razor: false,
            nmp: false,
            lmr: false,
            lmp: false,
            check_ext: false,
            variety: true,
        }
    }
}

// ----------------------------------------------------------------------------
// Zobrist hashing (full recompute per node: O(90), simple and bug-free).
// ----------------------------------------------------------------------------

fn zobrist() -> &'static ([[u64; CELLS]; 14], u64) {
    static Z: OnceLock<([[u64; CELLS]; 14], u64)> = OnceLock::new();
    Z.get_or_init(|| {
        // Deterministic splitmix64 fill so runs are reproducible.
        let mut s: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut table = [[0u64; CELLS]; 14];
        for piece in table.iter_mut() {
            for sq in piece.iter_mut() {
                *sq = next();
            }
        }
        (table, next())
    })
}

#[inline]
fn piece_index(p: Piece) -> usize {
    (p.kind as usize) * 2 + if p.color == Color::Red { 0 } else { 1 }
}

fn zkey(board: &Board, side: Color) -> u64 {
    let (table, side_key) = zobrist();
    let mut k = 0u64;
    for (sq, cell) in board.cells.iter().enumerate() {
        if let Some(p) = cell {
            k ^= table[piece_index(*p)][sq];
        }
    }
    if side == Color::Black {
        k ^= *side_key;
    }
    k
}

// ----------------------------------------------------------------------------
// Transposition table.
// ----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    depth: i16,
    score: i32,
    bound: Bound,
    best: Option<Move>,
}

struct Tt {
    slots: Vec<Option<TtEntry>>,
    mask: usize,
}

impl Tt {
    fn new(bits: u32) -> Self {
        let size = 1usize << bits;
        Tt {
            slots: vec![None; size],
            mask: size - 1,
        }
    }

    fn probe(&self, key: u64) -> Option<TtEntry> {
        match self.slots[(key as usize) & self.mask] {
            Some(e) if e.key == key => Some(e),
            _ => None,
        }
    }

    fn store(&mut self, key: u64, depth: i16, score: i32, bound: Bound, best: Option<Move>) {
        let slot = &mut self.slots[(key as usize) & self.mask];
        // Depth-preferred replacement.
        if let Some(e) = slot {
            if e.key == key && e.depth > depth {
                return;
            }
        }
        *slot = Some(TtEntry { key, depth, score, bound, best });
    }
}

// ----------------------------------------------------------------------------
// Evaluation.
// ----------------------------------------------------------------------------

// Piece-square tables in Red orientation: index [rank][file], rank 9 is Red's
// home edge, rank 0 is Black's. Black pieces read the rank-mirrored entry.
#[rustfmt::skip]
const PST_SOLDIER: [[i32; 9]; 10] = [
    [  9,  9,  9, 11, 13, 11,  9,  9,  9],
    [ 19, 24, 34, 42, 44, 42, 34, 24, 19],
    [ 19, 24, 32, 37, 37, 37, 32, 24, 19],
    [ 19, 23, 27, 29, 30, 29, 27, 23, 19],
    [ 14, 18, 20, 27, 29, 27, 20, 18, 14],
    [  7,  0, 13,  0, 16,  0, 13,  0,  7],
    [  7,  0,  7,  0, 15,  0,  7,  0,  7],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
];
#[rustfmt::skip]
const PST_HORSE: [[i32; 9]; 10] = [
    [  0,  2,  4,  6,  6,  6,  4,  2,  0],
    [  2,  6, 10, 12, 12, 12, 10,  6,  2],
    [  4, 10, 14, 16, 18, 16, 14, 10,  4],
    [  6, 12, 16, 20, 22, 20, 16, 12,  6],
    [  6, 14, 18, 22, 24, 22, 18, 14,  6],
    [  6, 14, 18, 22, 24, 22, 18, 14,  6],
    [  6, 12, 16, 20, 22, 20, 16, 12,  6],
    [  4, 10, 14, 16, 18, 16, 14, 10,  4],
    [  2,  6, 10, 12, 12, 12, 10,  6,  2],
    [  0,  2,  4,  6,  6,  6,  4,  2,  0],
];
#[rustfmt::skip]
const PST_CANNON: [[i32; 9]; 10] = [
    [  6,  4,  0, -2, -4, -2,  0,  4,  6],
    [  2,  2,  0, -4, -6, -4,  0,  2,  2],
    [  2,  2,  0, -2, -4, -2,  0,  2,  2],
    [  0,  4,  6,  6,  8,  6,  6,  4,  0],
    [  0,  2,  6,  8, 10,  8,  6,  2,  0],
    [  0,  2,  6,  8, 10,  8,  6,  2,  0],
    [  0,  4,  8, 10, 12, 10,  8,  4,  0],
    [  2,  4,  6,  8,  8,  8,  6,  4,  2],
    [  2,  2,  4,  6,  6,  6,  4,  2,  2],
    [  4,  4,  4,  6,  8,  6,  4,  4,  4],
];
#[rustfmt::skip]
const PST_CHARIOT: [[i32; 9]; 10] = [
    [ 12, 14, 14, 16, 16, 16, 14, 14, 12],
    [ 14, 16, 16, 20, 22, 20, 16, 16, 14],
    [ 12, 14, 14, 18, 20, 18, 14, 14, 12],
    [ 12, 16, 16, 20, 22, 20, 16, 16, 12],
    [ 12, 14, 14, 18, 20, 18, 14, 14, 12],
    [ 12, 16, 16, 18, 20, 18, 16, 16, 12],
    [ 10, 14, 14, 16, 18, 16, 14, 14, 10],
    [ 10, 12, 12, 16, 18, 16, 12, 12, 10],
    [  8, 10, 12, 16, 16, 16, 12, 10,  8],
    [ 10, 12, 12, 16, 16, 16, 12, 12, 10],
];

fn pst(kind: PieceKind, color: Color, sq: usize) -> i32 {
    let f = file_of(sq) as usize;
    let r = rank_of(sq) as usize;
    let rr = if color == Color::Red { r } else { 9 - r };
    match kind {
        PieceKind::Soldier => PST_SOLDIER[rr][f],
        PieceKind::Horse => PST_HORSE[rr][f],
        PieceKind::Cannon => PST_CANNON[rr][f],
        PieceKind::Chariot => PST_CHARIOT[rr][f],
        _ => 0, // advisor / elephant / general: confined; base value only
    }
}

/// Static evaluation from the perspective of `side` (higher = better).
fn evaluate(board: &Board, side: Color) -> i32 {
    let mut score = 0;
    for (sq, cell) in board.cells.iter().enumerate() {
        if let Some(p) = cell {
            let v = base_value(p.kind) + pst(p.kind, p.color, sq);
            score += if p.color == side { v } else { -v };
        }
    }
    score
}

// ----------------------------------------------------------------------------
// Engine.
// ----------------------------------------------------------------------------

pub struct SearchEngine {
    max_depth: u8,
    budget: Duration,
    tt: Tt,
    killers: [[Option<Move>; 2]; MAX_PLY],
    /// Butterfly history: quiet-move cutoff counters indexed [from][to].
    /// Boxed so the ~32 KB table is heap-allocated, not on the search stack.
    history: Box<[[i32; CELLS]; CELLS]>,
    /// Countermove table: indexed by the opponent's previous move [from][to],
    /// the quiet reply that last produced a cutoff against it.
    counter: Box<[[Option<Move>; CELLS]; CELLS]>,
    tuning: Tuning,
    rng: u64,
    // Set per search:
    deadline: Instant,
    nodes: u64,
    aborted: bool,
    /// How many times each position occurred in the actual game so far.
    game_counts: HashMap<u64, u32>,
    path: Vec<u64>,
    /// Parallel to `path`: did the move that produced that position give
    /// check? Used to score a repetition reached by perpetual check (长将)
    /// as a loss for the checking side, instead of a flat draw.
    path_check: Vec<bool>,
    /// Set by `search` on return to signal that the value just produced
    /// depended on the *path* (a repetition score), not the position alone.
    /// Such values must not be cached in the position-keyed TT, or they would
    /// be served for a different path (graph-history interaction).
    path_dep: bool,
}

impl SearchEngine {
    pub fn new(max_depth: u8, budget: Duration) -> Self {
        SearchEngine::new_tuned(max_depth, budget, Tuning::full())
    }

    /// Like [`Self::new`] but with explicit heuristic toggles (A/B harness).
    pub fn new_tuned(max_depth: u8, budget: Duration, tuning: Tuning) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let seed = nanos ^ 0x9E37_79B9_7F4A_7C15;
        SearchEngine {
            max_depth: max_depth.max(1),
            budget,
            tt: Tt::new(19), // ~512k entries
            killers: [[None; 2]; MAX_PLY],
            history: Box::new([[0; CELLS]; CELLS]),
            counter: Box::new([[None; CELLS]; CELLS]),
            tuning,
            rng: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
            deadline: Instant::now(),
            nodes: 0,
            aborted: false,
            game_counts: HashMap::new(),
            path: Vec::with_capacity(MAX_PLY),
            path_check: Vec::with_capacity(MAX_PLY),
            path_dep: false,
        }
    }

    /// Depth-bounded engine with an effectively unlimited clock (for tests).
    #[cfg(test)]
    pub fn fixed_depth(depth: u8) -> Self {
        SearchEngine::new(depth, Duration::from_secs(3600))
    }

    /// Pin the variety RNG so an A/B match is byte-for-byte reproducible.
    #[cfg(test)]
    pub fn set_seed(&mut self, s: u64) {
        self.rng = if s == 0 { 0x9E37_79B9_7F4A_7C15 } else { s };
    }

    /// Nodes visited by the last `best_move` (benchmarking).
    #[cfg(test)]
    pub fn nodes_searched(&self) -> u64 {
        self.nodes
    }

    /// Drive [`Self::repetition_score`] with a synthetic search path so the
    /// (intricate) cycle parity logic can be unit-tested deterministically
    /// without reaching a real perpetual inside the search.
    #[cfg(test)]
    pub fn repetition_score_probe(
        &mut self,
        path: &[u64],
        path_check: &[bool],
        key: u64,
        ply: usize,
        here_check: bool,
    ) -> i32 {
        self.path = path.to_vec();
        self.path_check = path_check.to_vec();
        self.repetition_score(key, ply, here_check)
    }

    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn time_up(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        // Check the clock only occasionally; it is relatively expensive.
        if self.nodes & 2047 == 0 && Instant::now() >= self.deadline {
            self.aborted = true;
        }
        self.aborted
    }

    /// Count how many times every position occurred in the actual game, so
    /// the search can detect a *threefold* repetition (rather than treating
    /// any transient transposition as an immediate draw).
    fn build_repetition(&mut self, state: &GameState) {
        self.game_counts.clear();
        let mut b = state.board;
        for (mv, captured) in state.history.iter().rev() {
            b.unmake(*mv, *captured);
        }
        let n = state.history.len();
        let mut side = if n.is_multiple_of(2) {
            state.side_to_move
        } else {
            state.side_to_move.opposite()
        };
        *self.game_counts.entry(zkey(&b, side)).or_insert(0) += 1;
        for (mv, _) in state.history.iter() {
            b.make(*mv);
            side = side.opposite();
            *self.game_counts.entry(zkey(&b, side)).or_insert(0) += 1;
        }
    }
}

/// MVV-LVA score for a capture (victim heavily outweighs attacker).
#[inline]
fn mvv_lva(board: &Board, m: Move) -> i32 {
    let victim = board.get(m.to).map(|p| base_value(p.kind)).unwrap_or(0);
    let attacker = board.get(m.from).map(|p| base_value(p.kind)).unwrap_or(0);
    victim * 16 - attacker
}

/// History "gravity" update: pulls the entry toward ±HISTORY_MAX by `bonus`
/// (positive on a cutoff, negative as a malus) while staying bounded, so no
/// periodic rescale is needed.
#[inline]
fn hist_bump(h: &mut i32, bonus: i32) {
    *h += bonus - *h * bonus.abs() / HISTORY_MAX;
}

/// True if `side` still has a chariot, cannon, or horse — i.e. real attacking
/// power, so a null move cannot be a zugzwang trap. (Bare K+A/E has none.)
fn has_attacking_material(board: &Board, side: Color) -> bool {
    board.cells.iter().flatten().any(|p| {
        p.color == side
            && matches!(
                p.kind,
                PieceKind::Chariot | PieceKind::Cannon | PieceKind::Horse
            )
    })
}

/// LMR reduction (in plies) for a late quiet move. Grows with depth and move
/// index; PV nodes reduce one ply less.
#[inline]
fn lmr_reduction(depth: i16, move_idx: usize, is_pv: bool) -> i16 {
    let mut r: i16 = 1;
    if depth >= 6 {
        r += 1;
    }
    if move_idx >= 6 {
        r += 1;
    }
    if move_idx >= 12 {
        r += 1;
    }
    if is_pv {
        r -= 1;
    }
    r.max(1)
}

/// Quiet-move count at a node past which late move pruning kicks in.
#[inline]
fn lmp_count(depth: i16) -> usize {
    (3 + depth * depth) as usize
}

/// Least-valuable pseudo attacker of `target` for `side`, as the capture move
/// that brings it there. Uses the authoritative `pseudo_moves` generator so
/// Xiangqi-specific reachability (horse legs, cannon screens, blocked sliders)
/// is exact and there is no second, divergent attack table to maintain.
fn least_valuable_attacker(board: &Board, target: usize, side: Color) -> Option<Move> {
    let mut best: Option<(i32, Move)> = None;
    for m in pseudo_moves(board, side) {
        if m.to != target {
            continue;
        }
        if let Some(p) = board.get(m.from) {
            let v = base_value(p.kind);
            if best.is_none_or(|(bv, _)| v < bv) {
                best = Some((v, m));
            }
        }
    }
    best.map(|(_, m)| m)
}

/// Recursive half of SEE: `side` is on move and may capture the piece now
/// standing on `target` with its least valuable attacker, or decline (max 0).
/// Restores the board exactly (every `make` is paired with an `unmake`).
fn see_recapture(board: &mut Board, target: usize, side: Color) -> i32 {
    let Some(mv) = least_valuable_attacker(board, target, side) else {
        return 0;
    };
    let victim = base_value(board.get(target).expect("occupied during SEE").kind);
    let cap = board.make(mv);
    let val = (victim - see_recapture(board, target, side.opposite())).max(0);
    board.unmake(mv, cap);
    val
}

/// Static Exchange Evaluation of a capture: material the mover nets if the
/// full capture sequence on `mv.to` is played out with least-valuable
/// attackers, each side free to stop. Pins/self-check are ignored (standard
/// SEE); piece values come from `base_value` (single source of truth).
fn see(board: &mut Board, mv: Move) -> i32 {
    let Some(victim) = board.get(mv.to) else {
        return 0; // not a capture
    };
    let Some(mover) = board.get(mv.from).map(|p| p.color) else {
        return 0;
    };
    let vval = base_value(victim.kind);
    let cap = board.make(mv);
    let s = vval - see_recapture(board, mv.to, mover.opposite());
    board.unmake(mv, cap);
    s
}

/// Order quiescence captures by MVV-LVA (SEE pruning is applied in the loop).
fn order_captures(board: &Board, moves: &mut [Move]) {
    moves.sort_by_key(|m| -mvv_lva(board, *m));
}

#[inline]
fn captures_only(board: &Board, moves: Vec<Move>) -> Vec<Move> {
    moves
        .into_iter()
        .filter(|m| board.get(m.to).is_some())
        .collect()
}

impl SearchEngine {
    fn quiescence(&mut self, board: &mut Board, side: Color, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if self.time_up() {
            return 0;
        }
        let stand = evaluate(board, side);
        if stand >= beta {
            return beta;
        }
        if stand > alpha {
            alpha = stand;
        }

        let pseudo = legal_moves(board, side);
        let mut caps = captures_only(board, pseudo);
        order_captures(board, &mut caps);
        for mv in caps {
            // Skip captures that lose material by static exchange: they cannot
            // raise alpha above the stand-pat and only inflate the q-tree.
            if self.tuning.see_qprune && see(board, mv) < 0 {
                continue;
            }
            let captured = board.make(mv);
            let score = -self.quiescence(board, side.opposite(), -beta, -alpha);
            board.unmake(mv, captured);
            if self.aborted {
                return 0;
            }
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }

    /// Score a just-detected threefold repetition. If the repeating cycle
    /// (taken from the most recent identical position on the search path) was
    /// a one-sided *perpetual check*, the checking side loses (CCA rule);
    /// otherwise — mutual perpetual check, or anything not all-checks — it is
    /// a draw (0). `here_check` is whether the move into this node checked.
    ///
    /// Only cycles wholly within the search path are classified; a repetition
    /// formed against pre-root game history falls back to a draw here (the
    /// authoritative ruling is `GameState::status`).
    fn repetition_score(&self, key: u64, ply: usize, here_check: bool) -> i32 {
        let m = self.path.len();
        let Some(p) = (0..m).rev().find(|&t| self.path[t] == key) else {
            return 0; // repetition via game history only: treat as draw
        };
        let k = m - p; // cycle length in plies (even, >= 2)
        // Cycle move j (0..k) is by `side` when j is even, else the opponent.
        // Its check flag is path_check at the destination position, except the
        // final move into this node, which is `here_check`.
        let mut all_self = true; // every `side` move in the cycle is a check
        let mut all_opp = true; // every opponent move in the cycle is a check
        for j in 0..k {
            let checks = if j < k - 1 {
                self.path_check[p + 1 + j]
            } else {
                here_check
            };
            if j % 2 == 0 {
                all_self &= checks;
            } else {
                all_opp &= checks;
            }
        }
        match (all_self, all_opp) {
            (true, true) => 0,                       // mutual perpetual check
            (true, false) => -MATE + ply as i32,     // `side` perpetually checks: loses
            (false, true) => MATE - ply as i32,      // opponent perpetually checks: loses
            (false, false) => 0,                     // not perpetual check
        }
    }

    /// Full move ordering for an interior node: TT move, winning/equal
    /// captures (SEE ≥ 0) by MVV-LVA, the two killers, the countermove for
    /// `prev`, quiet moves by butterfly history, then losing captures last.
    fn order(
        &self,
        board: &mut Board,
        moves: &mut [Move],
        tt_move: Option<Move>,
        ply: usize,
        prev: Option<Move>,
    ) {
        let killers = if ply < MAX_PLY {
            self.killers[ply]
        } else {
            [None, None]
        };
        let cm = if self.tuning.countermove {
            prev.and_then(|p| self.counter[p.from][p.to])
        } else {
            None
        };
        let mut keyed: Vec<(i32, Move)> = Vec::with_capacity(moves.len());
        for &m in moves.iter() {
            let s = if Some(m) == tt_move {
                -3_000_000
            } else if board.get(m.to).is_some() {
                let mvv = mvv_lva(board, m);
                if self.tuning.see_order {
                    let sx = see(board, m);
                    if sx >= 0 {
                        -2_000_000 - mvv // winning / equal capture
                    } else {
                        2_000_000 - sx // losing capture: dead last
                    }
                } else {
                    -2_000_000 - mvv // original: all captures ahead of quiets
                }
            } else if Some(m) == killers[0] {
                -1_900_000
            } else if Some(m) == killers[1] {
                -1_800_000
            } else if Some(m) == cm {
                -1_700_000
            } else if self.tuning.history {
                -self.history[m.from][m.to]
            } else {
                0
            };
            keyed.push((s, m));
        }
        keyed.sort_by_key(|&(s, _)| s);
        for (slot, (_, m)) in moves.iter_mut().zip(keyed) {
            *slot = m;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        board: &mut Board,
        side: Color,
        depth: i16,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        prev: Option<Move>,
    ) -> i32 {
        self.nodes += 1;
        if self.time_up() {
            self.path_dep = false;
            return 0;
        }

        let key = zkey(board, side);
        // Did the move that led to this node give check (i.e. is the side to
        // move now in check)? Recorded into `path_check` for perpetual-check
        // scoring of repetitions.
        let here_check = ply > 0 && in_check(board, side);

        // Threefold repetition: this position plus two prior occurrences (in
        // the real game and/or on the current search line). Under CCA rules a
        // repetition reached by *perpetual check* is a loss for the checking
        // side, not a draw. (Perpetual chase is left to the game-level rule;
        // the search treats it as a draw — see module notes.)
        if ply > 0 {
            let in_game = self.game_counts.get(&key).copied().unwrap_or(0) as usize;
            let in_path = self.path.iter().filter(|&&k| k == key).count();
            if in_game + in_path >= 2 {
                self.path_dep = true; // a repetition score is path-dependent
                return self.repetition_score(key, ply, here_check);
            }
        }
        // Default for every non-repetition return below (TT hit, quiescence,
        // mate). The normal-completion path overwrites this with the
        // subtree's aggregate flag.
        self.path_dep = false;

        let alpha_orig = alpha;
        if let Some(e) = self.tt.probe(key) {
            if e.depth >= depth {
                let s = adjust_from_tt(e.score, ply);
                match e.bound {
                    Bound::Exact => return s,
                    Bound::Lower if s >= beta => return s,
                    Bound::Upper if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        if depth <= 0 {
            return self.quiescence(board, side, alpha, beta);
        }

        let mut moves = legal_moves(board, side);
        if moves.is_empty() {
            // No legal reply: a loss in Xiangqi (mate or stalemate).
            return -MATE + ply as i32;
        }

        // PV vs zero-window node: PVS searches the first child with a full
        // window and the rest with a null window, so `beta - alpha > 1`
        // identifies a PV node. Selective pruning is restricted to non-PV
        // nodes that are not in check and outside the mate zone.
        let is_pv = beta - alpha > 1;
        let mate_zone = alpha <= -MATE_THRESHOLD || beta >= MATE_THRESHOLD;
        let prunable = !is_pv && !here_check && !mate_zone;
        let eval = if prunable {
            evaluate(board, side)
        } else {
            0
        };

        // Reverse futility / static null move: so far ahead that even giving
        // back `RFP_MARGIN` per remaining ply still beats beta.
        if self.tuning.rfp
            && prunable
            && depth <= RFP_MAX_DEPTH
            && eval - RFP_MARGIN * depth as i32 >= beta
        {
            return eval;
        }

        // Razoring: so far below alpha at shallow depth that only a tactical
        // shot could save it — let quiescence confirm a fail-low.
        if self.tuning.razor && prunable && depth <= RAZOR_MAX_DEPTH && eval + RAZOR_MARGIN < alpha
        {
            let q = self.quiescence(board, side, alpha, beta);
            if self.aborted {
                self.path_dep = false;
                return 0;
            }
            if q < alpha {
                return q;
            }
        }

        // Null-move pruning: pass the move; if the opponent still cannot reach
        // beta with a reduced search, this node is a fail-high. The null child
        // pushes its *own* (opponent-to-move) key, so the path stays balanced;
        // a repetition-tainted null result is distrusted via `path_dep` rather
        // than trusted as a cutoff (the CCA invariant — see module notes).
        if self.tuning.nmp
            && prunable
            && depth >= 3
            && eval >= beta
            && has_attacking_material(board, side)
        {
            let r = 2 + depth / 4;
            let nd = (depth - 1 - r).max(0);
            let nscore =
                -self.search(board, side.opposite(), nd, -beta, -beta + 1, ply + 1, None);
            let null_dep = self.path_dep;
            self.path_dep = false;
            if self.aborted {
                return 0;
            }
            if !null_dep && nscore >= beta {
                // A null search cannot prove a real mate; clamp to beta there.
                return if nscore >= MATE_THRESHOLD { beta } else { nscore };
            }
        }

        let tt_move = self.tt.probe(key).and_then(|e| e.best);
        self.order(board, &mut moves, tt_move, ply, prev);

        self.path.push(key);
        self.path_check.push(here_check);
        let mut best_score = -INF;
        let mut best_move = None;
        let mut first = true;
        let mut move_idx: usize = 0;
        // Quiet moves already tried at this node (for the history malus on a
        // later cutoff). Captures are excluded — history is a quiet-move stat.
        let mut quiets: Vec<Move> = Vec::new();
        // OR of every searched child's path-dependence flag. If set, this
        // node's value was influenced by a repetition score and must not be
        // cached in the position-keyed TT.
        let mut subtree_dep = false;

        for mv in moves {
            // `quiet` is decided before the move is made (a piece on the
            // destination ⇒ capture); history/killers and the LMR/LMP gates
            // are quiet-move stats only.
            let quiet = board.get(mv.to).is_none();

            // Late move pruning: at shallow non-PV nodes, once enough moves
            // have been tried, skip the ordered-last quiets. Never while in
            // check, and not before a non-losing score exists (mate defence).
            if self.tuning.lmp
                && prunable
                && quiet
                && depth <= LMP_MAX_DEPTH
                && move_idx >= lmp_count(depth)
                && best_score > -MATE_THRESHOLD
            {
                move_idx += 1;
                continue;
            }

            let captured = board.make(mv);
            let gives_check = in_check(board, side.opposite());
            // Check extension: stay one ply deeper down forcing lines.
            let ext: i16 = if self.tuning.check_ext && gives_check && ply < MAX_PLY {
                1
            } else {
                0
            };
            let new_depth = depth - 1 + ext;

            let score = if first {
                -self.search(board, side.opposite(), new_depth, -beta, -alpha, ply + 1, Some(mv))
            } else {
                // LMR: search late quiet (non-check, non-extended) moves
                // reduced; only re-search at full depth if they beat alpha.
                let reduce = if self.tuning.lmr
                    && quiet
                    && !gives_check
                    && ext == 0
                    && depth >= 3
                    && move_idx >= 3
                {
                    lmr_reduction(depth, move_idx, is_pv)
                } else {
                    0
                };
                let d = (new_depth - reduce).max(1);
                let mut s = -self.search(
                    board,
                    side.opposite(),
                    d,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    Some(mv),
                );
                if reduce > 0 && s > alpha {
                    // Reduced search beat alpha — confirm at full depth.
                    s = -self.search(
                        board,
                        side.opposite(),
                        new_depth,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        Some(mv),
                    );
                }
                if s > alpha && s < beta {
                    // PV: re-search with the full window.
                    s = -self.search(
                        board,
                        side.opposite(),
                        new_depth,
                        -beta,
                        -alpha,
                        ply + 1,
                        Some(mv),
                    );
                }
                s
            };
            board.unmake(mv, captured);
            first = false;
            move_idx += 1;
            // The child set `self.path_dep` on return; fold it in.
            subtree_dep |= self.path_dep;

            if self.aborted {
                self.path.pop();
                self.path_check.pop();
                self.path_dep = false;
                return 0;
            }
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if quiet {
                    if ply < MAX_PLY {
                        let k = &mut self.killers[ply];
                        if k[0] != Some(mv) {
                            k[1] = k[0];
                            k[0] = Some(mv);
                        }
                    }
                    if let Some(p) = prev {
                        self.counter[p.from][p.to] = Some(mv);
                    }
                    let bonus = (depth as i32) * (depth as i32);
                    hist_bump(&mut self.history[mv.from][mv.to], bonus);
                    for q in &quiets {
                        hist_bump(&mut self.history[q.from][q.to], -bonus);
                    }
                }
                break;
            }
            if quiet {
                quiets.push(mv);
            }
        }
        self.path.pop();
        self.path_check.pop();

        let bound = if best_score <= alpha_orig {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        // A repetition-influenced value is path-dependent: never cache it in
        // the position-keyed TT (it would be served on a different path).
        if !subtree_dep {
            self.tt
                .store(key, depth, adjust_to_tt(best_score, ply), bound, best_move);
        }
        self.path_dep = subtree_dep;
        best_score
    }
}

// Mate scores must be stored relative to the node, not the root.
fn adjust_to_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_THRESHOLD {
        score + ply as i32
    } else if score < -MATE_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}
fn adjust_from_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_THRESHOLD {
        score - ply as i32
    } else if score < -MATE_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

impl Engine for SearchEngine {
    fn name(&self) -> String {
        format!("AlphaBeta (depth≤{}, {:?})", self.max_depth, self.budget)
    }

    fn best_move(&mut self, state: &GameState) -> Option<Move> {
        let side = state.side_to_move;
        let mut root_board = state.board;
        let root_moves = legal_moves(&mut root_board, side);
        if root_moves.is_empty() {
            return None;
        }
        if root_moves.len() == 1 {
            return Some(root_moves[0]);
        }

        self.build_repetition(state);
        self.path.clear();
        self.path_check.clear();
        self.deadline = Instant::now() + self.budget;
        self.nodes = 0;
        self.aborted = false;
        self.killers = [[None; 2]; MAX_PLY];
        // Quiet-move ordering stats start fresh each move so play stays
        // deterministic given the position (no carry-over between turns).
        *self.history = [[0; CELLS]; CELLS];
        *self.counter = [[None; CELLS]; CELLS];

        let mut best = root_moves[0];
        let mut best_pool = vec![root_moves[0]];

        // Iterative deepening: each completed depth refines the move and
        // seeds the next iteration's ordering via the transposition table.
        for depth in 1..=self.max_depth as i16 {
            let key = zkey(&root_board, side);
            let tt_move = self.tt.probe(key).and_then(|e| e.best);
            let mut moves = root_moves.clone();
            self.order(&mut root_board, &mut moves, tt_move.or(Some(best)), 0, None);

            // Root is searched with a full window per move so the scores are
            // exact: this keeps the equal-best pool honest (a null-window /
            // PVS probe returns fail-low bounds that spuriously look "equal").
            // Alpha-beta + PVS still prune inside the deep interior search.
            let mut best_score = -INF;
            let mut local_best = moves[0];
            let mut scored: Vec<(Move, i32)> = Vec::with_capacity(moves.len());
            let mut completed = true;
            let mut root_dep = false; // any root line influenced by repetition?

            for mv in moves.iter() {
                let captured = root_board.make(*mv);
                let score = -self.search(
                    &mut root_board,
                    side.opposite(),
                    depth - 1,
                    -INF,
                    INF,
                    1,
                    Some(*mv),
                );
                root_board.unmake(*mv, captured);
                root_dep |= self.path_dep;

                if self.aborted {
                    completed = false;
                    break;
                }
                if score > best_score {
                    best_score = score;
                    local_best = *mv;
                }
                scored.push((*mv, score));
            }

            if completed {
                best = local_best;
                // Equally-best moves give natural variety. In the opening,
                // widen the pool to every move within `OPENING_MARGIN` of the
                // best (near-best, so still sound) so games don't always start
                // identically; afterwards require an exact tie, leaving normal
                // play strength unchanged.
                let cutoff = if self.tuning.variety && state.history.len() < OPENING_PLIES {
                    best_score - OPENING_MARGIN
                } else {
                    best_score
                };
                best_pool = scored
                    .iter()
                    .filter(|(_, s)| *s >= cutoff)
                    .map(|(m, _)| *m)
                    .collect();
                if best_pool.is_empty() {
                    best_pool = vec![local_best];
                }
                if !root_dep {
                    self.tt
                        .store(key, depth, best_score, Bound::Exact, Some(local_best));
                }
            } else {
                break; // ran out of time; keep the last completed depth
            }
        }

        // Vary play among equally-best moves.
        let pick = (self.next_rand() as usize) % best_pool.len();
        Some(best_pool.get(pick).copied().unwrap_or(best))
    }
}

#[cfg(test)]
mod see_tests {
    //! Static Exchange Evaluation on hand-built positions. Coordinates are in
    //! Red orientation (rank 9 = Red home, rank 0 = Black home); piece values
    //! come from `base_value` (Chariot 1000, Soldier 100).
    use super::*;

    fn put(b: &mut Board, f: i32, r: i32, kind: PieceKind, color: Color) {
        b.cells[idx(f, r)] = Some(Piece { kind, color });
    }
    fn mv(f0: i32, r0: i32, f1: i32, r1: i32) -> Move {
        Move { from: idx(f0, r0), to: idx(f1, r1) }
    }

    /// Generals parked in their palaces, off file 4, so they never attack the
    /// exchange square and `pseudo_moves` always has a side to enumerate.
    fn with_generals(b: &mut Board) {
        put(b, 3, 9, PieceKind::General, Color::Red);
        put(b, 5, 0, PieceKind::General, Color::Black);
    }

    #[test]
    fn undefended_capture_wins_the_victim() {
        let mut b = Board::empty();
        with_generals(&mut b);
        put(&mut b, 4, 5, PieceKind::Chariot, Color::Red);
        put(&mut b, 4, 3, PieceKind::Soldier, Color::Black); // undefended
        assert_eq!(see(&mut b, mv(4, 5, 4, 3)), 100);
    }

    #[test]
    fn rook_takes_pawn_defended_by_rook_is_losing() {
        let mut b = Board::empty();
        with_generals(&mut b);
        put(&mut b, 4, 5, PieceKind::Chariot, Color::Red);
        put(&mut b, 4, 3, PieceKind::Soldier, Color::Black);
        put(&mut b, 4, 1, PieceKind::Chariot, Color::Black); // recaptures
        // +100 (pawn) − 1000 (own rook lost) = −900.
        assert_eq!(see(&mut b, mv(4, 5, 4, 3)), -900);
    }

    #[test]
    fn equal_rook_trade_is_zero() {
        let mut b = Board::empty();
        with_generals(&mut b);
        put(&mut b, 4, 5, PieceKind::Chariot, Color::Red);
        put(&mut b, 4, 3, PieceKind::Chariot, Color::Black);
        put(&mut b, 4, 1, PieceKind::Chariot, Color::Black); // recaptures
        assert_eq!(see(&mut b, mv(4, 5, 4, 3)), 0);
    }

    #[test]
    fn recapturer_declines_when_recapture_loses() {
        // Red Sx pawn; Black rook *could* recapture but a Red rook x-rays
        // behind it, so taking back loses the rook — Black declines and the
        // exchange nets Red the pawn.
        let mut b = Board::empty();
        with_generals(&mut b);
        put(&mut b, 4, 4, PieceKind::Soldier, Color::Red);
        put(&mut b, 4, 3, PieceKind::Soldier, Color::Black);
        put(&mut b, 4, 1, PieceKind::Chariot, Color::Black);
        put(&mut b, 4, 0, PieceKind::Chariot, Color::Red); // x-ray defender
        assert_eq!(see(&mut b, mv(4, 4, 4, 3)), 100);
    }

    #[test]
    fn see_restores_the_board_exactly() {
        let mut b = Board::empty();
        with_generals(&mut b);
        put(&mut b, 4, 5, PieceKind::Chariot, Color::Red);
        put(&mut b, 4, 3, PieceKind::Soldier, Color::Black);
        put(&mut b, 4, 1, PieceKind::Chariot, Color::Black);
        let before = b.cells;
        let _ = see(&mut b, mv(4, 5, 4, 3));
        assert!(b.cells == before, "SEE must leave the board untouched");
    }
}
