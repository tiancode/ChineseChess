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
use crate::moves::legal_moves;

const INF: i32 = 32_001;
const MATE: i32 = 30_000;
const MATE_THRESHOLD: i32 = MATE - 256;
const MAX_PLY: usize = 64;

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

fn base_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::General => 0, // both always present -> handled via mate
        PieceKind::Chariot => 1000,
        PieceKind::Cannon => 500,
        PieceKind::Horse => 450,
        PieceKind::Advisor => 200,
        PieceKind::Elephant => 200,
        PieceKind::Soldier => 100,
    }
}

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
    rng: u64,
    // Set per search:
    deadline: Instant,
    nodes: u64,
    aborted: bool,
    /// How many times each position occurred in the actual game so far.
    game_counts: HashMap<u64, u32>,
    path: Vec<u64>,
}

impl SearchEngine {
    pub fn new(max_depth: u8, budget: Duration) -> Self {
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
            rng: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
            deadline: Instant::now(),
            nodes: 0,
            aborted: false,
            game_counts: HashMap::new(),
            path: Vec::with_capacity(MAX_PLY),
        }
    }

    /// Depth-bounded engine with an effectively unlimited clock (for tests).
    #[cfg(test)]
    pub fn fixed_depth(depth: u8) -> Self {
        SearchEngine::new(depth, Duration::from_secs(3600))
    }

    /// Nodes visited by the last `best_move` (benchmarking).
    #[cfg(test)]
    pub fn nodes_searched(&self) -> u64 {
        self.nodes
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

/// Order: TT move, then captures by MVV-LVA, then killers, then the rest.
fn order_moves(
    board: &Board,
    moves: &mut [Move],
    tt_move: Option<Move>,
    killers: &[Option<Move>; 2],
) {
    moves.sort_by_key(|m| {
        if Some(*m) == tt_move {
            return -1_000_000;
        }
        if let Some(victim) = board.get(m.to) {
            let attacker = board.get(m.from).map(|p| base_value(p.kind)).unwrap_or(0);
            return -(100_000 + base_value(victim.kind) * 16 - attacker);
        }
        if Some(*m) == killers[0] || Some(*m) == killers[1] {
            return -50_000;
        }
        0
    });
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
        order_moves(board, &mut caps, None, &[None, None]);
        for mv in caps {
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

    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        board: &mut Board,
        side: Color,
        depth: i16,
        mut alpha: i32,
        beta: i32,
        ply: usize,
    ) -> i32 {
        self.nodes += 1;
        if self.time_up() {
            return 0;
        }

        let key = zkey(board, side);

        // Threefold-repetition draw: count this position's prior occurrences
        // in the real game plus on the current search line; if reaching it
        // again makes three, it is a draw.
        if ply > 0 {
            let in_game = self.game_counts.get(&key).copied().unwrap_or(0) as usize;
            let in_path = self.path.iter().filter(|&&k| k == key).count();
            if in_game + in_path >= 2 {
                return 0;
            }
        }

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

        let tt_move = self.tt.probe(key).and_then(|e| e.best);
        let killers = if ply < MAX_PLY {
            self.killers[ply]
        } else {
            [None, None]
        };
        order_moves(board, &mut moves, tt_move, &killers);

        self.path.push(key);
        let mut best_score = -INF;
        let mut best_move = None;
        let mut first = true;

        for mv in moves {
            let captured = board.make(mv);
            let score = if first {
                -self.search(board, side.opposite(), depth - 1, -beta, -alpha, ply + 1)
            } else {
                // PVS: null-window probe, re-search on a fail-high.
                let s = -self.search(
                    board,
                    side.opposite(),
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                );
                if s > alpha && s < beta {
                    -self.search(board, side.opposite(), depth - 1, -beta, -alpha, ply + 1)
                } else {
                    s
                }
            };
            board.unmake(mv, captured);
            first = false;

            if self.aborted {
                self.path.pop();
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
                // Beta cutoff: remember quiet killer moves.
                if board.get(mv.to).is_none() && ply < MAX_PLY {
                    let k = &mut self.killers[ply];
                    if k[0] != Some(mv) {
                        k[1] = k[0];
                        k[0] = Some(mv);
                    }
                }
                break;
            }
        }
        self.path.pop();

        let bound = if best_score <= alpha_orig {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt
            .store(key, depth, adjust_to_tt(best_score, ply), bound, best_move);
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
        self.deadline = Instant::now() + self.budget;
        self.nodes = 0;
        self.aborted = false;
        self.killers = [[None; 2]; MAX_PLY];

        let mut best = root_moves[0];
        let mut best_pool = vec![root_moves[0]];

        // Iterative deepening: each completed depth refines the move and
        // seeds the next iteration's ordering via the transposition table.
        for depth in 1..=self.max_depth as i16 {
            let key = zkey(&root_board, side);
            let tt_move = self.tt.probe(key).and_then(|e| e.best);
            let mut moves = root_moves.clone();
            order_moves(&root_board, &mut moves, tt_move.or(Some(best)), &[None, None]);

            // Root is searched with a full window per move so the scores are
            // exact: this keeps the equal-best pool honest (a null-window /
            // PVS probe returns fail-low bounds that spuriously look "equal").
            // Alpha-beta + PVS still prune inside the deep interior search.
            let mut best_score = -INF;
            let mut local_best = moves[0];
            let mut pool: Vec<Move> = Vec::new();
            let mut completed = true;

            for mv in moves.iter() {
                let captured = root_board.make(*mv);
                let score =
                    -self.search(&mut root_board, side.opposite(), depth - 1, -INF, INF, 1);
                root_board.unmake(*mv, captured);

                if self.aborted {
                    completed = false;
                    break;
                }
                if score > best_score {
                    best_score = score;
                    local_best = *mv;
                    pool.clear();
                    pool.push(*mv);
                } else if score == best_score {
                    pool.push(*mv);
                }
            }

            if completed {
                best = local_best;
                best_pool = if pool.is_empty() { vec![local_best] } else { pool };
                self.tt
                    .store(key, depth, best_score, Bound::Exact, Some(local_best));
            } else {
                break; // ran out of time; keep the last completed depth
            }
        }

        // Vary play among equally-best moves.
        let pick = (self.next_rand() as usize) % best_pool.len();
        Some(best_pool.get(pick).copied().unwrap_or(best))
    }
}
