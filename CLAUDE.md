# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A Chinese Chess (Xiangqi) desktop program: Rust + `egui`/`eframe` GUI with a pluggable AI engine. Code comments are English; all user-facing UI strings are Chinese.

## Commands

```bash
cargo run --release          # play (release is required for the AI to be responsive)
cargo test                   # full suite (~22 tests: rules/perft, engine tactics, UI logic)
cargo test <name>            # single test, e.g. `cargo test perft_matches_known_values`
cargo clippy --all-targets   # expected to be warning-free; keep it that way
cargo test --release engine_benchmark -- --ignored --nocapture   # node-count benchmark
```

The release profile sets `lto = true` / `opt-level = 3`; debug-mode AI is much slower.

## Core invariant: the coordinate system

Every module depends on this — get it wrong and bugs are subtle:

- 9 files (`x`/file `0..=8`) × 10 ranks (`y`/rank `0..=9`); `index = rank * 9 + file`.
- Rank 0 is the **top** (Black's home); rank 9 is the **bottom** (Red's home).
- **Red moves first and sits at the bottom**, so Red soldiers advance toward *smaller* ranks (`forward(Red) == -1`).
- Helpers (`idx`, `file_of`, `rank_of`, `in_palace`, `crossed_river`, `own_half`, `forward`) live in `board.rs`; use them rather than recomputing.

## Architecture

Data/rules layer (`board.rs`, `moves.rs`, `game.rs`) is pure and engine/UI-agnostic.

- `Board::make`/`unmake` return/restore the captured piece for cheap reversible application — used both by the game and inside the search.
- `moves.rs`: `pseudo_moves` is *also* the attack generator (`is_attacked`/`in_check` run it for the opponent). `legal_moves` filters pseudo-moves by make → check self-check **and** the flying-general rule (`generals_face`) → unmake.
- `game.rs::GameState` owns turn tracking, undo `history`, move `log` notation, and end-game detection. Xiangqi rule: no legal move ⇒ the side to move *loses* whether checkmated or stalemated. Threefold repetition is scored as a **draw** — a deliberate simplification; real Xiangqi punishes perpetual check/chase as a loss, and that is intentionally not enforced.

### Two independent, non-shared hashing schemes

This trips people up. They are deliberately separate and **both full-recompute** (no incremental Zobrist update — correctness over speed):

- `game.rs::position_hash` — FNV-1a, drives `GameState`'s threefold-repetition detection.
- `ai/search.rs::zkey` — Zobrist, drives the transposition table *and* the engine's own in-search repetition counting (`build_repetition` replays game history into `game_counts`; `path` tracks the current search line).

Changing one does not affect the other.

### Pluggable engine

`ai/mod.rs` defines `trait Engine { name(); best_move(&GameState) -> Option<Move> }`. `make_engine(difficulty, perspective)` is the **single** place to swap engines; it maps difficulty 1–5 to `(max_depth, time_budget)`. The UI, threading, and legality checks all adapt automatically — to add a stronger engine, implement `Engine` and return it from `make_engine`, change nothing else.

`ai/search.rs` is the bundled engine: iterative deepening + alpha-beta/PVS, depth-preferred transposition table, quiescence search, MVV-LVA + killer move ordering, piece-square evaluation. Mate scores are stored TT-relative to the node via `adjust_to_tt`/`adjust_from_tt` (not relative to the root). The root searches every move with a full window (PVS only in the interior) so the equal-best pool used for move variety stays honest.

### UI / threading (`ui/app.rs`)

- The engine is held as `Arc<Mutex<Box<dyn Engine + Send>>>`, **persisted across moves so the TT is reused**. It is rebuilt (fresh TT) only on new game or difficulty change, and only while idle.
- AI search runs in a spawned thread; `poll_ai` is called every frame. A monotonic `req_id` is bumped on resign/undo/new-game/load/human-move; stale results whose id ≠ current `req_id` are discarded.
- Board flip: the human side always sits at the bottom (`flip()` when human is Black). `to_screen`/`from_screen` are exact inverses; all click handling goes through `handle_screen_click` so the renderer and tests exercise the identical mapping.
- `#[cfg(test)]` `run_ai_blocking` / `XiangqiApp::headless()` allow synchronous end-to-end UI tests with no threads or egui context.

### Save format

`game.rs::SaveGame` is JSON, versioned by `SAVE_FORMAT`; `from_save` rejects newer formats, wrong cell counts, and out-of-range history indices instead of panicking. The repetition `hashes` are **not serialized** — `from_save` rewinds the saved history to the start and replays it to rebuild them, so draw detection survives a load without changing the on-disk schema.

## CJK fonts

`main.rs::load_cjk_font` probes known OS font paths. If none is found the program still runs but pieces render as Latin letters (K/A/E/H/R/C/P, color conveys side) — keep that fallback working when touching rendering.
