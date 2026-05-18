# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A Chinese Chess (Xiangqi) desktop program: Rust + `egui`/`eframe` GUI with a pluggable AI engine. Code comments are English; all user-facing UI strings are Chinese.

## Commands

```bash
cargo run --release          # play (release is required for the AI to be responsive)
cargo test                   # default suite (~34 tests: rules/perft, engine tactics, UI logic)
cargo test <name>            # single test, e.g. `cargo test perft_matches_known_values`
cargo clippy --all-targets   # expected to be warning-free; keep it that way
cargo test --release engine_benchmark -- --ignored --nocapture       # node-count benchmark
cargo test alphazero_sidecar -- --ignored                            # exercises the Python sidecar (needs torch)
```

Two tests are `#[ignore]`d so the default suite needs no PyTorch: `engine_benchmark`
and `alphazero_sidecar_returns_legal_move`. The release profile sets `lto = true` /
`opt-level = 3`; debug-mode AI is much slower.

The AlphaZero trainer is a separate Python project (`alphazero/`, see its own
`README.md`):

```bash
python -m alphazero.tests.test_perft     # validate the Python rules port vs Rust perft
python -m alphazero.pipeline --smoke     # ~1 min end-to-end self-play→train→gate sanity run
python -m alphazero.serve --selftest     # verify the Rust↔Python sidecar protocol
```

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
- `game.rs::GameState` owns turn tracking, undo `history`, move `log` notation, and end-game detection. Xiangqi rule: no legal move ⇒ the side to move *loses* whether checkmated or stalemated. Repetition is judged by the **CCA / Asian rules**, not scored as a flat draw: `repetition_judgment` classifies each side's offence over the repeated cycle (`tag_move` → 长将 perpetual check = level 2, 长捉 perpetual chase = level 1, idle = 0) and `repetition_status` applies the decision table — a one-sided perpetual check/chase *loses*; only a mutual or idle repetition draws. A 120-ply (`NO_CAPTURE_PLY_LIMIT`) no-capture run is a draw.

### Two independent, non-shared hashing schemes

This trips people up. They are deliberately separate and **both full-recompute** (no incremental Zobrist update — correctness over speed):

- `game.rs::position_hash` — FNV-1a, drives `GameState`'s threefold-repetition detection.
- `ai/search.rs::zkey` — Zobrist, drives the transposition table *and* the engine's own in-search repetition counting (`build_repetition` replays game history into `game_counts`; `path` tracks the current search line).

Changing one does not affect the other.

### Pluggable engine

`ai/mod.rs` defines `trait Engine { name(); best_move(&GameState) -> Option<Move> }` and an `EngineKind { AlphaBeta, AlphaZero }`. `make_engine(kind, difficulty, perspective)` is the **single** place engines are constructed: difficulty 1–5 maps to `(max_depth, time_budget)` for `AlphaBeta` and to an MCTS `sims` count for `AlphaZero` (no time budget — per-move wall-clock is `sims` × sidecar inference speed). The UI, threading, and legality checks all adapt automatically — to add another engine, add an `EngineKind` arm, implement `Engine`, and return it from `make_engine`, change nothing else.

`ai/search.rs` is the bundled engine: iterative deepening + alpha-beta/PVS, depth-preferred transposition table, quiescence search, MVV-LVA + killer move ordering, piece-square evaluation. Mate scores are stored TT-relative to the node via `adjust_to_tt`/`adjust_from_tt` (not relative to the root). The root searches every move with a full window (PVS only in the interior) so the equal-best pool used for move variety stays honest.

### UI / threading (`ui/app.rs`)

- The engine is held as `Arc<Mutex<Box<dyn Engine + Send>>>`, **persisted across moves so the TT is reused**. It is rebuilt (fresh TT) only on new game or difficulty change, and only while idle.
- AI search runs in a spawned thread; `poll_ai` is called every frame. A monotonic `req_id` is bumped on resign/undo/new-game/load/human-move; stale results whose id ≠ current `req_id` are discarded.
- Board flip: the human side always sits at the bottom (`flip()` when human is Black). `to_screen`/`from_screen` are exact inverses; all click handling goes through `handle_screen_click` so the renderer and tests exercise the identical mapping.
- `#[cfg(test)]` `run_ai_blocking` / `XiangqiApp::headless()` allow synchronous end-to-end UI tests with no threads or egui context.

### Save format

`game.rs::SaveGame` is JSON, versioned by `SAVE_FORMAT`; `from_save` rejects newer formats, wrong cell counts, and out-of-range history indices instead of panicking. The repetition `hashes` are **not serialized** — `from_save` rewinds the saved history to the start and replays it to rebuild them, so draw detection survives a load without changing the on-disk schema.

## The AlphaZero subproject (`alphazero/`)

A standalone PyTorch AlphaZero self-play trainer for Xiangqi, **independent of the Rust crate**. It re-implements the rules in pure Python (`alphazero/xiangqi/`) and is validated bit-for-bit against the Rust engine's perft (44 / 1920 / 79666, `alphazero/tests/test_perft.py`). Terminal scoring matches the Rust rules exactly (no-legal-reply loses; repetition / 120-ply no-capture draw). Key pieces: `net.py` (ResNet dual-head, 8100 flat `from*to` actions), `encoding.py` (16 planes, canonicalised to the side to move), `mcts.py` (PUCT, apply/undo on a cloned `GameState`), `pipeline.py` (self-play → train → arena gate). Tuning lives in `config.py`; `Config.smoke()` shrinks it for `--smoke`. Full layout/run/tuning docs are in `alphazero/README.md`.

**Rust ↔ Python bridge (do not re-implement the net/MCTS in Rust).** `src/ai/alphazero.rs::AlphaZeroEngine` lazily spawns a long-lived `python -m alphazero.serve` sidecar and talks line-delimited JSON over stdin/stdout: it sends the full move-index history (the encoding is *identical* to the Rust `Move`, no coordinate conversion), the sidecar replays it onto a fresh `GameState`, runs MCTS, and returns the chosen move. The sidecar is respawned after an I/O failure; if no interpreter is found the side simply gets no move rather than crashing. Configuration is via optional env vars: `XIANGQI_PYTHON` (interpreter), `XIANGQI_AZ_DIR` (repo root), `XIANGQI_AZ_CKPT` (default `alphazero/checkpoints/best.pt`). The trained policy loss must be computed in fp32 — under AMP the logits are fp16 and the `-1e9` illegal-move mask overflows half precision.

## CJK fonts

`main.rs::load_cjk_font` probes known OS font paths. If none is found the program still runs but pieces render as Latin letters (K/A/E/H/R/C/P, color conveys side) — keep that fallback working when touching rendering.
