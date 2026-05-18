//! AlphaZero engine: a thin client over the Python MCTS sidecar
//! (`alphazero/serve.py`).
//!
//! The trained network is a PyTorch state-dict, and real playing strength
//! comes from MCTS + the net. Rather than re-implement either in Rust, we
//! spawn the tested Python `serve.py` once and talk to it over line-delimited
//! JSON: send the whole game history (move indices, identical to our `Move`
//! encoding), get back the chosen move.
//!
//! Configuration via environment (all optional, sensible defaults):
//! - `XIANGQI_PYTHON` — interpreter (default: try `python3`, then `python`).
//! - `XIANGQI_AZ_DIR` — repo root to run from (default: this crate's dir).
//! - `XIANGQI_AZ_CKPT` — checkpoint path (default
//!   `alphazero/checkpoints/best.pt`, relative to the repo root).

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use super::Engine;
use crate::board::Move;
use crate::game::GameState;

struct Sidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

pub struct AlphaZeroEngine {
    /// MCTS simulations per move (mapped from the UI difficulty).
    sims: u32,
    /// Lazily spawned on the first move; respawned after an I/O failure.
    sidecar: Option<Sidecar>,
}

impl AlphaZeroEngine {
    pub fn new(sims: u32) -> Self {
        AlphaZeroEngine {
            sims: sims.max(1),
            sidecar: None,
        }
    }

    fn kill(&mut self) {
        if let Some(mut sc) = self.sidecar.take() {
            let _ = sc.child.kill();
            let _ = sc.child.wait();
        }
    }
}

fn repo_dir() -> String {
    std::env::var("XIANGQI_AZ_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_string())
}

fn ckpt() -> String {
    std::env::var("XIANGQI_AZ_CKPT")
        .unwrap_or_else(|_| "alphazero/checkpoints/best.pt".to_string())
}

fn interpreters() -> Vec<String> {
    match std::env::var("XIANGQI_PYTHON") {
        Ok(p) => vec![p],
        Err(_) => vec!["python3".to_string(), "python".to_string()],
    }
}

/// Spawn `python -m alphazero.serve` and wait for its `{"ready": true}`
/// handshake. Returns `None` if no interpreter works (the game then simply
/// gets no move from this side rather than crashing).
fn spawn_sidecar() -> Option<Sidecar> {
    let dir = repo_dir();
    let ckpt = ckpt();
    for py in interpreters() {
        let mut child = match Command::new(&py)
            .arg("-m")
            .arg("alphazero.serve")
            .arg("--ckpt")
            .arg(&ckpt)
            .current_dir(&dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => continue, // interpreter not found; try the next
        };

        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = child.kill();
            continue;
        };
        let mut stdout = BufReader::new(stdout);

        // Read until the ready line (model load can take a few seconds; a
        // broken environment exits and read_line returns 0, so we move on).
        let mut ready = false;
        let mut line = String::new();
        for _ in 0..64 {
            line.clear();
            match stdout.read_line(&mut line) {
                Ok(0) | Err(_) => break, // process exited / pipe error
                Ok(_) => {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) {
                        if v.get("ready").and_then(serde_json::Value::as_bool) == Some(true) {
                            ready = true;
                            break;
                        }
                    }
                }
            }
        }

        if ready {
            return Some(Sidecar { child, stdin, stdout });
        }
        let _ = child.kill();
        let _ = child.wait();
    }
    None
}

fn parse_move(resp: &str) -> Option<Move> {
    let v: serde_json::Value = serde_json::from_str(resp.trim()).ok()?;
    let arr = v.get("move")?.as_array()?; // `null` (terminal) -> None
    if arr.len() != 2 {
        return None;
    }
    Some(Move {
        from: arr[0].as_u64()? as usize,
        to: arr[1].as_u64()? as usize,
    })
}

impl Engine for AlphaZeroEngine {
    fn name(&self) -> String {
        format!("AlphaZero (MCTS sims={})", self.sims)
    }

    fn best_move(&mut self, state: &GameState) -> Option<Move> {
        // Whole game history as flat board indices — identical to our `Move`
        // encoding, so the sidecar reconstructs the exact position (including
        // repetition / no-capture counters) by replaying it.
        let moves: Vec<[usize; 2]> = state.history.iter().map(|(m, _)| [m.from, m.to]).collect();
        let req = serde_json::json!({ "moves": moves, "sims": self.sims });
        let line = serde_json::to_string(&req).ok()?;

        // One round-trip; on any I/O failure drop the child and respawn once.
        for attempt in 0..2 {
            if self.sidecar.is_none() {
                self.sidecar = spawn_sidecar();
            }
            let Some(sc) = self.sidecar.as_mut() else {
                return None; // no usable Python interpreter
            };

            let exchange = (|| -> std::io::Result<String> {
                sc.stdin.write_all(line.as_bytes())?;
                sc.stdin.write_all(b"\n")?;
                sc.stdin.flush()?;
                let mut resp = String::new();
                if sc.stdout.read_line(&mut resp)? == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "sidecar closed",
                    ));
                }
                Ok(resp)
            })();

            match exchange {
                Ok(resp) => return parse_move(&resp),
                Err(_) => {
                    self.kill();
                    if attempt == 1 {
                        return None;
                    }
                }
            }
        }
        None
    }
}

impl Drop for AlphaZeroEngine {
    fn drop(&mut self) {
        self.kill();
    }
}
