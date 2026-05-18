//! egui front-end: board rendering, mouse input, control panel, AI threading.

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;

use eframe::egui;

use crate::ai::{make_engine, Engine, EngineKind};
use crate::board::*;
use crate::game::{DrawReason, GameState, GameStatus, SaveGame};

/// Who controls each side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum GameMode {
    /// Human plays one colour, an engine plays the other.
    #[default]
    HumanVsAi,
    /// Both colours are engines; the match auto-advances.
    AiVsAi,
}

/// Blank frame around the board, in points.
const MARGIN: f32 = 36.0;
/// Breathing room kept above and below the board so it does not touch the
/// window edges when maximised.
const OUTER_MARGIN: f32 = 14.0;
/// Minimum cell size: the board scales freely with the window (filling the
/// space when maximised) but never shrinks below this on a small window.
const CELL_MIN: f32 = 26.0;

/// Board coord -> on-screen cell indices, accounting for board flip
/// (the human side always sits at the bottom).
fn to_screen(flip: bool, f: i32, r: i32) -> (i32, i32) {
    if flip {
        (8 - f, 9 - r)
    } else {
        (f, r)
    }
}

/// Exact inverse of [`to_screen`].
fn from_screen(flip: bool, sf: i32, sr: i32) -> (i32, i32) {
    if flip {
        (8 - sf, 9 - sr)
    } else {
        (sf, sr)
    }
}

/// A combo box for choosing an [`EngineKind`]. `label` must be unique within
/// the panel (egui derives the widget id from it).
fn engine_combo(ui: &mut egui::Ui, label: &str, kind: &mut EngineKind) {
    egui::ComboBox::from_label(label)
        .selected_text(kind.label())
        .show_ui(ui, |ui| {
            for k in [EngineKind::AlphaBeta, EngineKind::AlphaZero] {
                ui.selectable_value(kind, k, k.label());
            }
        });
}

type SharedEngine = Arc<Mutex<Box<dyn Engine + Send>>>;

pub struct XiangqiApp {
    game: GameState,
    human_color: Color,
    pending_color: Color, // side the human takes on the next New Game
    difficulty: u8,
    selected: Option<usize>,
    status: GameStatus,
    /// Set when a side resigns; overrides the computed status until reset.
    forced: Option<GameStatus>,

    /// Active mode; `pending_mode` is applied on the next New Game (the same
    /// deferred-apply pattern as `pending_color`).
    mode: GameMode,
    pending_mode: GameMode,
    /// Active per-side engine kinds; the `pending_*` ones are applied on New
    /// Game. In Human-vs-AI only the AI side's kind is used.
    red_kind: EngineKind,
    black_kind: EngineKind,
    pending_red_kind: EngineKind,
    pending_black_kind: EngineKind,

    thinking: bool,
    req_id: u64,
    ai_rx: Option<Receiver<(u64, Option<Move>)>>,
    /// Set when an engine yields no move on a non-terminal position (e.g. the
    /// AlphaZero sidecar could not start). Stops the auto-loop from spinning;
    /// cleared on New Game / Undo / Load / a difficulty change.
    ai_stalled: bool,

    /// One engine per colour, persisted across moves so each keeps its state
    /// (e.g. the AlphaBeta transposition table). Only the AI sides are used.
    engine_red: SharedEngine,
    engine_black: SharedEngine,
    /// Difficulty the current engines were built for.
    engine_diff: u8,
    /// Whether a CJK font loaded; if not, pieces are drawn as letters.
    cjk_ok: bool,

    save_path: String,
    message: String,
}

impl XiangqiApp {
    fn bootstrap(cjk_ok: bool) -> Self {
        let difficulty = 3;
        let mut app = XiangqiApp {
            game: GameState::new(),
            human_color: Color::Red,
            pending_color: Color::Red,
            difficulty,
            selected: None,
            status: GameStatus::Ongoing,
            forced: None,
            mode: GameMode::HumanVsAi,
            pending_mode: GameMode::HumanVsAi,
            red_kind: EngineKind::AlphaBeta,
            black_kind: EngineKind::AlphaBeta,
            pending_red_kind: EngineKind::AlphaBeta,
            pending_black_kind: EngineKind::AlphaBeta,
            thinking: false,
            req_id: 0,
            ai_rx: None,
            ai_stalled: false,
            engine_red: Arc::new(Mutex::new(make_engine(
                EngineKind::AlphaBeta,
                difficulty,
                Color::Red,
            ))),
            engine_black: Arc::new(Mutex::new(make_engine(
                EngineKind::AlphaBeta,
                difficulty,
                Color::Black,
            ))),
            engine_diff: difficulty,
            cjk_ok,
            save_path: "xiangqi_save.json".to_owned(),
            message: String::new(),
        };
        app.status = app.game.status();
        app
    }

    pub fn new(_cc: &eframe::CreationContext<'_>, cjk_ok: bool) -> Self {
        Self::bootstrap(cjk_ok)
    }

    /// Construct without eframe, for tests.
    #[cfg(test)]
    pub fn headless() -> Self {
        Self::bootstrap(true)
    }

    /// Rebuild both engines (fresh state). Called on a new game or when the
    /// difficulty changes. Replacing the `Arc`s drops the old engines; an
    /// AlphaZero engine's `Drop` shuts down its Python sidecar, and a fresh
    /// one re-spawns lazily on first use.
    fn rebuild_engines(&mut self) {
        self.engine_red = Arc::new(Mutex::new(make_engine(
            self.red_kind,
            self.difficulty,
            Color::Red,
        )));
        self.engine_black = Arc::new(Mutex::new(make_engine(
            self.black_kind,
            self.difficulty,
            Color::Black,
        )));
        self.engine_diff = self.difficulty;
        self.ai_stalled = false; // a config change is a fresh chance to run
    }

    /// The AI side in Human-vs-AI (meaningless in AI-vs-AI, where both sides
    /// are engines — callers gate on [`GameMode::HumanVsAi`] first).
    fn ai_color(&self) -> Color {
        self.human_color.opposite()
    }

    /// Whether `c` is engine-controlled: both sides in AI-vs-AI, otherwise the
    /// non-human side.
    fn is_ai(&self, c: Color) -> bool {
        match self.mode {
            GameMode::AiVsAi => true,
            GameMode::HumanVsAi => c != self.human_color,
        }
    }

    fn kind_for(&self, c: Color) -> EngineKind {
        match c {
            Color::Red => self.red_kind,
            Color::Black => self.black_kind,
        }
    }

    fn engine_for(&self, c: Color) -> &SharedEngine {
        match c {
            Color::Red => &self.engine_red,
            Color::Black => &self.engine_black,
        }
    }

    /// True when the human side sits at the bottom-flipped board. There is no
    /// human in AI-vs-AI, so the board stays in Red's orientation.
    fn flip(&self) -> bool {
        self.mode == GameMode::HumanVsAi && self.human_color == Color::Black
    }

    /// Status to display: a resignation overrides the computed status.
    fn eff_status(&self) -> GameStatus {
        self.forced.unwrap_or(self.status)
    }

    fn game_over(&self) -> bool {
        matches!(
            self.eff_status(),
            GameStatus::Win(_) | GameStatus::Stalemate(_) | GameStatus::Draw(_)
        )
    }

    fn resign(&mut self) {
        if self.game_over() {
            return;
        }
        // The human concedes; the AI side wins.
        self.forced = Some(GameStatus::Win(self.ai_color()));
        self.selected = None;
        self.req_id += 1;
    }

    fn new_game(&mut self) {
        self.game = GameState::new();
        self.human_color = self.pending_color;
        self.mode = self.pending_mode;
        self.red_kind = self.pending_red_kind;
        self.black_kind = self.pending_black_kind;
        self.selected = None;
        self.forced = None;
        self.thinking = false;
        self.ai_rx = None;
        self.req_id += 1;
        self.rebuild_engines();
        self.status = self.game.status();
        self.message.clear();
    }

    fn undo(&mut self) {
        if self.thinking {
            return;
        }
        if !self.game.undo() {
            return;
        }
        // In Human-vs-AI, step back past the AI's reply too so it is the
        // human's turn again. In AI-vs-AI a single ply is the natural step.
        if self.mode == GameMode::HumanVsAi
            && self.game.side_to_move != self.human_color
            && !self.game.history.is_empty()
        {
            self.game.undo();
        }
        self.selected = None;
        self.forced = None;
        self.req_id += 1;
        self.ai_stalled = false;
        self.status = self.game.status();
        self.message.clear();
    }

    fn save(&mut self) {
        match serde_json::to_string_pretty(&self.game.to_save()) {
            Ok(json) => match std::fs::write(&self.save_path, json) {
                Ok(()) => self.message = format!("已保存到 {}", self.save_path),
                Err(e) => self.message = format!("保存失败: {e}"),
            },
            Err(e) => self.message = format!("序列化失败: {e}"),
        }
    }

    fn load(&mut self) {
        let data = match std::fs::read_to_string(&self.save_path) {
            Ok(d) => d,
            Err(e) => {
                self.message = format!("读取失败: {e}");
                return;
            }
        };
        let save: SaveGame = match serde_json::from_str(&data) {
            Ok(s) => s,
            Err(e) => {
                self.message = format!("解析失败: {e}");
                return;
            }
        };
        match GameState::from_save(save) {
            Some(g) => {
                self.game = g;
                self.selected = None;
                self.forced = None;
                self.thinking = false;
                self.ai_rx = None;
                self.req_id += 1;
                self.ai_stalled = false;
                self.status = self.game.status();
                self.message = format!("已从 {} 载入", self.save_path);
            }
            None => self.message = "存档数据无效".to_owned(),
        }
    }

    /// Spawn a background search if the side to move is engine-controlled.
    /// In AI-vs-AI this fires for whichever colour is on move, so the match
    /// auto-advances: `poll_ai` applies the move and flips the turn, and the
    /// next frame starts the other engine.
    fn maybe_start_ai(&mut self, ctx: &egui::Context) {
        if self.thinking || self.game_over() || self.ai_stalled {
            return;
        }
        let side = self.game.side_to_move;
        if !self.is_ai(side) {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let game = self.game.clone();
        let id = self.req_id;
        let engine = Arc::clone(self.engine_for(side));
        thread::spawn(move || {
            let mv = match engine.lock() {
                Ok(mut e) => e.best_move(&game),
                Err(_) => None, // poisoned: engine panicked previously
            };
            let _ = tx.send((id, mv));
        });
        self.ai_rx = Some(rx);
        self.thinking = true;
        ctx.request_repaint();
    }

    /// Poll for a finished AI search and apply its move.
    fn poll_ai(&mut self, ctx: &egui::Context) {
        if !self.thinking {
            return;
        }
        let result = self.ai_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some((id, mv)) = result {
            self.thinking = false;
            self.ai_rx = None;
            if id == self.req_id {
                match mv {
                    Some(mv) if self.game.is_legal(mv) => {
                        self.game.apply(mv);
                        self.status = self.game.status();
                    }
                    // No usable move on a live position means the engine
                    // failed (most often: the AlphaZero sidecar could not
                    // start). Stop the auto-loop instead of respawning every
                    // frame; New Game / Undo / a difficulty change retries.
                    _ if !self.game_over() => {
                        self.ai_stalled = true;
                        self.message = "AI 无法走子（引擎无响应或环境异常）。\
                             请重开对局，或检查 Python、torch 与模型文件。"
                            .to_owned();
                    }
                    _ => {}
                }
            }
        } else {
            ctx.request_repaint(); // keep polling
        }
    }

    fn human_can_move(&self) -> bool {
        self.mode == GameMode::HumanVsAi
            && !self.thinking
            && !self.game_over()
            && self.game.side_to_move == self.human_color
    }

    fn legal_targets(&mut self) -> Vec<usize> {
        let Some(sel) = self.selected else {
            return Vec::new();
        };
        self.game
            .legal_moves()
            .into_iter()
            .filter(|m| m.from == sel)
            .map(|m| m.to)
            .collect()
    }

    fn on_click_square(&mut self, sq: usize) {
        if !self.human_can_move() {
            return;
        }
        if let Some(sel) = self.selected {
            let mv = Move { from: sel, to: sq };
            if self.game.is_legal(mv) {
                self.game.apply(mv);
                self.selected = None;
                self.req_id += 1;
                self.status = self.game.status();
                return;
            }
        }
        // (Re)select one of our own pieces, otherwise clear.
        match self.game.board.get(sq) {
            Some(p) if p.color == self.human_color => self.selected = Some(sq),
            _ => self.selected = None,
        }
    }

    /// Translate an on-screen intersection (cell indices) to a board square
    /// and handle it. Shared by the renderer and tests so the flip mapping
    /// is exercised exactly as the UI uses it.
    fn handle_screen_click(&mut self, sf: i32, sr: i32) {
        if !(0..FILES as i32).contains(&sf) || !(0..RANKS as i32).contains(&sr) {
            return;
        }
        let (f, r) = from_screen(self.flip(), sf, sr);
        self.on_click_square(idx(f, r));
    }

    /// Synchronous engine move for the side to move, for tests (no threads /
    /// egui context). Call it repeatedly to play a whole AI-vs-AI game.
    #[cfg(test)]
    fn run_ai_blocking(&mut self) {
        let side = self.game.side_to_move;
        if self.thinking || self.game_over() || !self.is_ai(side) {
            return;
        }
        let mv = self.engine_for(side).lock().unwrap().best_move(&self.game);
        if let Some(mv) = mv {
            if self.game.is_legal(mv) {
                self.game.apply(mv);
                self.status = self.game.status();
            }
        }
    }

    fn status_text(&self) -> String {
        let side = |c: Color| if c == Color::Red { "红方" } else { "黑方" };
        match self.eff_status() {
            GameStatus::Win(c) => {
                let how = if self.forced.is_some() { "认输" } else { "将死" };
                format!("{}！{} 胜", how, side(c))
            }
            GameStatus::Stalemate(c) => format!("困毙！{} 胜", side(c)),
            GameStatus::Draw(DrawReason::Repetition) => "和棋（三次重复局面）".to_owned(),
            GameStatus::Draw(DrawReason::NoCapture) => "和棋（60 回合无吃子）".to_owned(),
            GameStatus::Check(c) => format!("{} 被将军", side(c)),
            GameStatus::Ongoing => {
                let stm = self.game.side_to_move;
                let who = if self.mode == GameMode::HumanVsAi && stm == self.human_color {
                    "你".to_owned()
                } else {
                    let k = self.kind_for(stm).label();
                    if self.thinking {
                        format!("{k} 思考中…")
                    } else {
                        k.to_owned()
                    }
                };
                format!("轮到 {}（{}）", side(stm), who)
            }
        }
    }
}

impl eframe::App for XiangqiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_ai(ctx);
        // Apply a difficulty change while idle (rebuilds both engines).
        if !self.thinking && self.difficulty != self.engine_diff {
            self.rebuild_engines();
        }
        self.maybe_start_ai(ctx);

        self.control_panel(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_board(ui);
        });
    }
}

impl XiangqiApp {
    fn control_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("controls")
            .resizable(false)
            .exact_width(252.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("中国象棋");
                ui.separator();

                let mut st = egui::RichText::new(self.status_text()).strong();
                if self.game_over() {
                    st = st.size(20.0).color(super::PIECE_RED);
                }
                ui.label(st);
                ui.add_space(6.0);

                ui.label("对战模式：");
                ui.horizontal(|ui| {
                    ui.radio_value(
                        &mut self.pending_mode,
                        GameMode::HumanVsAi,
                        "人机对战",
                    );
                    ui.radio_value(&mut self.pending_mode, GameMode::AiVsAi, "AI 对战");
                });

                match self.pending_mode {
                    GameMode::HumanVsAi => {
                        ui.label("你执：");
                        ui.horizontal(|ui| {
                            ui.radio_value(
                                &mut self.pending_color,
                                Color::Red,
                                "红（先手）",
                            );
                            ui.radio_value(
                                &mut self.pending_color,
                                Color::Black,
                                "黑（后手）",
                            );
                        });
                        // The engine plays the other colour.
                        let sel = if self.pending_color == Color::Red {
                            &mut self.pending_black_kind
                        } else {
                            &mut self.pending_red_kind
                        };
                        engine_combo(ui, "AI 引擎", sel);
                    }
                    GameMode::AiVsAi => {
                        engine_combo(ui, "红方引擎", &mut self.pending_red_kind);
                        engine_combo(ui, "黑方引擎", &mut self.pending_black_kind);
                    }
                }
                if ui.button("开始新对局").clicked() {
                    self.new_game();
                }

                ui.add_space(8.0);
                ui.add(
                    egui::Slider::new(&mut self.difficulty, 1..=5)
                        .text("AI 难度（1–5；AlphaZero 为 MCTS 模拟数）"),
                );

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("悔棋").clicked() {
                        self.undo();
                    }
                    let can_resign =
                        self.mode == GameMode::HumanVsAi && !self.game_over();
                    if ui
                        .add_enabled(can_resign, egui::Button::new("认输"))
                        .clicked()
                    {
                        self.resign();
                    }
                });

                ui.separator();
                ui.label("存档文件名：");
                ui.text_edit_singleline(&mut self.save_path);
                ui.horizontal(|ui| {
                    if ui.button("保存").clicked() {
                        self.save();
                    }
                    if ui.button("载入").clicked() {
                        self.load();
                    }
                });
                if !self.message.is_empty() {
                    ui.label(egui::RichText::new(&self.message).italics());
                }

                ui.separator();
                ui.label(format!("着法记录（{} 步）", self.game.log.len()));
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (n, mv) in self.game.log.iter().enumerate() {
                            ui.label(format!("{:>3}. {}", n + 1, mv));
                        }
                    });
            });
    }

    fn draw_board(&mut self, ui: &mut egui::Ui) {
        // Take the whole panel and size the board to fit it, so the board
        // grows when the window is maximised and shrinks (to a floor) when
        // it is small. `cell` is derived from the *allocated* rect rather
        // than a pre-query, which keeps the board strictly inside the
        // visible region (available_size() can over-report and would
        // otherwise let a maximised board spill off-window).
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), egui::Sense::click());
        let area = response.rect;
        let (cols, rows) = (FILES as f32 - 1.0, RANKS as f32 - 1.0);
        let cell = ((area.width() - 2.0 * MARGIN) / cols)
            .min((area.height() - 2.0 * MARGIN - 2.0 * OUTER_MARGIN) / rows)
            .max(CELL_MIN);

        let board_w = cols * cell + 2.0 * MARGIN;
        let board_h = rows * cell + 2.0 * MARGIN;

        // Centre the board within the allocated area.
        let board_min = area.min
            + egui::vec2(
                ((area.width() - board_w) * 0.5).max(0.0),
                ((area.height() - board_h) * 0.5).max(0.0),
            );
        let rect = egui::Rect::from_min_size(board_min, egui::vec2(board_w, board_h));
        let origin = rect.min + egui::vec2(MARGIN, MARGIN);

        // When the human plays Black the board is flipped so their pieces sit
        // at the bottom. `point` maps a board coord to its screen position;
        // the click handler applies the exact inverse.
        let flip = self.flip();
        let cjk = self.cjk_ok;
        let point = |f: i32, r: i32| -> egui::Pos2 {
            let (sf, sr) = to_screen(flip, f, r);
            egui::pos2(origin.x + sf as f32 * cell, origin.y + sr as f32 * cell)
        };

        // Colors.
        let bg = egui::Color32::from_rgb(0xEC, 0xCF, 0x93);
        let line = egui::Color32::from_rgb(0x5A, 0x3A, 0x1E);
        let disc = super::PIECE_CREAM;
        let red = super::PIECE_RED;
        let black = egui::Color32::from_rgb(0x20, 0x20, 0x20);
        let sel_col = egui::Color32::from_rgb(0x1E, 0x88, 0xE5);
        let moved_col = egui::Color32::from_rgb(0xE6, 0xA2, 0x17);
        let dot_col = egui::Color32::from_rgba_unmultiplied(0x1E, 0x88, 0xE5, 170);
        let last_col = egui::Color32::from_rgba_unmultiplied(0xF1, 0xC4, 0x0F, 90);
        let check_col = egui::Color32::from_rgba_unmultiplied(0xE7, 0x4C, 0x3C, 110);
        let stroke = |w: f32| egui::Stroke::new(w, line);

        painter.rect_filled(rect, 4.0, bg);

        // Horizontal lines (full width).
        for r in 0..RANKS as i32 {
            painter.line_segment([point(0, r), point(8, r)], stroke(1.4));
        }
        // Vertical lines, broken at the river except the two borders.
        for f in 0..FILES as i32 {
            if f == 0 || f == 8 {
                painter.line_segment([point(f, 0), point(f, 9)], stroke(1.4));
            } else {
                painter.line_segment([point(f, 0), point(f, 4)], stroke(1.4));
                painter.line_segment([point(f, 5), point(f, 9)], stroke(1.4));
            }
        }
        // Palace diagonals.
        for top in [0, 7] {
            painter.line_segment([point(3, top), point(5, top + 2)], stroke(1.4));
            painter.line_segment([point(5, top), point(3, top + 2)], stroke(1.4));
        }
        // River text (Latin fallback when no CJK font is available).
        let river_y = (point(0, 4).y + point(0, 5).y) / 2.0;
        if cjk {
            painter.text(
                egui::pos2(point(1, 0).x, river_y),
                egui::Align2::CENTER_CENTER,
                "楚 河",
                egui::FontId::proportional(cell * 0.44),
                line,
            );
            painter.text(
                egui::pos2(point(7, 0).x, river_y),
                egui::Align2::CENTER_CENTER,
                "漢 界",
                egui::FontId::proportional(cell * 0.44),
                line,
            );
        } else {
            painter.text(
                egui::pos2(point(4, 0).x, river_y),
                egui::Align2::CENTER_CENTER,
                "R I V E R",
                egui::FontId::proportional(cell * 0.40),
                line,
            );
            painter.text(
                egui::pos2(rect.center().x, rect.min.y + 12.0),
                egui::Align2::CENTER_CENTER,
                "No CJK font: pieces shown as letters (K/A/E/H/R/C/P)",
                egui::FontId::proportional((cell * 0.22).max(11.0)),
                egui::Color32::from_rgb(0xB0, 0x30, 0x20),
            );
        }

        // Coordinate labels, anchored to the screen edges so they stay
        // readable whether or not the board is flipped.
        let label_size = (cell * 0.27).max(11.0);
        let label_off = cell * 0.37;
        let bottom_r = if flip { 0 } else { 9 };
        for f in 0..FILES as i32 {
            let p = point(f, bottom_r);
            painter.text(
                egui::pos2(p.x, p.y + label_off),
                egui::Align2::CENTER_CENTER,
                (b'A' + f as u8) as char,
                egui::FontId::proportional(label_size),
                line,
            );
        }
        let left_f = if flip { 8 } else { 0 };
        for r in 0..RANKS as i32 {
            let p = point(left_f, r);
            painter.text(
                egui::pos2(p.x - label_off, p.y),
                egui::Align2::CENTER_CENTER,
                format!("{}", RANKS as i32 - r),
                egui::FontId::proportional(label_size),
                line,
            );
        }

        // Last-move highlight (faint trail discs on the from/to squares).
        if let Some(mv) = self.game.last_move {
            for sq in [mv.from, mv.to] {
                let c = point(file_of(sq), rank_of(sq));
                painter.circle_filled(c, cell * 0.46, last_col);
            }
        }

        // Check highlight on the general in check.
        if let GameStatus::Check(c) = self.eff_status() {
            if let Some(g) = self.game.board.find_general(c) {
                let p = point(file_of(g), rank_of(g));
                painter.circle_filled(p, cell * 0.48, check_col);
            }
        }

        let targets = self.legal_targets();
        let moved_to = self.game.last_move.map(|m| m.to);

        // Pieces.
        for i in 0..CELLS {
            let Some(piece) = self.game.board.get(i) else {
                continue;
            };
            let c = point(file_of(i), rank_of(i));
            let pc = if piece.color == Color::Red { red } else { black };
            painter.circle_filled(c, cell * 0.42, disc);
            painter.circle_stroke(c, cell * 0.42, egui::Stroke::new(2.0, pc));
            // Ring the piece that just moved so the last move is easy to spot.
            if moved_to == Some(i) {
                painter.circle_stroke(c, cell * 0.5, egui::Stroke::new(3.0, moved_col));
            }
            if self.selected == Some(i) {
                painter.circle_stroke(c, cell * 0.46, egui::Stroke::new(3.0, sel_col));
            }
            painter.text(
                c,
                egui::Align2::CENTER_CENTER,
                if cjk { piece.glyph() } else { piece.ascii() },
                egui::FontId::proportional(if cjk { cell * 0.5 } else { cell * 0.43 }),
                pc,
            );
        }

        // Legal-target markers.
        for &t in &targets {
            let c = point(file_of(t), rank_of(t));
            if self.game.board.get(t).is_some() {
                painter.circle_stroke(c, cell * 0.46, egui::Stroke::new(3.0, dot_col));
            } else {
                painter.circle_filled(c, (cell * 0.12).max(4.0), dot_col);
            }
        }

        // Click -> nearest on-screen intersection.
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let sff = ((pos.x - origin.x) / cell).round();
                let srr = ((pos.y - origin.y) / cell).round();
                if (0.0..FILES as f32).contains(&sff) && (0.0..RANKS as f32).contains(&srr) {
                    let (sf, sr) = (sff as i32, srr as i32);
                    let cell_pt = egui::pos2(
                        origin.x + sf as f32 * cell,
                        origin.y + sr as f32 * cell,
                    );
                    if cell_pt.distance(pos) <= cell * 0.5 {
                        self.handle_screen_click(sf, sr);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod ui_tests {
    use super::*;

    #[test]
    fn flip_mapping_is_exact_inverse() {
        for &flip in &[false, true] {
            for f in 0..FILES as i32 {
                for r in 0..RANKS as i32 {
                    let (sf, sr) = to_screen(flip, f, r);
                    assert!((0..FILES as i32).contains(&sf));
                    assert!((0..RANKS as i32).contains(&sr));
                    assert_eq!(from_screen(flip, sf, sr), (f, r));
                }
            }
        }
    }

    #[test]
    fn select_then_move_red() {
        let mut app = XiangqiApp::headless(); // human Red, Red to move
        app.handle_screen_click(0, 6); // Red soldier (file 0, rank 6)
        assert_eq!(app.selected, Some(idx(0, 6)));
        app.handle_screen_click(0, 5); // forward one step
        assert_eq!(app.selected, None);
        assert_eq!(app.game.history.len(), 1);
        assert_eq!(app.game.side_to_move, Color::Black);
        assert_eq!(
            app.game.last_move,
            Some(Move { from: idx(0, 6), to: idx(0, 5) })
        );
    }

    #[test]
    fn illegal_target_clears_selection_without_moving() {
        let mut app = XiangqiApp::headless();
        app.handle_screen_click(0, 6);
        assert_eq!(app.selected, Some(idx(0, 6)));
        app.handle_screen_click(4, 4); // empty, not a legal target
        assert_eq!(app.selected, None);
        assert!(app.game.history.is_empty());
    }

    #[test]
    fn undo_steps_back_past_ai_reply() {
        let mut app = XiangqiApp::headless();
        app.difficulty = 1;
        app.rebuild_engines();
        app.handle_screen_click(0, 6);
        app.handle_screen_click(0, 5);
        assert_eq!(app.game.side_to_move, Color::Black);
        app.run_ai_blocking();
        assert_eq!(app.game.history.len(), 2);
        assert_eq!(app.game.side_to_move, Color::Red);
        app.undo();
        assert_eq!(app.game.history.len(), 0);
        assert_eq!(app.game.side_to_move, Color::Red);
        assert_eq!(app.selected, None);
    }

    #[test]
    fn resign_ends_the_game() {
        let mut app = XiangqiApp::headless(); // human Red
        app.resign();
        assert!(app.game_over());
        assert_eq!(app.eff_status(), GameStatus::Win(Color::Black));
        assert!(!app.human_can_move());
        assert!(app.status_text().contains("认输"));
    }

    #[test]
    fn ai_vs_ai_alphabeta_auto_plays() {
        let mut app = XiangqiApp::headless();
        app.difficulty = 1;
        app.pending_mode = GameMode::AiVsAi;
        app.new_game();
        assert_eq!(app.mode, GameMode::AiVsAi);
        assert!(!app.flip(), "no human side -> Red's orientation");
        assert!(!app.human_can_move());

        for _ in 0..6 {
            if app.game_over() {
                break;
            }
            let before = app.game.history.len();
            app.run_ai_blocking();
            assert_eq!(
                app.game.history.len(),
                before + 1,
                "the side to move should have played"
            );
        }
        assert!(app.game.history.len() >= 4);
        // Turn strictly alternates from Red.
        let expect = if app.game.history.len().is_multiple_of(2) {
            Color::Red
        } else {
            Color::Black
        };
        assert_eq!(app.game.side_to_move, expect);
    }

    #[test]
    fn ai_moves_first_when_human_is_black() {
        let mut app = XiangqiApp::headless();
        app.pending_color = Color::Black;
        app.new_game(); // human Black, AI Red, Red to move
        app.difficulty = 1;
        app.rebuild_engines();
        assert!(app.flip());
        app.run_ai_blocking();
        assert_eq!(app.game.history.len(), 1);
        assert_eq!(app.game.side_to_move, Color::Black);
    }

    #[test]
    fn flipped_board_click_maps_correctly() {
        let mut app = XiangqiApp::headless();
        app.pending_color = Color::Black;
        app.new_game();
        app.difficulty = 1;
        app.rebuild_engines();
        app.run_ai_blocking(); // AI (Red) moves; human (Black) to move, flipped
        assert_eq!(app.game.side_to_move, Color::Black);

        // Black soldier at board (0,3) is drawn at screen cell to_screen(0,3).
        let (sf, sr) = to_screen(true, 0, 3);
        app.handle_screen_click(sf, sr);
        assert_eq!(app.selected, Some(idx(0, 3)));
        let (tf, tr) = to_screen(true, 0, 4);
        app.handle_screen_click(tf, tr);
        assert_eq!(
            app.game.last_move,
            Some(Move { from: idx(0, 3), to: idx(0, 4) })
        );
    }

    #[test]
    fn app_save_then_load_roundtrips() {
        let path = std::env::temp_dir()
            .join(format!("xiangqi_ui_test_{}.json", std::process::id()));
        let p = path.to_string_lossy().to_string();

        let mut a = XiangqiApp::headless();
        a.handle_screen_click(0, 6);
        a.handle_screen_click(0, 5);
        a.save_path = p.clone();
        a.save();
        assert!(a.message.starts_with("已保存"), "save msg: {}", a.message);

        let mut b = XiangqiApp::headless();
        b.save_path = p.clone();
        b.load();
        assert!(b.message.starts_with("已从"), "load msg: {}", b.message);
        assert_eq!(b.game.board.cells, a.game.board.cells);
        assert_eq!(b.game.side_to_move, a.game.side_to_move);
        assert_eq!(b.game.log, a.game.log);

        let _ = std::fs::remove_file(&path);
    }
}
