//! egui front-end: board rendering, mouse input, control panel, AI threading.

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;

use eframe::egui;

use crate::ai::{make_engine, Engine};
use crate::board::*;
use crate::game::{DrawReason, GameState, GameStatus, SaveGame};

const MARGIN: f32 = 38.0;
const CELL: f32 = 60.0;

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

    thinking: bool,
    req_id: u64,
    ai_rx: Option<Receiver<(u64, Option<Move>)>>,

    /// Persisted across moves so the transposition table is reused.
    engine: SharedEngine,
    /// Difficulty the current `engine` was built for.
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
            thinking: false,
            req_id: 0,
            ai_rx: None,
            engine: Arc::new(Mutex::new(make_engine(difficulty, Color::Black))),
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

    /// Rebuild the engine (fresh transposition table). Called on a new game
    /// or when the difficulty changes.
    fn rebuild_engine(&mut self) {
        self.engine = Arc::new(Mutex::new(make_engine(self.difficulty, self.ai_color())));
        self.engine_diff = self.difficulty;
    }

    fn ai_color(&self) -> Color {
        self.human_color.opposite()
    }

    /// True when the human side sits at the bottom-flipped board.
    fn flip(&self) -> bool {
        self.human_color == Color::Black
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
        self.selected = None;
        self.forced = None;
        self.thinking = false;
        self.ai_rx = None;
        self.req_id += 1;
        self.rebuild_engine();
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
        // Step back past the AI's reply too, so it is the human's turn again.
        if self.game.side_to_move != self.human_color && !self.game.history.is_empty() {
            self.game.undo();
        }
        self.selected = None;
        self.forced = None;
        self.req_id += 1;
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
                self.status = self.game.status();
                self.message = format!("已从 {} 载入", self.save_path);
            }
            None => self.message = "存档数据无效".to_owned(),
        }
    }

    /// Spawn a background search if it is the AI's move.
    fn maybe_start_ai(&mut self, ctx: &egui::Context) {
        if self.thinking || self.game_over() {
            return;
        }
        if self.game.side_to_move != self.ai_color() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let game = self.game.clone();
        let id = self.req_id;
        let engine = Arc::clone(&self.engine);
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
                if let Some(mv) = mv {
                    if self.game.is_legal(mv) {
                        self.game.apply(mv);
                        self.status = self.game.status();
                    }
                }
            }
        } else {
            ctx.request_repaint(); // keep polling
        }
    }

    fn human_can_move(&self) -> bool {
        !self.thinking && !self.game_over() && self.game.side_to_move == self.human_color
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

    /// Synchronous AI move, for tests (no threads / egui context).
    #[cfg(test)]
    fn run_ai_blocking(&mut self) {
        if self.thinking || self.game_over() || self.game.side_to_move != self.ai_color() {
            return;
        }
        let mv = self.engine.lock().unwrap().best_move(&self.game);
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
                let who = if self.game.side_to_move == self.human_color {
                    "你"
                } else if self.thinking {
                    "AI 思考中…"
                } else {
                    "AI"
                };
                format!("轮到 {}（{}）", side(self.game.side_to_move), who)
            }
        }
    }
}

impl eframe::App for XiangqiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_ai(ctx);
        // Apply a difficulty change while idle (rebuilds the engine/TT).
        if !self.thinking && self.difficulty != self.engine_diff {
            self.rebuild_engine();
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
                    st = st.size(20.0).color(egui::Color32::from_rgb(0xC0, 0x2A, 0x1B));
                }
                ui.label(st);
                ui.add_space(6.0);

                ui.label("新对局执棋：");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.pending_color, Color::Red, "红（先手）");
                    ui.radio_value(&mut self.pending_color, Color::Black, "黑（后手）");
                });
                if ui.button("开始新对局").clicked() {
                    self.new_game();
                }

                ui.add_space(8.0);
                ui.add(
                    egui::Slider::new(&mut self.difficulty, 1..=5)
                        .text("AI 难度（1–5）"),
                );

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("悔棋").clicked() {
                        self.undo();
                    }
                    if ui
                        .add_enabled(!self.game_over(), egui::Button::new("认输"))
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
        let board_w = (FILES as f32 - 1.0) * CELL + 2.0 * MARGIN;
        let board_h = (RANKS as f32 - 1.0) * CELL + 2.0 * MARGIN;
        let (response, painter) =
            ui.allocate_painter(egui::vec2(board_w, board_h), egui::Sense::click());
        let rect = response.rect;
        let origin = rect.min + egui::vec2(MARGIN, MARGIN);

        // When the human plays Black the board is flipped so their pieces sit
        // at the bottom. `point` maps a board coord to its screen position;
        // the click handler applies the exact inverse.
        let flip = self.flip();
        let cjk = self.cjk_ok;
        let point = |f: i32, r: i32| -> egui::Pos2 {
            let (sf, sr) = to_screen(flip, f, r);
            egui::pos2(origin.x + sf as f32 * CELL, origin.y + sr as f32 * CELL)
        };

        // Colors.
        let bg = egui::Color32::from_rgb(0xEC, 0xCF, 0x93);
        let line = egui::Color32::from_rgb(0x5A, 0x3A, 0x1E);
        let disc = egui::Color32::from_rgb(0xF7, 0xE7, 0xC1);
        let red = egui::Color32::from_rgb(0xC0, 0x2A, 0x1B);
        let black = egui::Color32::from_rgb(0x20, 0x20, 0x20);
        let sel_col = egui::Color32::from_rgb(0x1E, 0x88, 0xE5);
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
                egui::FontId::proportional(26.0),
                line,
            );
            painter.text(
                egui::pos2(point(7, 0).x, river_y),
                egui::Align2::CENTER_CENTER,
                "漢 界",
                egui::FontId::proportional(26.0),
                line,
            );
        } else {
            painter.text(
                egui::pos2(point(4, 0).x, river_y),
                egui::Align2::CENTER_CENTER,
                "R I V E R",
                egui::FontId::proportional(24.0),
                line,
            );
            painter.text(
                egui::pos2(rect.center().x, rect.min.y + 12.0),
                egui::Align2::CENTER_CENTER,
                "No CJK font: pieces shown as letters (K/A/E/H/R/C/P)",
                egui::FontId::proportional(13.0),
                egui::Color32::from_rgb(0xB0, 0x30, 0x20),
            );
        }

        // Coordinate labels, anchored to the screen edges so they stay
        // readable whether or not the board is flipped.
        let bottom_r = if flip { 0 } else { 9 };
        for f in 0..FILES as i32 {
            let p = point(f, bottom_r);
            painter.text(
                egui::pos2(p.x, p.y + 22.0),
                egui::Align2::CENTER_CENTER,
                (b'A' + f as u8) as char,
                egui::FontId::proportional(16.0),
                line,
            );
        }
        let left_f = if flip { 8 } else { 0 };
        for r in 0..RANKS as i32 {
            let p = point(left_f, r);
            painter.text(
                egui::pos2(p.x - 22.0, p.y),
                egui::Align2::CENTER_CENTER,
                format!("{}", RANKS as i32 - r),
                egui::FontId::proportional(16.0),
                line,
            );
        }

        // Last-move highlight.
        if let Some(mv) = self.game.last_move {
            for sq in [mv.from, mv.to] {
                let c = point(file_of(sq), rank_of(sq));
                painter.circle_filled(c, CELL * 0.46, last_col);
            }
        }

        // Check highlight on the general in check.
        if let GameStatus::Check(c) = self.eff_status() {
            if let Some(g) = self.game.board.find_general(c) {
                let p = point(file_of(g), rank_of(g));
                painter.circle_filled(p, CELL * 0.48, check_col);
            }
        }

        let targets = self.legal_targets();

        // Pieces.
        for i in 0..CELLS {
            let Some(piece) = self.game.board.get(i) else {
                continue;
            };
            let c = point(file_of(i), rank_of(i));
            let pc = if piece.color == Color::Red { red } else { black };
            painter.circle_filled(c, CELL * 0.42, disc);
            painter.circle_stroke(c, CELL * 0.42, egui::Stroke::new(2.0, pc));
            if self.selected == Some(i) {
                painter.circle_stroke(c, CELL * 0.46, egui::Stroke::new(3.0, sel_col));
            }
            painter.text(
                c,
                egui::Align2::CENTER_CENTER,
                if cjk { piece.glyph() } else { piece.ascii() },
                egui::FontId::proportional(if cjk { 30.0 } else { 26.0 }),
                pc,
            );
        }

        // Legal-target markers.
        for &t in &targets {
            let c = point(file_of(t), rank_of(t));
            if self.game.board.get(t).is_some() {
                painter.circle_stroke(c, CELL * 0.46, egui::Stroke::new(3.0, dot_col));
            } else {
                painter.circle_filled(c, 7.0, dot_col);
            }
        }

        // Click -> nearest on-screen intersection.
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let sff = ((pos.x - origin.x) / CELL).round();
                let srr = ((pos.y - origin.y) / CELL).round();
                if (0.0..FILES as f32).contains(&sff) && (0.0..RANKS as f32).contains(&srr) {
                    let (sf, sr) = (sff as i32, srr as i32);
                    let cell = egui::pos2(
                        origin.x + sf as f32 * CELL,
                        origin.y + sr as f32 * CELL,
                    );
                    if cell.distance(pos) <= CELL * 0.5 {
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
        app.rebuild_engine();
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
    fn ai_moves_first_when_human_is_black() {
        let mut app = XiangqiApp::headless();
        app.pending_color = Color::Black;
        app.new_game(); // human Black, AI Red, Red to move
        app.difficulty = 1;
        app.rebuild_engine();
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
        app.rebuild_engine();
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
