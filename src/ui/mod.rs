pub mod app;
pub use app::XiangqiApp;

use eframe::egui;

/// Piece palette, shared so the board pieces, the game-over banner, and the
/// window icon all stay in sync if the colours are ever retuned.
pub const PIECE_CREAM: egui::Color32 = egui::Color32::from_rgb(0xF7, 0xE7, 0xC1);
pub const PIECE_RED: egui::Color32 = egui::Color32::from_rgb(0xC0, 0x2A, 0x1B);
