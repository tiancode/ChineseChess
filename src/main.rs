//! Chinese Chess (Xiangqi) with an egui UI and a pluggable AI.

mod ai;
mod board;
mod game;
mod moves;
mod ui;

#[cfg(test)]
mod tests;

use eframe::egui;
use std::sync::Arc;

/// Try to locate a CJK-capable font so the piece glyphs render. Falls back to
/// `None` (egui's default font) if nothing is found.
fn load_cjk_font() -> Option<Vec<u8>> {
    const CANDIDATES: &[&str] = &[
        // Linux
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        // macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        // Windows
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simsun.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
    ];
    CANDIDATES.iter().find_map(|p| std::fs::read(p).ok())
}

/// Returns true if a CJK font was loaded (so Chinese glyphs render).
fn install_fonts(ctx: &egui::Context) -> bool {
    let Some(bytes) = load_cjk_font() else {
        eprintln!("warning: no CJK font found; pieces will render as letters.");
        return false;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert("cjk".to_owned(), Arc::new(egui::FontData::from_owned(bytes)));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "cjk".to_owned());
    }
    ctx.set_fonts(fonts);
    true
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([880.0, 700.0])
            .with_min_inner_size([760.0, 660.0])
            .with_title("中国象棋"),
        ..Default::default()
    };

    eframe::run_native(
        "中国象棋",
        options,
        Box::new(|cc| {
            let cjk_ok = install_fonts(&cc.egui_ctx);
            Ok(Box::new(ui::XiangqiApp::new(cc, cjk_ok)))
        }),
    )
}
