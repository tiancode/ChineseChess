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

/// Render the General piece ("将") as a window/taskbar icon, in the same
/// cream-disc / red-ink style as the on-board pieces. Reuses the CJK font
/// located by [`load_cjk_font`]; returns `None` (no custom icon) if none is
/// available or the glyph cannot be rasterised, mirroring the text fallback.
fn build_icon() -> Option<egui::IconData> {
    use ab_glyph::{Font, FontVec};

    let bytes = load_cjk_font()?;
    let font = FontVec::try_from_vec(bytes).ok()?;

    const S: usize = 128;
    // As a fraction of the icon size: cream disc radius, red rim radius, and
    // the glyph height (sized to sit comfortably inside the disc).
    const DISC_R: f32 = 0.44;
    const RIM_R: f32 = 0.48;
    const GLYPH_SCALE: f32 = 0.66;

    let mut rgba = vec![0u8; S * S * 4];
    let center = S as f32 / 2.0;
    let r_in = S as f32 * DISC_R;
    let r_out = S as f32 * RIM_R;
    let with_alpha = |c: egui::Color32| [c.r(), c.g(), c.b(), 0xFF];
    let cream = with_alpha(ui::PIECE_CREAM);
    let red = with_alpha(ui::PIECE_RED);

    // Cream disc with a red rim; outside the disc stays transparent.
    for y in 0..S {
        for x in 0..S {
            let dx = x as f32 + 0.5 - center;
            let dy = y as f32 + 0.5 - center;
            let d = (dx * dx + dy * dy).sqrt();
            let px = if d <= r_in {
                cream
            } else if d <= r_out {
                red
            } else {
                continue;
            };
            let o = (y * S + x) * 4;
            rgba[o..o + 4].copy_from_slice(&px);
        }
    }

    // Rasterise '将' centred on the disc, alpha-blended in red.
    let glyph = font.glyph_id('将').with_scale(S as f32 * GLYPH_SCALE);
    if let Some(outlined) = font.outline_glyph(glyph) {
        let b = outlined.px_bounds();
        let base_x = center - (b.max.x - b.min.x) / 2.0;
        let base_y = center - (b.max.y - b.min.y) / 2.0;
        outlined.draw(|gx, gy, cov| {
            let xi = (base_x + gx as f32).round() as i32;
            let yi = (base_y + gy as f32).round() as i32;
            if xi < 0 || yi < 0 || xi >= S as i32 || yi >= S as i32 {
                return;
            }
            let o = (yi as usize * S + xi as usize) * 4;
            let a = cov.clamp(0.0, 1.0);
            let inv = 1.0 - a;
            for (dst, &src) in rgba[o..o + 3].iter_mut().zip(&red[..3]) {
                *dst = (*dst as f32 * inv + src as f32 * a).round() as u8;
            }
            rgba[o + 3] = (rgba[o + 3] as f32).max(a * 255.0).round() as u8;
        });
    }

    Some(egui::IconData {
        rgba,
        width: S as u32,
        height: S as u32,
    })
}

fn main() -> eframe::Result<()> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([880.0, 700.0])
        .with_min_inner_size([640.0, 560.0])
        .with_title("中国象棋");
    if let Some(icon) = build_icon() {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
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
