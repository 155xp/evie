//! Colours, fonts and egui styling.

use egui::{Color32, FontData, FontDefinitions, FontFamily, FontId, RichText, Stroke, Visuals};
use std::sync::Arc;

pub const BG: Color32 = Color32::from_rgb(9, 11, 14);
pub const PANEL: Color32 = Color32::from_rgb(14, 17, 22);
pub const RAISED: Color32 = Color32::from_rgb(21, 25, 32);
pub const BORDER: Color32 = Color32::from_rgb(32, 38, 48);
pub const TEXT: Color32 = Color32::from_rgb(222, 227, 235);
pub const DIM: Color32 = Color32::from_rgb(132, 142, 158);
pub const FAINT: Color32 = Color32::from_rgb(78, 87, 101);
pub const ACCENT: Color32 = Color32::from_rgb(45, 212, 191);
pub const UP: Color32 = Color32::from_rgb(245, 166, 35);
pub const DOWN: Color32 = Color32::from_rgb(236, 72, 153);
pub const DANGER: Color32 = Color32::from_rgb(248, 113, 113);

pub fn mono() -> FontFamily {
    FontFamily::Name("mono".into())
}

pub fn strong() -> FontFamily {
    FontFamily::Name("strong".into())
}

pub fn mono_text(s: impl Into<String>, size: f32) -> RichText {
    RichText::new(s).family(mono()).size(size)
}

pub fn heading(s: impl Into<String>) -> RichText {
    RichText::new(s.into().to_uppercase()).family(strong()).size(10.5).color(DIM)
}

/// Colour for a signed, normalised activation in [-1, 1].
pub fn act_color(v: f32) -> Color32 {
    let t = v.abs().clamp(0.0, 1.0).sqrt();
    let base = if v >= 0.0 { UP } else { DOWN };
    lerp(RAISED, base, t)
}

pub fn lerp(a: Color32, b: Color32, t: f32) -> Color32 {
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(m(a.r(), b.r()), m(a.g(), b.g()), m(a.b(), b.b()))
}

fn system_cjk() -> Option<(Vec<u8>, u32)> {
    // Chinese tokens are common in Qwen's vocabulary; use a system CJK font when one exists.
    let candidates = [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
        ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 2),
        ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc", 2),
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
        ("/System/Library/Fonts/PingFang.ttc", 0),
        ("C:\\Windows\\Fonts\\msyh.ttc", 0),
    ];
    candidates.iter().find_map(|(p, i)| std::fs::read(p).ok().map(|b| (b, *i)))
}

pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert("inter".into(), Arc::new(FontData::from_static(include_bytes!("../../assets/fonts/Inter-Regular.ttf"))));
    fonts.font_data.insert("inter-semibold".into(), Arc::new(FontData::from_static(include_bytes!("../../assets/fonts/Inter-SemiBold.ttf"))));
    fonts.font_data.insert("jbmono".into(), Arc::new(FontData::from_static(include_bytes!("../../assets/fonts/JetBrainsMono-Regular.ttf"))));
    let mut fallback = vec!["emoji-icon-font".to_string(), "NotoEmoji-Regular".to_string()];
    if let Some((bytes, index)) = system_cjk() {
        let mut fd = FontData::from_owned(bytes);
        fd.index = index;
        fonts.font_data.insert("cjk".into(), Arc::new(fd));
        fallback.insert(0, "cjk".into());
    }
    let fallback: Vec<String> = fallback.into_iter().filter(|f| fonts.font_data.contains_key(f)).collect();
    let family = |first: &str| std::iter::once(first.to_string()).chain(fallback.iter().cloned()).collect::<Vec<_>>();
    fonts.families.insert(FontFamily::Proportional, family("inter"));
    fonts.families.insert(FontFamily::Monospace, family("jbmono"));
    fonts.families.insert(mono(), family("jbmono"));
    fonts.families.insert(strong(), family("inter-semibold"));
    ctx.set_fonts(fonts);

    let mut v = Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = RAISED;
    v.extreme_bg_color = BG;
    v.faint_bg_color = RAISED;
    v.override_text_color = Some(TEXT);
    v.selection.bg_fill = ACCENT.gamma_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.inactive.bg_fill = RAISED;
    v.widgets.inactive.weak_bg_fill = RAISED;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.hovered.bg_fill = Color32::from_rgb(28, 34, 43);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(28, 34, 43);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.6));
    v.widgets.active.bg_fill = Color32::from_rgb(34, 41, 52);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(34, 41, 52);
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.corner_radius = 6.into();
    }
    v.window_corner_radius = 10.into();
    ctx.set_visuals_of(egui::Theme::Dark, v);
    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = egui::vec2(8.0, 6.0);
        s.spacing.button_padding = egui::vec2(10.0, 5.0);
        s.spacing.slider_width = 150.0;
        s.text_styles.insert(egui::TextStyle::Body, FontId::new(13.5, FontFamily::Proportional));
        s.text_styles.insert(egui::TextStyle::Button, FontId::new(13.0, FontFamily::Proportional));
        s.text_styles.insert(egui::TextStyle::Small, FontId::new(11.0, FontFamily::Proportional));
        s.text_styles.insert(egui::TextStyle::Monospace, FontId::new(12.5, mono()));
        s.text_styles.insert(egui::TextStyle::Heading, FontId::new(17.0, strong()));
    });
}
