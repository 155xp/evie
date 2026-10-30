//! Monochrome palette, fonts and egui styling.

use egui::{Color32, FontData, FontDefinitions, FontFamily, FontId, RichText, Stroke, Visuals};
use std::sync::Arc;

pub const BG: Color32 = Color32::from_rgb(0, 0, 0);
pub const RAISED: Color32 = Color32::from_rgb(14, 14, 14);
pub const HOVER: Color32 = Color32::from_rgb(26, 26, 26);
pub const BORDER: Color32 = Color32::from_rgb(34, 34, 34);
pub const TEXT: Color32 = Color32::from_rgb(242, 242, 242);
pub const DIM: Color32 = Color32::from_rgb(140, 140, 140);
pub const FAINT: Color32 = Color32::from_rgb(82, 82, 82);

pub fn mono() -> FontFamily {
    FontFamily::Name("mono".into())
}

pub fn strong() -> FontFamily {
    FontFamily::Name("strong".into())
}

pub fn mono_text(s: impl Into<String>, size: f32) -> RichText {
    RichText::new(s).family(mono()).size(size)
}

/// Background for a token or cell with a signed activation in [-1, 1]: brighter = stronger.
pub fn act_fill(v: f32) -> Color32 {
    let t = v.abs().clamp(0.0, 1.0).powf(0.8);
    Color32::from_white_alpha((t * 170.0) as u8)
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
    v.panel_fill = BG;
    v.window_fill = RAISED;
    v.extreme_bg_color = RAISED;
    v.faint_bg_color = RAISED;
    v.code_bg_color = RAISED;
    v.override_text_color = Some(TEXT);
    v.selection.bg_fill = Color32::from_white_alpha(56);
    v.selection.stroke = Stroke::new(1.0, TEXT);
    v.hyperlink_color = TEXT;
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.window_shadow = egui::epaint::Shadow::NONE;
    v.popup_shadow = egui::epaint::Shadow::NONE;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.inactive.bg_fill = RAISED;
    v.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.hovered.bg_fill = HOVER;
    v.widgets.hovered.weak_bg_fill = HOVER;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_gray(70));
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.active.bg_fill = Color32::from_gray(40);
    v.widgets.active.weak_bg_fill = Color32::from_gray(40);
    v.widgets.active.bg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.active.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.open.weak_bg_fill = HOVER;
    v.slider_trailing_fill = true;
    for w in [&mut v.widgets.noninteractive, &mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.corner_radius = 6.into();
        w.expansion = 0.0;
    }
    v.window_corner_radius = 8.into();
    v.menu_corner_radius = 8.into();
    ctx.set_visuals_of(egui::Theme::Dark, v);
    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = egui::vec2(8.0, 6.0);
        s.spacing.button_padding = egui::vec2(10.0, 5.0);
        s.spacing.interact_size.y = 26.0;
        s.text_styles.insert(egui::TextStyle::Body, FontId::new(13.5, FontFamily::Proportional));
        s.text_styles.insert(egui::TextStyle::Button, FontId::new(13.0, FontFamily::Proportional));
        s.text_styles.insert(egui::TextStyle::Small, FontId::new(11.0, FontFamily::Proportional));
        s.text_styles.insert(egui::TextStyle::Monospace, FontId::new(12.5, mono()));
        s.text_styles.insert(egui::TextStyle::Heading, FontId::new(16.0, strong()));
    });
}
