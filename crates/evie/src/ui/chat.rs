//! Right panel: streaming chat. Messages render as token chips tinted by the selected neuron's
//! activation; hovering a token shows the model's whole state at that token in the atlas.

use super::App;
use super::theme::*;
use crate::state::{Message, Trace, TraceTok};
use egui::{Color32, Frame, Key, KeyboardShortcut, Margin, Modifiers, RichText, ScrollArea, Stroke, Ui};
use evie_core::Role;
use std::ops::Range;

pub fn show(app: &mut App, ui: &mut Ui) {
    let generating = app.shared.chat.lock().generating;
    ui.horizontal(|ui| {
        ui.label(heading("Chat"));
        let steering = app.shared.rack.lock().items.iter().filter(|i| i.enabled).count();
        if steering > 0 {
            ui.label(RichText::new(format!("steered by {steering} neuron{}", if steering == 1 { "" } else { "s" })).size(11.5).color(UP));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add_enabled(!generating, egui::Button::new(RichText::new("clear").size(12.0)).small()).clicked() {
                let mut c = app.shared.chat.lock();
                c.messages.clear();
                drop(c);
                *app.shared.trace.write() = Default::default();
                app.pinned_token = None;
            }
            settings(app, ui);
        });
    });
    ui.add_space(4.0);

    egui::Panel::bottom("composer").frame(Frame::new().inner_margin(Margin { top: 8, ..Default::default() })).show(ui, |ui| composer(app, ui, generating));

    let messages = app.shared.chat.lock().messages.clone();
    let shared = app.shared.clone();
    let trace = shared.trace.read();
    let sel = app.selected;
    let n = app.shared.engine().map(|e| e.n_neurons()).unwrap_or(1);
    ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
        ui.set_width(ui.available_width());
        if messages.is_empty() {
            ui.add_space(20.0);
            ui.label(RichText::new("Say something to the model. Every token it reads or writes lights up the atlas.").size(12.5).color(FAINT));
            ui.add_space(6.0);
            for s in ["Who are you?", "What do you think about humans?", "Tell me a short story about the sea."] {
                if ui.add(egui::Button::new(RichText::new(s).size(12.5).color(DIM)).frame(true)).clicked() {
                    app.shared.send_user(s, None);
                }
            }
        }
        for m in &messages {
            message(app, ui, m, &trace, sel, n);
            ui.add_space(10.0);
        }
    });
}

fn settings(app: &mut App, ui: &mut Ui) {
    ui.menu_button(RichText::new("settings").size(12.0), |ui| {
        ui.set_min_width(260.0);
        let mut c = app.shared.chat.lock();
        let s = &mut c.settings;
        ui.add(egui::Slider::new(&mut s.temperature, 0.0..=1.5).text("temperature"));
        ui.add(egui::Slider::new(&mut s.max_new, 16..=1024).text("max tokens"));
        ui.add(egui::Slider::new(&mut s.repetition_penalty, 1.0..=1.5).text("repetition penalty"));
        ui.checkbox(&mut s.compare, "also generate an unsteered reply")
            .on_hover_text("Runs a baseline alongside in the same batch, so you can see exactly what the steering changed.");
        ui.separator();
        ui.label(RichText::new("system prompt").size(11.5).color(DIM));
        ui.add(egui::TextEdit::multiline(&mut c.system).desired_rows(3).hint_text("(none)"));
    });
}

fn composer(app: &mut App, ui: &mut Ui, generating: bool) {
    ui.horizontal(|ui| {
        let w = ui.available_width() - 70.0;
        let edit = egui::TextEdit::multiline(&mut app.input)
            .desired_rows(2)
            .desired_width(w)
            .hint_text("Message the model…  (Enter to send)")
            .return_key(Some(KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter)));
        let r = ui.add(edit);
        let enter = r.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) && !i.modifiers.shift);
        ui.vertical(|ui| {
            if generating {
                if ui.add_sized([60.0, 40.0], egui::Button::new(RichText::new("stop").color(DANGER))).clicked() {
                    app.shared.request_stop();
                }
            } else if (ui.add_sized([60.0, 40.0], egui::Button::new(RichText::new("send").color(ACCENT))).clicked() || enter) && !app.input.trim().is_empty() {
                let text = std::mem::take(&mut app.input);
                app.pinned_token = None;
                app.shared.send_user(&text, None);
            }
        });
    });
}

/// The message's own tokens, minus chat-template scaffolding.
fn content_range(tokens: &[TraceTok], r: &Range<usize>) -> Range<usize> {
    let (mut s, mut e) = (r.start.min(tokens.len()), r.end.min(tokens.len()));
    let is = |i: usize, t: &str| tokens.get(i).is_some_and(|x| x.text == t);
    if s < e && is(s, "<|im_start|>") {
        s += 1;
        if s < e && ["user", "assistant", "system"].iter().any(|t| is(s, t)) {
            s += 1;
        }
        if s < e && is(s, "\n") {
            s += 1;
        }
    }
    if e > s && is(e - 1, "\n") {
        e -= 1;
    }
    if e > s && is(e - 1, "<|im_end|>") {
        e -= 1;
    }
    s..e
}

fn message(app: &mut App, ui: &mut Ui, m: &Message, trace: &Trace, sel: Option<(usize, usize)>, n: usize) {
    let (who, color) = match m.role {
        Role::User => (m.via.clone().unwrap_or_else(|| "you".into()), DIM),
        Role::Assistant => ("model".to_string(), ACCENT),
        Role::System => ("system".to_string(), FAINT),
    };
    ui.horizontal(|ui| {
        ui.label(RichText::new(who).family(strong()).size(11.5).color(color));
        if !m.steering.is_empty() {
            ui.label(RichText::new(m.steering.join("  ")).family(mono()).size(10.5).color(UP.gamma_multiply(0.8)));
        }
    });
    let fill = if m.role == Role::User { RAISED } else { Color32::TRANSPARENT };
    Frame::new().fill(fill).corner_radius(8).inner_margin(Margin::symmetric(if m.role == Role::User { 10 } else { 0 }, 6)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        let range = m.tokens.as_ref().map(|r| content_range(&trace.tokens, r)).filter(|r| r.end <= trace.frames.len() && !r.is_empty());
        match range {
            Some(r) => tokens(app, ui, trace, r, sel, n),
            None => {
                ui.label(RichText::new(&m.text).size(13.5));
            }
        }
        if !m.done {
            let t = ui.input(|i| i.time);
            let a = ((t * 3.0).sin() * 0.5 + 0.5) as f32;
            ui.label(RichText::new("▍").color(ACCENT.gamma_multiply(a)));
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));
        }
    });
    if let Some(b) = &m.baseline {
        Frame::new().stroke(Stroke::new(1.0, BORDER)).corner_radius(8).inner_margin(Margin::symmetric(10, 6)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("unsteered").family(strong()).size(10.5).color(FAINT));
            ui.label(RichText::new(b).size(12.5).color(DIM));
        });
    }
}

fn tokens(app: &mut App, ui: &mut Ui, trace: &Trace, r: Range<usize>, sel: Option<(usize, usize)>, n: usize) {
    let shown = app.shown_token();
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 2.0);
        for t in r {
            let tok = &trace.tokens[t];
            let v = sel.and_then(|(l, nn)| trace.frames.get(t).map(|f| Trace::decode(f[l * n + nn])));
            let text = tok.text.replace('\n', "↵");
            let bg = match v {
                Some(v) if v.abs() > 0.02 => act_color(v).gamma_multiply(0.55 + 0.45 * v.abs().sqrt()),
                _ => Color32::TRANSPARENT,
            };
            let rich = RichText::new(text).size(13.5).background_color(bg).color(if tok.text.starts_with("<|") { FAINT } else { TEXT });
            let resp = ui.add(egui::Label::new(rich).sense(egui::Sense::click()));
            if shown == Some(t) && (app.pinned_token.is_some() || app.hovered_token.is_some()) {
                ui.painter().rect_stroke(resp.rect.expand(1.0), 3, Stroke::new(1.0, ACCENT), egui::StrokeKind::Outside);
            }
            if resp.hovered() {
                app.hovered_token = Some(t);
            }
            if resp.clicked() {
                app.pinned_token = if app.pinned_token == Some(t) { None } else { Some(t) };
            }
            let resp = match v {
                Some(v) => resp.on_hover_text(RichText::new(format!("token {t} · activation {v:+.3}")).family(mono()).size(11.5)),
                None => resp,
            };
            let _ = resp;
            if tok.text.contains('\n') {
                ui.end_row();
            }
        }
    });
}
