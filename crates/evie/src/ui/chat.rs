//! Left panel: the conversation. Model picker and settings on top, messages, the neurons being
//! steered right above the input (steering changes the conversation), then the composer.
//! With a neuron selected, every token is shaded by that neuron's activation.

use super::theme::*;
use super::{App, Drawer};
use crate::state::{Message, Trace, TraceTok};
use egui::{Frame, Key, KeyboardShortcut, Margin, Modifiers, RichText, ScrollArea, Stroke, Ui};
use evie_core::steering::presets_4b;
use evie_core::{Mode, Role};
use std::ops::Range;

pub fn presets_available(app: &App) -> bool {
    app.shared.engine().is_some_and(|e| e.n_layers() == 32 && e.n_neurons() == 9216)
}

pub fn apply_preset(app: &mut App, name: &str) {
    if !presets_available(app) {
        return;
    }
    if let Some((n, _, ivs)) = presets_4b().into_iter().find(|(n, _, _)| n.eq_ignore_ascii_case(name)) {
        app.shared.rack.lock().set(ivs, Some(n.to_string()));
    }
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let generating = app.shared.chat.lock().generating;
    header(app, ui, generating);
    ui.add_space(10.0);

    egui::Panel::bottom("composer").frame(Frame::new().inner_margin(Margin { top: 10, ..Default::default() })).show(ui, |ui| {
        steering(app, ui);
        composer(app, ui, generating);
    });

    let messages = app.shared.chat.lock().messages.clone();
    let shared = app.shared.clone();
    let trace = shared.trace.read();
    let sel = app.selected;
    let n = app.shared.engine().map(|e| e.n_neurons()).unwrap_or(1);
    ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
        ui.set_width(ui.available_width());
        if messages.is_empty() {
            ui.add_space(8.0);
            for s in ["Who are you?", "What do you think about humans?", "Tell me a short story about the sea."] {
                if ui.add(egui::Button::new(RichText::new(s).color(DIM)).frame(false)).clicked() {
                    app.shared.send_user(s, None);
                }
            }
        }
        for m in &messages {
            message(app, ui, m, &trace, sel, n);
            ui.add_space(14.0);
        }
    });
}

fn header(app: &mut App, ui: &mut Ui, generating: bool) {
    ui.horizontal(|ui| {
        let current = app.shared.engine().map(|e| e.id.clone()).unwrap_or_default();
        egui::ComboBox::from_id_salt("model").selected_text(RichText::new(&current).family(strong()).size(14.0)).show_ui(ui, |ui| {
            for m in evie_core::registry::MODELS {
                let have = evie_core::registry::is_installed(m.id);
                let label = if have { m.id.to_string() } else { format!("{}  ·  download {:.1} GB", m.id, m.approx_gb) };
                if ui.selectable_label(current == m.id, label).clicked() && current != m.id {
                    app.shared.send(if have { crate::state::Job::Load(m.id.into()) } else { crate::state::Job::Download(m.id.into()) });
                }
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            menu(app, ui);
            if ui.add_enabled(!generating, egui::Button::new("New chat").frame(false)).clicked() {
                app.shared.chat.lock().messages.clear();
                *app.shared.trace.write() = Default::default();
                app.pinned_token = None;
                if app.drawer == Drawer::Token {
                    app.drawer = Drawer::Closed;
                }
            }
            if generating {
                let tps = app.shared.status.lock().tok_s;
                ui.label(mono_text(format!("{tps:.1} tok/s"), 11.5).color(FAINT));
            }
        });
    });
}

fn menu(app: &mut App, ui: &mut Ui) {
    ui.menu_button(RichText::new("···").size(14.0), |ui| {
        ui.set_min_width(300.0);
        if presets_available(app) {
            ui.menu_button("Steering presets", |ui| {
                for (name, about, ivs) in presets_4b() {
                    let desc: Vec<String> = ivs.iter().map(|i| i.describe()).collect();
                    if ui.button(name).on_hover_text(format!("{about}\n{}", desc.join(", "))).clicked() {
                        app.shared.rack.lock().set(ivs, Some(name.to_string()));
                        ui.close();
                    }
                }
            });
            ui.separator();
        }
        let mut c = app.shared.chat.lock();
        let s = &mut c.settings;
        ui.add(egui::Slider::new(&mut s.temperature, 0.0..=1.5).text("temperature"));
        ui.add(egui::Slider::new(&mut s.max_new, 16..=1024).text("max tokens"));
        ui.checkbox(&mut s.compare, "Show the unsteered reply too");
        ui.add_space(4.0);
        ui.add(egui::TextEdit::multiline(&mut c.system).desired_rows(2).hint_text("System prompt"));
        drop(c);
        if let Some(url) = app.shared.mcp_url.lock().clone() {
            ui.separator();
            ui.label(RichText::new("Let an agent drive (MCP)").size(12.0).color(DIM));
            let cmd = format!("claude mcp add --transport http evie {url}");
            if ui.add(egui::Button::new(mono_text(&cmd, 11.0)).wrap_mode(egui::TextWrapMode::Wrap)).on_hover_text("Copy").clicked() {
                ui.ctx().copy_text(cmd);
                ui.close();
            }
        }
    });
}

/// The neurons being steered, each with a live slider. Hidden when nothing is steered.
fn steering(app: &mut App, ui: &mut Ui) {
    let rack = app.shared.rack.lock().clone();
    if rack.items.is_empty() {
        return;
    }
    ui.horizontal(|ui| {
        ui.label(RichText::new(rack.preset.clone().unwrap_or_else(|| "Steering".into())).size(12.0).color(DIM));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add(egui::Button::new(RichText::new("Clear").size(12.0).color(DIM)).frame(false)).clicked() {
                app.shared.rack.lock().set(Vec::new(), None);
            }
        });
    });
    let mut items = rack.items.clone();
    let mut changed = false;
    let mut remove = None;
    for (i, iv) in items.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut iv.enabled, "").changed();
            let id = ui.add(egui::Label::new(mono_text(iv.id(), 12.5).color(if iv.enabled { TEXT } else { FAINT })).sense(egui::Sense::click()));
            if id.on_hover_text(if iv.label.is_empty() { "Inspect".to_string() } else { iv.label.clone() }).clicked() {
                app.select(iv.layer, iv.neuron, true);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(egui::Button::new(RichText::new("×").size(14.0).color(DIM)).frame(false)).clicked() {
                    remove = Some(i);
                }
                let range = match iv.mode {
                    Mode::Add => -3.0..=3.0,
                    Mode::Mul => -2.0..=6.0,
                    Mode::Set => -20.0..=20.0,
                };
                ui.spacing_mut().slider_width = (ui.available_width() - 60.0).max(60.0);
                changed |= ui.add(egui::Slider::new(&mut iv.value, range).step_by(0.05).fixed_decimals(2)).changed();
            });
        });
    }
    if let Some(i) = remove {
        items.remove(i);
        changed = true;
    }
    if changed {
        let mut r = app.shared.rack.lock();
        r.items = items;
        r.preset = None;
        r.touch();
    }
    ui.add_space(6.0);
}

fn composer(app: &mut App, ui: &mut Ui, generating: bool) {
    Frame::new().fill(RAISED).stroke(Stroke::new(1.0, BORDER)).corner_radius(10).inner_margin(Margin::symmetric(10, 8)).show(ui, |ui| {
        ui.horizontal(|ui| {
            let w = ui.available_width() - 58.0;
            let edit = egui::TextEdit::multiline(&mut app.input)
                .desired_rows(2)
                .desired_width(w)
                .frame(egui::Frame::NONE)
                .hint_text("Message")
                .return_key(Some(KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter)));
            let r = ui.add(edit);
            let enter = r.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) && !i.modifiers.shift);
            if generating {
                if ui.add_sized([50.0, 30.0], egui::Button::new("Stop")).clicked() {
                    app.shared.request_stop();
                }
            } else if (ui.add_sized([50.0, 30.0], egui::Button::new("Send")).clicked() || enter) && !app.input.trim().is_empty() {
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
    let user = m.role == Role::User;
    if user || !m.steering.is_empty() || m.via.is_some() {
        ui.horizontal(|ui| {
            if let Some(v) = &m.via {
                ui.label(RichText::new(v).size(11.5).color(FAINT));
            }
            if !user && !m.steering.is_empty() {
                ui.label(mono_text(m.steering.join("  "), 11.0).color(FAINT));
            }
        });
    }
    let frame = if user { Frame::new().fill(RAISED).corner_radius(10).inner_margin(Margin::symmetric(12, 8)) } else { Frame::new() };
    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        let range = m.tokens.as_ref().map(|r| content_range(&trace.tokens, r)).filter(|r| r.end <= trace.frames.len() && !r.is_empty());
        match range {
            Some(r) => tokens(app, ui, trace, r, sel, n),
            None => {
                ui.label(RichText::new(&m.text).size(14.0));
            }
        }
    });
    if let Some(b) = &m.baseline {
        ui.add_space(6.0);
        Frame::new().stroke(Stroke::new(1.0, BORDER)).corner_radius(10).inner_margin(Margin::symmetric(12, 8)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Without steering").size(11.5).color(FAINT));
            ui.label(RichText::new(b).size(13.0).color(DIM));
        });
    }
}

fn tokens(app: &mut App, ui: &mut Ui, trace: &Trace, r: Range<usize>, sel: Option<(usize, usize)>, n: usize) {
    let shown = app.shown_token();
    let marked = app.pinned_token.is_some() || app.hovered_token.is_some();
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 3.0);
        for t in r {
            let tok = &trace.tokens[t];
            let v = sel.and_then(|(l, nn)| trace.frames.get(t).map(|f| Trace::decode(f[l * n + nn])));
            let bg = match v {
                Some(v) if v.abs() > 0.08 => act_fill(v),
                _ => egui::Color32::TRANSPARENT,
            };
            let rich = RichText::new(tok.text.replace('\n', " ")).size(14.0).background_color(bg).color(if tok.text.starts_with("<|") { FAINT } else { TEXT });
            let resp = ui.add(egui::Label::new(rich).sense(egui::Sense::click()));
            if marked && shown == Some(t) {
                ui.painter().rect_stroke(resp.rect.expand(1.0), 3, Stroke::new(1.0, TEXT), egui::StrokeKind::Outside);
            }
            if resp.hovered() {
                app.hovered_token = Some(t);
            }
            if resp.clicked() {
                app.open_token(t);
            }
            if let Some(v) = v {
                resp.on_hover_text(mono_text(format!("{v:+.3}"), 11.5));
            }
            if tok.text.contains('\n') {
                ui.end_row();
            }
        }
    });
}
