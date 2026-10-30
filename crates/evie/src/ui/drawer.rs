//! The drawer under the atlas: search results, one neuron, or one token. Hidden otherwise.

use super::theme::*;
use super::{App, Drawer, chip, token_row};
use crate::state::{AJob, Trace};
use egui::{Align2, FontId, Pos2, RichText, ScrollArea, Sense, Stroke, Ui, Vec2};
use evie_core::{Engine, Intervention};
use std::time::{Duration, Instant};

pub fn show(app: &mut App, ui: &mut Ui, eng: &Engine) {
    // Claim the panel's whole height so it doesn't shrink while a view is still loading.
    ui.set_min_height(ui.available_height());
    match app.drawer {
        Drawer::Results => results(app, ui, eng),
        Drawer::Neuron => neuron(app, ui, eng),
        Drawer::Token => token(app, ui, eng),
        Drawer::Closed => {}
    }
}

/// Title row with an optional back link and a close button. Returns true if closed.
fn title(app: &mut App, ui: &mut Ui, text: RichText, back: bool) -> bool {
    let mut closed = false;
    ui.horizontal(|ui| {
        if back && ui.add(egui::Button::new(RichText::new("‹ Results").color(DIM)).frame(false)).clicked() {
            app.drawer = Drawer::Results;
            app.from_results = false;
        }
        ui.label(text);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add(egui::Button::new(RichText::new("×").size(16.0).color(DIM)).frame(false)).on_hover_text("Close (Esc)").clicked() {
                closed = true;
            }
        });
    });
    if closed {
        app.close_drawer();
    }
    closed
}

fn results(app: &mut App, ui: &mut Ui, eng: &Engine) {
    let s = app.shared.search.lock().clone();
    let head = if s.running { s.status.clone() } else { format!("“{}”  ·  {}", s.query, s.status) };
    if title(app, ui, RichText::new(head).color(DIM), false) {
        return;
    }
    if s.running {
        ui.spinner();
        return;
    }
    ui.add_space(6.0);
    let lens = app.lens.label().to_string();
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        for r in &s.results {
            let resp = ui
                .horizontal(|ui| {
                    ui.set_width(ui.available_width());
                    ui.add_sized([96.0, 20.0], egui::Label::new(mono_text(format!("L{} N{}", r.layer, r.neuron), 12.5)));
                    ui.add_sized([14.0, 20.0], egui::Label::new(mono_text(if r.sign > 0 { "+" } else { "−" }, 13.0).color(DIM)));
                    let ranks: Vec<String> = r.lenses.iter().map(|(l, rank, _)| format!("{l} {rank}")).collect();
                    ui.add_sized([150.0, 20.0], egui::Label::new(mono_text(ranks.join(" · "), 11.0).color(FAINT)))
                        .on_hover_text("Rank of the best matching word in this pole's readout, per lens (0 = top token)");
                    if let Some((_, _, top)) = r.lenses.iter().find(|(l, _, _)| *l == lens).or(r.lenses.first()) {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for t in top.iter().take(7) {
                            chip(ui, t);
                        }
                    }
                })
                .response
                .interact(Sense::click());
            if resp.hovered() {
                ui.painter().rect_filled(resp.rect.expand(2.0), 4, egui::Color32::from_white_alpha(10));
            }
            if resp.clicked() {
                app.select(r.layer, r.neuron, true);
                app.from_results = true;
                app.dose_edit = if r.sign > 0 { 1.0 } else { -1.0 };
            }
        }
        if s.results.is_empty() && !s.matched.is_empty() {
            ui.label(RichText::new("No neurons found.").color(FAINT));
        }
    });
    let _ = eng;
}

fn neuron(app: &mut App, ui: &mut Ui, eng: &Engine) {
    let Some((l, n)) = app.selected else {
        app.drawer = Drawer::Closed;
        return;
    };
    let back = app.from_results;
    if title(app, ui, mono_text(format!("L{l} N{n}"), 15.0), back) {
        return;
    }
    ui.add_space(8.0);
    let nn = eng.n_neurons();
    ui.columns(2, |cols| {
        let ui = &mut cols[0];
        sparkline(app, ui, l, n, nn);
        ui.add_space(10.0);
        steer(app, ui, l, n, eng);
        let ui = &mut cols[1];
        ScrollArea::vertical().id_salt("reads").auto_shrink([false, false]).show(ui, |ui| {
            for sel in eng.lens_list() {
                let label = sel.label().to_string();
                ui.label(mono_text(&label, 11.5).color(if sel == app.lens { TEXT } else { FAINT }));
                match app.shared.readout(&sel, l, n) {
                    Some(r) => {
                        token_row(ui, "+", r.up.iter().take(9).map(|(t, _)| eng.token_str(*t)));
                        token_row(ui, "−", r.down.iter().take(9).map(|(t, _)| eng.token_str(*t)));
                    }
                    None => {
                        ui.spinner();
                    }
                }
                ui.add_space(6.0);
            }
        });
    });
}

fn sparkline(app: &mut App, ui: &mut Ui, l: usize, n: usize, nn: usize) {
    let shared = app.shared.clone();
    let trace = shared.trace.read();
    let vals: Vec<f32> = trace.frames.iter().map(|f| Trace::decode(f[l * nn + n])).collect();
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 64.0), Sense::click());
    let p = ui.painter_at(rect);
    let mid = rect.center().y;
    p.line_segment([Pos2::new(rect.left(), mid), Pos2::new(rect.right(), mid)], Stroke::new(1.0, BORDER));
    if vals.is_empty() {
        p.text(rect.center(), Align2::CENTER_CENTER, "activation over the conversation", FontId::new(11.0, mono()), FAINT);
        return;
    }
    let w = rect.width() / vals.len() as f32;
    let max = vals.iter().fold(0.05f32, |a, v| a.max(v.abs()));
    for (i, v) in vals.iter().enumerate() {
        let h = (v / max) * (rect.height() * 0.46);
        let x = rect.left() + i as f32 * w;
        let r = egui::Rect::from_min_max(Pos2::new(x, mid.min(mid - h)), Pos2::new(x + w.max(1.0), mid.max(mid - h)));
        p.rect_filled(r, 0, if *v >= 0.0 { TEXT } else { DIM });
    }
    if let Some(t) = app.shown_token() {
        let x = rect.left() + (t as f32 + 0.5) * w;
        p.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(1.0, FAINT));
    }
    if let Some(pos) = resp.hover_pos() {
        let t = (((pos.x - rect.left()) / w) as usize).min(vals.len() - 1);
        app.hovered_token = Some(t);
        let tok = trace.tokens.get(t).map(|x| x.text.clone()).unwrap_or_default();
        resp.on_hover_text(mono_text(format!("token {t} {tok:?}  {:+.3}", vals[t]), 11.5));
    }
}

fn steer(app: &mut App, ui: &mut Ui, l: usize, n: usize, eng: &Engine) {
    let existing = app.shared.rack.lock().get(l, n).cloned();
    ui.horizontal(|ui| {
        ui.spacing_mut().slider_width = (ui.available_width() - 150.0).max(80.0);
        let r = ui.add(egui::Slider::new(&mut app.dose_edit, -3.0..=3.0).step_by(0.05).fixed_decimals(2));
        if r.changed()
            && let Some(mut iv) = existing.clone()
        {
            iv.value = app.dose_edit;
            app.shared.rack.lock().upsert(iv);
        }
        if existing.is_some() {
            if ui.button("Stop steering").clicked() {
                app.shared.rack.lock().remove(l, n);
            }
        } else if ui.button("Steer").clicked() {
            app.shared.rack.lock().upsert(Intervention::add(l, n, app.dose_edit));
        }
    });
    ui.label(RichText::new(format!("dose 1 adds {:+.1} to this neuron on every token", eng.dose_unit(l, n))).size(11.5).color(FAINT));
}

fn token(app: &mut App, ui: &mut Ui, eng: &Engine) {
    let tok = app.pinned_token.and_then(|t| app.shared.trace.read().tokens.get(t).map(|x| (t, x.text.clone())));
    let Some((t, tok)) = tok else {
        app.drawer = Drawer::Closed;
        return;
    };
    if title(app, ui, mono_text(format!("token {t}  {tok:?}"), 14.0), false) {
        app.pinned_token = None;
        return;
    }
    ui.label(RichText::new("What each layer is disposed to say at this token").size(12.0).color(FAINT));
    // Debounce: recompute once the token has been stable briefly.
    let have = app.shared.workspace.lock().as_ref().map(|w| w.token);
    if have != Some(t) {
        match app.workspace_req {
            Some((rt, at)) if rt == t => {
                if at.elapsed() > Duration::from_millis(150) {
                    app.shared.analyse(AJob::Workspace(t));
                    app.workspace_req = Some((t, Instant::now() + Duration::from_secs(3600)));
                }
            }
            _ => app.workspace_req = Some((t, Instant::now())),
        }
        ui.ctx().request_repaint_after(Duration::from_millis(60));
        ui.spinner();
        return;
    }
    let ws = app.shared.workspace.lock().clone();
    let Some(ws) = ws else { return };
    ui.add_space(6.0);
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        egui::Grid::new("ws").num_columns(ws.rows.len() + 1).spacing([18.0, 4.0]).show(ui, |ui| {
            ui.label("");
            for (lens, _) in &ws.rows {
                ui.label(mono_text(lens, 11.5).color(FAINT));
            }
            ui.end_row();
            for layer in 0..eng.n_layers() {
                ui.label(mono_text(format!("L{layer}"), 11.5).color(FAINT));
                for (_, rows) in &ws.rows {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for (tk, _) in rows[layer].iter().take(4) {
                            chip(ui, eng.token_str(*tk));
                        }
                    });
                }
                ui.end_row();
            }
        });
    });
}
