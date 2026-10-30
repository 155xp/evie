//! Bottom panel: the selected neuron (lens readouts, activation trace, steering), the residual-stream
//! workspace at the shown token, and the agent/MCP activity feed.

use super::theme::*;
use super::{App, Tab, chip, token_row};
use crate::state::{AJob, Trace};
use egui::{Align2, Color32, FontId, Frame, Margin, Pos2, RichText, ScrollArea, Sense, Stroke, Ui, Vec2};
use evie_core::{Intervention, LensSel};
use std::time::{Duration, Instant};

pub fn show(app: &mut App, ui: &mut Ui) {
    let Some(eng) = app.shared.engine() else { return };
    ui.horizontal(|ui| {
        for (t, name) in [(Tab::Neuron, "Neuron"), (Tab::Workspace, "Workspace"), (Tab::Agent, "Agent / MCP")] {
            if ui.selectable_label(app.tab == t, RichText::new(name).family(strong()).size(12.5)).clicked() {
                app.tab = t;
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            for sel in eng.lens_list().into_iter().rev() {
                if ui.selectable_label(app.lens == sel, RichText::new(format!("{} lens", sel.label())).family(mono()).size(11.0)).clicked() {
                    app.lens = sel;
                }
            }
            ui.label(RichText::new("hover readouts:").size(11.0).color(FAINT));
        });
    });
    ui.add_space(6.0);
    match app.tab {
        Tab::Neuron => neuron(app, ui, &eng),
        Tab::Workspace => workspace(app, ui, &eng),
        Tab::Agent => agent(app, ui),
    }
}

fn neuron(app: &mut App, ui: &mut Ui, eng: &evie_core::Engine) {
    let Some((l, n)) = app.selected else {
        ui.label(RichText::new("Hover the atlas to read neurons through the lenses; click one to inspect and steer it.").color(FAINT));
        ui.label(RichText::new("Each neuron's readout is unembed(J · W_down[:, n]) — what pushing it makes the model disposed to say.").size(11.5).color(FAINT));
        return;
    };
    let nn = eng.n_neurons();
    ui.columns(2, |cols| {
        let ui = &mut cols[0];
        ui.horizontal(|ui| {
            ui.label(mono_text(format!("L{l} N{n}"), 20.0).strong());
            let shown = app.shown_token();
            if let Some(v) = shown.and_then(|t| app.shared.trace.read().frames.get(t).map(|f| Trace::decode(f[l * nn + n]))) {
                ui.label(mono_text(format!("{v:+.3}"), 14.0).color(if v >= 0.0 { UP } else { DOWN })).on_hover_text("activation at the shown token, relative to this neuron's typical range");
            }
        });
        let unit = eng.dose_unit(l, n);
        ui.label(
            RichText::new(format!("‖W_down‖ {:.3} · median ‖resid L{l}‖ {:.1} · dose 1 = {unit:+.2}", eng.col_norms[l][n], eng.units[l]))
                .family(mono())
                .size(10.5)
                .color(FAINT),
        );
        ui.add_space(4.0);
        sparkline(app, ui, l, n, nn);
        ui.add_space(6.0);
        steer_controls(app, ui, l, n);
        let ui = &mut cols[1];
        ScrollArea::vertical().id_salt("reads").auto_shrink([false, false]).show(ui, |ui| {
            for sel in eng.lens_list() {
                let label = sel.label().to_string();
                match app.shared.readout(&sel, l, n) {
                    Some(r) => {
                        token_row(ui, &format!("{label:>5} +"), r.up.iter().take(10).map(|(t, _)| eng.token_str(*t)), UP);
                        token_row(ui, &format!("{label:>5} −"), r.down.iter().take(10).map(|(t, _)| eng.token_str(*t)), DOWN);
                    }
                    None => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(RichText::new(format!("reading through the {label} lens…")).size(11.5).color(FAINT));
                        });
                    }
                }
                ui.add_space(4.0);
            }
        });
    });
}

fn sparkline(app: &mut App, ui: &mut Ui, l: usize, n: usize, nn: usize) {
    let trace = app.shared.trace.read();
    let vals: Vec<f32> = trace.frames.iter().map(|f| Trace::decode(f[l * nn + n])).collect();
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 54.0), Sense::click());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 6, BG);
    let mid = rect.center().y;
    p.line_segment([Pos2::new(rect.left(), mid), Pos2::new(rect.right(), mid)], Stroke::new(1.0, BORDER));
    if vals.is_empty() {
        p.text(rect.center(), Align2::CENTER_CENTER, "activation over the conversation", FontId::new(11.0, mono()), FAINT);
        return;
    }
    let w = rect.width() / vals.len().max(1) as f32;
    let max = vals.iter().fold(0.05f32, |a, v| a.max(v.abs()));
    for (i, v) in vals.iter().enumerate() {
        let h = (v / max) * (rect.height() * 0.46);
        let x = rect.left() + i as f32 * w;
        let r = egui::Rect::from_min_max(Pos2::new(x, mid.min(mid - h)), Pos2::new(x + w.max(1.0), mid.max(mid - h)));
        p.rect_filled(r, 0, if *v >= 0.0 { UP } else { DOWN });
    }
    if let Some(t) = app.shown_token() {
        let x = rect.left() + (t as f32 + 0.5) * w;
        p.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(1.0, ACCENT));
    }
    if let Some(pos) = resp.hover_pos() {
        let t = (((pos.x - rect.left()) / w) as usize).min(vals.len() - 1);
        app.hovered_token = Some(t);
        let tok = trace.tokens.get(t).map(|x| x.text.clone()).unwrap_or_default();
        resp.on_hover_text(RichText::new(format!("token {t} {tok:?}: {:+.3}", vals[t])).family(mono()));
    }
}

fn steer_controls(app: &mut App, ui: &mut Ui, l: usize, n: usize) {
    let existing = app.shared.rack.lock().get(l, n).cloned();
    Frame::new().fill(RAISED).corner_radius(8).inner_margin(Margin::symmetric(10, 8)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(heading("Push this neuron"));
            if existing.is_some() {
                ui.label(RichText::new("live in rack").size(11.0).color(ACCENT));
            }
        });
        ui.spacing_mut().slider_width = ui.available_width() - 70.0;
        let r = ui.add(egui::Slider::new(&mut app.dose_edit, -3.0..=3.0).step_by(0.05).fixed_decimals(2).text("dose"));
        if r.changed()
            && let Some(mut iv) = existing.clone()
        {
            iv.value = app.dose_edit;
            app.shared.rack.lock().upsert(iv);
        }
        ui.horizontal(|ui| {
            for d in [-1.0f32, -0.6, 0.6, 1.0] {
                if ui.small_button(format!("{d:+}")).clicked() {
                    app.dose_edit = d;
                    let mut iv = existing.clone().unwrap_or_else(|| Intervention::add(l, n, d));
                    iv.value = d;
                    iv.enabled = true;
                    app.shared.rack.lock().upsert(iv);
                }
            }
            if existing.is_some() {
                if ui.button(RichText::new("remove").color(DANGER)).clicked() {
                    app.shared.rack.lock().remove(l, n);
                }
            } else if ui.button(RichText::new("steer").color(ACCENT)).clicked() {
                app.shared.rack.lock().upsert(Intervention::add(l, n, app.dose_edit));
            }
        });
    });
}

fn workspace(app: &mut App, ui: &mut Ui, eng: &evie_core::Engine) {
    let Some(t) = app.shown_token() else {
        ui.label(RichText::new("The workspace view reads the residual stream at a token through every lens, layer by layer.").color(FAINT));
        return;
    };
    // Debounce: only recompute once the shown token has been stable briefly.
    let have = app.shared.workspace.lock().as_ref().map(|w| w.token);
    if have != Some(t) {
        match app.workspace_req {
            Some((rt, at)) if rt == t => {
                if at.elapsed() > Duration::from_millis(180) {
                    app.shared.analyse(AJob::Workspace(t));
                    app.workspace_req = Some((t, Instant::now() + Duration::from_secs(3600)));
                }
            }
            _ => app.workspace_req = Some((t, Instant::now())),
        }
        ui.ctx().request_repaint_after(Duration::from_millis(60));
    }
    let tok = app.shared.trace.read().tokens.get(t).map(|x| x.text.clone()).unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("what the residual stream at token {t} {tok:?} is disposed to say")).size(12.0).color(DIM));
        if have != Some(t) {
            ui.spinner();
        }
    });
    let ws = app.shared.workspace.lock().clone();
    let Some(ws) = ws else { return };
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        egui::Grid::new("ws").num_columns(ws.rows.len() + 1).spacing([14.0, 3.0]).striped(true).show(ui, |ui| {
            ui.label(RichText::new("layer").size(11.0).color(FAINT));
            for (lens, _) in &ws.rows {
                ui.label(mono_text(format!("{lens} lens"), 11.0).color(if LensSel::parse(lens) == app.lens { ACCENT } else { DIM }));
            }
            ui.end_row();
            for layer in 0..eng.n_layers() {
                ui.label(mono_text(format!("L{layer}"), 11.0).color(FAINT));
                for (_, rows) in &ws.rows {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        for (tk, _) in rows[layer].iter().take(4) {
                            chip(ui, eng.token_str(*tk), Color32::from_rgb(120, 170, 255));
                        }
                    });
                }
                ui.end_row();
            }
        });
    });
}

fn agent(app: &mut App, ui: &mut Ui) {
    let url = app.shared.mcp_url.lock().clone();
    ui.columns(2, |cols| {
        let ui = &mut cols[0];
        match &url {
            Some(url) => {
                ui.label(RichText::new("Let an agent explore the model with you").family(strong()).size(14.0));
                ui.label(RichText::new("evie serves the Model Context Protocol. Agents can search neurons, read them through the lenses, steer, and chat — and you watch it happen here.").size(12.0).color(DIM));
                ui.add_space(6.0);
                for (label, cmd) in [
                    ("Claude Code", format!("claude mcp add --transport http evie {url}")),
                    ("Codex / others (stdio)", format!("evie mcp-stdio --url {url}")),
                ] {
                    ui.label(RichText::new(label).size(11.5).color(FAINT));
                    ui.horizontal(|ui| {
                        Frame::new().fill(BG).corner_radius(6).inner_margin(Margin::symmetric(8, 5)).show(ui, |ui| {
                            ui.label(mono_text(&cmd, 11.5));
                        });
                        if ui.small_button("copy").clicked() {
                            ui.ctx().copy_text(cmd.clone());
                        }
                    });
                }
            }
            None => {
                ui.label(RichText::new("MCP server is off (start without --no-mcp to enable).").color(FAINT));
            }
        }
        let ui = &mut cols[1];
        ui.label(heading("Activity"));
        ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            let log = app.shared.activity.lock().clone();
            if log.is_empty() {
                ui.label(RichText::new("No agent activity yet.").size(12.0).color(FAINT));
            }
            for a in log.iter() {
                ui.horizontal_wrapped(|ui| {
                    ui.label(mono_text(format!("{:>4}s", a.at.elapsed().as_secs()), 10.5).color(FAINT));
                    ui.label(RichText::new(&a.who).family(strong()).size(11.5).color(if a.who == "evie" { DIM } else { ACCENT }));
                    ui.label(RichText::new(&a.text).size(12.0));
                });
            }
        });
    });
}
