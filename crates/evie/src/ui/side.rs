//! Left panel: the steering rack, community presets and concept search.

use super::theme::*;
use super::{App, chip};
use crate::state::AJob;
use egui::{Frame, Margin, RichText, ScrollArea, Stroke, Ui};
use evie_core::steering::presets_4b;
use evie_core::Mode;

pub fn presets_available(app: &App) -> bool {
    app.shared.engine().is_some_and(|e| e.n_layers() == 32 && e.n_neurons() == 9216)
}

pub fn apply_preset(app: &mut App, name: &str) {
    if !presets_available(app) {
        app.shared.log("evie", format!("preset {name:?} needs Qwen3.5-4B"));
        return;
    }
    if let Some((n, _, ivs)) = presets_4b().into_iter().find(|(n, _, _)| n.eq_ignore_ascii_case(name)) {
        app.shared.rack.lock().set(ivs.clone(), Some(n.to_string()));
        if let Some(first) = ivs.first() {
            app.select(first.layer, first.neuron, false);
        }
    }
}

pub fn show(app: &mut App, ui: &mut Ui) {
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        rack(app, ui);
        ui.add_space(14.0);
        ui.separator();
        ui.add_space(8.0);
        search(app, ui);
    });
}

fn rack(app: &mut App, ui: &mut Ui) {
    let rack = app.shared.rack.lock().clone();
    ui.horizontal(|ui| {
        ui.label(heading("Steering"));
        if let Some(p) = &rack.preset {
            ui.label(RichText::new(p).size(11.5).color(ACCENT));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if !rack.items.is_empty() && ui.small_button("clear").clicked() {
                app.shared.rack.lock().set(Vec::new(), None);
            }
            let has_presets = presets_available(app);
            ui.add_enabled_ui(has_presets, |ui| {
                ui.menu_button(RichText::new("presets").size(12.0), |ui| {
                    ui.set_min_width(300.0);
                    for (name, about, ivs) in presets_4b() {
                        let desc: Vec<String> = ivs.iter().map(|i| i.describe()).collect();
                        let r = ui.button(RichText::new(name).size(13.0)).on_hover_text(format!("{about}\n{}", desc.join(", ")));
                        if r.clicked() {
                            app.shared.rack.lock().set(ivs, Some(name.to_string()));
                            ui.close();
                        }
                    }
                })
                .response
                .on_disabled_hover_text("Community recipes are for Qwen3.5-4B");
            });
        });
    });
    ui.add_space(4.0);
    if rack.items.is_empty() {
        ui.label(RichText::new("No neurons steered. Click a neuron in the atlas and push it, load a preset, or let an agent do it over MCP.").size(12.0).color(FAINT));
        return;
    }
    let mut remove = None;
    let mut changed = false;
    let mut items = rack.items.clone();
    for (i, iv) in items.iter_mut().enumerate() {
        let col = if iv.mode == Mode::Add && iv.value < 0.0 { DOWN } else { UP };
        Frame::new()
            .fill(RAISED)
            .stroke(Stroke::new(1.0, if iv.enabled { col.gamma_multiply(0.45) } else { BORDER }))
            .corner_radius(8)
            .inner_margin(Margin::symmetric(10, 7))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    changed |= ui.checkbox(&mut iv.enabled, "").changed();
                    let id = ui.add(egui::Label::new(mono_text(iv.id(), 13.0).color(if iv.enabled { TEXT } else { FAINT })).sense(egui::Sense::click()));
                    if id.on_hover_text("inspect").clicked() {
                        app.select(iv.layer, iv.neuron, true);
                    }
                    if !iv.label.is_empty() {
                        ui.label(RichText::new(&iv.label).size(11.5).color(DIM));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("x").on_hover_text("remove").clicked() {
                            remove = Some(i);
                        }
                        egui::ComboBox::from_id_salt(("mode", i)).width(52.0).selected_text(RichText::new(mode_name(iv.mode)).size(11.0)).show_ui(ui, |ui| {
                            for m in [Mode::Add, Mode::Mul, Mode::Set] {
                                changed |= ui.selectable_value(&mut iv.mode, m, mode_name(m)).changed();
                            }
                        });
                    });
                });
                let range = match iv.mode {
                    Mode::Add => -3.0..=3.0,
                    Mode::Mul => -2.0..=6.0,
                    Mode::Set => -20.0..=20.0,
                };
                ui.spacing_mut().slider_width = ui.available_width() - 70.0;
                changed |= ui.add(egui::Slider::new(&mut iv.value, range).step_by(0.05).fixed_decimals(2)).changed();
                if let Some(g) = &iv.gate {
                    let det: Vec<String> = g.detectors.iter().map(|d| format!("L{} N{}", d.layer, d.neuron)).collect();
                    ui.label(RichText::new(format!("gated: fires after Σ[{}] > {:.1}", det.join(" + "), g.threshold)).size(11.0).color(DIM));
                }
            });
        ui.add_space(4.0);
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
}

fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::Add => "add",
        Mode::Mul => "×",
        Mode::Set => "set",
    }
}

fn search(app: &mut App, ui: &mut Ui) {
    let Some(eng) = app.shared.engine() else { return };
    ui.label(heading("Find neurons"));
    ui.label(RichText::new("Rank all neurons by how strongly their readout says these words.").size(11.5).color(FAINT));
    ui.add_space(4.0);
    let mut go = false;
    ui.horizontal(|ui| {
        let r = ui.add(egui::TextEdit::singleline(&mut app.search_input).hint_text("pirate, 海盗").desired_width(ui.available_width() - 64.0));
        if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            go = true;
        }
        go |= ui.button("find").clicked();
    });
    ui.horizontal(|ui| {
        ui.label(RichText::new("lens").size(11.5).color(DIM));
        for sel in eng.lens_list() {
            let on = app.lens == sel;
            if ui.selectable_label(on, RichText::new(sel.label()).family(mono()).size(11.5)).clicked() {
                app.lens = sel;
            }
        }
    });
    if go && !app.search_input.trim().is_empty() {
        app.shared.search.lock().query = app.search_input.clone();
        app.shared.analyse(AJob::Search(app.search_input.clone(), app.lens.clone()));
    }
    let s = app.shared.search.lock().clone();
    if s.running {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(RichText::new(&s.status).size(11.5).color(DIM));
        });
    } else if !s.status.is_empty() {
        ui.label(RichText::new(&s.status).size(11.5).color(FAINT));
    }
    if !s.matched.is_empty() {
        ui.label(RichText::new(format!("tokens: {}", s.matched.join(" "))).family(mono()).size(10.5).color(FAINT));
    }
    ui.add_space(4.0);
    let lens_label = app.lens.label().to_string();
    for r in &s.results {
        let col = if r.sign > 0 { UP } else { DOWN };
        let selected = app.selected == Some((r.layer, r.neuron));
        let resp = Frame::new()
            .fill(if selected { RAISED } else { PANEL })
            .stroke(Stroke::new(1.0, if selected { ACCENT.gamma_multiply(0.6) } else { BORDER }))
            .corner_radius(7)
            .inner_margin(Margin::symmetric(8, 5))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(mono_text(format!("L{} N{}", r.layer, r.neuron), 12.5));
                    ui.label(mono_text(if r.sign > 0 { "+" } else { "−" }, 13.0).color(col));
                    ui.label(mono_text(format!("z {:.1}", r.z), 11.0).color(DIM));
                    for (lens, rank, _) in &r.lenses {
                        let c = if *rank == 0 { ACCENT } else if *rank < 10 { TEXT } else { FAINT };
                        ui.label(mono_text(format!("{lens}:{rank}"), 10.5).color(c)).on_hover_text(format!("best rank of a query word under the {lens} lens (0 = top token)"));
                    }
                });
                if let Some((_, _, top)) = r.lenses.iter().find(|(l, _, _)| *l == lens_label).or(r.lenses.first()) {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                        for t in top.iter().take(6) {
                            chip(ui, t, col);
                        }
                    });
                }
            })
            .response
            .interact(egui::Sense::click());
        if resp.clicked() {
            app.select(r.layer, r.neuron, true);
            app.dose_edit = if r.sign > 0 { 1.0 } else { -1.0 };
        }
        ui.add_space(3.0);
    }
}
