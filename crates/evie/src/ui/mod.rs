//! The evie window: neuron atlas, chat, steering rack, search and neuron inspector.

mod atlas;
mod chat;
mod inspector;
mod side;
pub mod theme;

use crate::state::{Phase, Shared};
use atlas::{AtlasView, Geometry, Images};
use egui::{Align2, Color32, FontId, Frame, Margin, RichText, Stroke, Vec2};
use evie_core::LensSel;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use theme::*;

/// Scripted actions for automated screenshots (`--shot`).
#[derive(Debug, Clone, Default)]
pub struct Script {
    pub shot: Option<PathBuf>,
    pub prompt: Option<String>,
    pub preset: Option<String>,
    pub select: Option<(usize, usize)>,
    pub search: Option<String>,
    pub steer: Vec<evie_core::Intervention>,
    pub tab: Option<String>,
    pub delay: f32,
    pub exit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Neuron,
    Workspace,
    Agent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtlasMode {
    Token,
    Peak,
}

pub struct App {
    pub shared: Arc<Shared>,
    view: AtlasView,
    geo: Option<Geometry>,
    pub selected: Option<(usize, usize)>,
    /// Token pinned by clicking in the chat (None = follow the latest token).
    pub pinned_token: Option<usize>,
    /// Token under the pointer in the chat this frame.
    pub hovered_token: Option<usize>,
    pub lens: LensSel,
    pub tab: Tab,
    pub mode: AtlasMode,
    pub input: String,
    pub search_input: String,
    // atlas animation (texture layout)
    target: Vec<f32>,
    display: Vec<f32>,
    values_img: Arc<Vec<u8>>,
    values_gen: u64,
    shown_key: (u64, usize, AtlasMode),
    marks_img: Arc<Vec<u8>>,
    marks_gen: u64,
    marks_key: (u64, u64, Option<(usize, usize)>),
    peak: Vec<i8>,
    peak_frames: (u64, usize),
    last_frame: Instant,
    pub glow: f32,
    script: Script,
    script_state: ScriptState,
    pub workspace_req: Option<(usize, Instant)>,
    pub dose_edit: f32,
    pending_focus: Option<(usize, usize)>,
    /// Frames in the trace, sampled once per UI frame (avoids re-locking the trace mid-frame).
    trace_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScriptState {
    WaitModel,
    Setup,
    WaitReply,
    Settle(Instant),
    Shooting,
    Done,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, shared: Arc<Shared>, script: Script) -> Self {
        theme::install(&cc.egui_ctx);
        if let Some(rs) = cc.wgpu_render_state.as_ref() {
            atlas::install(rs);
            log::info!("rendering with {:?}", rs.adapter.get_info().name);
        }
        let _ = shared.ctx.set(cc.egui_ctx.clone());
        Self {
            shared,
            view: AtlasView::default(),
            geo: None,
            selected: None,
            pinned_token: None,
            hovered_token: None,
            lens: LensSel::Named("J".into()),
            tab: Tab::Neuron,
            mode: AtlasMode::Token,
            input: String::new(),
            search_input: String::new(),
            target: Vec::new(),
            display: Vec::new(),
            values_img: Arc::new(Vec::new()),
            values_gen: 0,
            shown_key: (u64::MAX, usize::MAX, AtlasMode::Token),
            marks_img: Arc::new(Vec::new()),
            marks_gen: 0,
            marks_key: (u64::MAX, u64::MAX, None),
            peak: Vec::new(),
            peak_frames: (u64::MAX, 0),
            last_frame: Instant::now(),
            glow: 1.6,
            script_state: if script.shot.is_some() { ScriptState::WaitModel } else { ScriptState::Done },
            script,
            workspace_req: None,
            dose_edit: 1.0,
            pending_focus: None,
            trace_len: 0,
        }
    }

    /// The token the atlas shows: hovered in chat, else pinned, else the newest.
    pub fn shown_token(&self) -> Option<usize> {
        let n = self.trace_len;
        if n == 0 {
            return None;
        }
        self.hovered_token.or(self.pinned_token).map(|t| t.min(n - 1)).or(Some(n - 1))
    }

    pub fn select(&mut self, layer: usize, neuron: usize, focus: bool) {
        self.selected = Some((layer, neuron));
        self.tab = Tab::Neuron;
        if let Some(r) = self.shared.rack.lock().get(layer, neuron) {
            self.dose_edit = r.value;
        }
        if focus {
            self.pending_focus = Some((layer, neuron));
        }
    }

    fn update_atlas_images(&mut self, ctx: &egui::Context) {
        let Some(geo) = self.geo else { return };
        let (tw, th) = geo.tex_size();
        let n_tex = (tw * th) as usize;
        if self.target.len() != n_tex {
            self.target = vec![0.0; n_tex];
            self.display = vec![0.0; n_tex];
        }
        let shown = self.shown_token();
        let trace = self.shared.trace.read();
        // Peak mode: strongest activation of each neuron over the whole trace.
        if self.mode == AtlasMode::Peak && (self.peak_frames.0 != trace.version || self.peak_frames.1 != trace.frames.len()) {
            let start = if self.peak_frames.0 == trace.version || self.peak.len() == geo.layers * geo.neurons { self.peak_frames.1.min(trace.frames.len()) } else { 0 };
            if self.peak.len() != geo.layers * geo.neurons || start == 0 {
                self.peak = vec![0; geo.layers * geo.neurons];
            }
            for f in &trace.frames[start..] {
                for (p, &v) in self.peak.iter_mut().zip(f.iter()) {
                    if v.unsigned_abs() > p.unsigned_abs() {
                        *p = v;
                    }
                }
            }
            self.peak_frames = (trace.version, trace.frames.len());
            self.shown_key.0 = u64::MAX;
        }
        let key = (trace.version, shown.unwrap_or(usize::MAX), self.mode);
        if key != self.shown_key {
            self.shown_key = key;
            let src: Option<&[i8]> = match self.mode {
                AtlasMode::Peak if !self.peak.is_empty() => Some(&self.peak),
                _ => shown.and_then(|t| trace.frames.get(t)).map(|f| &f[..]),
            };
            match src {
                Some(f) if f.len() == geo.layers * geo.neurons => {
                    for layer in 0..geo.layers {
                        for nn in 0..geo.neurons {
                            self.target[geo.texel_index(layer, nn)] = f[layer * geo.neurons + nn] as f32 / 127.0;
                        }
                    }
                }
                _ => self.target.iter_mut().for_each(|v| *v = 0.0),
            }
        }
        drop(trace);
        // Ease the displayed values toward the target (activity "afterglow").
        let dt = self.last_frame.elapsed().as_secs_f32().min(0.1);
        self.last_frame = Instant::now();
        let k = 1.0 - (-dt * 18.0).exp();
        let mut moving = false;
        let img = Arc::make_mut(&mut self.values_img);
        img.resize(n_tex, 0);
        for ((d, &t), o) in self.display.iter_mut().zip(&self.target).zip(img.iter_mut()) {
            let diff = t - *d;
            if diff.abs() > 0.004 {
                *d += diff * k;
                moving = true;
            } else {
                *d = t;
            }
            *o = ((*d * 127.0).round() as i8) as u8;
        }
        self.values_gen += 1;
        if moving {
            ctx.request_repaint_after(Duration::from_millis(14));
        }

        // Markers: steered neurons and search hits.
        let rack_v = self.shared.rack.lock().version;
        let search_v = self.shared.search.lock().version;
        let mkey = (rack_v, search_v, None);
        if mkey != self.marks_key || self.marks_img.len() != n_tex {
            self.marks_key = mkey;
            let img = Arc::make_mut(&mut self.marks_img);
            img.clear();
            img.resize(n_tex, 0);
            for r in self.shared.search.lock().results.iter().filter(|r| r.layer < geo.layers && r.neuron < geo.neurons) {
                img[geo.texel_index(r.layer, r.neuron)] |= atlas::mark::HIT;
            }
            for iv in self.shared.rack.lock().items.iter().filter(|i| i.enabled && i.layer < geo.layers && i.neuron < geo.neurons) {
                let up = match iv.mode {
                    evie_core::Mode::Add => iv.value >= 0.0,
                    _ => true,
                };
                img[geo.texel_index(iv.layer, iv.neuron)] |= if up { atlas::mark::STEER_UP } else { atlas::mark::STEER_DOWN };
            }
            self.marks_gen += 1;
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        let status = self.shared.status.lock().clone();
        ui.horizontal_centered(|ui| {
            ui.add_space(4.0);
            let (r, _) = ui.allocate_exact_size(Vec2::splat(22.0), egui::Sense::hover());
            ui.painter().rect_filled(r, 6, Color32::from_rgb(13, 148, 136));
            ui.painter().text(r.center(), Align2::CENTER_CENTER, "E", FontId::new(13.0, strong()), Color32::WHITE);
            ui.label(RichText::new("evie").family(strong()).size(16.0));
            ui.label(RichText::new("neuron microscope").size(12.0).color(FAINT));
            ui.add_space(12.0);

            let current = self.shared.engine().map(|e| e.id.clone());
            let installed = evie_core::registry::installed();
            egui::ComboBox::from_id_salt("model")
                .selected_text(RichText::new(current.clone().unwrap_or_else(|| "no model".into())).family(mono()).size(12.5))
                .show_ui(ui, |ui| {
                    for m in evie_core::registry::MODELS {
                        let have = installed.iter().any(|x| x.id == m.id);
                        let label = if have { m.id.to_string() } else { format!("{} — download {:.1} GB", m.id, m.approx_gb) };
                        if ui.selectable_label(current.as_deref() == Some(m.id), label).on_hover_text(m.about).clicked() {
                            self.shared.send(if have { crate::state::Job::Load(m.id.into()) } else { crate::state::Job::Download(m.id.into()) });
                        }
                    }
                });

            if let Some(eng) = self.shared.engine() {
                ui.label(
                    RichText::new(format!("{} layers · {} neurons/layer · {} total", eng.n_layers(), eng.n_neurons(), group(eng.n_layers() * eng.n_neurons())))
                        .size(11.5)
                        .color(DIM),
                );
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(6.0);
                let mcp = self.shared.mcp_url.lock().clone();
                match mcp {
                    Some(url) => {
                        let r = ui.add(egui::Label::new(RichText::new(format!("● MCP {}", url.trim_start_matches("http://"))).family(mono()).size(11.5).color(ACCENT)).sense(egui::Sense::click()));
                        if r.on_hover_text("Agents (Claude Code, Codex…) can read and steer neurons here. Click to copy.").clicked() {
                            ui.ctx().copy_text(url);
                        }
                    }
                    None => {
                        ui.label(RichText::new("○ MCP off").size(11.5).color(FAINT));
                    }
                }
                ui.add_space(10.0);
                let generating = self.shared.chat.lock().generating;
                if generating || status.tok_s > 0.0 {
                    ui.label(RichText::new(format!("{:.1} tok/s", status.tok_s)).family(mono()).size(11.5).color(if generating { UP } else { DIM }));
                }
                match status.phase {
                    Phase::Loading | Phase::Downloading => {
                        if let Some(p) = status.progress {
                            ui.add(egui::ProgressBar::new(p).desired_width(140.0).desired_height(6.0));
                        } else {
                            ui.spinner();
                        }
                        ui.label(RichText::new(&status.message).size(11.5).color(DIM));
                    }
                    Phase::Failed => {
                        ui.label(RichText::new(&status.message).size(11.5).color(DANGER));
                    }
                    _ => {}
                }
            });
        });
    }

    fn atlas_panel(&mut self, ui: &mut egui::Ui) {
        let Some(eng) = self.shared.engine() else {
            self.welcome(ui);
            return;
        };
        let geo = Geometry::new(eng.n_layers(), eng.n_neurons());
        if self.geo != Some(geo) {
            self.geo = Some(geo);
            self.shown_key.0 = u64::MAX;
            self.marks_key.0 = u64::MAX;
            self.peak_frames = (u64::MAX, 0);
        }
        self.update_atlas_images(ui.ctx());

        // Header row: what the atlas is showing + mode toggles.
        let shown = self.shown_token();
        let trace_len = self.shared.trace.read().frames.len();
        ui.horizontal(|ui| {
            ui.label(heading("Neuron atlas"));
            let what = match (self.mode, shown) {
                (AtlasMode::Peak, _) => format!("peak activation over {trace_len} tokens"),
                (_, Some(t)) => {
                    let tok = self.shared.trace.read().tokens.get(t).map(|x| x.text.clone()).unwrap_or_default();
                    format!("token {t} {:?}{}", tok, if self.hovered_token.is_none() && self.pinned_token.is_none() { " · live" } else { "" })
                }
                _ => "send a message to see the model think".into(),
            };
            ui.label(RichText::new(what).family(mono()).size(11.5).color(DIM));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.selectable_value(&mut self.mode, AtlasMode::Peak, RichText::new("peak").size(11.5));
                ui.selectable_value(&mut self.mode, AtlasMode::Token, RichText::new("token").size(11.5));
                if self.pinned_token.is_some() && ui.small_button("follow live").clicked() {
                    self.pinned_token = None;
                }
                ui.add(egui::Slider::new(&mut self.glow, 0.5..=6.0).show_value(false).text(RichText::new("gain").size(11.0).color(DIM)));
            });
        });

        let images = Images { values: self.values_img.clone(), values_gen: self.values_gen, marks: self.marks_img.clone(), marks_gen: self.marks_gen };
        let resp = self.view.show(ui, &geo, images, self.selected, self.glow);
        if let Some(focus) = self.pending_focus.take().or_else(|| self.shared.focus.lock().take()) {
            self.selected = Some(focus);
            self.view.focus(resp.rect, &geo, focus.0, focus.1);
        }
        if let Some((l, n)) = resp.clicked {
            self.select(l, n, false);
        }
        self.atlas_overlay(ui, &resp, &geo, &eng);
    }

    fn atlas_overlay(&mut self, ui: &mut egui::Ui, resp: &atlas::AtlasResponse, geo: &Geometry, eng: &evie_core::Engine) {
        let painter = ui.painter_at(resp.rect);
        let z = self.view.zoom;
        // Layer labels above each tile.
        for layer in 0..geo.layers {
            let o = geo.tile_origin(layer);
            let p = self.view.to_screen(resp.rect, o) - Vec2::new(0.0, 3.0);
            if resp.rect.expand(40.0).contains(p) {
                let size = (z * geo.gap_y * 0.42).clamp(8.0, 15.0);
                painter.text(p, Align2::LEFT_BOTTOM, format!("L{layer}"), FontId::new(size, mono()), DIM);
            }
        }
        // Steered neurons: rings that stay visible at any zoom.
        let rack = self.shared.rack.lock().items.clone();
        let t = ui.input(|i| i.time) as f32;
        for iv in rack.iter().filter(|i| i.enabled && i.layer < geo.layers && i.neuron < geo.neurons) {
            let c = self.view.to_screen(resp.rect, geo.cell_center(iv.layer, iv.neuron));
            let up = !(iv.mode == evie_core::Mode::Add && iv.value < 0.0);
            let col = if up { UP } else { DOWN };
            let r = (z * 0.9).max(5.0) + 1.5 * (t * 3.0).sin().abs();
            painter.circle_stroke(c, r, Stroke::new(1.6, col));
            painter.circle_stroke(c, r + 4.0, Stroke::new(1.0, col.gamma_multiply(0.35)));
            if z < 6.0 {
                painter.text(c + Vec2::new(r + 5.0, 0.0), Align2::LEFT_CENTER, iv.describe(), FontId::new(10.5, mono()), col);
            }
        }
        if !rack.is_empty() {
            ui.ctx().request_repaint_after(Duration::from_millis(33));
        }
        if let Some((l, n)) = self.selected {
            let c = self.view.to_screen(resp.rect, geo.cell_center(l, n));
            painter.circle_stroke(c, (z * 0.8).max(7.0), Stroke::new(2.0, ACCENT));
            painter.circle_stroke(c, (z * 0.8).max(7.0) + 5.0, Stroke::new(1.0, ACCENT.gamma_multiply(0.4)));
        }
        // Hover card.
        if let Some((l, n)) = self.view.hover {
            let shown = self.shown_token();
            let v = shown.and_then(|t| self.shared.trace.read().frames.get(t).map(|f| crate::state::Trace::decode(f[l * geo.neurons + n])));
            let read = self.shared.readout(&self.lens, l, n);
            let unit = eng.dose_unit(l, n);
            resp.response.clone().on_hover_ui_at_pointer(|ui| {
                ui.set_max_width(360.0);
                ui.horizontal(|ui| {
                    ui.label(mono_text(format!("L{l} N{n}"), 14.0).strong());
                    if let Some(v) = v {
                        ui.label(mono_text(format!("{v:+.3}"), 12.5).color(if v >= 0.0 { UP } else { DOWN }));
                    }
                    if let Some(iv) = self.shared.rack.lock().get(l, n) {
                        ui.label(mono_text(format!("steered {}", iv.describe()), 11.0).color(ACCENT));
                    }
                });
                match read {
                    Some(r) => {
                        token_row(ui, &format!("{} +", self.lens.label()), r.up.iter().take(7).map(|(t, _)| eng.token_str(*t)), UP);
                        token_row(ui, &format!("{} −", self.lens.label()), r.down.iter().take(7).map(|(t, _)| eng.token_str(*t)), DOWN);
                    }
                    None => {
                        ui.label(RichText::new("reading neuron…").size(11.0).color(FAINT));
                    }
                }
                ui.label(RichText::new(format!("dose 1.0 = {unit:+.2} activation · click to inspect")).size(10.5).color(FAINT));
            });
        }
    }

    fn welcome(&mut self, ui: &mut egui::Ui) {
        let status = self.shared.status.lock().clone();
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.22);
            ui.label(RichText::new("evie").family(strong()).size(44.0));
            ui.label(RichText::new("read and steer every neuron of a language model").size(15.0).color(DIM));
            ui.add_space(26.0);
            match status.phase {
                Phase::Loading | Phase::Downloading => {
                    ui.add(egui::ProgressBar::new(status.progress.unwrap_or(0.0)).desired_width(420.0).animate(status.progress.is_none()));
                    ui.label(RichText::new(&status.message).color(DIM));
                }
                _ => {
                    if status.phase == Phase::Failed {
                        ui.label(RichText::new(&status.message).color(DANGER));
                        ui.add_space(10.0);
                    }
                    for m in evie_core::registry::MODELS {
                        let installed = evie_core::registry::is_installed(m.id);
                        let label = if installed { format!("Load {}", m.id) } else { format!("Download {} ({:.1} GB)", m.id, m.approx_gb) };
                        if ui.add_sized([360.0, 34.0], egui::Button::new(RichText::new(label).size(14.0))).on_hover_text(m.about).clicked() {
                            self.shared.send(if installed { crate::state::Job::Load(m.id.into()) } else { crate::state::Job::Download(m.id.into()) });
                        }
                        ui.label(RichText::new(m.about).size(11.5).color(FAINT));
                        ui.add_space(8.0);
                    }
                }
            }
        });
    }

    fn run_script(&mut self, ctx: &egui::Context) {
        let ready = self.shared.engine().is_some();
        match self.script_state {
            ScriptState::Done | ScriptState::Shooting => {}
            ScriptState::WaitModel => {
                if ready {
                    self.script_state = ScriptState::Setup;
                }
                ctx.request_repaint_after(Duration::from_millis(100));
            }
            ScriptState::Setup => {
                if let Some(p) = self.script.preset.clone() {
                    side::apply_preset(self, &p);
                }
                if !self.script.steer.is_empty() {
                    self.shared.rack.lock().set(self.script.steer.clone(), None);
                }
                if let Some((l, n)) = self.script.select {
                    self.select(l, n, true);
                }
                if let Some(q) = self.script.search.clone() {
                    self.search_input = q.clone();
                    self.shared.analyse(crate::state::AJob::Search(q, self.lens.clone()));
                }
                match self.script.tab.as_deref() {
                    Some("workspace") => self.tab = Tab::Workspace,
                    Some("agent") => self.tab = Tab::Agent,
                    _ => {}
                }
                if let Some(p) = self.script.prompt.clone() {
                    self.shared.send_user(&p, None);
                    self.script_state = ScriptState::WaitReply;
                } else {
                    self.script_state = ScriptState::Settle(Instant::now());
                }
                ctx.request_repaint();
            }
            ScriptState::WaitReply => {
                let c = self.shared.chat.lock();
                if !c.generating {
                    self.script_state = ScriptState::Settle(Instant::now());
                }
                drop(c);
                ctx.request_repaint_after(Duration::from_millis(100));
            }
            ScriptState::Settle(t) => {
                let searching = self.shared.search.lock().running;
                if t.elapsed().as_secs_f32() >= self.script.delay.max(1.5) && !searching {
                    self.script_state = ScriptState::Shooting;
                    let path = self.script.shot.clone().unwrap();
                    let exit = self.script.exit;
                    let ctx2 = ctx.clone();
                    ctx.request_screenshot(move |img| {
                        let (w, h) = (img.size[0] as u32, img.size[1] as u32);
                        let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                        match image::save_buffer(&path, &bytes, w, h, image::ColorType::Rgba8) {
                            Ok(()) => log::info!("screenshot saved to {}", path.display()),
                            Err(e) => log::error!("screenshot failed: {e}"),
                        }
                        if exit {
                            ctx2.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                }
                ctx.request_repaint_after(Duration::from_millis(100));
            }
        }
    }
}

pub fn group(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A row of token chips with a small label.
pub fn token_row<'a>(ui: &mut egui::Ui, label: &str, toks: impl Iterator<Item = &'a str>, color: Color32) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(4.0, 4.0);
        ui.label(mono_text(label, 10.5).color(FAINT));
        for t in toks {
            chip(ui, t, color);
        }
    });
}

pub fn chip(ui: &mut egui::Ui, text: &str, color: Color32) -> egui::Response {
    let shown = text.replace('\n', "↵").replace(' ', "·");
    Frame::new()
        .fill(color.gamma_multiply(0.13))
        .stroke(Stroke::new(1.0, color.gamma_multiply(0.35)))
        .corner_radius(4)
        .inner_margin(Margin::symmetric(5, 1))
        .show(ui, |ui| ui.add(egui::Label::new(mono_text(shown, 11.5).color(theme::lerp(color, TEXT, 0.45))).wrap_mode(egui::TextWrapMode::Extend)))
        .response
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.trace_len = self.shared.trace.read().frames.len();
        self.run_script(&ctx);
        self.hovered_token = None;

        egui::Panel::top("top")
            .exact_size(42.0)
            .frame(Frame::new().fill(BG).inner_margin(Margin::symmetric(10, 0)).stroke(Stroke::new(1.0, BORDER)))
            .show(ui, |ui| self.top_bar(ui));

        if self.shared.engine().is_some() {
            egui::Panel::left("side")
                .resizable(true)
                .default_size(320.0)
                .min_size(260.0)
                .frame(Frame::new().fill(PANEL).inner_margin(Margin::same(12)).stroke(Stroke::new(1.0, BORDER)))
                .show(ui, |ui| side::show(self, ui));
            egui::Panel::right("chat")
                .resizable(true)
                .default_size(430.0)
                .min_size(320.0)
                .frame(Frame::new().fill(PANEL).inner_margin(Margin::same(12)).stroke(Stroke::new(1.0, BORDER)))
                .show(ui, |ui| chat::show(self, ui));
            egui::Panel::bottom("inspector")
                .resizable(true)
                .default_size(300.0)
                .min_size(160.0)
                .frame(Frame::new().fill(PANEL).inner_margin(Margin::same(12)).stroke(Stroke::new(1.0, BORDER)))
                .show(ui, |ui| inspector::show(self, ui));
        }
        egui::CentralPanel::default().frame(Frame::new().fill(BG).inner_margin(Margin::same(10))).show(ui, |ui| self.atlas_panel(ui));
    }
}
