//! The evie window: conversation on the left, the neuron atlas filling the rest, and a drawer
//! under the atlas that appears only when there is something to show (search results, a neuron,
//! or a token).

mod atlas;
mod chat;
mod drawer;
pub mod theme;

use crate::state::{AJob, Phase, Shared};
use atlas::{AtlasView, Geometry, Images};
use egui::{Align2, FontId, Frame, Margin, RichText, Stroke, Vec2};
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

/// What the drawer under the atlas shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drawer {
    Closed,
    Results,
    Neuron,
    Token,
}

const GAIN: f32 = 1.4;

pub struct App {
    pub shared: Arc<Shared>,
    view: AtlasView,
    geo: Option<Geometry>,
    pub selected: Option<(usize, usize)>,
    /// Token pinned by clicking in the chat (None = follow the newest token).
    pub pinned_token: Option<usize>,
    /// Token under the pointer in the chat this frame.
    pub hovered_token: Option<usize>,
    pub lens: LensSel,
    pub drawer: Drawer,
    /// The neuron view was opened from search results (offer a way back).
    pub from_results: bool,
    pub input: String,
    pub search_input: String,
    // atlas animation (texture layout)
    target: Vec<f32>,
    display: Vec<f32>,
    values_img: Arc<Vec<u8>>,
    values_gen: u64,
    shown_key: (u64, usize),
    marks_img: Arc<Vec<u8>>,
    marks_gen: u64,
    marks_key: (u64, u64),
    last_frame: Instant,
    script: Script,
    script_state: ScriptState,
    pub workspace_req: Option<(usize, Instant)>,
    pub dose_edit: f32,
    pending_focus: Option<(usize, usize)>,
    /// Frames in the trace, sampled once per UI frame.
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
            drawer: Drawer::Closed,
            from_results: false,
            input: String::new(),
            search_input: String::new(),
            target: Vec::new(),
            display: Vec::new(),
            values_img: Arc::new(Vec::new()),
            values_gen: 0,
            shown_key: (u64::MAX, usize::MAX),
            marks_img: Arc::new(Vec::new()),
            marks_gen: 0,
            marks_key: (u64::MAX, u64::MAX),
            last_frame: Instant::now(),
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
        self.drawer = Drawer::Neuron;
        if let Some(r) = self.shared.rack.lock().get(layer, neuron) {
            self.dose_edit = r.value;
        }
        if focus {
            self.pending_focus = Some((layer, neuron));
        }
    }

    pub fn open_token(&mut self, t: usize) {
        self.pinned_token = Some(t);
        self.drawer = Drawer::Token;
    }

    pub fn close_drawer(&mut self) {
        self.drawer = Drawer::Closed;
        self.selected = None;
        self.from_results = false;
    }

    pub fn run_search(&mut self) {
        let q = self.search_input.trim().to_string();
        if q.is_empty() {
            return;
        }
        self.shared.search.lock().query = q.clone();
        self.shared.analyse(AJob::Search(q, self.lens.clone()));
        self.drawer = Drawer::Results;
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
        let key = (trace.version, shown.unwrap_or(usize::MAX));
        if key != self.shown_key {
            self.shown_key = key;
            match shown.and_then(|t| trace.frames.get(t)).filter(|f| f.len() == geo.layers * geo.neurons) {
                Some(f) => {
                    for layer in 0..geo.layers {
                        for nn in 0..geo.neurons {
                            self.target[geo.texel_index(layer, nn)] = f[layer * geo.neurons + nn] as f32 / 127.0;
                        }
                    }
                }
                None => self.target.iter_mut().for_each(|v| *v = 0.0),
            }
        }
        drop(trace);
        // Ease the displayed values toward the target so activity flows between tokens.
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
            ctx.request_repaint_after(Duration::from_millis(16));
        }

        // Markers: steered neurons and search hits.
        let rack_v = self.shared.rack.lock().version;
        let search_v = self.shared.search.lock().version;
        if (rack_v, search_v) != self.marks_key || self.marks_img.len() != n_tex {
            self.marks_key = (rack_v, search_v);
            let img = Arc::make_mut(&mut self.marks_img);
            img.clear();
            img.resize(n_tex, 0);
            for r in self.shared.search.lock().results.iter().filter(|r| r.layer < geo.layers && r.neuron < geo.neurons) {
                img[geo.texel_index(r.layer, r.neuron)] |= atlas::mark::HIT;
            }
            for iv in self.shared.rack.lock().items.iter().filter(|i| i.enabled && i.layer < geo.layers && i.neuron < geo.neurons) {
                img[geo.texel_index(iv.layer, iv.neuron)] |= atlas::mark::STEER_UP;
            }
            self.marks_gen += 1;
        }
    }

    fn main_area(&mut self, ui: &mut egui::Ui) {
        let Some(eng) = self.shared.engine() else {
            self.welcome(ui);
            return;
        };
        let geo = Geometry::new(eng.n_layers(), eng.n_neurons());
        if self.geo != Some(geo) {
            self.geo = Some(geo);
            self.shown_key.0 = u64::MAX;
            self.marks_key.0 = u64::MAX;
        }
        self.update_atlas_images(ui.ctx());

        // Search bar + lens switch.
        ui.horizontal(|ui| {
            let lens_w = 46.0 * eng.lens_list().len() as f32 + 10.0;
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.search_input)
                    .hint_text("Find neurons by word, e.g. ocean, sea")
                    .desired_width(ui.available_width() - lens_w)
                    .margin(Margin::symmetric(10, 6)),
            );
            if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.run_search();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                for sel in eng.lens_list().into_iter().rev() {
                    let on = self.lens == sel;
                    let txt = mono_text(sel.label(), 12.0).color(if on { TEXT } else { FAINT });
                    if ui.add(egui::Button::selectable(on, txt)).on_hover_text(lens_help(&sel)).clicked() {
                        self.lens = sel;
                    }
                }
            });
        });
        ui.add_space(8.0);

        if self.drawer != Drawer::Closed {
            egui::Panel::bottom("drawer")
                .resizable(true)
                .default_size(340.0)
                .min_size(150.0)
                .frame(Frame::new().fill(BG).inner_margin(Margin { top: 12, ..Default::default() }).stroke(Stroke::new(1.0, BORDER)))
                .show(ui, |ui| drawer::show(self, ui, &eng));
        }

        let images = Images { values: self.values_img.clone(), values_gen: self.values_gen, marks: self.marks_img.clone(), marks_gen: self.marks_gen };
        let resp = self.view.show(ui, &geo, images, self.selected, GAIN);
        if let Some(focus) = self.pending_focus.take().or_else(|| self.shared.focus.lock().take()) {
            self.selected = Some(focus);
            self.drawer = Drawer::Neuron;
            self.view.focus(resp.rect, &geo, focus.0, focus.1);
        }
        if let Some((l, n)) = resp.clicked {
            self.from_results = false;
            self.select(l, n, false);
        }
        self.atlas_overlay(ui, &resp, &geo, &eng);
    }

    fn atlas_overlay(&mut self, ui: &mut egui::Ui, resp: &atlas::AtlasResponse, geo: &Geometry, eng: &evie_core::Engine) {
        let painter = ui.painter_at(resp.rect);
        let z = self.view.zoom;
        for layer in 0..geo.layers {
            let p = self.view.to_screen(resp.rect, geo.tile_origin(layer)) - Vec2::new(0.0, 3.0);
            if resp.rect.expand(40.0).contains(p) {
                let size = (z * geo.gap_y * 0.42).clamp(8.0, 14.0);
                painter.text(p, Align2::LEFT_BOTTOM, format!("L{layer}"), FontId::new(size, mono()), FAINT);
            }
        }
        // Steered neurons stay visible at any zoom.
        for iv in self.shared.rack.lock().items.iter().filter(|i| i.enabled && i.layer < geo.layers && i.neuron < geo.neurons) {
            let c = self.view.to_screen(resp.rect, geo.cell_center(iv.layer, iv.neuron));
            let r = (z * 0.9).max(6.0);
            painter.circle_stroke(c, r, Stroke::new(1.5, TEXT));
            if z < 6.0 {
                let galley = painter.layout_no_wrap(iv.describe(), FontId::new(11.0, mono()), TEXT);
                let at = c + Vec2::new(r + 6.0, -galley.size().y / 2.0);
                painter.rect_filled(egui::Rect::from_min_size(at, galley.size()).expand2(Vec2::new(4.0, 2.0)), 4, BG);
                painter.galley(at, galley, TEXT);
            }
        }
        if let Some((l, n)) = self.selected {
            let c = self.view.to_screen(resp.rect, geo.cell_center(l, n));
            let r = (z * 0.8).max(8.0);
            painter.circle_stroke(c, r, Stroke::new(1.0, TEXT));
            painter.circle_stroke(c, r + 4.0, Stroke::new(1.0, DIM));
        }
        // Which token the atlas shows.
        if let Some(t) = self.shown_token() {
            let tok = self.shared.trace.read().tokens.get(t).map(|x| x.text.clone()).unwrap_or_default();
            let live = self.hovered_token.is_none() && self.pinned_token.is_none();
            let label = format!("{}token {t}  {:?}", if live { "● " } else { "" }, tok);
            painter.text(resp.rect.left_bottom() + Vec2::new(4.0, -4.0), Align2::LEFT_BOTTOM, label, FontId::new(11.5, mono()), DIM);
        }
        if let Some((l, n)) = self.view.hover.filter(|_| self.script.shot.is_none()) {
            let v = self.shown_token().and_then(|t| self.shared.trace.read().frames.get(t).map(|f| crate::state::Trace::decode(f[l * geo.neurons + n])));
            let read = self.shared.readout(&self.lens, l, n);
            resp.response.clone().on_hover_ui_at_pointer(|ui| {
                ui.set_max_width(380.0);
                ui.horizontal(|ui| {
                    ui.label(mono_text(format!("L{l} N{n}"), 13.5));
                    if let Some(v) = v {
                        ui.label(mono_text(format!("{v:+.3}"), 12.0).color(DIM));
                    }
                });
                match read {
                    Some(r) => {
                        token_row(ui, "+", r.up.iter().take(8).map(|(t, _)| eng.token_str(*t)));
                        token_row(ui, "−", r.down.iter().take(8).map(|(t, _)| eng.token_str(*t)));
                    }
                    None => {
                        ui.label(RichText::new("reading…").size(11.5).color(FAINT));
                    }
                }
            });
        }
    }

    fn welcome(&mut self, ui: &mut egui::Ui) {
        let status = self.shared.status.lock().clone();
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.3);
            match status.phase {
                Phase::Loading | Phase::Downloading => {
                    ui.add(egui::ProgressBar::new(status.progress.unwrap_or(0.0)).desired_width(360.0).desired_height(4.0).animate(status.progress.is_none()));
                    ui.add_space(6.0);
                    ui.label(RichText::new(&status.message).size(12.5).color(DIM));
                }
                _ => {
                    if status.phase == Phase::Failed {
                        ui.label(RichText::new(&status.message).color(DIM));
                        ui.add_space(12.0);
                    }
                    for m in evie_core::registry::MODELS {
                        let installed = evie_core::registry::is_installed(m.id);
                        let label = if installed { m.id.to_string() } else { format!("{}  ·  download {:.1} GB", m.id, m.approx_gb) };
                        if ui.add_sized([340.0, 36.0], egui::Button::new(RichText::new(label).size(14.0))).clicked() {
                            self.shared.send(if installed { crate::state::Job::Load(m.id.into()) } else { crate::state::Job::Download(m.id.into()) });
                        }
                        ui.add_space(6.0);
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
                    chat::apply_preset(self, &p);
                }
                if !self.script.steer.is_empty() {
                    self.shared.rack.lock().set(self.script.steer.clone(), None);
                }
                if let Some(q) = self.script.search.clone() {
                    self.search_input = q;
                    self.run_search();
                }
                if let Some((l, n)) = self.script.select {
                    self.select(l, n, true);
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
                if !self.shared.chat.lock().generating {
                    if self.script.tab.as_deref() == Some("token") && self.trace_len > 0 {
                        self.open_token(self.trace_len - 1);
                    }
                    self.script_state = ScriptState::Settle(Instant::now());
                }
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

fn lens_help(sel: &LensSel) -> &'static str {
    match sel {
        LensSel::Logit => "Logit lens: decode a direction with the model's own unembedding",
        LensSel::Named(n) if n == "J" => "Jacobian lens: transport through the average downstream Jacobian first",
        LensSel::Named(n) if n == "R" => "R-lens: relevance-propagation Jacobian, cleaner on early layers",
        _ => "Jacobian-style lens",
    }
}

/// A row of token chips after a short label.
pub fn token_row<'a>(ui: &mut egui::Ui, label: &str, toks: impl Iterator<Item = &'a str>) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(4.0, 4.0);
        ui.add_sized([22.0, 18.0], egui::Label::new(mono_text(label, 11.5).color(FAINT)));
        for t in toks {
            chip(ui, t);
        }
    });
}

pub fn chip(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let shown = text.replace('\n', "↵").replace(' ', "·");
    Frame::new()
        .fill(RAISED)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(4)
        .inner_margin(Margin::symmetric(5, 1))
        .show(ui, |ui| ui.add(egui::Label::new(mono_text(shown, 11.5)).wrap_mode(egui::TextWrapMode::Extend)))
        .response
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.trace_len = self.shared.trace.read().frames.len();
        self.run_script(&ctx);
        self.hovered_token = None;
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.close_drawer();
        }
        if self.shared.engine().is_some() {
            egui::Panel::left("chat")
                .resizable(true)
                .default_size(440.0)
                .min_size(340.0)
                .frame(Frame::new().fill(BG).inner_margin(Margin::same(16)).stroke(Stroke::new(1.0, BORDER)))
                .show(ui, |ui| chat::show(self, ui));
        }
        egui::CentralPanel::default().frame(Frame::new().fill(BG).inner_margin(Margin::same(14))).show(ui, |ui| self.main_area(ui));
    }
}
