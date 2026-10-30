//! State shared by the GUI, the inference worker, the analysis worker and the MCP server.

use crossbeam_channel::Sender;
use evie_core::analysis::Readout;
use evie_core::{Engine, Intervention, LensSel, Role};
use parking_lot::{Mutex, RecursiveRwLock, RwLock};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    NoModel,
    Downloading,
    Loading,
    Ready,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Status {
    pub phase: Phase,
    pub message: String,
    pub progress: Option<f32>,
    pub tok_s: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Rack {
    pub items: Vec<Intervention>,
    pub version: u64,
    pub preset: Option<String>,
}

impl Rack {
    pub fn set(&mut self, items: Vec<Intervention>, preset: Option<String>) {
        self.items = items;
        self.preset = preset;
        self.version += 1;
    }
    pub fn touch(&mut self) {
        self.version += 1;
    }
    /// Adds or replaces the intervention on the same neuron.
    pub fn upsert(&mut self, iv: Intervention) {
        match self.items.iter_mut().find(|i| i.layer == iv.layer && i.neuron == iv.neuron) {
            Some(slot) => *slot = iv,
            None => self.items.push(iv),
        }
        self.preset = None;
        self.version += 1;
    }
    pub fn remove(&mut self, layer: usize, neuron: usize) {
        self.items.retain(|i| !(i.layer == layer && i.neuron == neuron));
        self.preset = None;
        self.version += 1;
    }
    pub fn get(&self, layer: usize, neuron: usize) -> Option<&Intervention> {
        self.items.iter().find(|i| i.layer == layer && i.neuron == neuron)
    }
}

#[derive(Debug, Clone)]
pub struct GenSettings {
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    /// Also generate an unsteered reply side by side (same batch, nearly free).
    pub compare: bool,
}

impl Default for GenSettings {
    fn default() -> Self {
        Self { max_new: 320, temperature: 0.0, top_p: 0.95, repetition_penalty: 1.0, seed: 0, compare: false }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub text: String,
    /// Unsteered reply generated alongside (compare mode).
    pub baseline: Option<String>,
    /// Interventions active when this reply was generated.
    pub steering: Vec<String>,
    /// This message's tokens in the trace, if it was part of the last forward pass.
    pub tokens: Option<Range<usize>>,
    pub done: bool,
    /// Who sent it ("you", "claude", ...), for messages that came through MCP.
    pub via: Option<String>,
}

impl Message {
    pub fn new(role: Role, text: impl Into<String>) -> Self {
        Self { role, text: text.into(), baseline: None, steering: Vec::new(), tokens: None, done: true, via: None }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Chat {
    pub system: String,
    pub messages: Vec<Message>,
    pub generating: bool,
    pub settings: GenSettings,
    /// Bumped whenever a reply finishes (MCP waits on it).
    pub finished: u64,
}

#[derive(Debug, Clone)]
pub struct TraceTok {
    pub id: u32,
    pub text: String,
    pub generated: bool,
}

/// Everything the last forward pass saw: tokens, every neuron's activation, residual stream.
#[derive(Default)]
pub struct Trace {
    pub tokens: Vec<TraceTok>,
    /// Per token, `layers × neurons` activations normalised by each neuron's range and
    /// companded: `q = round(127 · sign(v) · sqrt(min(|v|, 1)))`.
    pub frames: Vec<Arc<[i8]>>,
    /// Per token, the residual stream leaving every layer (`layers × hidden`).
    pub resid: Vec<Arc<[half::f16]>>,
    /// Per-token gate scores for gated interventions (display only).
    pub version: u64,
}

impl Trace {
    pub fn decode(q: i8) -> f32 {
        let v = q as f32 / 127.0;
        v * v.abs()
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub layer: usize,
    pub neuron: usize,
    pub sign: i8,
    pub z: f32,
    /// (lens label, best rank of a query word in this pole, top tokens of this pole)
    pub lenses: Vec<(String, usize, Vec<String>)>,
}

#[derive(Debug, Clone, Default)]
pub struct SearchState {
    pub query: String,
    pub lens: Option<LensSel>,
    pub running: bool,
    pub status: String,
    pub matched: Vec<String>,
    pub results: Vec<SearchResult>,
    pub version: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Workspace {
    pub token: usize,
    /// Per lens label: per layer top tokens.
    pub rows: Vec<(String, Vec<Vec<(u32, f32)>>)>,
}

#[derive(Debug, Clone)]
pub struct Activity {
    pub at: Instant,
    pub who: String,
    pub text: String,
}

pub enum Job {
    /// Generate the assistant's reply to the conversation.
    Chat,
    /// Load a model by id.
    Load(String),
    /// Download (then load) a model by id.
    Download(String),
}

pub enum AJob {
    Readouts(LensSel, Vec<(usize, usize)>),
    Search(String, LensSel),
    Workspace(usize),
    Calibrate,
}

pub struct Shared {
    pub engine: RwLock<Option<Arc<Engine>>>,
    pub status: Mutex<Status>,
    pub rack: Mutex<Rack>,
    pub chat: Mutex<Chat>,
    /// Recursive so a UI frame can re-read it while a writer waits (a fair lock would deadlock).
    pub trace: RecursiveRwLock<Trace>,
    pub reads: Mutex<HashMap<(LensSel, usize, usize), Readout>>,
    pub reads_pending: Mutex<HashSet<(LensSel, usize, usize)>>,
    pub search: Mutex<SearchState>,
    pub workspace: Mutex<Option<Workspace>>,
    pub activity: Mutex<VecDeque<Activity>>,
    /// A neuron an agent asked the UI to show.
    pub focus: Mutex<Option<(usize, usize)>>,
    pub stop: AtomicBool,
    pub ctx: OnceLock<egui::Context>,
    pub mcp_url: Mutex<Option<String>>,
    jobs: Sender<Job>,
    ajobs: Sender<AJob>,
}

impl Shared {
    pub fn new(jobs: Sender<Job>, ajobs: Sender<AJob>) -> Self {
        Self {
            engine: RwLock::new(None),
            status: Mutex::new(Status { phase: Phase::NoModel, message: String::new(), progress: None, tok_s: 0.0 }),
            rack: Mutex::new(Rack::default()),
            chat: Mutex::new(Chat::default()),
            trace: RecursiveRwLock::new(Trace::default()),
            reads: Mutex::new(HashMap::new()),
            reads_pending: Mutex::new(HashSet::new()),
            search: Mutex::new(SearchState::default()),
            workspace: Mutex::new(None),
            activity: Mutex::new(VecDeque::new()),
            focus: Mutex::new(None),
            stop: AtomicBool::new(false),
            ctx: OnceLock::new(),
            mcp_url: Mutex::new(None),
            jobs,
            ajobs,
        }
    }

    pub fn engine(&self) -> Option<Arc<Engine>> {
        self.engine.read().clone()
    }

    pub fn repaint(&self) {
        if let Some(c) = self.ctx.get() {
            c.request_repaint();
        }
    }

    pub fn set_status(&self, phase: Phase, message: impl Into<String>, progress: Option<f32>) {
        {
            let mut s = self.status.lock();
            s.phase = phase;
            s.message = message.into();
            s.progress = progress;
        }
        self.repaint();
    }

    pub fn send(&self, job: Job) {
        let _ = self.jobs.send(job);
    }

    pub fn analyse(&self, job: AJob) {
        let _ = self.ajobs.send(job);
    }

    pub fn log(&self, who: &str, text: impl Into<String>) {
        let mut a = self.activity.lock();
        a.push_back(Activity { at: Instant::now(), who: who.to_string(), text: text.into() });
        while a.len() > 200 {
            a.pop_front();
        }
        drop(a);
        self.repaint();
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Appends a user message and starts generating a reply. Returns false if busy.
    pub fn send_user(&self, text: &str, via: Option<&str>) -> bool {
        let mut c = self.chat.lock();
        if c.generating || text.trim().is_empty() {
            return false;
        }
        let mut m = Message::new(Role::User, text.trim());
        m.via = via.map(str::to_string);
        c.messages.push(m);
        c.generating = true;
        drop(c);
        self.send(Job::Chat);
        self.repaint();
        true
    }

    /// Cached readout, requesting it in the background if missing.
    pub fn readout(&self, lens: &LensSel, layer: usize, neuron: usize) -> Option<Readout> {
        let key = (lens.clone(), layer, neuron);
        if let Some(r) = self.reads.lock().get(&key) {
            return Some(r.clone());
        }
        if self.reads_pending.lock().insert(key) {
            self.analyse(AJob::Readouts(lens.clone(), vec![(layer, neuron)]));
        }
        None
    }
}
