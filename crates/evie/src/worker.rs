//! The inference thread: loads models and generates replies, streaming tokens and per-token neuron
//! frames into the shared state while applying the live steering rack.

use crate::state::{Job, Message, Phase, Shared, TraceTok};
use crossbeam_channel::Receiver;
use evie_core::steering::GateState;
use evie_core::{ChatMsg, Engine, Hooks, Plan, Role, SampleParams, Sampler, Seq};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub fn spawn(shared: Arc<Shared>, rx: Receiver<Job>) {
    std::thread::Builder::new()
        .name("evie-infer".into())
        .spawn(move || {
            for job in rx {
                let r = match job {
                    Job::Load(id) => load(&shared, &id),
                    Job::Download(id) => download(&shared, &id).and_then(|_| load(&shared, &id)),
                    Job::Chat => generate(&shared),
                };
                if let Err(e) = r {
                    log::error!("{e:#}");
                    let mut c = shared.chat.lock();
                    if c.generating {
                        c.generating = false;
                        c.finished += 1;
                        if let Some(m) = c.messages.last_mut().filter(|m| m.role == Role::Assistant && !m.done) {
                            m.done = true;
                            m.text += &format!("\n\n[error: {e:#}]");
                        }
                    }
                    drop(c);
                    let ready = shared.engine.read().is_some();
                    shared.set_status(if ready { Phase::Ready } else { Phase::Failed }, format!("{e:#}"), None);
                }
            }
        })
        .expect("spawn inference thread");
}

fn download(shared: &Shared, id: &str) -> anyhow::Result<()> {
    shared.set_status(Phase::Downloading, format!("downloading {id}"), Some(0.0));
    evie_core::registry::download(id, &|p| {
        let frac = if p.total > 0 { p.done as f32 / p.total as f32 } else { 0.0 };
        let mb = |b: u64| b as f64 / 1e6;
        shared.set_status(Phase::Downloading, format!("{} · {:.0} / {:.0} MB", p.file, mb(p.done), mb(p.total)), Some(frac));
    })
}

fn load(shared: &Shared, id: &str) -> anyhow::Result<()> {
    *shared.engine.write() = None;
    shared.reads.lock().clear();
    shared.reads_pending.lock().clear();
    *shared.trace.write() = Default::default();
    shared.set_status(Phase::Loading, format!("loading {id}"), None);
    let t = Instant::now();
    let eng = Engine::load(id, &|s| shared.set_status(Phase::Loading, format!("{id}: {s}"), None))?;
    let eng = Arc::new(eng);
    {
        let e = eng.clone();
        std::thread::spawn(move || e.model.prefetch());
    }
    shared.set_status(Phase::Loading, format!("{id}: calibrating neuron ranges (first launch only)"), Some(0.0));
    eng.ensure_calibrated(&|p| shared.set_status(Phase::Loading, format!("{id}: calibrating neuron ranges (first launch only)"), Some(p)))?;
    *shared.engine.write() = Some(eng.clone());
    shared.set_status(Phase::Ready, format!("{id} ready in {:.1}s", t.elapsed().as_secs_f32()), None);
    shared.log("evie", format!("loaded {id}: {} layers × {} neurons", eng.n_layers(), eng.n_neurons()));
    shared.analyse(crate::state::AJob::Calibrate);
    Ok(())
}

/// Records activations (normalised + companded) and residuals for one stream.
struct Capture {
    rows: usize,
    n: usize,
    d: usize,
    scale: Option<Arc<Vec<f32>>>,
    frames: Vec<Vec<i8>>,
    resid: Vec<Vec<half::f16>>,
}

impl Capture {
    fn new(eng: &Engine, rows: usize) -> Self {
        let (layers, n, d) = (eng.n_layers(), eng.n_neurons(), eng.model.cfg.hidden);
        Self {
            rows,
            n,
            d,
            scale: eng.act_scale.read().clone(),
            frames: vec![vec![0i8; layers * n]; rows],
            resid: vec![vec![half::f16::ZERO; layers * d]; rows],
        }
    }

    fn mlp(&mut self, layer: usize, acts: &[f32]) {
        let n = self.n;
        let scale = self.scale.as_deref().map(|s| &s[layer * n..(layer + 1) * n]);
        self.frames.par_iter_mut().enumerate().take(self.rows).for_each(|(r, f)| {
            let src = &acts[r * n..(r + 1) * n];
            let dst = &mut f[layer * n..(layer + 1) * n];
            for (j, (o, &a)) in dst.iter_mut().zip(src).enumerate() {
                let v = match scale {
                    Some(s) => a / s[j],
                    None => a,
                };
                let q = (v.abs().min(1.0).sqrt() * 127.0).round();
                *o = if v < 0.0 { -q as i8 } else { q as i8 };
            }
        });
    }

    fn resid(&mut self, layer: usize, x: &[f32]) {
        let d = self.d;
        for (r, dst) in self.resid.iter_mut().enumerate().take(self.rows) {
            for (o, &v) in dst[layer * d..(layer + 1) * d].iter_mut().zip(&x[r * d..(r + 1) * d]) {
                *o = half::f16::from_f32(v);
            }
        }
    }
}

struct Lane<'a> {
    plan: &'a Plan,
    gate: &'a mut GateState,
    capture: Option<&'a mut Capture>,
}

struct Probe<'a> {
    lanes: Vec<Lane<'a>>,
    n: usize,
}

impl Hooks for Probe<'_> {
    fn mlp(&mut self, layer: usize, seq: usize, _pos0: usize, acts: &mut [f32]) {
        let lane = &mut self.lanes[seq];
        lane.plan.apply(layer, acts, self.n, lane.gate);
        if let Some(c) = lane.capture.as_mut() {
            c.mlp(layer, acts);
        }
    }

    fn resid(&mut self, layer: usize, seq: usize, _pos0: usize, x: &[f32]) {
        if let Some(c) = self.lanes[seq].capture.as_mut() {
            c.resid(layer, x);
        }
    }
}

fn is_stop(eng: &Engine, t: u32) -> bool {
    t == eng.tok.im_end || t == eng.tok.endoftext || eng.model.cfg.eos_ids.contains(&t)
}

fn generate(shared: &Shared) -> anyhow::Result<()> {
    let eng = shared.engine().ok_or_else(|| anyhow::anyhow!("no model loaded"))?;
    shared.stop.store(false, Ordering::SeqCst);

    // Conversation → prompt, remembering where each message's tokens start.
    let (msgs, settings, system) = {
        let c = shared.chat.lock();
        (c.messages.clone(), c.settings.clone(), c.system.clone())
    };
    let mut chat_msgs: Vec<ChatMsg> = Vec::new();
    if !system.trim().is_empty() {
        chat_msgs.push(ChatMsg::system(system.clone()));
    }
    let sys_offset = chat_msgs.len();
    chat_msgs.extend(msgs.iter().map(|m| ChatMsg { role: m.role, content: m.text.clone() }));
    let ids = eng.tok.encode(&eng.tok.chat(&chat_msgs, true))?;
    let mut bounds = Vec::with_capacity(msgs.len());
    let mut prev = if sys_offset > 0 { eng.tok.encode(&eng.tok.chat(&chat_msgs[..sys_offset], false))?.len() } else { 0 };
    for i in 0..msgs.len() {
        let end = eng.tok.encode(&eng.tok.chat(&chat_msgs[..sys_offset + i + 1], false))?.len().min(ids.len());
        bounds.push(prev..end);
        prev = end;
    }

    let (mut ivs, mut version) = {
        let r = shared.rack.lock();
        (r.items.clone(), r.version)
    };
    let mut plan = eng.compile(&ivs);
    let baseline = Plan::default();
    let mut gate = plan.new_state();
    let mut bgate = baseline.new_state();
    let compare = settings.compare && !plan.is_empty();

    {
        let mut c = shared.chat.lock();
        for (m, b) in c.messages.iter_mut().zip(&bounds) {
            m.tokens = Some(b.clone());
        }
        let mut reply = Message::new(Role::Assistant, "");
        reply.done = false;
        reply.steering = ivs.iter().filter(|i| i.enabled).map(|i| i.describe()).collect();
        reply.baseline = compare.then(String::new);
        reply.tokens = Some(ids.len()..ids.len());
        c.messages.push(reply);
    }
    {
        let mut tr = shared.trace.write();
        tr.tokens = ids.iter().map(|&id| TraceTok { id, text: eng.token_str(id).to_string(), generated: false }).collect();
        tr.frames.clear();
        tr.resid.clear();
        tr.version += 1;
    }
    shared.repaint();

    let n = eng.n_neurons();
    let t0 = Instant::now();
    let mut cache = eng.model.new_cache();
    let mut bcache = eng.model.new_cache();
    let mut cap = Capture::new(&eng, ids.len());
    plan.begin(&mut gate, ids.len());
    baseline.begin(&mut bgate, ids.len());
    let logits = {
        let mut lanes = vec![Lane { plan: &plan, gate: &mut gate, capture: Some(&mut cap) }];
        let mut seqs = vec![Seq { cache: &mut cache, tokens: &ids }];
        if compare {
            lanes.push(Lane { plan: &baseline, gate: &mut bgate, capture: None });
            seqs.push(Seq { cache: &mut bcache, tokens: &ids });
        }
        eng.model.forward(&mut seqs, &mut Probe { lanes, n }, true)
    };
    push_frames(shared, cap);
    let prefill = t0.elapsed();
    log::info!("prefill {} tokens in {prefill:.2?}", ids.len());

    let sp = SampleParams {
        temperature: settings.temperature,
        top_p: settings.top_p,
        min_p: 0.0,
        repetition_penalty: settings.repetition_penalty,
        seed: settings.seed,
    };
    let mut sampler = Sampler::new(sp);
    let mut bsampler = Sampler::new(sp);
    let mut out: Vec<u32> = Vec::new();
    let mut bout: Vec<u32> = Vec::new();
    let mut next = Some(sampler.sample(&logits[0], &ids));
    let mut bnext = if compare { Some(bsampler.sample(&logits[1], &ids)) } else { None };
    let t_dec = Instant::now();
    let mut steps = 0usize;

    loop {
        if shared.stop.load(Ordering::SeqCst) {
            break;
        }
        if let Some(t) = next {
            if is_stop(&eng, t) || out.len() >= settings.max_new {
                next = None;
            } else {
                out.push(t);
            }
        }
        if let Some(t) = bnext {
            if is_stop(&eng, t) || bout.len() >= settings.max_new {
                bnext = None;
            } else {
                bout.push(t);
            }
        }
        {
            let text = eng.tok.decode(&out);
            let btext = compare.then(|| eng.tok.decode(&bout));
            let mut c = shared.chat.lock();
            if let Some(m) = c.messages.last_mut() {
                m.text = text;
                if let Some(b) = btext {
                    m.baseline = Some(b);
                }
                m.tokens = Some(ids.len()..ids.len() + out.len());
            }
        }
        shared.repaint();
        if next.is_none() && bnext.is_none() {
            break;
        }

        // Live steering: pick up rack edits between tokens.
        {
            let r = shared.rack.lock();
            if r.version != version {
                ivs = r.items.clone();
                version = r.version;
                plan = eng.compile(&ivs);
                gate = plan.new_state();
            }
        }

        let a = next.map(|t| [t]);
        let b = bnext.map(|t| [t]);
        let mut cap = Capture::new(&eng, if a.is_some() { 1 } else { 0 });
        let mut lanes = Vec::new();
        let mut seqs = Vec::new();
        let mut which = Vec::new();
        if let Some(tok) = a.as_ref() {
            plan.begin(&mut gate, 1);
            lanes.push(Lane { plan: &plan, gate: &mut gate, capture: Some(&mut cap) });
            seqs.push(Seq { cache: &mut cache, tokens: tok });
            which.push(0);
        }
        if let Some(tok) = b.as_ref() {
            baseline.begin(&mut bgate, 1);
            lanes.push(Lane { plan: &baseline, gate: &mut bgate, capture: None });
            seqs.push(Seq { cache: &mut bcache, tokens: tok });
            which.push(1);
        }
        let logits = eng.model.forward(&mut seqs, &mut Probe { lanes, n }, true);
        if let Some(&t) = a.as_ref().map(|x| &x[0]) {
            {
                let mut tr = shared.trace.write();
                tr.tokens.push(TraceTok { id: t, text: eng.token_str(t).to_string(), generated: true });
            }
            push_frames(shared, cap);
        }
        for (slot, lg) in which.iter().zip(&logits) {
            if *slot == 0 {
                let hist: Vec<u32> = ids.iter().chain(&out).copied().collect();
                next = Some(sampler.sample(lg, &hist));
            } else {
                let hist: Vec<u32> = ids.iter().chain(&bout).copied().collect();
                bnext = Some(bsampler.sample(lg, &hist));
            }
        }
        steps += 1;
        let tps = steps as f32 / t_dec.elapsed().as_secs_f32().max(1e-6);
        shared.status.lock().tok_s = tps;
    }

    let mut c = shared.chat.lock();
    if let Some(m) = c.messages.last_mut() {
        m.done = true;
    }
    c.generating = false;
    c.finished += 1;
    drop(c);
    log::info!("generated {} tokens in {:.2?}", out.len(), t_dec.elapsed());
    shared.repaint();
    Ok(())
}

fn push_frames(shared: &Shared, cap: Capture) {
    let mut tr = shared.trace.write();
    for (f, r) in cap.frames.into_iter().zip(cap.resid) {
        tr.frames.push(f.into());
        tr.resid.push(r.into());
    }
    tr.version += 1;
}
