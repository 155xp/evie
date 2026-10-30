//! Background analysis: lens readouts, concept search and residual-stream (workspace) views.

use crate::state::{AJob, SearchResult, Shared, Workspace};
use crossbeam_channel::Receiver;
use evie_core::analysis::{self, Readout};
use evie_core::{Engine, LensSel};
use std::sync::Arc;

pub fn spawn(shared: Arc<Shared>, rx: Receiver<AJob>) {
    std::thread::Builder::new()
        .name("evie-analyst".into())
        .spawn(move || {
            while let Ok(job) = rx.recv() {
                // Coalesce queued readout requests into one batched pass over the unembedding.
                let mut jobs = vec![job];
                while let Ok(j) = rx.try_recv() {
                    jobs.push(j);
                }
                let mut readouts: Vec<(LensSel, (usize, usize))> = Vec::new();
                let mut rest = Vec::new();
                for j in jobs {
                    match j {
                        AJob::Readouts(lens, ns) => readouts.extend(ns.into_iter().map(|n| (lens.clone(), n))),
                        other => rest.push(other),
                    }
                }
                let Some(eng) = shared.engine() else { continue };
                if !readouts.is_empty() {
                    run_readouts(&shared, &eng, readouts);
                }
                for j in rest {
                    match j {
                        AJob::Search(q, lens) => search(&shared, &eng, &q, &lens),
                        AJob::Workspace(t) => workspace(&shared, &eng, t),
                        AJob::Calibrate => warm(&shared, &eng),
                        AJob::Readouts(..) => {}
                    }
                }
            }
        })
        .expect("spawn analyst thread");
}

fn run_readouts(shared: &Shared, eng: &Engine, mut reqs: Vec<(LensSel, (usize, usize))>) {
    reqs.sort_by(|a, b| a.0.label().cmp(b.0.label()));
    reqs.dedup();
    let mut i = 0;
    while i < reqs.len() {
        let lens = reqs[i].0.clone();
        let mut ns = Vec::new();
        while i < reqs.len() && reqs[i].0 == lens {
            ns.push(reqs[i].1);
            i += 1;
        }
        let ns: Vec<(usize, usize)> = ns.into_iter().filter(|&(l, n)| l < eng.n_layers() && n < eng.n_neurons()).collect();
        let res = eng.readouts(&lens, &ns, 16);
        let mut cache = shared.reads.lock();
        let mut pending = shared.reads_pending.lock();
        for (n, r) in ns.into_iter().zip(res) {
            pending.remove(&(lens.clone(), n.0, n.1));
            cache.insert((lens.clone(), n.0, n.1), r);
        }
    }
    shared.repaint();
}

/// Comma/space separated words → neurons whose readout ranks them highly, with exact ranks per lens.
pub fn search_now(eng: &Engine, query: &str, lens: &LensSel, limit: usize) -> (Vec<String>, Vec<SearchResult>) {
    let words: Vec<String> = query.split(|c: char| c == ',' || c == ';' || c.is_whitespace()).map(str::to_string).filter(|w| !w.is_empty()).collect();
    let toks = eng.word_tokens(&words);
    let matched: Vec<String> = toks.iter().map(|&t| format!("{:?}", eng.token_str(t))).collect();
    if toks.is_empty() {
        return (matched, Vec::new());
    }
    let hits = eng.search(lens, &toks, limit);
    let neurons: Vec<(usize, usize)> = hits.iter().map(|h| (h.layer, h.neuron)).collect();
    let mut results: Vec<SearchResult> =
        hits.iter().map(|h| SearchResult { layer: h.layer, neuron: h.neuron, sign: h.sign, z: h.z, lenses: Vec::new() }).collect();
    for sel in eng.lens_list() {
        let ranks = analysis::token_ranks(&eng.model, eng.lens(&sel), &neurons, &toks);
        let reads: Vec<Readout> = eng.readouts(&sel, &neurons, 8);
        for ((r, rk), rd) in results.iter_mut().zip(ranks).zip(reads) {
            let (rank, top) = if r.sign > 0 { (rk.0, rd.up) } else { (rk.1, rd.down) };
            r.lenses.push((sel.label().to_string(), rank, top.iter().map(|(t, _)| eng.token_str(*t).to_string()).collect()));
        }
    }
    (matched, results)
}

fn search(shared: &Shared, eng: &Engine, query: &str, lens: &LensSel) {
    {
        let mut s = shared.search.lock();
        s.running = true;
        s.status = if eng.sigma_ready(lens) { "searching…".into() } else { format!("indexing {} lens (one-time)…", lens.label()) };
    }
    shared.repaint();
    eng.sigma(lens, &|p| {
        shared.search.lock().status = format!("indexing {} lens (one-time) {:.0}%", lens.label(), p * 100.0);
        shared.repaint();
    });
    let t = std::time::Instant::now();
    let (matched, results) = search_now(eng, query, lens, 40);
    let mut s = shared.search.lock();
    s.running = false;
    s.status = if matched.is_empty() {
        format!("no vocabulary tokens match {query:?}")
    } else {
        format!("{} neurons · {:.0} ms", results.len(), t.elapsed().as_secs_f32() * 1000.0)
    };
    s.matched = matched;
    s.results = results;
    s.version += 1;
    drop(s);
    shared.repaint();
}

/// What each layer's residual stream at `token` is disposed to say, through every lens.
pub fn workspace_rows(shared: &Shared, eng: &Engine, token: usize, k: usize) -> Option<Workspace> {
    let resid = shared.trace.read().resid.get(token).cloned()?;
    let d = eng.model.cfg.hidden;
    let rows: Vec<(usize, Vec<f32>)> = (0..eng.n_layers()).map(|l| (l, resid[l * d..(l + 1) * d].iter().map(|v| v.to_f32()).collect())).collect();
    let mut out = Vec::new();
    for sel in eng.lens_list() {
        out.push((sel.label().to_string(), analysis::resid_readouts(&eng.model, eng.lens(&sel), &rows, k)));
    }
    Some(Workspace { token, rows: out })
}

fn workspace(shared: &Shared, eng: &Engine, token: usize) {
    if let Some(ws) = workspace_rows(shared, eng, token, 6) {
        *shared.workspace.lock() = Some(ws);
        shared.repaint();
    }
}

/// Precomputes search statistics for every lens so the first search is instant.
fn warm(shared: &Shared, eng: &Engine) {
    for sel in eng.lens_list() {
        if !eng.sigma_ready(&sel) {
            shared.search.lock().status = format!("indexing {} lens…", sel.label());
            shared.repaint();
            eng.sigma(&sel, &|_| {});
        }
    }
    let mut s = shared.search.lock();
    if !s.running {
        s.status = "ready — search any word, e.g. pirate, 海盗".into();
    }
    drop(s);
    shared.repaint();
}
