//! Reading neurons and residuals through the logit lens and Jacobian-style lenses, and searching
//! all neurons for the ones whose readout ranks given words highly.
//!
//! A neuron's readout under a lens is `unembed(J_l · W_down[:, n])` (its + pole) and the negation
//! (its − pole). For search, each token's standardized logit for every neuron is
//!
//! ```text
//!   z_t(n) = a_t · u_n / sqrt(u_nᵀ M u_n),   u_n = J_l w_n,  a_t = (W_U[t] − mean) ⊙ g,  M = cov(W_U ⊙ g)
//! ```
//!
//! which is scale-free (the final RMSNorm cancels) and computed for all neurons at once by pulling
//! `a_t` back through `J_lᵀ` and taking one transposed pass over `W_down`.

use crate::kernels::{Kind16, Mat16, matmul, matvec_t, topk, transpose_to_f32};
use crate::lens::Lens;
use crate::model::Model;
use rayon::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct Readout {
    /// Tokens the + pole pushes up (logits descending).
    pub up: Vec<(u32, f32)>,
    /// Tokens the − pole pushes up, i.e. the most negative logits of the + pole.
    pub down: Vec<(u32, f32)>,
}

/// Full lens logits (`n × vocab`) for `n` residual-basis vectors living at `layer`.
pub fn lens_logits(model: &Model, lens: Option<&Lens>, layer: usize, vecs: &[f32], n: usize) -> Vec<f32> {
    let t = match lens {
        Some(l) => l.transport(layer, vecs, n),
        None => vecs.to_vec(),
    };
    model.unembed(&t, n)
}

/// Write directions (`W_down` columns) of the given neurons, one row each.
pub fn write_dirs(model: &Model, neurons: &[(usize, usize)]) -> Vec<f32> {
    let d = model.cfg.hidden;
    let mut out = vec![0f32; neurons.len() * d];
    out.par_chunks_mut(d).zip(neurons.par_iter()).for_each(|(o, &(l, n))| model.layers[l].mlp.down.col_to_f32(n, o));
    out
}

/// Lens logits for neurons' write directions (`neurons.len() × vocab`), batched by layer.
pub fn neuron_logits(model: &Model, lens: Option<&Lens>, neurons: &[(usize, usize)]) -> Vec<f32> {
    let d = model.cfg.hidden;
    let dirs = write_dirs(model, neurons);
    let mut transported = vec![0f32; neurons.len() * d];
    let mut layers: Vec<usize> = neurons.iter().map(|x| x.0).collect();
    layers.sort_unstable();
    layers.dedup();
    for l in layers {
        let idx: Vec<usize> = (0..neurons.len()).filter(|&i| neurons[i].0 == l).collect();
        let mut block = Vec::with_capacity(idx.len() * d);
        for &i in &idx {
            block.extend_from_slice(&dirs[i * d..(i + 1) * d]);
        }
        let t = match lens {
            Some(lz) => lz.transport(l, &block, idx.len()),
            None => block,
        };
        for (j, &i) in idx.iter().enumerate() {
            transported[i * d..(i + 1) * d].copy_from_slice(&t[j * d..(j + 1) * d]);
        }
    }
    model.unembed(&transported, neurons.len())
}

/// Top-`k` tokens for both poles of every neuron.
pub fn neuron_readouts(model: &Model, lens: Option<&Lens>, neurons: &[(usize, usize)], k: usize) -> Vec<Readout> {
    let v = model.lm_head.rows;
    let mut out = Vec::with_capacity(neurons.len());
    for chunk in neurons.chunks(48) {
        let logits = neuron_logits(model, lens, chunk);
        out.par_extend(logits.par_chunks(v).map(|row| {
            let neg: Vec<f32> = row.iter().map(|x| -x).collect();
            Readout { up: topk(row, k), down: topk(&neg, k).into_iter().map(|(t, x)| (t, -x)).collect() }
        }));
    }
    out
}

/// For each neuron: (best rank of any query token in the + pole, best rank in the − pole), 0 = top.
pub fn token_ranks(model: &Model, lens: Option<&Lens>, neurons: &[(usize, usize)], tokens: &[u32]) -> Vec<(usize, usize)> {
    let v = model.lm_head.rows;
    let mut out = Vec::with_capacity(neurons.len());
    for chunk in neurons.chunks(48) {
        let logits = neuron_logits(model, lens, chunk);
        out.par_extend(logits.par_chunks(v).map(|row| {
            let best = |sign: f32| {
                tokens
                    .iter()
                    .filter(|&&t| (t as usize) < row.len())
                    .map(|&t| {
                        let x = sign * row[t as usize];
                        row.iter().filter(|&&y| sign * y > x).count()
                    })
                    .min()
                    .unwrap_or(usize::MAX)
            };
            (best(1.0), best(-1.0))
        }));
    }
    out
}

/// Lens readouts of residual-stream vectors: `(layer, vector)` → top-`k` tokens.
pub fn resid_readouts(model: &Model, lens: Option<&Lens>, resids: &[(usize, Vec<f32>)], k: usize) -> Vec<Vec<(u32, f32)>> {
    let d = model.cfg.hidden;
    let mut transported = vec![0f32; resids.len() * d];
    for (i, (l, r)) in resids.iter().enumerate() {
        let t = match lens {
            Some(lz) => lz.transport(*l, r, 1),
            None => r.clone(),
        };
        transported[i * d..(i + 1) * d].copy_from_slice(&t);
    }
    let logits = model.unembed(&transported, resids.len());
    logits.par_chunks(model.lm_head.rows).map(|row| topk(row, k)).collect()
}

/// Mean and (gain-weighted) covariance of the unembedding rows, for standardizing lens logits.
pub struct UnembedStats {
    pub mean: Vec<f32>,
    /// `cov((W_U − mean) ⊙ g)` scaled by `1 / m_scale` and stored as f16.
    pub m16: Mat16,
    pub m_scale: f32,
}

impl UnembedStats {
    pub fn compute(model: &Model) -> Self {
        let w = &model.lm_head;
        let (v, d) = (w.rows, w.cols);
        let g = &model.final_norm;
        let sums: Vec<f64> = (0..v)
            .into_par_iter()
            .fold(
                || vec![0f64; d],
                |mut acc, r| {
                    for (a, &x) in acc.iter_mut().zip(w.row(r)) {
                        *a += crate::kernels::to_f32(w.kind, x) as f64;
                    }
                    acc
                },
            )
            .reduce(|| vec![0f64; d], |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            });
        let mean: Vec<f32> = sums.iter().map(|s| (s / v as f64) as f32).collect();
        // Covariance from an evenly strided subsample of the vocabulary.
        let s = 16384.min(v);
        let stride = v / s;
        let mut xt = vec![0f32; d * s]; // (d × s): column j = centered, gain-weighted row j*stride
        let mut row = vec![0f32; d];
        for j in 0..s {
            w.row_to_f32(j * stride, &mut row);
            for i in 0..d {
                xt[i * s + j] = (row[i] - mean[i]) * g[i];
            }
        }
        let xt16: Vec<u16> = xt.iter().map(|&x| half::f16::from_f32(x * 64.0).to_bits()).collect();
        let xm = Mat16::from_vec(d, s, Kind16::F16, xt16);
        let mut cov = vec![0f32; d * d];
        matmul(&xm, &xt, d, &mut cov);
        let norm = 1.0 / (s as f32 * 64.0);
        cov.iter_mut().for_each(|c| *c *= norm);
        let m_scale = cov.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-30);
        let m16 = Mat16::from_vec(d, d, Kind16::F16, cov.iter().map(|&c| half::f16::from_f32(c / m_scale).to_bits()).collect());
        Self { mean, m16, m_scale }
    }

    /// `a_t = (W_U[t] − mean) ⊙ g`
    pub fn centered(&self, model: &Model, t: u32) -> Vec<f32> {
        let mut r = vec![0f32; model.cfg.hidden];
        model.lm_head.row_to_f32(t as usize, &mut r);
        r.iter_mut().zip(&self.mean).zip(&model.final_norm).for_each(|((x, m), g)| *x = (*x - m) * g);
        r
    }
}

/// `sqrt(u_nᵀ M u_n)` for every neuron (layer-major) under `lens` — the per-neuron logit spread.
pub fn neuron_sigma(model: &Model, lens: Option<&Lens>, stats: &UnembedStats, progress: &dyn Fn(f32)) -> Vec<f32> {
    let (nl, n, d) = (model.cfg.n_layers, model.cfg.intermediate, model.cfg.hidden);
    let mut sigma = vec![0f32; nl * n];
    for l in 0..nl {
        let wt = transpose_to_f32(&model.layers[l].mlp.down); // n × d
        let u = match lens {
            Some(lz) => lz.transport(l, &wt, n),
            None => wt,
        };
        let mut mu = vec![0f32; n * d];
        matmul(&stats.m16, &u, n, &mut mu);
        sigma[l * n..(l + 1) * n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, s)| *s = (crate::kernels::dot(&u[i * d..(i + 1) * d], &mu[i * d..(i + 1) * d]) * stats.m_scale).max(1e-30).sqrt());
        progress((l + 1) as f32 / nl as f32);
    }
    sigma
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub layer: usize,
    pub neuron: usize,
    /// +1: the neuron's + pole promotes the words; −1: its − pole does.
    pub sign: i8,
    pub z: f32,
}

/// Standardized scores of `tokens` for every neuron (best token per neuron and pole), top `limit`.
pub fn search(model: &Model, lens: Option<&Lens>, stats: &UnembedStats, sigma: &[f32], tokens: &[u32], limit: usize) -> Vec<Hit> {
    if tokens.is_empty() {
        return Vec::new();
    }
    let (nl, n) = (model.cfg.n_layers, model.cfg.intermediate);
    let a: Vec<Vec<f32>> = tokens.iter().map(|&t| stats.centered(model, t)).collect();
    let per_layer: Vec<Vec<Hit>> = (0..nl)
        .map(|l| {
            let pulled: Vec<Vec<f32>> = a
                .iter()
                .map(|at| match lens {
                    Some(lz) => lz.pullback(l, at),
                    None => at.clone(),
                })
                .collect();
            let refs: Vec<&[f32]> = pulled.iter().map(Vec::as_slice).collect();
            let scores = matvec_t(&model.layers[l].mlp.down, &refs); // per token: n
            let sig = &sigma[l * n..(l + 1) * n];
            let mut best: Vec<Hit> = (0..n)
                .flat_map(|i| {
                    let (mut up, mut dn) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                    for s in &scores {
                        let z = s[i] / sig[i];
                        up = up.max(z);
                        dn = dn.max(-z);
                    }
                    [Hit { layer: l, neuron: i, sign: 1, z: up }, Hit { layer: l, neuron: i, sign: -1, z: dn }]
                })
                .collect();
            best.sort_by(|x, y| y.z.total_cmp(&x.z));
            best.truncate(limit);
            best
        })
        .collect();
    let mut all: Vec<Hit> = per_layer.into_iter().flatten().collect();
    all.sort_by(|x, y| y.z.total_cmp(&x.z));
    all.truncate(limit);
    all
}
