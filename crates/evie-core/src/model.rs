//! Qwen3.5 text decoder (hybrid Gated DeltaNet + gated full attention) with activation hooks.
//!
//! Mirrors `transformers.models.qwen3_5.modeling_qwen3_5` but keeps activations in f32 throughout.
//! A forward call can carry several independent sequences ("segments"); their projections are batched
//! together while attention / recurrent state stays per sequence.

use crate::config::{Config, LayerKind};
use crate::kernels::{self, Mat16, add_inplace, matmul, rmsnorm, rmsnorm_rows, sigmoid, silu, softplus};
use crate::weights::SafeTensors;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::path::Path;

pub struct Mlp {
    pub gate: Mat16,
    pub up: Mat16,
    /// `[hidden, intermediate]`: column `n` is neuron n's write direction.
    pub down: Mat16,
}

pub struct DeltaNet {
    pub qkv: Mat16,
    pub z: Mat16,
    pub b: Mat16,
    pub a: Mat16,
    /// Depthwise causal conv weights, `[conv_dim][kernel]`.
    pub conv: Vec<f32>,
    /// `-exp(A_log)` per value head.
    pub neg_a: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
    pub out: Mat16,
}

pub struct Attention {
    pub q: Mat16,
    pub k: Mat16,
    pub v: Mat16,
    pub o: Mat16,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

pub enum Mixer {
    Linear(DeltaNet),
    Full(Attention),
}

pub struct Layer {
    /// Pre-mixer RMSNorm gain, already `1 + w`.
    pub ln1: Vec<f32>,
    /// Pre-MLP RMSNorm gain, already `1 + w`.
    pub ln2: Vec<f32>,
    pub mixer: Mixer,
    pub mlp: Mlp,
}

pub struct Model {
    pub cfg: Config,
    pub embed: Mat16,
    pub lm_head: Mat16,
    pub layers: Vec<Layer>,
    /// Final RMSNorm gain, already `1 + w`.
    pub final_norm: Vec<f32>,
    inv_freq: Vec<f32>,
    st: SafeTensors,
}

/// Per-sequence inference state.
#[derive(Clone)]
pub struct Cache {
    /// Number of tokens already consumed.
    pub pos: usize,
    pub layers: Vec<LayerCache>,
}

#[derive(Clone)]
pub enum LayerCache {
    Linear {
        /// Last `kernel - 1` conv inputs, oldest first, each `conv_dim` wide.
        conv: Vec<f32>,
        /// Recurrent state `[v_heads][k_dim][v_dim]`.
        state: Vec<f32>,
    },
    Full {
        /// `[pos][kv_heads][head_dim]`
        k: Vec<f32>,
        v: Vec<f32>,
    },
}

impl Cache {
    pub fn new(cfg: &Config) -> Self {
        let layers = cfg
            .layer_kinds
            .iter()
            .map(|k| match k {
                LayerKind::Linear => LayerCache::Linear {
                    conv: vec![0.0; (cfg.conv_kernel - 1) * cfg.conv_dim()],
                    state: vec![0.0; cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim],
                },
                LayerKind::Full => LayerCache::Full { k: Vec::new(), v: Vec::new() },
            })
            .collect();
        Self { pos: 0, layers }
    }

    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerCache::Linear { conv, state } => (conv.len() + state.len()) * 4,
                LayerCache::Full { k, v } => (k.len() + v.len()) * 4,
            })
            .sum()
    }
}

/// One sequence in a forward call: its state and the new tokens to append.
pub struct Seq<'a> {
    pub cache: &'a mut Cache,
    pub tokens: &'a [u32],
}

/// Observation / intervention points. Rows are the sequence's new positions `pos0..pos0 + rows`.
pub trait Hooks {
    /// MLP neuron activations `silu(gate) * up` (`rows × intermediate`), editable in place.
    fn mlp(&mut self, _layer: usize, _seq: usize, _pos0: usize, _acts: &mut [f32]) {}
    /// Residual stream leaving `layer` (`rows × hidden`).
    fn resid(&mut self, _layer: usize, _seq: usize, _pos0: usize, _x: &[f32]) {}
}

pub struct NoHooks;
impl Hooks for NoHooks {}

impl Model {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg = Config::load(&dir.join("config.json"))?;
        let st = SafeTensors::open_dir(dir)?;
        let prefix = if st.has("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else if st.has("model.embed_tokens.weight") {
            "model."
        } else {
            bail!("unrecognised checkpoint layout in {}", dir.display());
        };
        let p = |s: &str| format!("{prefix}{s}");
        let plus_one = |v: Vec<f32>| v.into_iter().map(|w| 1.0 + w).collect::<Vec<_>>();

        let embed = st.mat16(&p("embed_tokens.weight"))?;
        let lm_head = if st.has("lm_head.weight") && !cfg.tie_embeddings { st.mat16("lm_head.weight")? } else { embed.clone() };
        if embed.rows < cfg.vocab || embed.cols != cfg.hidden {
            bail!("embedding shape {}x{} does not match config", embed.rows, embed.cols);
        }

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for (i, kind) in cfg.layer_kinds.iter().enumerate() {
            let l = |s: &str| p(&format!("layers.{i}.{s}"));
            let mixer = match kind {
                LayerKind::Linear => {
                    let conv = st.f32s(&l("linear_attn.conv1d.weight"))?;
                    if conv.len() != cfg.conv_dim() * cfg.conv_kernel {
                        bail!("layer {i}: conv weight size {}", conv.len());
                    }
                    Mixer::Linear(DeltaNet {
                        qkv: st.mat16(&l("linear_attn.in_proj_qkv.weight"))?,
                        z: st.mat16(&l("linear_attn.in_proj_z.weight"))?,
                        b: st.mat16(&l("linear_attn.in_proj_b.weight"))?,
                        a: st.mat16(&l("linear_attn.in_proj_a.weight"))?,
                        conv,
                        neg_a: st.f32s(&l("linear_attn.A_log"))?.into_iter().map(|a| -a.exp()).collect(),
                        dt_bias: st.f32s(&l("linear_attn.dt_bias"))?,
                        norm: st.f32s(&l("linear_attn.norm.weight"))?,
                        out: st.mat16(&l("linear_attn.out_proj.weight"))?,
                    })
                }
                LayerKind::Full => Mixer::Full(Attention {
                    q: st.mat16(&l("self_attn.q_proj.weight"))?,
                    k: st.mat16(&l("self_attn.k_proj.weight"))?,
                    v: st.mat16(&l("self_attn.v_proj.weight"))?,
                    o: st.mat16(&l("self_attn.o_proj.weight"))?,
                    q_norm: plus_one(st.f32s(&l("self_attn.q_norm.weight"))?),
                    k_norm: plus_one(st.f32s(&l("self_attn.k_norm.weight"))?),
                }),
            };
            layers.push(Layer {
                ln1: plus_one(st.f32s(&l("input_layernorm.weight"))?),
                ln2: plus_one(st.f32s(&l("post_attention_layernorm.weight"))?),
                mixer,
                mlp: Mlp {
                    gate: st.mat16(&l("mlp.gate_proj.weight"))?,
                    up: st.mat16(&l("mlp.up_proj.weight"))?,
                    down: st.mat16(&l("mlp.down_proj.weight"))?,
                },
            });
        }
        let final_norm = plus_one(st.f32s(&p("norm.weight"))?);
        let half = cfg.rotary_dim / 2;
        let inv_freq = (0..half)
            .map(|i| 1.0 / (cfg.rope_theta as f32).powf((2 * i) as f32 / cfg.rotary_dim as f32))
            .collect();
        Ok(Self { cfg, embed, lm_head, layers, final_norm, inv_freq, st })
    }

    /// Faults every text-model weight page into memory (call from a background thread).
    pub fn prefetch(&self) {
        let names: Vec<String> = self.st.names().filter(|n| !n.contains("visual") && !n.starts_with("mtp")).cloned().collect();
        self.st.prefetch(&names);
    }

    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.cfg)
    }

    /// Runs the new tokens of every sequence through the model, advancing their caches.
    /// Returns next-token logits for the last new token of each sequence (if `logits`).
    pub fn forward(&self, seqs: &mut [Seq], hooks: &mut dyn Hooks, logits: bool) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let (d, n) = (cfg.hidden, cfg.intermediate);
        let lens: Vec<usize> = seqs.iter().map(|s| s.tokens.len()).collect();
        let mut offs = vec![0usize; seqs.len() + 1];
        for (i, l) in lens.iter().enumerate() {
            offs[i + 1] = offs[i] + l;
        }
        let total = offs[seqs.len()];
        if total == 0 {
            return vec![Vec::new(); seqs.len()];
        }

        let mut x = vec![0f32; total * d];
        {
            let toks: Vec<u32> = seqs.iter().flat_map(|s| s.tokens.iter().copied()).collect();
            x.par_chunks_mut(d).zip(toks.par_iter()).for_each(|(row, &t)| self.embed.row_to_f32(t as usize, row));
        }
        let mut h = vec![0f32; total * d];
        let mut mixed = vec![0f32; total * d];
        let mut g = vec![0f32; total * n];
        let mut u = vec![0f32; total * n];

        for (li, layer) in self.layers.iter().enumerate() {
            rmsnorm_rows(&x, &layer.ln1, cfg.eps, &mut h, d);
            match &layer.mixer {
                Mixer::Linear(dn) => self.deltanet(dn, li, &h, seqs, &offs, &mut mixed),
                Mixer::Full(at) => self.attention(at, li, &h, seqs, &offs, &mut mixed),
            }
            add_inplace(&mut x, &mixed);

            rmsnorm_rows(&x, &layer.ln2, cfg.eps, &mut h, d);
            matmul(&layer.mlp.gate, &h, total, &mut g);
            matmul(&layer.mlp.up, &h, total, &mut u);
            g.par_chunks_mut(1 << 14).zip(u.par_chunks(1 << 14)).for_each(|(a, b)| {
                for (p, q) in a.iter_mut().zip(b) {
                    *p = silu(*p) * q;
                }
            });
            for (s, seq) in seqs.iter().enumerate() {
                hooks.mlp(li, s, seq.cache.pos, &mut g[offs[s] * n..offs[s + 1] * n]);
            }
            matmul(&layer.mlp.down, &g, total, &mut mixed);
            add_inplace(&mut x, &mixed);
            for (s, seq) in seqs.iter().enumerate() {
                hooks.resid(li, s, seq.cache.pos, &x[offs[s] * d..offs[s + 1] * d]);
            }
        }
        for (seq, l) in seqs.iter_mut().zip(&lens) {
            seq.cache.pos += l;
        }
        if !logits {
            return vec![Vec::new(); seqs.len()];
        }
        let live: Vec<usize> = (0..seqs.len()).filter(|&s| lens[s] > 0).collect();
        let mut last = vec![0f32; live.len() * d];
        for (i, &s) in live.iter().enumerate() {
            let r = offs[s + 1] - 1;
            last[i * d..(i + 1) * d].copy_from_slice(&x[r * d..(r + 1) * d]);
        }
        let out = self.unembed(&last, live.len());
        let mut res = vec![Vec::new(); seqs.len()];
        let v = self.lm_head.rows;
        for (i, &s) in live.iter().enumerate() {
            res[s] = out[i * v..(i + 1) * v].to_vec();
        }
        res
    }

    /// Final norm + LM head on `n` residual vectors (`n × hidden`) → `n × vocab` logits.
    pub fn unembed(&self, resid: &[f32], n: usize) -> Vec<f32> {
        let d = self.cfg.hidden;
        let mut normed = vec![0f32; n * d];
        rmsnorm_rows(resid, &self.final_norm, self.cfg.eps, &mut normed, d);
        let mut out = vec![0f32; n * self.lm_head.rows];
        matmul(&self.lm_head, &normed, n, &mut out);
        out
    }

    fn deltanet(&self, dn: &DeltaNet, li: usize, h: &[f32], seqs: &mut [Seq], offs: &[usize], out: &mut [f32]) {
        let cfg = &self.cfg;
        let total = h.len() / cfg.hidden;
        let c = cfg.conv_dim();
        let (hk, hv, dk, dv) = (cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim);
        let key_dim = hk * dk;
        let vd = hv * dv;
        let kern = cfg.conv_kernel;

        let mut qkv = vec![0f32; total * c];
        matmul(&dn.qkv, h, total, &mut qkv);
        let mut z = vec![0f32; total * vd];
        matmul(&dn.z, h, total, &mut z);
        let mut bb = vec![0f32; total * hv];
        matmul(&dn.b, h, total, &mut bb);
        let mut aa = vec![0f32; total * hv];
        matmul(&dn.a, h, total, &mut aa);

        let mut core = vec![0f32; total * vd];
        for (s, seq) in seqs.iter_mut().enumerate() {
            let (r0, nrows) = (offs[s], offs[s + 1] - offs[s]);
            if nrows == 0 {
                continue;
            }
            let LayerCache::Linear { conv, state } = &mut seq.cache.layers[li] else {
                unreachable!("layer {li} is not linear attention")
            };

            // Causal depthwise conv over [previous kernel-1 inputs ; new inputs], then SiLU.
            let rows = &mut qkv[r0 * c..(r0 + nrows) * c];
            let mut ext = Vec::with_capacity((kern - 1 + nrows) * c);
            ext.extend_from_slice(conv);
            ext.extend_from_slice(rows);
            rows.par_chunks_mut(c).enumerate().for_each(|(t, row)| {
                for (ch, o) in row.iter_mut().enumerate() {
                    let w = &dn.conv[ch * kern..(ch + 1) * kern];
                    let mut acc = 0.0;
                    for (j, &wj) in w.iter().enumerate() {
                        acc += wj * ext[(t + j) * c + ch];
                    }
                    *o = silu(acc);
                }
            });
            conv.copy_from_slice(&ext[nrows * c..]);

            // L2-normalise q and k per key head; q also gets 1/sqrt(dk).
            let qscale = 1.0 / (dk as f32).sqrt();
            rows.par_chunks_mut(c).for_each(|row| {
                for head in 0..hk {
                    for (base, scale) in [(head * dk, qscale), (key_dim + head * dk, 1.0)] {
                        let v = &mut row[base..base + dk];
                        let inv = 1.0 / (kernels::dot(v, v) + 1e-6).sqrt();
                        for x in v.iter_mut() {
                            *x *= inv * scale;
                        }
                    }
                }
            });

            let rows = &*rows;
            let bseg = &bb[r0 * hv..(r0 + nrows) * hv];
            let aseg = &aa[r0 * hv..(r0 + nrows) * hv];
            let zseg = &z[r0 * vd..(r0 + nrows) * vd];
            let ratio = hv / hk;
            let eps = cfg.eps;
            // Each value head owns its slice of the state and of the output columns.
            let outs: Vec<Vec<f32>> = state
                .par_chunks_mut(dk * dv)
                .enumerate()
                .map(|(head, st)| {
                    let kh = head / ratio;
                    let mut o = vec![0f32; nrows * dv];
                    let mut kv = vec![0f32; dv];
                    for t in 0..nrows {
                        let row = &rows[t * c..(t + 1) * c];
                        let q = &row[kh * dk..(kh + 1) * dk];
                        let k = &row[key_dim + kh * dk..key_dim + (kh + 1) * dk];
                        let v = &row[2 * key_dim + head * dv..2 * key_dim + (head + 1) * dv];
                        let decay = (dn.neg_a[head] * softplus(aseg[t * hv + head] + dn.dt_bias[head])).exp();
                        let beta = sigmoid(bseg[t * hv + head]);
                        recur_step(st, q, k, v, decay, beta, &mut kv, &mut o[t * dv..(t + 1) * dv]);
                        // Gated RMSNorm: norm(o) * w * silu(z)
                        let ot = &mut o[t * dv..(t + 1) * dv];
                        let zt = &zseg[t * vd + head * dv..t * vd + (head + 1) * dv];
                        let inv = 1.0 / (kernels::dot(ot, ot) / dv as f32 + eps).sqrt();
                        for ((x, &w), &zz) in ot.iter_mut().zip(&dn.norm).zip(zt) {
                            *x = *x * inv * w * silu(zz);
                        }
                    }
                    o
                })
                .collect();
            for (head, o) in outs.iter().enumerate() {
                for t in 0..nrows {
                    let dst = (r0 + t) * vd + head * dv;
                    core[dst..dst + dv].copy_from_slice(&o[t * dv..(t + 1) * dv]);
                }
            }
        }
        matmul(&dn.out, &core, total, out);
    }

    fn attention(&self, at: &Attention, li: usize, h: &[f32], seqs: &mut [Seq], offs: &[usize], out: &mut [f32]) {
        let cfg = &self.cfg;
        let total = h.len() / cfg.hidden;
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let mut qg = vec![0f32; total * nh * hd * 2];
        matmul(&at.q, h, total, &mut qg);
        let mut kk = vec![0f32; total * nkv * hd];
        matmul(&at.k, h, total, &mut kk);
        let mut vv = vec![0f32; total * nkv * hd];
        matmul(&at.v, h, total, &mut vv);

        let mut o = vec![0f32; total * nh * hd];
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        for (s, seq) in seqs.iter_mut().enumerate() {
            let (r0, nrows) = (offs[s], offs[s + 1] - offs[s]);
            if nrows == 0 {
                continue;
            }
            let pos0 = seq.cache.pos;
            // q: per head [query | gate]; normalise + rotate the query half.
            qg[r0 * nh * hd * 2..(r0 + nrows) * nh * hd * 2]
                .par_chunks_mut(nh * hd * 2)
                .enumerate()
                .for_each(|(t, row)| {
                    for head in 0..nh {
                        let q = &mut row[head * 2 * hd..head * 2 * hd + hd];
                        let tmp = q.to_vec();
                        rmsnorm(&tmp, &at.q_norm, cfg.eps, q);
                        self.rope(q, pos0 + t);
                    }
                });
            kk[r0 * nkv * hd..(r0 + nrows) * nkv * hd].par_chunks_mut(nkv * hd).enumerate().for_each(|(t, row)| {
                for head in 0..nkv {
                    let k = &mut row[head * hd..(head + 1) * hd];
                    let tmp = k.to_vec();
                    rmsnorm(&tmp, &at.k_norm, cfg.eps, k);
                    self.rope(k, pos0 + t);
                }
            });
            let LayerCache::Full { k: kc, v: vc } = &mut seq.cache.layers[li] else {
                unreachable!("layer {li} is not full attention")
            };
            kc.extend_from_slice(&kk[r0 * nkv * hd..(r0 + nrows) * nkv * hd]);
            vc.extend_from_slice(&vv[r0 * nkv * hd..(r0 + nrows) * nkv * hd]);
            let (kc, vc) = (&*kc, &*vc);
            let qg = &qg;
            o[r0 * nh * hd..(r0 + nrows) * nh * hd].par_chunks_mut(hd).enumerate().for_each(|(i, oh)| {
                let (t, head) = (i / nh, i % nh);
                let kvh = head / group;
                let qrow = &qg[(r0 + t) * nh * hd * 2..];
                let q = &qrow[head * 2 * hd..head * 2 * hd + hd];
                let gate = &qrow[head * 2 * hd + hd..head * 2 * hd + 2 * hd];
                let ctx = pos0 + t + 1;
                let mut sc: Vec<f32> = (0..ctx).map(|j| kernels::dot(q, &kc[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd]) * scale).collect();
                kernels::softmax_inplace(&mut sc);
                oh.fill(0.0);
                for (j, &p) in sc.iter().enumerate() {
                    let vrow = &vc[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd];
                    for (a, &b) in oh.iter_mut().zip(vrow) {
                        *a += p * b;
                    }
                }
                for (a, &g) in oh.iter_mut().zip(gate) {
                    *a *= sigmoid(g);
                }
            });
        }
        matmul(&at.o, &o, total, out);
    }

    /// Partial rotary embedding (rotate-half convention) on the first `rotary_dim` dims.
    fn rope(&self, x: &mut [f32], pos: usize) {
        let half = self.cfg.rotary_dim / 2;
        for i in 0..half {
            let ang = pos as f32 * self.inv_freq[i];
            let (s, c) = ang.sin_cos();
            let (a, b) = (x[i], x[i + half]);
            x[i] = a * c - b * s;
            x[i + half] = b * c + a * s;
        }
    }

    pub fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    pub fn n_neurons(&self) -> usize {
        self.cfg.intermediate
    }

    /// Neuron `n`'s write direction (column of `down_proj`) in layer `layer`.
    pub fn write_dir(&self, layer: usize, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; self.cfg.hidden];
        self.layers[layer].mlp.down.col_to_f32(n, &mut v);
        v
    }

    pub fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.st.shape(name).map(|s| s.to_vec())
    }
}

/// One gated-delta-rule step for one head; `o` receives `S^T q` after the update.
#[inline(always)]
fn recur_body(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], decay: f32, beta: f32, kv: &mut [f32], o: &mut [f32]) {
    let dv = v.len();
    kv.fill(0.0);
    for (i, &ki) in k.iter().enumerate() {
        let row = &s[i * dv..(i + 1) * dv];
        for (a, &b) in kv.iter_mut().zip(row) {
            *a += b * ki;
        }
    }
    // delta = (v - decay * S^T k) * beta
    for (a, &vj) in kv.iter_mut().zip(v) {
        *a = (vj - decay * *a) * beta;
    }
    o.fill(0.0);
    for (i, (&ki, &qi)) in k.iter().zip(q).enumerate() {
        let row = &mut s[i * dv..(i + 1) * dv];
        for ((sv, &dj), oj) in row.iter_mut().zip(kv.iter()).zip(o.iter_mut()) {
            *sv = *sv * decay + ki * dj;
            *oj += *sv * qi;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn recur_avx2(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], decay: f32, beta: f32, kv: &mut [f32], o: &mut [f32]) {
    recur_body(s, q, k, v, decay, beta, kv, o)
}

#[allow(clippy::too_many_arguments)]
fn recur_step(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], decay: f32, beta: f32, kv: &mut [f32], o: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if kernels::simd() {
        // SAFETY: CPU features checked at runtime.
        unsafe { recur_avx2(s, q, k, v, decay, beta, kv, o) };
        return;
    }
    recur_body(s, q, k, v, decay, beta, kv, o)
}

/// Loads a model directory and checks it is a Qwen3.5 text model we can run.
pub fn load(dir: &Path) -> Result<Model> {
    Model::load(dir).with_context(|| format!("loading model from {}", dir.display()))
}
