//! A loaded model plus everything needed to read and steer it.

use crate::analysis::{self, Hit, Readout, UnembedStats};
use crate::calib::Calib;
use crate::lens::Lens;
use crate::model::Model;
use crate::registry;
use crate::steering::{Intervention, Plan};
use crate::tokenizer::Tok;
use anyhow::{Context, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Which readout to use: the plain logit lens or one of the model's fitted lenses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LensSel {
    Logit,
    Named(String),
}

impl LensSel {
    pub fn label(&self) -> &str {
        match self {
            LensSel::Logit => "logit",
            LensSel::Named(n) => n,
        }
    }
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "" | "logit" | "l" => LensSel::Logit,
            other => LensSel::Named(other.to_uppercase().replace("JPP", "J++")),
        }
    }
}

pub struct Engine {
    pub id: String,
    pub model: Model,
    pub tok: Tok,
    /// Median residual norm leaving each layer: the dose unit.
    pub units: Vec<f32>,
    /// ‖W_down[:, n]‖ per layer.
    pub col_norms: Vec<Vec<f32>>,
    pub lenses: Vec<Lens>,
    /// Per-neuron activation scale (layer-major), filled in by a background calibration.
    pub act_scale: RwLock<Option<Arc<Vec<f32>>>>,
    pub cache_dir: PathBuf,
    stats: OnceLock<UnembedStats>,
    sigma: Mutex<HashMap<LensSel, Arc<Vec<f32>>>>,
}

impl Engine {
    pub fn load(id: &str, status: &dyn Fn(&str)) -> Result<Self> {
        let dir = registry::model_dir(id);
        status("mapping weights");
        let model = Model::load(&dir).with_context(|| format!("loading {id}"))?;
        status("loading tokenizer");
        let tok = Tok::load(&dir.join("tokenizer.json"), model.cfg.vocab)?;
        let cache_dir = registry::cache_dir(id);
        status("measuring column norms");
        let col_norms: Vec<Vec<f32>> = model.layers.iter().map(|l| l.mlp.down.col_norms()).collect();
        let (units, act_scale) = match Calib::load(&cache_dir) {
            Some(c) if c.median_resid.len() == model.cfg.n_layers => (c.median_resid, Some(Arc::new(c.act_scale))),
            _ => {
                status("measuring dose units (reference passage)");
                (Calib::measure_units(&model, &tok)?, None)
            }
        };
        let mut lenses = Vec::new();
        for (name, path) in registry::installed_lenses(id) {
            status(&format!("loading {name} lens"));
            match Lens::load(&path, name, model.cfg.n_layers, model.cfg.hidden) {
                Ok(l) => lenses.push(l),
                Err(e) => log::warn!("skipping lens {name}: {e:#}"),
            }
        }
        Ok(Self {
            id: id.to_string(),
            model,
            tok,
            units,
            col_norms,
            lenses,
            act_scale: RwLock::new(act_scale),
            cache_dir,
            stats: OnceLock::new(),
            sigma: Mutex::new(HashMap::new()),
        })
    }

    pub fn n_layers(&self) -> usize {
        self.model.cfg.n_layers
    }

    pub fn n_neurons(&self) -> usize {
        self.model.cfg.intermediate
    }

    /// Activation change produced by a dose of 1.0 on this neuron.
    pub fn dose_unit(&self, layer: usize, neuron: usize) -> f32 {
        self.units[layer] / self.col_norms[layer][neuron].max(1e-12)
    }

    pub fn compile(&self, ivs: &[Intervention]) -> Plan {
        Plan::compile(ivs, self.n_layers(), self.n_neurons(), |l, n| self.dose_unit(l, n))
    }

    pub fn lens(&self, sel: &LensSel) -> Option<&Lens> {
        match sel {
            LensSel::Logit => None,
            LensSel::Named(n) => self.lenses.iter().find(|l| l.name.eq_ignore_ascii_case(n)),
        }
    }

    /// The logit lens plus every loaded lens.
    pub fn lens_list(&self) -> Vec<LensSel> {
        std::iter::once(LensSel::Logit).chain(self.lenses.iter().map(|l| LensSel::Named(l.name.clone()))).collect()
    }

    /// Runs the activation-scale calibration if it isn't cached yet.
    pub fn ensure_calibrated(&self, progress: &dyn Fn(f32)) -> Result<()> {
        if self.act_scale.read().is_some() {
            return Ok(());
        }
        let (scale, tokens) = Calib::measure_scales(&self.model, &self.tok, progress)?;
        let calib = Calib { version: 1, median_resid: self.units.clone(), act_scale: scale, act_tokens: tokens };
        if let Err(e) = calib.save(&self.cache_dir) {
            log::warn!("could not cache calibration: {e:#}");
        }
        *self.act_scale.write() = Some(Arc::new(calib.act_scale));
        Ok(())
    }

    pub fn readouts(&self, sel: &LensSel, neurons: &[(usize, usize)], k: usize) -> Vec<Readout> {
        analysis::neuron_readouts(&self.model, self.lens(sel), neurons, k)
    }

    pub fn stats(&self) -> &UnembedStats {
        self.stats.get_or_init(|| UnembedStats::compute(&self.model))
    }

    /// Per-neuron logit spread under `sel`, computed once and cached on disk.
    pub fn sigma(&self, sel: &LensSel, progress: &dyn Fn(f32)) -> Arc<Vec<f32>> {
        if let Some(s) = self.sigma.lock().get(sel) {
            return s.clone();
        }
        let path = self.cache_dir.join(format!("sigma-{}.f32", sel.label().replace('+', "p")));
        let expected = self.n_layers() * self.n_neurons();
        let cached = std::fs::read(&path).ok().filter(|b| b.len() == expected * 4);
        let sigma = match cached {
            Some(b) => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            None => {
                let s = analysis::neuron_sigma(&self.model, self.lens(sel), self.stats(), progress);
                std::fs::create_dir_all(&self.cache_dir).ok();
                let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(&path, bytes).ok();
                s
            }
        };
        let arc = Arc::new(sigma);
        self.sigma.lock().insert(sel.clone(), arc.clone());
        arc
    }

    pub fn sigma_ready(&self, sel: &LensSel) -> bool {
        self.sigma.lock().contains_key(sel)
    }

    /// Token ids for a list of words (all case/space variants that are single tokens).
    pub fn word_tokens(&self, words: &[String]) -> Vec<u32> {
        let mut ids: Vec<u32> = words.iter().flat_map(|w| self.tok.word_tokens(w)).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    pub fn search(&self, sel: &LensSel, tokens: &[u32], limit: usize) -> Vec<Hit> {
        let sigma = self.sigma(sel, &|_| {});
        analysis::search(&self.model, self.lens(sel), self.stats(), &sigma, tokens, limit)
    }

    pub fn token_str(&self, id: u32) -> &str {
        self.tok.display.get(id as usize).map(String::as_str).unwrap_or("")
    }
}
