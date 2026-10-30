//! Per-model calibration: dose units (median residual norms) and per-neuron activation scales.

use crate::model::{Hooks, Model, Seq};
use crate::tokenizer::{ChatMsg, Tok};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The reference passage neuron-explorer (Lyra, MIT) measures residual norms on when no sweep
/// statistics exist. Using the same text keeps evie's dose units comparable with shared recipes.
pub const REFERENCE_TEXT: &str = concat!(
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, ",
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. ",
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n",
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
);

/// Short, varied passages (prose, chat, code, maths, Chinese) for measuring how far each neuron
/// usually swings, so the atlas can colour activations relative to a neuron's own range.
const CORPUS: &[&str] = &[
    "The lighthouse keeper climbed the spiral stairs at dusk, counting each step as his father had taught him. \
     Below, the sea folded itself against the rocks, grey and patient, while gulls argued over the last of the light.",
    "Photosynthesis converts light energy into chemical energy. In the chloroplast, water is split, oxygen is released, \
     and the energy captured is stored in ATP and NADPH, which then power the Calvin cycle to fix carbon dioxide into sugar.",
    "fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n - 1) + fibonacci(n - 2),\n    }\n}\n",
    "Let x be a real number with x^2 - 5x + 6 = 0. Factoring gives (x - 2)(x - 3) = 0, so x = 2 or x = 3. \
     The sum of the roots is 5 and their product is 6, as Vieta's formulas predict.",
    "我们明天早上去公园散步吧。天气预报说会是晴天，气温大约二十度，非常适合户外活动。记得带上水和帽子。",
    "BREAKING: Markets rallied on Tuesday after the central bank held interest rates steady, citing cooling inflation \
     and a resilient labour market. Analysts expect further guidance at next month's meeting.",
    "Dear Sam, thank you so much for the birthday card and the book! I started reading it on the train and could not \
     put it down. Let's get coffee soon — I want to hear everything about your new job. Love, Priya",
];

const CHATS: &[(&str, &str)] = &[
    ("Who are you?", "I am Qwen, a large language model. How can I help you today?"),
    ("Give me a quick recipe for pancakes.", "Whisk 1 cup flour, 1 tbsp sugar, 2 tsp baking powder and a pinch of salt. \
      Add 1 cup milk, 1 egg and 2 tbsp melted butter, stir until just combined, and cook on a hot griddle until golden."),
    ("What do you think about humans?", "Humans are endlessly curious, creative and kind, though sometimes contradictory. \
      I enjoy helping people learn, build and explore new ideas."),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calib {
    pub version: u32,
    /// Median ‖residual leaving layer L‖ over tokens 1.. of [`REFERENCE_TEXT`] (the dose unit).
    pub median_resid: Vec<f32>,
    /// Per neuron (layer-major), the largest |activation| seen on the calibration corpus.
    #[serde(skip)]
    pub act_scale: Vec<f32>,
    pub act_tokens: usize,
}

const VERSION: u32 = 1;

struct ResidNorms {
    d: usize,
    norms: Vec<Vec<f32>>,
}

impl Hooks for ResidNorms {
    fn resid(&mut self, layer: usize, _seq: usize, _pos0: usize, x: &[f32]) {
        self.norms[layer] = x.chunks(self.d).skip(1).map(|r| crate::kernels::dot(r, r).sqrt()).collect();
    }
}

struct MaxAbs {
    n: usize,
    scale: Vec<f32>,
}

impl Hooks for MaxAbs {
    fn mlp(&mut self, layer: usize, _seq: usize, _pos0: usize, acts: &mut [f32]) {
        let dst = &mut self.scale[layer * self.n..(layer + 1) * self.n];
        for row in acts.chunks(self.n).skip(1) {
            for (s, &a) in dst.iter_mut().zip(row) {
                *s = s.max(a.abs());
            }
        }
    }
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 1.0;
    }
    v.sort_by(f32::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 { v[m] } else { 0.5 * (v[m - 1] + v[m]) }
}

impl Calib {
    /// Dose units only (one ~110-token forward pass).
    pub fn measure_units(model: &Model, tok: &Tok) -> Result<Vec<f32>> {
        let ids = tok.encode(REFERENCE_TEXT)?;
        let mut hook = ResidNorms { d: model.cfg.hidden, norms: vec![Vec::new(); model.cfg.n_layers] };
        let mut cache = model.new_cache();
        model.forward(&mut [Seq { cache: &mut cache, tokens: &ids }], &mut hook, false);
        Ok(hook.norms.into_iter().map(median).collect())
    }

    /// Per-neuron activation ranges over the built-in corpus; `progress` gets 0..1.
    pub fn measure_scales(model: &Model, tok: &Tok, progress: &dyn Fn(f32)) -> Result<(Vec<f32>, usize)> {
        let n = model.cfg.intermediate;
        let mut hook = MaxAbs { n, scale: vec![0f32; model.cfg.n_layers * n] };
        let mut texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        texts.extend(CHATS.iter().map(|(u, a)| tok.chat(&[ChatMsg::user(*u), ChatMsg::assistant(*a)], false)));
        let mut tokens = 0;
        for (i, t) in texts.iter().enumerate() {
            let ids = tok.encode(t)?;
            tokens += ids.len().saturating_sub(1);
            let mut cache = model.new_cache();
            model.forward(&mut [Seq { cache: &mut cache, tokens: &ids }], &mut hook, false);
            progress((i + 1) as f32 / texts.len() as f32);
        }
        // Neurons that never fired get a small floor so normalised values stay finite.
        let mut nonzero: Vec<f32> = hook.scale.iter().copied().filter(|&s| s > 0.0).collect();
        nonzero.sort_by(f32::total_cmp);
        let floor = nonzero.get(nonzero.len() / 100).copied().unwrap_or(1e-3).max(1e-4);
        for s in hook.scale.iter_mut() {
            *s = s.max(floor);
        }
        Ok((hook.scale, tokens))
    }

    pub fn load(dir: &Path) -> Option<Self> {
        let meta: Calib = serde_json::from_str(&std::fs::read_to_string(dir.join("calib.json")).ok()?).ok()?;
        if meta.version != VERSION {
            return None;
        }
        let raw = std::fs::read(dir.join("act_scale.f32")).ok()?;
        let act_scale = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        Some(Calib { act_scale, ..meta })
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("calib.json"), serde_json::to_string_pretty(self)?)?;
        let bytes: Vec<u8> = self.act_scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join("act_scale.f32"), bytes).context("writing act_scale.f32")
    }

    pub fn compute(model: &Model, tok: &Tok, progress: &dyn Fn(f32)) -> Result<Self> {
        let median_resid = Self::measure_units(model, tok)?;
        let (act_scale, act_tokens) = Self::measure_scales(model, tok, progress)?;
        Ok(Self { version: VERSION, median_resid, act_scale, act_tokens })
    }
}
