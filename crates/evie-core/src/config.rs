//! Qwen3.5 text-model configuration (`config.json`, possibly nested under `text_config`).

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet (linear attention with a recurrent state).
    Linear,
    /// Gated full softmax attention with a KV cache.
    Full,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub n_layers: usize,
    pub layer_kinds: Vec<LayerKind>,
    pub vocab: usize,
    pub eps: f32,
    pub tie_embeddings: bool,

    // full attention
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,

    // gated deltanet
    pub conv_kernel: usize,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,

    pub eos_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct RawRope {
    rope_theta: f64,
    #[serde(default = "one")]
    partial_rotary_factor: f64,
}

fn one() -> f64 {
    1.0
}

#[derive(Deserialize)]
struct RawText {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    rope_parameters: Option<RawRope>,
    rope_theta: Option<f64>,
    partial_rotary_factor: Option<f64>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let root: serde_json::Value = serde_json::from_str(&text)?;
        let text_cfg = root.get("text_config").unwrap_or(&root);
        let raw: RawText = serde_json::from_value(text_cfg.clone()).context("parsing Qwen3.5 text config")?;

        let (rope_theta, partial) = match &raw.rope_parameters {
            Some(r) => (r.rope_theta, r.partial_rotary_factor),
            None => (raw.rope_theta.unwrap_or(10_000_000.0), raw.partial_rotary_factor.unwrap_or(1.0)),
        };
        let layer_kinds = raw
            .layer_types
            .iter()
            .map(|t| match t.as_str() {
                "linear_attention" => Ok(LayerKind::Linear),
                "full_attention" => Ok(LayerKind::Full),
                other => bail!("unsupported layer type {other:?}"),
            })
            .collect::<Result<Vec<_>>>()?;
        if layer_kinds.len() != raw.num_hidden_layers {
            bail!("layer_types has {} entries for {} layers", layer_kinds.len(), raw.num_hidden_layers);
        }
        let tie = raw
            .tie_word_embeddings
            .or_else(|| root.get("tie_word_embeddings").and_then(|v| v.as_bool()))
            .unwrap_or(false);
        let mut eos_ids = Vec::new();
        for v in [raw.eos_token_id.as_ref(), root.get("eos_token_id")].into_iter().flatten() {
            match v {
                serde_json::Value::Number(n) => eos_ids.extend(n.as_u64().map(|x| x as u32)),
                serde_json::Value::Array(a) => eos_ids.extend(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32)),
                _ => {}
            }
        }
        eos_ids.sort_unstable();
        eos_ids.dedup();

        let rotary_dim = ((raw.head_dim as f64) * partial).round() as usize;
        Ok(Config {
            hidden: raw.hidden_size,
            intermediate: raw.intermediate_size,
            n_layers: raw.num_hidden_layers,
            layer_kinds,
            vocab: raw.vocab_size,
            eps: raw.rms_norm_eps as f32,
            tie_embeddings: tie,
            n_heads: raw.num_attention_heads,
            n_kv_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rotary_dim,
            rope_theta,
            conv_kernel: raw.linear_conv_kernel_dim,
            lin_k_heads: raw.linear_num_key_heads,
            lin_v_heads: raw.linear_num_value_heads,
            lin_k_dim: raw.linear_key_head_dim,
            lin_v_dim: raw.linear_value_head_dim,
            eos_ids,
        })
    }

    pub fn n_neurons(&self) -> usize {
        self.n_layers * self.intermediate
    }

    /// Channels that go through the short causal convolution: q, k and v of a DeltaNet layer.
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim
    }
}
