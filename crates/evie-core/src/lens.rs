//! Jacobian-style lenses (J-lens, R-lens, J++): per-layer linear maps `J_l` that transport a
//! residual-stream vector at layer `l` into the readout basis, decoded with the model's own
//! final norm + unembedding:  `lens_l(h) = unembed(J_l @ h)`  (Gurnee et al., 2026).

use crate::kernels::{Kind16, Mat16, matmul};
use crate::pickle::{TorchFile, Val};
use anyhow::{Context, Result, bail};
use std::path::Path;

pub struct Lens {
    pub name: String,
    /// Layer whose output basis the maps target; layers past it read out with the identity.
    pub target_layer: usize,
    pub mats: Vec<Option<Mat16>>,
    pub n_prompts: usize,
    pub method: String,
}

impl Lens {
    pub fn load(path: &Path, name: &str, n_layers: usize, d: usize) -> Result<Self> {
        let tf = TorchFile::open(path)?;
        let Some(Val::Dict(items)) = tf.root.get("J") else { bail!("{}: no `J` dict (not a lens file?)", path.display()) };
        let mut mats: Vec<Option<Mat16>> = vec![None; n_layers];
        let mut max_src = 0;
        for (k, v) in items {
            let layer = k.as_int().context("non-integer layer key")? as usize;
            let Val::Tensor(t) = v else { continue };
            if layer >= n_layers {
                bail!("{}: layer {layer} out of range for a {n_layers}-layer model", path.display());
            }
            if t.shape != [d, d] {
                bail!("{}: layer {layer} map has shape {:?}, expected [{d}, {d}]", path.display(), t.shape);
            }
            let kind = match t.dtype.as_str() {
                "HalfStorage" => Kind16::F16,
                "BFloat16Storage" => Kind16::Bf16,
                other => bail!("{}: unsupported lens dtype {other}", path.display()),
            };
            mats[layer] = Some(Mat16::from_vec(d, d, kind, tf.tensor_u16(t)?));
            max_src = max_src.max(layer);
        }
        let prov = tf.root.get("provenance");
        let target_layer = prov
            .and_then(|p| p.get("target_layer"))
            .and_then(Val::as_int)
            .map(|t| t as usize)
            .unwrap_or((max_src + 1).min(n_layers - 1));
        let method = prov
            .and_then(|p| p.get("method").or_else(|| p.get("config_json")))
            .and_then(Val::as_str)
            .unwrap_or("jacobian")
            .to_string();
        let n_prompts = tf.root.get("n_prompts").and_then(Val::as_int).unwrap_or(0) as usize;
        Ok(Self { name: name.to_string(), target_layer, mats, n_prompts, method })
    }

    pub fn map(&self, layer: usize) -> Option<&Mat16> {
        if layer > self.target_layer {
            return None;
        }
        self.mats.get(layer).and_then(Option::as_ref)
    }

    /// `J_l @ x` for `n` row vectors at `layer` (identity where the lens has no map).
    pub fn transport(&self, layer: usize, x: &[f32], n: usize) -> Vec<f32> {
        match self.map(layer) {
            Some(m) => {
                let mut y = vec![0f32; n * m.rows];
                matmul(m, x, n, &mut y);
                y
            }
            None => x.to_vec(),
        }
    }

    /// `J_lᵀ @ a` for a single vector (pulls a readout-basis direction back to layer `l`).
    pub fn pullback(&self, layer: usize, a: &[f32]) -> Vec<f32> {
        match self.map(layer) {
            Some(m) => crate::kernels::matvec_t(m, &[a])[0].clone(),
            None => a.to_vec(),
        }
    }
}
