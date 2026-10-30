//! Next-token sampling: greedy by default, with temperature / top-p / min-p / repetition penalty.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SampleParams {
    /// 0 = greedy.
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self { temperature: 0.0, top_p: 1.0, min_p: 0.0, repetition_penalty: 1.0, seed: 0 }
    }
}

pub struct Sampler {
    pub params: SampleParams,
    rng: u64,
}

impl Sampler {
    pub fn new(params: SampleParams) -> Self {
        Self { params, rng: params.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03 }
    }

    fn uniform(&mut self) -> f32 {
        // splitmix64
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let p = self.params;
        let penalize = p.repetition_penalty != 1.0 && !history.is_empty();
        let adjusted;
        let logits = if penalize {
            let mut l = logits.to_vec();
            let mut seen = history.to_vec();
            seen.sort_unstable();
            seen.dedup();
            for t in seen {
                if let Some(v) = l.get_mut(t as usize) {
                    *v = if *v > 0.0 { *v / p.repetition_penalty } else { *v * p.repetition_penalty };
                }
            }
            adjusted = l;
            &adjusted[..]
        } else {
            logits
        };
        if p.temperature <= 0.0 {
            return crate::kernels::topk(logits, 1)[0].0;
        }
        let cand = crate::kernels::topk(logits, 512);
        let top = cand[0].1;
        let mut probs: Vec<(u32, f32)> = cand.iter().map(|&(t, l)| (t, ((l - top) / p.temperature).exp())).collect();
        let total: f32 = probs.iter().map(|x| x.1).sum();
        probs.iter_mut().for_each(|x| x.1 /= total);
        if p.min_p > 0.0 {
            let floor = probs[0].1 * p.min_p;
            probs.retain(|x| x.1 >= floor);
        }
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            let mut keep = 0;
            for x in &probs {
                acc += x.1;
                keep += 1;
                if acc >= p.top_p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
        }
        let total: f32 = probs.iter().map(|x| x.1).sum();
        let mut r = self.uniform() * total;
        for &(t, pr) in &probs {
            r -= pr;
            if r <= 0.0 {
                return t;
            }
        }
        probs.last().map(|x| x.0).unwrap_or(cand[0].0)
    }
}
