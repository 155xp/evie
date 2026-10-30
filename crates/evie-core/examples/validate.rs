//! Compares evie's forward pass with a dump from `tools/reference.py`:
//! `cargo run --release --example validate -- <model_dir> <reference.json> [prompt]`.

use evie_core::{ChatMsg, Hooks, Model, Seq, Tok};
use serde_json::Value;

struct Capture {
    acts: Vec<Vec<f32>>,
    norms: Vec<Vec<f32>>,
    n: usize,
    d: usize,
}

impl Hooks for Capture {
    fn mlp(&mut self, layer: usize, _seq: usize, _pos0: usize, acts: &mut [f32]) {
        self.acts[layer] = acts.to_vec();
    }
    fn resid(&mut self, layer: usize, _seq: usize, _pos0: usize, x: &[f32]) {
        self.norms[layer] = x.chunks(self.d).map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt()).collect();
    }
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let r: Value = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Who are you?".into());
    let model = Model::load(&dir)?;
    let tok = Tok::load(&dir.join("tokenizer.json"), model.cfg.vocab)?;

    let text = tok.chat(&[ChatMsg::user(prompt)], true);
    println!("chat text matches: {}", text == r["text"].as_str().unwrap());
    let ids = tok.encode(&text)?;
    let ref_ids: Vec<u32> = r["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    println!("token ids match:   {} ({} tokens)", ids == ref_ids, ids.len());

    let (nl, n, d) = (model.cfg.n_layers, model.cfg.intermediate, model.cfg.hidden);
    let mut cap = Capture { acts: vec![vec![]; nl], norms: vec![vec![]; nl], n, d };
    let mut cache = model.new_cache();
    let logits = model.forward(&mut [Seq { cache: &mut cache, tokens: &ref_ids }], &mut cap, true).remove(0);

    let mut worst_norm = 0f32;
    for l in 0..nl {
        let rn = floats(&r["resid_norms"][l]);
        for (a, b) in cap.norms[l].iter().zip(&rn) {
            worst_norm = worst_norm.max((a - b).abs() / b.max(1e-3));
        }
    }
    println!("resid norm max rel err: {worst_norm:.2e}");

    let stride = r["act_stride"].as_u64().unwrap() as usize;
    let (mut num, mut den) = (0f64, 0f64);
    for l in 0..nl {
        for (t, row) in r["acts"][l].as_array().unwrap().iter().enumerate() {
            for (j, v) in floats(row).iter().enumerate() {
                let mine = cap.acts[l][t * cap.n + j * stride];
                num += ((mine - v) as f64).powi(2);
                den += (*v as f64).powi(2);
            }
        }
    }
    println!("mlp act rel rms err: {:.2e}", (num / den).sqrt());

    let ls = r["logits_stride"].as_u64().unwrap() as usize;
    let rl = floats(&r["logits"]);
    let max_err = rl.iter().enumerate().map(|(i, v)| (logits[i * ls] - v).abs()).fold(0f32, f32::max);
    println!("logits max abs err (strided sample): {max_err:.3e}");
    let top_ref: Vec<u32> = r["top"].as_array().unwrap().iter().map(|x| x[0].as_u64().unwrap() as u32).collect();
    let top_mine: Vec<u32> = evie_core::kernels::topk(&logits, 20).iter().map(|x| x.0).collect();
    println!("top-1 match: {}  top-10 overlap: {}/10", top_ref[0] == top_mine[0], top_mine[..10].iter().filter(|t| top_ref[..10].contains(t)).count());

    let mut out = Vec::new();
    let mut lg = logits;
    for _ in 0..r["greedy"].as_array().unwrap().len() {
        let next = evie_core::kernels::topk(&lg, 1)[0].0;
        out.push(next);
        lg = model.forward(&mut [Seq { cache: &mut cache, tokens: &[next] }], &mut evie_core::NoHooks, true).remove(0);
    }
    let rg: Vec<u32> = r["greedy"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let agree = out.iter().zip(&rg).take_while(|(a, b)| a == b).count();
    println!("greedy agreement: {agree}/{} tokens", rg.len());
    println!("  ref : {:?}", tok.decode(&rg));
    println!("  mine: {:?}", tok.decode(&out));
    Ok(())
}
