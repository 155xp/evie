//! GPU vs CPU agreement + GPU decode speed: `cargo run --release --example gpu_check -- <model_dir>`.

use evie_core::gpu::{GpuModel, GpuSeq};
use evie_core::{ChatMsg, Hooks, Model, NoHooks, Seq, Tok};
use std::time::Instant;

struct Acts(Vec<Vec<f32>>, Vec<Vec<f32>>);
impl Hooks for Acts {
    fn mlp(&mut self, layer: usize, _s: usize, _p: usize, a: &mut [f32]) {
        self.0[layer] = a.to_vec();
    }
    fn resid(&mut self, layer: usize, _s: usize, _p: usize, x: &[f32]) {
        self.1[layer] = x.to_vec();
    }
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Who are you?".into());
    let model = Model::load(&dir)?;
    let tok = Tok::load(&dir.join("tokenizer.json"), model.cfg.vocab)?;
    let t = Instant::now();
    let gpu = GpuModel::new(&model, 2048, &|s| eprintln!("  {s}"))?;
    eprintln!("GPU {} ready in {:.1?}, weights {:.2} GB", gpu.adapter_name, t.elapsed(), gpu.weight_bytes as f64 / 1e9);
    let ids = tok.encode(&tok.chat(&[ChatMsg::user(prompt)], true))?;

    // CPU reference (exact bf16 weights).
    let mut cache = model.new_cache();
    let mut acts = Acts(vec![Vec::new(); model.cfg.n_layers], vec![Vec::new(); model.cfg.n_layers]);
    let cpu = model.forward(&mut [Seq { cache: &mut cache, tokens: &ids }], &mut acts, true).remove(0);

    // GPU: scale = 1 so captured frames are sqrt-companded raw activations (clipped at 1).
    gpu.reset_slot(0);
    gpu.set_ops(0, &[]);
    let t = Instant::now();
    let out = gpu.forward(&[GpuSeq { slot: 0, pos0: 0, tokens: &ids }], true, true)?;
    eprintln!("GPU prefill {} tokens in {:.1?}", ids.len(), t.elapsed());
    let g = &out.logits[0];
    let max_err = cpu.iter().zip(g).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let top_c: Vec<u32> = evie_core::kernels::topk(&cpu, 10).iter().map(|x| x.0).collect();
    let top_g: Vec<u32> = evie_core::kernels::topk(g, 10).iter().map(|x| x.0).collect();
    println!("logits max abs diff {max_err:.3}  (logit scale ~{:.1})", cpu.iter().fold(0f32, |a, b| a.max(b.abs())));
    println!("top-1 match {}  top-10 overlap {}/10", top_c[0] == top_g[0], top_g.iter().filter(|t| top_c.contains(t)).count());
    // Compare captured activations of the last row (decode the sqrt companding; only |v| < 1 is exact).
    let n = model.cfg.intermediate;
    let last = ids.len() - 1;
    let (mut num, mut den, mut cnt) = (0f64, 0f64, 0);
    for l in 0..model.cfg.n_layers {
        for j in 0..n {
            let c = acts.0[l][last * n + j];
            if c.abs() < 0.9 {
                let q = out.frames[last][l * n + j] as f32 / 127.0;
                let gv = q * q.abs();
                num += ((gv - c) as f64).powi(2);
                den += (c as f64).powi(2);
                cnt += 1;
            }
        }
    }
    println!("captured activations rel rms err {:.3} over {cnt} neurons", (num / den).sqrt());

    // Residual stream per layer at the last row: where does the GPU first diverge?
    let d = model.cfg.hidden;
    let mut line = String::new();
    for l in 0..model.cfg.n_layers {
        let c = &acts.1[l][last * d..(last + 1) * d];
        let g = &out.resid[last][l * d..(l + 1) * d];
        let (mut nn, mut dd) = (0f64, 0f64);
        for (a, b) in c.iter().zip(g) {
            nn += ((b.to_f32() - a) as f64).powi(2);
            dd += (*a as f64).powi(2);
        }
        line += &format!(" L{l}:{:.3}", (nn / dd).sqrt());
    }
    println!("resid rel err per layer:{line}");

    // Greedy decode on both.
    let steps = 32;
    let mut cpu_out = Vec::new();
    let mut lg = cpu;
    let t = Instant::now();
    for _ in 0..steps {
        let next = evie_core::kernels::topk(&lg, 1)[0].0;
        cpu_out.push(next);
        lg = model.forward(&mut [Seq { cache: &mut cache, tokens: &[next] }], &mut NoHooks, true).remove(0);
    }
    let cpu_dt = t.elapsed();
    let mut gpu_out = Vec::new();
    let mut lg = out.logits[0].clone();
    let t = Instant::now();
    for i in 0..steps {
        let next = evie_core::kernels::topk(&lg, 1)[0].0;
        gpu_out.push(next);
        lg = gpu.forward(&[GpuSeq { slot: 0, pos0: ids.len() + i, tokens: &[next] }], true, true)?.logits.remove(0);
    }
    let gpu_dt = t.elapsed();
    let agree = cpu_out.iter().zip(&gpu_out).take_while(|(a, b)| a == b).count();
    println!("greedy agreement {agree}/{steps}");
    println!("  cpu: {:?}", tok.decode(&cpu_out));
    println!("  gpu: {:?}", tok.decode(&gpu_out));
    println!("decode speed: cpu {:.1} tok/s, gpu {:.1} tok/s (with full activation capture)", steps as f64 / cpu_dt.as_secs_f64(), steps as f64 / gpu_dt.as_secs_f64());
    Ok(())
}
