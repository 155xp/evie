//! Kernel micro-benchmarks against a real checkpoint: `cargo run --release --example bench -- <model_dir>`.

use evie_core::Model;
use evie_core::kernels::{matmul, matvec};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let model = Model::load(&dir)?;
    model.prefetch();
    let lm = &model.lm_head;
    let d = model.cfg.hidden;
    let x = vec![0.01f32; d];
    let mut y = vec![0f32; lm.rows];
    for _ in 0..3 {
        let t = Instant::now();
        matvec(lm, &x, &mut y);
        let dt = t.elapsed();
        let gb = (lm.rows * lm.cols * 2) as f64 / 1e9;
        println!("lm_head matvec {}x{}: {:.2?} = {:.1} GB/s", lm.rows, lm.cols, dt, gb / dt.as_secs_f64());
    }
    let w = &model.layers[0].mlp.gate;
    let xs = vec![0.01f32; w.cols];
    let mut ys = vec![0f32; w.rows];
    let t = Instant::now();
    for _ in 0..200 {
        matvec(w, &xs, &mut ys);
    }
    let dt = t.elapsed() / 200;
    println!("mlp.gate matvec {}x{}: {:.2?} = {:.1} GB/s", w.rows, w.cols, dt, (w.rows * w.cols * 2) as f64 / 1e9 / dt.as_secs_f64());
    for n in [4, 16, 64, 256] {
        let xs = vec![0.01f32; n * w.cols];
        let mut ys = vec![0f32; n * w.rows];
        let t = Instant::now();
        matmul(w, &xs, n, &mut ys);
        let dt = t.elapsed();
        println!("mlp.gate matmul n={n}: {:.2?} = {:.0} GFLOP/s", dt, (2 * n * w.rows * w.cols) as f64 / 1e9 / dt.as_secs_f64());
    }
    // One decode step's worth of forward work, timed by part.
    let mut cache = model.new_cache();
    let ids: Vec<u32> = (0..16).map(|i| 1000 + i).collect();
    model.forward(&mut [evie_core::Seq { cache: &mut cache, tokens: &ids }], &mut evie_core::NoHooks, true);
    let t = Instant::now();
    for _ in 0..10 {
        model.forward(&mut [evie_core::Seq { cache: &mut cache, tokens: &[42] }], &mut evie_core::NoHooks, false);
    }
    println!("forward (no logits): {:.2?}/token", t.elapsed() / 10);
    let t = Instant::now();
    for _ in 0..10 {
        model.forward(&mut [evie_core::Seq { cache: &mut cache, tokens: &[42] }], &mut evie_core::NoHooks, true);
    }
    println!("forward (+logits): {:.2?}/token", t.elapsed() / 10);
    Ok(())
}
