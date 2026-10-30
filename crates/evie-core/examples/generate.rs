//! Greedy generation smoke test: `cargo run --release --example generate -- <model_dir> "<prompt>"`.

use evie_core::{ChatMsg, Model, NoHooks, Seq, Tok};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let prompt = args.get(2).cloned().unwrap_or_else(|| "Who are you?".into());
    let max_new: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(48);

    let t = Instant::now();
    let model = Model::load(&dir)?;
    let tok = Tok::load(&dir.join("tokenizer.json"), model.cfg.vocab)?;
    eprintln!("loaded in {:.2?}", t.elapsed());
    let t = Instant::now();
    model.prefetch();
    eprintln!("prefetched in {:.2?}", t.elapsed());

    let text = tok.chat(&[ChatMsg::user(prompt)], true);
    let ids = tok.encode(&text)?;
    let mut cache = model.new_cache();
    let t = Instant::now();
    let mut logits = model.forward(&mut [Seq { cache: &mut cache, tokens: &ids }], &mut NoHooks, true).remove(0);
    eprintln!("prefill {} tokens in {:.2?}", ids.len(), t.elapsed());

    let mut out = Vec::new();
    let t = Instant::now();
    for _ in 0..max_new {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        if next == tok.im_end || next == tok.endoftext {
            break;
        }
        out.push(next);
        logits = model.forward(&mut [Seq { cache: &mut cache, tokens: &[next] }], &mut NoHooks, true).remove(0);
    }
    let dt = t.elapsed();
    println!("{}", tok.decode(&out));
    eprintln!("decode {} tokens in {:.2?} ({:.1} tok/s)", out.len(), dt, out.len() as f64 / dt.as_secs_f64());
    Ok(())
}
