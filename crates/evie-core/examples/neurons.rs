//! Prints neuron readouts in the format of a9lim's J-lens post:
//! `cargo run --release --example neurons -- <model-id> <layer> <neuron> [words...]`.

use evie_core::{Engine, LensSel};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let t = Instant::now();
    let eng = Engine::load(&args[1], &|s| eprintln!("  {s}"))?;
    eprintln!("engine ready in {:.2?}; lenses: {:?}", t.elapsed(), eng.lenses.iter().map(|l| (&l.name, l.target_layer)).collect::<Vec<_>>());
    let (layer, neuron): (usize, usize) = (args[2].parse()?, args[3].parse()?);
    let words: Vec<String> = args[4..].to_vec();
    let toks = eng.word_tokens(&words);
    println!("L{layer} n{neuron}  unit(dose=1) = {:+.2}   words {:?} -> {:?}", eng.dose_unit(layer, neuron), words,
        toks.iter().map(|&t| eng.token_str(t)).collect::<Vec<_>>());
    for sel in eng.lens_list() {
        let t = Instant::now();
        let r = &eng.readouts(&sel, &[(layer, neuron)], 10)[0];
        let ranks = evie_core::analysis::token_ranks(&eng.model, eng.lens(&sel), &[(layer, neuron)], &toks)[0];
        let fmt = |v: &[(u32, f32)]| v.iter().map(|(t, _)| format!("{:?}", eng.token_str(*t))).collect::<Vec<_>>().join(" ");
        println!("  {:<5} + best rank {:>6} | {}", sel.label(), ranks.0, fmt(&r.up));
        println!("  {:<5} - best rank {:>6} | {}   ({:.0?})", "", ranks.1, fmt(&r.down), t.elapsed());
    }
    Ok(())
}
