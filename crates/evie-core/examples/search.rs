//! Concept search over every neuron: `cargo run --release --example search -- <model-id> <lens> <words...>`.

use evie_core::{Engine, LensSel};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let eng = Engine::load(&args[1], &|_| {})?;
    let sel = LensSel::parse(&args[2]);
    let words: Vec<String> = args[3..].to_vec();
    let toks = eng.word_tokens(&words);
    let t = Instant::now();
    eng.sigma(&sel, &|p| if (p * 32.0).fract() < 0.04 { eprint!("\rsigma {:.0}%", p * 100.0) });
    eprintln!("\rsigma ready in {:.1?}", t.elapsed());
    let t = Instant::now();
    let hits = eng.search(&sel, &toks, 12);
    eprintln!("search in {:.1?}", t.elapsed());
    let neurons: Vec<(usize, usize)> = hits.iter().map(|h| (h.layer, h.neuron)).collect();
    for sel2 in eng.lens_list() {
        let ranks = evie_core::analysis::token_ranks(&eng.model, eng.lens(&sel2), &neurons, &toks);
        let reads = eng.readouts(&sel2, &neurons, 6);
        println!("--- exact ranks under {} lens", sel2.label());
        for ((h, r), rd) in hits.iter().zip(&ranks).zip(&reads) {
            let (rank, toks) = if h.sign > 0 { (r.0, &rd.up) } else { (r.1, &rd.down) };
            println!("  L{:<2} N{:<5} {}  z={:5.1}  best rank {:>5} | {}", h.layer, h.neuron, if h.sign > 0 { '+' } else { '-' }, h.z, rank,
                toks.iter().map(|(t, _)| format!("{:?}", eng.token_str(*t))).collect::<Vec<_>>().join(" "));
        }
    }
    Ok(())
}
