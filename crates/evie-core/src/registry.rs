//! Where models, lenses and caches live, which models evie knows about, and downloading them.

use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub struct LensSpec {
    pub name: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub about: &'static str,
}

pub struct ModelSpec {
    pub id: &'static str,
    pub repo: &'static str,
    pub approx_gb: f32,
    pub about: &'static str,
    pub lenses: &'static [LensSpec],
}

pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "Qwen3.5-0.8B",
        repo: "Qwen/Qwen3.5-0.8B",
        approx_gb: 1.8,
        about: "24 layers × 3,584 neurons — fast, good for exploring",
        lenses: &[
            LensSpec {
                name: "J",
                file: "jlens.pt",
                url: "https://huggingface.co/neuronpedia/jacobian-lens/resolve/main/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
                about: "Jacobian lens (Neuronpedia fit, WikiText)",
            },
            LensSpec { name: "J++", file: "jpp.pt", url: "https://huggingface.co/neuronpedia/jacobian-lens/resolve/main/qwen3.5-0.8b/jpp/Salesforce-wikitext/Qwen3.5-0.8B_jpp_lens.pt", about: "J++ lens (Jacobian filtering, Ayonrinde & Lindsey)" },
        ],
    },
    ModelSpec {
        id: "Qwen3.5-4B",
        repo: "Qwen/Qwen3.5-4B",
        approx_gb: 9.3,
        about: "32 layers × 9,216 neurons — the model the neuron-steering community uses",
        lenses: &[
            LensSpec { name: "J", file: "jlens.pt", url: "https://huggingface.co/camilablank/workspace-lenses/resolve/main/qwen3.5-4b/j-lens/lens.pt", about: "Jacobian lens (camilablank, target L30)" },
            LensSpec { name: "R", file: "rlens.pt", url: "https://huggingface.co/camilablank/workspace-lenses/resolve/main/qwen3.5-4b/r-lens/lens.pt", about: "RelP lens (LRP-modified backward, cleaner early layers)" },
        ],
    },
];

pub fn spec(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id.eq_ignore_ascii_case(id))
}

/// `$EVIE_HOME`, else the platform data dir (`~/.local/share/evie` on Linux).
pub fn data_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("EVIE_HOME") {
        return PathBuf::from(p);
    }
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("evie")
}

pub fn model_dir(id: &str) -> PathBuf {
    data_dir().join("models").join(id)
}

pub fn lens_dir(id: &str) -> PathBuf {
    data_dir().join("lenses").join(id.to_lowercase())
}

pub fn cache_dir(id: &str) -> PathBuf {
    data_dir().join("cache").join(id.to_lowercase())
}

pub fn is_installed(id: &str) -> bool {
    let d = model_dir(id);
    d.join("config.json").exists()
        && d.join("tokenizer.json").exists()
        && std::fs::read_dir(&d).map(|r| r.flatten().any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))).unwrap_or(false)
        && !d.join(".partial").exists()
}

pub fn installed() -> Vec<&'static ModelSpec> {
    MODELS.iter().filter(|m| is_installed(m.id)).collect()
}

/// Lens files present for a model: (name, path).
pub fn installed_lenses(id: &str) -> Vec<(&'static str, PathBuf)> {
    let Some(s) = spec(id) else { return Vec::new() };
    s.lenses.iter().map(|l| (l.name, lens_dir(id).join(l.file))).filter(|(_, p)| p.exists()).collect()
}

#[derive(Debug, Clone)]
pub struct Progress {
    pub file: String,
    pub done: u64,
    pub total: u64,
}

fn fetch(url: &str, dest: &Path, on: &dyn Fn(Progress)) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    let part = dest.with_extension("part");
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let mut req = ureq::get(url);
    if have > 0 {
        req = req.header("Range", &format!("bytes={have}-"));
    }
    let resp = req.call().with_context(|| format!("GET {url}"))?;
    let resumed = resp.status() == 206;
    let len: u64 = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).unwrap_or(0);
    let start = if resumed { have } else { 0 };
    let total = start + len;
    let mut out = std::fs::OpenOptions::new().create(true).append(resumed).write(true).truncate(!resumed).open(&part)?;
    let mut body = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = start;
    let name = dest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    loop {
        let n = body.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        on(Progress { file: name.clone(), done, total });
    }
    out.flush()?;
    if total > 0 && done < total {
        bail!("download of {name} ended early ({done}/{total} bytes)");
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}

/// Downloads a model's text weights, tokenizer and its lenses (resumable).
pub fn download(id: &str, on: &dyn Fn(Progress)) -> Result<()> {
    let s = spec(id).with_context(|| format!("unknown model {id}"))?;
    let dir = model_dir(s.id);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(".partial"), b"")?;
    let base = format!("https://huggingface.co/{}/resolve/main", s.repo);
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"] {
        fetch(&format!("{base}/{f}"), &dir.join(f), on)?;
    }
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json"))?)?;
    let mut shards: Vec<String> =
        index["weight_map"].as_object().context("index.json has no weight_map")?.values().filter_map(|v| v.as_str().map(str::to_string)).collect();
    shards.sort();
    shards.dedup();
    for sh in shards {
        fetch(&format!("{base}/{sh}"), &dir.join(&sh), on)?;
    }
    std::fs::remove_file(dir.join(".partial")).ok();
    let ldir = lens_dir(s.id);
    std::fs::create_dir_all(&ldir)?;
    for l in s.lenses {
        fetch(l.url, &ldir.join(l.file), on)?;
    }
    Ok(())
}
