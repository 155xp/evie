//! Zero-copy access to (sharded) safetensors checkpoints via mmap.

use crate::kernels::{Kind16, Mat16};
use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
    F32,
}

#[derive(Debug, Clone)]
struct Entry {
    file: usize,
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

pub struct SafeTensors {
    files: Vec<Arc<Mmap>>,
    entries: HashMap<String, Entry>,
}

impl SafeTensors {
    /// Opens every `*.safetensors` file in `dir`.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        let mut files = Vec::new();
        let mut entries = HashMap::new();
        for (fi, path) in paths.iter().enumerate() {
            let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: the checkpoint is treated as read-only; nothing else in-process writes to it.
            let mmap = unsafe { Mmap::map(&file)? };
            if mmap.len() < 8 {
                bail!("{} is too small to be safetensors", path.display());
            }
            let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&mmap[8..8 + n]).with_context(|| format!("header of {}", path.display()))?;
            let base = 8 + n;
            for (name, v) in header {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = match v["dtype"].as_str() {
                    Some("BF16") => DType::BF16,
                    Some("F16") => DType::F16,
                    Some("F32") => DType::F32,
                    _ => continue, // not something we read (e.g. int buffers)
                };
                let shape = v["shape"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).unwrap_or_default();
                let off = v["data_offsets"].as_array().context("data_offsets")?;
                let start = base + off[0].as_u64().context("offset")? as usize;
                let end = base + off[1].as_u64().context("offset")? as usize;
                entries.insert(name, Entry { file: fi, dtype, shape, start, end });
            }
            files.push(Arc::new(mmap));
        }
        Ok(Self { files, entries })
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries.get(name).with_context(|| format!("missing tensor {name}"))
    }

    /// A 2-D bf16/f16 tensor as a matrix view straight into the mmap (copied only if misaligned).
    pub fn mat16(&self, name: &str) -> Result<Mat16> {
        let e = self.entry(name)?;
        let kind = match e.dtype {
            DType::BF16 => Kind16::Bf16,
            DType::F16 => Kind16::F16,
            DType::F32 => bail!("{name} is f32; expected a 16-bit matrix"),
        };
        let (rows, cols) = match e.shape.as_slice() {
            [r, c] => (*r, *c),
            [r, 1, c] => (*r, *c), // conv1d weights [C, 1, K]
            s => bail!("{name}: expected 2-D, got {s:?}"),
        };
        let bytes = &self.files[e.file][e.start..e.end];
        if bytes.len() != rows * cols * 2 {
            bail!("{name}: size mismatch");
        }
        if (bytes.as_ptr() as usize) % 2 == 0 {
            Ok(Mat16::from_mapped(rows, cols, kind, bytes.as_ptr() as *const u16, self.files[e.file].clone()))
        } else {
            let v: Vec<u16> = bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            Ok(Mat16::from_vec(rows, cols, kind, v))
        }
    }

    /// Any float tensor, flattened and converted to f32.
    pub fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        let bytes = &self.files[e.file][e.start..e.end];
        Ok(match e.dtype {
            DType::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            DType::BF16 => bytes.chunks_exact(2).map(|b| crate::kernels::bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            DType::F16 => bytes.chunks_exact(2).map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()).collect(),
        })
    }

    /// Touches every page of the given tensors so the first forward pass doesn't stall on page faults.
    pub fn prefetch(&self, names: &[String]) {
        use rayon::prelude::*;
        names.par_iter().for_each(|n| {
            if let Some(e) = self.entries.get(n) {
                let bytes = &self.files[e.file][e.start..e.end];
                let mut acc = 0u8;
                for chunk in bytes.chunks(4096) {
                    acc ^= chunk[0];
                }
                std::hint::black_box(acc);
            }
        });
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }
}
