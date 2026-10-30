//! Just enough of Python's pickle VM to read `torch.save` zip archives of tensors (dicts, lists,
//! ints, floats, strings and `_rebuild_tensor_v2` storages). Nothing is executed: unknown callables
//! become opaque values.

use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global(String, String),
    Storage { dtype: String, key: String },
    Tensor(TensorRef),
    Opaque,
}

#[derive(Debug, Clone)]
pub struct TensorRef {
    pub dtype: String,
    pub key: String,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

impl Val {
    pub fn get(&self, key: &str) -> Option<&Val> {
        match self {
            Val::Dict(items) => items.iter().find(|(k, _)| matches!(k, Val::Str(s) if s == key)).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Val::Int(i) => Some(*i),
            Val::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }
}

enum Item {
    Mark,
    V(Val),
}

fn pop(st: &mut Vec<Item>) -> Result<Val> {
    match st.pop() {
        Some(Item::V(v)) => Ok(v),
        _ => bail!("pickle: stack underflow"),
    }
}

fn pop_mark(st: &mut Vec<Item>) -> Result<Vec<Val>> {
    let mut out = Vec::new();
    loop {
        match st.pop() {
            Some(Item::Mark) => break,
            Some(Item::V(v)) => out.push(v),
            None => bail!("pickle: missing MARK"),
        }
    }
    out.reverse();
    Ok(out)
}

fn reduce(f: Val, args: Val) -> Val {
    let args = match args {
        Val::Tuple(a) => a,
        _ => return Val::Opaque,
    };
    match f {
        Val::Global(m, n) if m == "torch._utils" && n == "_rebuild_tensor_v2" => {
            let ints = |v: Option<&Val>| -> Vec<usize> {
                match v {
                    Some(Val::Tuple(t)) => t.iter().filter_map(|x| x.as_int()).map(|x| x as usize).collect(),
                    _ => Vec::new(),
                }
            };
            match args.first() {
                Some(Val::Storage { dtype, key }) => Val::Tensor(TensorRef {
                    dtype: dtype.clone(),
                    key: key.clone(),
                    offset: args.get(1).and_then(|v| v.as_int()).unwrap_or(0) as usize,
                    shape: ints(args.get(2)),
                    stride: ints(args.get(3)),
                }),
                _ => Val::Opaque,
            }
        }
        Val::Global(m, n) if m == "collections" && n == "OrderedDict" => Val::Dict(Vec::new()),
        _ => Val::Opaque,
    }
}

pub fn parse(data: &[u8]) -> Result<Val> {
    let mut st: Vec<Item> = Vec::new();
    let mut memo: HashMap<u32, Val> = HashMap::new();
    let mut i = 0usize;
    let take = |i: &mut usize, n: usize| -> Result<&[u8]> {
        let s = data.get(*i..*i + n).ok_or_else(|| anyhow!("pickle: truncated"))?;
        *i += n;
        Ok(s)
    };
    loop {
        let op = *data.get(i).ok_or_else(|| anyhow!("pickle: no STOP"))?;
        i += 1;
        match op {
            0x80 => i += 1,             // PROTO
            0x95 => i += 8,             // FRAME
            b'.' => return pop(&mut st), // STOP
            b'(' => st.push(Item::Mark),
            b'}' => st.push(Item::V(Val::Dict(Vec::new()))),
            b']' => st.push(Item::V(Val::List(Vec::new()))),
            b')' => st.push(Item::V(Val::Tuple(Vec::new()))),
            b'N' => st.push(Item::V(Val::None)),
            0x88 => st.push(Item::V(Val::Bool(true))),
            0x89 => st.push(Item::V(Val::Bool(false))),
            b'K' => {
                let v = take(&mut i, 1)?[0];
                st.push(Item::V(Val::Int(v as i64)))
            }
            b'M' => {
                let b = take(&mut i, 2)?;
                st.push(Item::V(Val::Int(u16::from_le_bytes([b[0], b[1]]) as i64)))
            }
            b'J' => {
                let b = take(&mut i, 4)?;
                st.push(Item::V(Val::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)))
            }
            0x8a => {
                // LONG1
                let n = take(&mut i, 1)?[0] as usize;
                let b = take(&mut i, n)?;
                let mut v: i64 = 0;
                for (k, &x) in b.iter().enumerate().take(8) {
                    v |= (x as i64) << (8 * k);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                st.push(Item::V(Val::Int(v)))
            }
            b'G' => {
                let b = take(&mut i, 8)?;
                st.push(Item::V(Val::Float(f64::from_be_bytes(b.try_into().unwrap()))))
            }
            b'X' | 0x8c | 0x8d => {
                let n = match op {
                    b'X' => u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as usize,
                    0x8c => take(&mut i, 1)?[0] as usize,
                    _ => u64::from_le_bytes(take(&mut i, 8)?.try_into().unwrap()) as usize,
                };
                let s = String::from_utf8_lossy(take(&mut i, n)?).into_owned();
                st.push(Item::V(Val::Str(s)))
            }
            b'B' | b'C' => {
                let n = if op == b'B' { u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as usize } else { take(&mut i, 1)?[0] as usize };
                st.push(Item::V(Val::Bytes(take(&mut i, n)?.to_vec())))
            }
            b'q' => {
                let k = take(&mut i, 1)?[0] as u32;
                if let Some(Item::V(v)) = st.last() {
                    memo.insert(k, v.clone());
                }
            }
            b'r' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap());
                if let Some(Item::V(v)) = st.last() {
                    memo.insert(k, v.clone());
                }
            }
            0x94 => {
                // MEMOIZE
                let k = memo.len() as u32;
                if let Some(Item::V(v)) = st.last() {
                    memo.insert(k, v.clone());
                }
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as u32;
                st.push(Item::V(memo.get(&k).cloned().context("pickle: bad memo")?))
            }
            b'j' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap());
                st.push(Item::V(memo.get(&k).cloned().context("pickle: bad memo")?))
            }
            b'c' => {
                let rest = &data[i..];
                let nl1 = rest.iter().position(|&b| b == b'\n').context("GLOBAL")?;
                let nl2 = rest[nl1 + 1..].iter().position(|&b| b == b'\n').context("GLOBAL")?;
                let m = String::from_utf8_lossy(&rest[..nl1]).into_owned();
                let n = String::from_utf8_lossy(&rest[nl1 + 1..nl1 + 1 + nl2]).into_owned();
                i += nl1 + nl2 + 2;
                st.push(Item::V(Val::Global(m, n)))
            }
            0x93 => {
                // STACK_GLOBAL
                let n = pop(&mut st)?;
                let m = pop(&mut st)?;
                st.push(Item::V(Val::Global(m.as_str().unwrap_or("").into(), n.as_str().unwrap_or("").into())))
            }
            b't' => {
                let items = pop_mark(&mut st)?;
                st.push(Item::V(Val::Tuple(items)))
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(pop(&mut st)?);
                }
                items.reverse();
                st.push(Item::V(Val::Tuple(items)))
            }
            b'l' => {
                let items = pop_mark(&mut st)?;
                st.push(Item::V(Val::List(items)))
            }
            b'd' => {
                let items = pop_mark(&mut st)?;
                let pairs = items.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].clone(), c[1].clone())).collect();
                st.push(Item::V(Val::Dict(pairs)))
            }
            b'a' => {
                let v = pop(&mut st)?;
                if let Some(Item::V(Val::List(l))) = st.last_mut() {
                    l.push(v);
                }
            }
            b'e' => {
                let items = pop_mark(&mut st)?;
                if let Some(Item::V(Val::List(l))) = st.last_mut() {
                    l.extend(items);
                }
            }
            b's' => {
                let v = pop(&mut st)?;
                let k = pop(&mut st)?;
                if let Some(Item::V(Val::Dict(d))) = st.last_mut() {
                    d.push((k, v));
                }
            }
            b'u' => {
                let items = pop_mark(&mut st)?;
                if let Some(Item::V(Val::Dict(d))) = st.last_mut() {
                    for c in items.chunks(2) {
                        if c.len() == 2 {
                            d.push((c[0].clone(), c[1].clone()));
                        }
                    }
                }
            }
            b'R' => {
                let args = pop(&mut st)?;
                let f = pop(&mut st)?;
                st.push(Item::V(reduce(f, args)))
            }
            0x81 => {
                // NEWOBJ
                let _args = pop(&mut st)?;
                let _cls = pop(&mut st)?;
                st.push(Item::V(Val::Opaque))
            }
            b'b' => {
                let _state = pop(&mut st)?; // BUILD: ignore object state
            }
            b'Q' => {
                let pid = pop(&mut st)?;
                let v = match pid {
                    Val::Tuple(t) if t.len() >= 3 => {
                        let dtype = match &t[1] {
                            Val::Global(_, n) => n.clone(),
                            other => format!("{other:?}"),
                        };
                        let key = t[2].as_str().map(str::to_string).unwrap_or_default();
                        Val::Storage { dtype, key }
                    }
                    _ => Val::Opaque,
                };
                st.push(Item::V(v))
            }
            b'0' => {
                st.pop();
            }
            other => bail!("pickle: unsupported opcode 0x{other:02x} at {}", i - 1),
        }
    }
}

/// A `torch.save` archive: the unpickled object plus raw storages.
pub struct TorchFile {
    pub root: Val,
    storages: HashMap<String, Vec<u8>>,
}

impl TorchFile {
    pub fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut z = zip::ZipArchive::new(std::io::BufReader::new(f)).with_context(|| format!("{} is not a torch zip", path.display()))?;
        let names: Vec<String> = z.file_names().filter_map(|n| n.ok().map(|n| n.to_string())).collect();
        let pkl = names.iter().find(|n| n.ends_with("data.pkl")).context("no data.pkl")?.clone();
        let prefix = pkl.trim_end_matches("data.pkl").to_string();
        let mut buf = Vec::new();
        z.by_name(&pkl)?.read_to_end(&mut buf)?;
        let root = parse(&buf)?;
        let mut storages = HashMap::new();
        for n in &names {
            if let Some(key) = n.strip_prefix(&format!("{prefix}data/")) {
                let mut b = Vec::new();
                z.by_name(n)?.read_to_end(&mut b)?;
                storages.insert(key.to_string(), b);
            }
        }
        Ok(Self { root, storages })
    }

    /// A contiguous 2-D f16/bf16 tensor's raw 16-bit elements.
    pub fn tensor_u16(&self, t: &TensorRef) -> Result<Vec<u16>> {
        let (rows, cols) = match t.shape.as_slice() {
            [r, c] => (*r, *c),
            s => bail!("expected 2-D tensor, got {s:?}"),
        };
        if t.stride != [cols, 1] {
            bail!("non-contiguous tensor (stride {:?})", t.stride);
        }
        let raw = self.storages.get(&t.key).with_context(|| format!("missing storage {}", t.key))?;
        let start = t.offset * 2;
        let bytes = raw.get(start..start + rows * cols * 2).context("storage too small")?;
        Ok(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    }
}
