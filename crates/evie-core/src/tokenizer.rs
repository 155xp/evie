//! Tokenizer + Qwen3.5 chat template (text-only subset of `chat_template.jinja`).

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMsg {
    pub role: Role,
    pub content: String,
}

impl ChatMsg {
    pub fn user(s: impl Into<String>) -> Self {
        Self { role: Role::User, content: s.into() }
    }
    pub fn assistant(s: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: s.into() }
    }
    pub fn system(s: impl Into<String>) -> Self {
        Self { role: Role::System, content: s.into() }
    }
}

pub struct Tok {
    inner: tokenizers::Tokenizer,
    /// Human-readable string for every token id (invalid UTF-8 fragments shown as `\xNN`).
    pub display: Vec<String>,
    pub im_start: u32,
    pub im_end: u32,
    pub endoftext: u32,
}

impl Tok {
    pub fn load(path: &Path, vocab: usize) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        let id = |s: &str| inner.token_to_id(s).ok_or_else(|| anyhow!("tokenizer has no {s}"));
        let (im_start, im_end, endoftext) = (id("<|im_start|>")?, id("<|im_end|>")?, id("<|endoftext|>")?);
        let byte_of = byte_decoder();
        let display = (0..vocab as u32)
            .into_par_iter()
            .map(|i| {
                let Some(piece) = inner.id_to_token(i) else { return String::new() };
                if inner.get_added_vocabulary().is_special_token(&piece) || piece.starts_with("<|") {
                    return piece;
                }
                let bytes: Option<Vec<u8>> = piece.chars().map(|c| byte_of.get(&c).copied()).collect();
                match bytes {
                    Some(b) => match String::from_utf8(b.clone()) {
                        Ok(s) => s,
                        Err(_) => b.iter().map(|x| format!("\\x{x:02X}")).collect(),
                    },
                    None => piece,
                }
            })
            .collect();
        Ok(Self { inner, display, im_start, im_end, endoftext })
    }

    /// Encodes text; special-token strings like `<|im_start|>` in the text are recognised.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, false).unwrap_or_default()
    }

    pub fn token_to_id(&self, s: &str) -> Option<u32> {
        self.inner.token_to_id(s)
    }

    pub fn vocab_size(&self) -> usize {
        self.display.len()
    }

    /// Renders messages the way Qwen3.5's template does with `enable_thinking=False`.
    pub fn chat(&self, msgs: &[ChatMsg], add_generation_prompt: bool) -> String {
        let last_user = msgs.iter().rposition(|m| m.role == Role::User);
        let mut s = String::new();
        for (i, m) in msgs.iter().enumerate() {
            let content = m.content.trim();
            match m.role {
                Role::System => {
                    if i == 0 {
                        s += &format!("<|im_start|>system\n{content}<|im_end|>\n");
                    }
                }
                Role::User => s += &format!("<|im_start|>user\n{content}<|im_end|>\n"),
                Role::Assistant => {
                    let body = strip_think(content);
                    if last_user.is_some_and(|u| i > u) {
                        s += &format!("<|im_start|>assistant\n<think>\n\n</think>\n\n{body}<|im_end|>\n");
                    } else {
                        s += &format!("<|im_start|>assistant\n{body}<|im_end|>\n");
                    }
                }
            }
        }
        if add_generation_prompt {
            s += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        }
        s
    }

    /// Token ids whose text, ignoring surrounding whitespace and case, equals `word`.
    /// Falls back to the first token of " word" when the word is not a single token.
    pub fn word_tokens(&self, word: &str) -> Vec<u32> {
        let w = word.trim().to_lowercase();
        if w.is_empty() {
            return Vec::new();
        }
        let mut ids: Vec<u32> = self
            .display
            .par_iter()
            .enumerate()
            .filter(|(_, d)| d.trim().to_lowercase() == w)
            .map(|(i, _)| i as u32)
            .collect();
        if ids.is_empty() {
            for variant in [format!(" {}", word.trim()), word.trim().to_string()] {
                if let Ok(t) = self.encode(&variant)
                    && let Some(&first) = t.first()
                {
                    ids.push(first);
                }
            }
            ids.sort_unstable();
            ids.dedup();
        }
        ids
    }
}

fn strip_think(content: &str) -> &str {
    match content.rfind("</think>") {
        Some(i) => content[i + "</think>".len()..].trim_start_matches('\n'),
        None => content,
    }
}

/// Inverse of GPT-2's `bytes_to_unicode` table used by byte-level BPE.
fn byte_decoder() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter().zip(cs).map(|(b, c)| (char::from_u32(c).unwrap(), b as u8)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_is_stripped_from_history() {
        assert_eq!(strip_think("<think>\nhm\n</think>\n\nHello"), "Hello");
        assert_eq!(strip_think("Hello"), "Hello");
    }
}
