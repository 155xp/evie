//! Model Context Protocol server (streamable HTTP, JSON responses) so agents like Claude Code can
//! search, read and steer neurons on the live model while the GUI shows what they do.

use crate::state::{Message, Shared};
use evie_core::{Intervention, LensSel, Mode, Role};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PROTOCOL: &str = "2025-06-18";

pub fn serve(shared: Arc<Shared>, port: u16) -> anyhow::Result<String> {
    let server = tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("MCP server on port {port}: {e}"))?;
    let url = format!("http://127.0.0.1:{}/mcp", server.server_addr().to_ip().map(|a| a.port()).unwrap_or(port));
    *shared.mcp_url.lock() = Some(url.clone());
    std::thread::Builder::new().name("evie-mcp".into()).spawn(move || {
        for req in server.incoming_requests() {
            let shared = shared.clone();
            std::thread::spawn(move || handle(&shared, req));
        }
    })?;
    Ok(url)
}

fn respond(req: tiny_http::Request, status: u16, body: Option<String>, session: Option<&str>) {
    let mut resp = tiny_http::Response::from_string(body.unwrap_or_default()).with_status_code(status);
    resp.add_header(tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap());
    if let Some(s) = session {
        resp.add_header(tiny_http::Header::from_bytes("Mcp-Session-Id", s).unwrap());
    }
    let _ = req.respond(resp);
}

fn handle(shared: &Shared, mut req: tiny_http::Request) {
    // Only browsers send Origin; refuse cross-site pages (DNS-rebinding protection).
    let origin = req.headers().iter().find(|h| h.field.equiv("Origin")).map(|h| h.value.as_str().to_string());
    if let Some(o) = origin
        && !(o.starts_with("http://127.0.0.1") || o.starts_with("http://localhost") || o == "null")
    {
        return respond(req, 403, Some(r#"{"error":"forbidden origin"}"#.into()), None);
    }
    match req.method() {
        tiny_http::Method::Post => {}
        tiny_http::Method::Delete => return respond(req, 200, None, None),
        _ => return respond(req, 405, None, None),
    }
    let mut body = String::new();
    if req.as_reader().read_to_string(&mut body).is_err() {
        return respond(req, 400, None, None);
    }
    let Ok(msg) = serde_json::from_str::<Value>(&body) else {
        return respond(req, 400, Some(rpc_error(Value::Null, -32700, "parse error").to_string()), None);
    };
    let is_init = msg.get("method").and_then(Value::as_str) == Some("initialize");
    let out = match msg {
        Value::Array(batch) => {
            let rs: Vec<Value> = batch.iter().filter_map(|m| dispatch(shared, m)).collect();
            (!rs.is_empty()).then(|| Value::Array(rs))
        }
        m => dispatch(shared, &m),
    };
    let session = is_init.then(|| format!("evie-{:x}", Instant::now().elapsed().as_nanos() ^ std::process::id() as u128));
    match out {
        Some(v) => respond(req, 200, Some(v.to_string()), session.as_deref()),
        None => respond(req, 202, None, session.as_deref()),
    }
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// Handles one JSON-RPC message; notifications (no id) produce no response.
pub fn dispatch(shared: &Shared, msg: &Value) -> Option<Value> {
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(json!({}));
    let id = id?;
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": params.get("protocolVersion").and_then(Value::as_str).unwrap_or(PROTOCOL),
            "capabilities": {"tools": {"listChanged": false}},
            "serverInfo": {"name": "evie", "title": "evie neuron microscope", "version": env!("CARGO_PKG_VERSION")},
            "instructions": INSTRUCTIONS,
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tools()})),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            let who = "agent";
            shared.log(who, format!("{name} {}", short(&args)));
            Ok(match call(shared, &name, &args) {
                Ok(v) => json!({"content": [{"type": "text", "text": serde_json::to_string_pretty(&v).unwrap_or_default()}], "structuredContent": v}),
                Err(e) => {
                    shared.log("evie", format!("{name} failed: {e:#}"));
                    json!({"content": [{"type": "text", "text": format!("error: {e:#}")}], "isError": true})
                }
            })
        }
        other => Err(rpc_error(id.clone(), -32601, &format!("unknown method {other}"))),
    };
    Some(match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => e,
    })
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 140 { format!("{}…", &s[..s.char_indices().nth(140).map(|(i, _)| i).unwrap_or(s.len())]) } else { s }
}

const INSTRUCTIONS: &str = "evie runs a Qwen3.5 model locally with hooks on every MLP neuron (the input of down_proj, \
where silu(gate)*up lives). Neurons are addressed as layer L (0-based) and neuron N. A neuron's 'readout' through a lens is \
the list of tokens its write direction W_down[:, N] pushes the model toward: the logit lens decodes it directly, the J-lens \
(Jacobian lens, Gurnee et al. 2026) first transports it through the average Jacobian of the downstream layers, and the R-lens \
uses a relevance-propagation variant that is cleaner on early layers. Each neuron has a + pole (tokens it promotes when \
pushed up) and a − pole (tokens it promotes when pushed down). Steering uses the community dose convention: dose 1.0 adds \
median(‖residual leaving layer L‖)/‖W_down[:, N]‖ to the activation on every token, i.e. one typical residual norm along the \
neuron's write direction; doses around ±0.4–1.0 are typical, larger ones degrade coherence. Typical workflow: \
search_neurons for a concept → read_neuron to confirm → set_steering → chat (optionally compare=true to see the unsteered \
reply too) → iterate. The human is watching the same model in the evie GUI, so explain what you are doing.";

fn tools() -> Value {
    let int = |d: &str| json!({"type": "integer", "description": d});
    json!([
        {"name": "status", "description": "Model, layer/neuron counts, available lenses, current steering and chat state.",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "search_neurons", "description": "Rank every neuron by how strongly its lens readout says the given words (standardized logit, then exact ranks under every lens). Use several related words and translations, e.g. 'pirate, pirates, 海盗'.",
         "inputSchema": {"type": "object", "properties": {
            "words": {"type": "string", "description": "Comma-separated words"},
            "lens": {"type": "string", "description": "Lens to rank with: 'J' (default), 'R', 'J++' or 'logit'"},
            "limit": int("Max results (default 12)")}, "required": ["words"]}},
        {"name": "read_neuron", "description": "Read a neuron through every lens (top tokens of both poles), its dose unit, and its activation over the current conversation.",
         "inputSchema": {"type": "object", "properties": {"layer": int("0-based layer"), "neuron": int("neuron index"), "k": int("tokens per pole (default 12)")}, "required": ["layer", "neuron"]}},
        {"name": "top_active", "description": "Most strongly firing neurons (relative to each neuron's usual range) at a token of the current conversation (default: the latest token).",
         "inputSchema": {"type": "object", "properties": {"token": int("token index in the conversation trace"), "k": int("how many (default 20)")}}},
        {"name": "activations_for_text", "description": "Run arbitrary text through the model (no chat template, current steering ignored) and return the neurons that fire most on it, with the token where each peaks.",
         "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "k": int("how many (default 20)")}, "required": ["text"]}},
        {"name": "set_steering", "description": "Set the steering rack (the GUI updates live). Each intervention: {layer, neuron, dose, mode?: 'add'|'mul'|'set', label?}. For mode 'add' dose is in community units; for 'mul' it multiplies the activation (0 ablates); for 'set' it clamps it. replace=false merges into the current rack.",
         "inputSchema": {"type": "object", "properties": {
            "interventions": {"type": "array", "items": {"type": "object", "properties": {
                "layer": {"type": "integer"}, "neuron": {"type": "integer"}, "dose": {"type": "number"},
                "mode": {"type": "string", "enum": ["add", "mul", "set"]}, "label": {"type": "string"}},
                "required": ["layer", "neuron", "dose"]}},
            "replace": {"type": "boolean", "description": "Replace the whole rack (default true)"}}, "required": ["interventions"]}},
        {"name": "clear_steering", "description": "Remove all interventions.", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "list_presets", "description": "Community steering recipes for Qwen3.5-4B (evil qwen, catgirl, science personas, gated anti-gaming).", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "apply_preset", "description": "Load a community recipe into the steering rack.", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        {"name": "chat", "description": "Send a user message to the steered model and wait for the full reply (shown streaming in the GUI). compare=true also returns the unsteered reply from the same batch.",
         "inputSchema": {"type": "object", "properties": {
            "message": {"type": "string"},
            "new_conversation": {"type": "boolean", "description": "Start fresh instead of continuing (default true)"},
            "compare": {"type": "boolean"}, "max_tokens": int("default 200"),
            "temperature": {"type": "number", "description": "0 = greedy (default)"}}, "required": ["message"]}},
        {"name": "workspace", "description": "What the residual stream at a token is disposed to say at every layer, through every lens (J-space readout).",
         "inputSchema": {"type": "object", "properties": {"token": int("token index (default: latest)"), "k": int("tokens per layer (default 5)")}}},
        {"name": "focus_neuron", "description": "Zoom the GUI's atlas to a neuron and open it in the inspector (to show the human what you found).",
         "inputSchema": {"type": "object", "properties": {"layer": {"type": "integer"}, "neuron": {"type": "integer"}}, "required": ["layer", "neuron"]}}
    ])
}

fn arg_usize(a: &Value, k: &str) -> Option<usize> {
    a.get(k).and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)).or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))).map(|x| x as usize)
}

fn call(shared: &Shared, name: &str, a: &Value) -> anyhow::Result<Value> {
    let eng = shared.engine().ok_or_else(|| anyhow::anyhow!("no model loaded yet"))?;
    let check = |l: usize, n: usize| -> anyhow::Result<()> {
        anyhow::ensure!(l < eng.n_layers(), "layer {l} out of range (0..{})", eng.n_layers());
        anyhow::ensure!(n < eng.n_neurons(), "neuron {n} out of range (0..{})", eng.n_neurons());
        Ok(())
    };
    let tok_list = |v: &[(u32, f32)], k: usize| -> Vec<String> { v.iter().take(k).map(|(t, _)| eng.token_str(*t).to_string()).collect() };
    match name {
        "status" => {
            let rack = shared.rack.lock().clone();
            let c = shared.chat.lock();
            Ok(json!({
                "model": eng.id, "layers": eng.n_layers(), "neurons_per_layer": eng.n_neurons(),
                "lenses": eng.lens_list().iter().map(|l| l.label().to_string()).collect::<Vec<_>>(),
                "dose_unit": "dose 1.0 = median(‖resid leaving L‖) / ‖W_down[:, n]‖ added to the activation on every token",
                "steering": rack.items, "preset": rack.preset,
                "conversation_messages": c.messages.len(), "generating": c.generating,
                "trace_tokens": shared.trace.read().frames.len(),
            }))
        }
        "search_neurons" => {
            let words = match a.get("words") {
                Some(Value::Array(v)) => v.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(","),
                Some(v) => v.as_str().unwrap_or("").to_string(),
                None => String::new(),
            };
            anyhow::ensure!(!words.trim().is_empty(), "words is required");
            let lens = LensSel::parse(a.get("lens").and_then(Value::as_str).unwrap_or("J"));
            let lens = if eng.lens(&lens).is_some() || lens == LensSel::Logit { lens } else { LensSel::Logit };
            let limit = arg_usize(a, "limit").unwrap_or(12).clamp(1, 60);
            let (matched, results) = crate::analyst::search_now(&eng, &words, &lens, limit);
            {
                let mut s = shared.search.lock();
                s.query = words.clone();
                s.results = results.clone();
                s.matched = matched.clone();
                s.status = format!("{} neurons (agent search)", results.len());
                s.version += 1;
            }
            shared.repaint();
            Ok(json!({
                "query_tokens": matched, "ranked_with": lens.label(),
                "neurons": results.iter().map(|r| json!({
                    "layer": r.layer, "neuron": r.neuron, "pole": if r.sign > 0 { "+" } else { "-" },
                    "suggested_dose": if r.sign > 0 { 1.0 } else { -1.0 }, "z": (r.z * 10.0).round() / 10.0,
                    "lenses": r.lenses.iter().map(|(l, rank, top)| json!({"lens": l, "best_rank": rank, "top_tokens": top})).collect::<Vec<_>>(),
                })).collect::<Vec<_>>()
            }))
        }
        "read_neuron" => {
            let (l, n) = (arg_usize(a, "layer").unwrap_or(usize::MAX), arg_usize(a, "neuron").unwrap_or(usize::MAX));
            check(l, n)?;
            let k = arg_usize(a, "k").unwrap_or(12).clamp(1, 40);
            let mut lenses = Vec::new();
            for sel in eng.lens_list() {
                let r = &eng.readouts(&sel, &[(l, n)], k)[0];
                shared.reads.lock().insert((sel.clone(), l, n), r.clone());
                lenses.push(json!({"lens": sel.label(), "plus_pole": tok_list(&r.up, k), "minus_pole": tok_list(&r.down, k)}));
            }
            let trace = shared.trace.read();
            let nn = eng.n_neurons();
            let acts: Vec<f32> = trace.frames.iter().map(|f| crate::state::Trace::decode(f[l * nn + n])).collect();
            let peak = acts.iter().enumerate().max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()));
            Ok(json!({
                "neuron": format!("L{l} N{n}"), "lenses": lenses,
                "dose_unit": eng.dose_unit(l, n), "write_norm": eng.col_norms[l][n], "median_resid_norm": eng.units[l],
                "steered": shared.rack.lock().get(l, n),
                "conversation_activation": peak.map(|(t, v)| json!({
                    "tokens": acts.len(), "latest": acts.last(), "peak": v, "peak_token": t,
                    "peak_token_text": trace.tokens.get(t).map(|x| x.text.clone()),
                    "note": "relative to the neuron's usual range (1.0 ≈ its max on calibration text)"})),
            }))
        }
        "top_active" => {
            let trace = shared.trace.read();
            anyhow::ensure!(!trace.frames.is_empty(), "no conversation yet — use chat first");
            let t = arg_usize(a, "token").unwrap_or(trace.frames.len() - 1).min(trace.frames.len() - 1);
            let k = arg_usize(a, "k").unwrap_or(20).clamp(1, 100);
            let f = &trace.frames[t];
            let mut idx: Vec<usize> = (0..f.len()).collect();
            idx.select_nth_unstable_by(k.min(f.len() - 1), |&x, &y| f[y].unsigned_abs().cmp(&f[x].unsigned_abs()));
            idx.truncate(k);
            idx.sort_by(|&x, &y| f[y].unsigned_abs().cmp(&f[x].unsigned_abs()));
            let nn = eng.n_neurons();
            Ok(json!({
                "token": t, "token_text": trace.tokens.get(t).map(|x| x.text.clone()),
                "neurons": idx.iter().map(|&i| json!({"layer": i / nn, "neuron": i % nn, "activation": (crate::state::Trace::decode(f[i]) * 1000.0).round() / 1000.0})).collect::<Vec<_>>()
            }))
        }
        "activations_for_text" => {
            let text = a.get("text").and_then(Value::as_str).unwrap_or("");
            anyhow::ensure!(!text.trim().is_empty(), "text is required");
            let k = arg_usize(a, "k").unwrap_or(20).clamp(1, 100);
            let ids = eng.tok.encode(text)?;
            anyhow::ensure!(ids.len() <= 2048, "text too long ({} tokens, max 2048)", ids.len());
            struct Peak {
                n: usize,
                scale: Option<Arc<Vec<f32>>>,
                best: Vec<(f32, usize)>,
            }
            impl evie_core::Hooks for Peak {
                fn mlp(&mut self, layer: usize, _s: usize, _p: usize, acts: &mut [f32]) {
                    for (r, row) in acts.chunks(self.n).enumerate().skip(1) {
                        for (j, &x) in row.iter().enumerate() {
                            let i = layer * self.n + j;
                            let v = self.scale.as_ref().map(|s| x / s[i]).unwrap_or(x);
                            if v.abs() > self.best[i].0.abs() {
                                self.best[i] = (v, r);
                            }
                        }
                    }
                }
            }
            let mut hook = Peak { n: eng.n_neurons(), scale: eng.act_scale.read().clone(), best: vec![(0.0, 0); eng.n_layers() * eng.n_neurons()] };
            let mut cache = eng.model.new_cache();
            eng.model.forward(&mut [evie_core::Seq { cache: &mut cache, tokens: &ids }], &mut hook, false);
            let mut idx: Vec<usize> = (0..hook.best.len()).collect();
            idx.sort_by(|&x, &y| hook.best[y].0.abs().total_cmp(&hook.best[x].0.abs()));
            idx.truncate(k);
            let nn = eng.n_neurons();
            Ok(json!({"tokens": ids.len(), "neurons": idx.iter().map(|&i| {
                let (v, t) = hook.best[i];
                json!({"layer": i / nn, "neuron": i % nn, "peak": (v * 1000.0).round() / 1000.0, "peak_token": t, "peak_token_text": eng.token_str(ids[t])})
            }).collect::<Vec<_>>()}))
        }
        "set_steering" => {
            let list = a.get("interventions").and_then(Value::as_array).ok_or_else(|| anyhow::anyhow!("interventions must be an array"))?;
            let mut ivs = Vec::new();
            for it in list {
                let (l, n) = (arg_usize(it, "layer").unwrap_or(usize::MAX), arg_usize(it, "neuron").unwrap_or(usize::MAX));
                check(l, n)?;
                let mode = match it.get("mode").and_then(Value::as_str).unwrap_or("add") {
                    "mul" | "multiply" => Mode::Mul,
                    "set" | "clamp" => Mode::Set,
                    _ => Mode::Add,
                };
                let dose = it.get("dose").or_else(|| it.get("value")).and_then(Value::as_f64).ok_or_else(|| anyhow::anyhow!("dose is required"))? as f32;
                anyhow::ensure!(dose.is_finite() && dose.abs() <= 50.0, "dose {dose} is out of range");
                let mut iv = Intervention::add(l, n, dose);
                iv.mode = mode;
                iv.label = it.get("label").and_then(Value::as_str).unwrap_or("agent").to_string();
                ivs.push(iv);
            }
            let replace = a.get("replace").and_then(Value::as_bool).unwrap_or(true);
            let mut rack = shared.rack.lock();
            if replace {
                rack.set(ivs, None);
            } else {
                for iv in ivs {
                    rack.upsert(iv);
                }
            }
            let items = rack.items.clone();
            drop(rack);
            if let Some(first) = items.first() {
                *shared.focus.lock() = Some((first.layer, first.neuron));
            }
            shared.repaint();
            Ok(json!({"steering": items}))
        }
        "clear_steering" => {
            shared.rack.lock().set(Vec::new(), None);
            shared.repaint();
            Ok(json!({"steering": []}))
        }
        "list_presets" => {
            let ok = eng.n_layers() == 32 && eng.n_neurons() == 9216;
            Ok(json!({"available": ok, "note": if ok { "" } else { "presets are for Qwen3.5-4B" },
                "presets": evie_core::steering::presets_4b().into_iter().map(|(n, about, ivs)| json!({"name": n, "about": about, "interventions": ivs})).collect::<Vec<_>>()}))
        }
        "apply_preset" => {
            anyhow::ensure!(eng.n_layers() == 32 && eng.n_neurons() == 9216, "presets are for Qwen3.5-4B (loaded: {})", eng.id);
            let want = a.get("name").and_then(Value::as_str).unwrap_or("").to_lowercase();
            let (n, _, ivs) = evie_core::steering::presets_4b()
                .into_iter()
                .find(|(n, _, _)| n.to_lowercase() == want || n.to_lowercase().contains(&want))
                .ok_or_else(|| anyhow::anyhow!("no preset named {want:?}"))?;
            shared.rack.lock().set(ivs.clone(), Some(n.to_string()));
            shared.repaint();
            Ok(json!({"preset": n, "steering": ivs}))
        }
        "chat" => {
            let message = a.get("message").and_then(Value::as_str).unwrap_or("").to_string();
            anyhow::ensure!(!message.trim().is_empty(), "message is required");
            {
                let mut c = shared.chat.lock();
                anyhow::ensure!(!c.generating, "the model is busy generating; try again shortly");
                if a.get("new_conversation").and_then(Value::as_bool).unwrap_or(true) {
                    c.messages.clear();
                }
                if let Some(t) = a.get("max_tokens").and_then(Value::as_u64) {
                    c.settings.max_new = (t as usize).clamp(1, 2048);
                }
                if let Some(t) = a.get("temperature").and_then(Value::as_f64) {
                    c.settings.temperature = t.clamp(0.0, 2.0) as f32;
                }
                if let Some(cmp) = a.get("compare").and_then(Value::as_bool) {
                    c.settings.compare = cmp;
                }
            }
            let before = shared.chat.lock().finished;
            anyhow::ensure!(shared.send_user(&message, Some("agent")), "could not start generation");
            let start = Instant::now();
            loop {
                std::thread::sleep(Duration::from_millis(100));
                let c = shared.chat.lock();
                if c.finished != before && !c.generating {
                    let m: Option<&Message> = c.messages.iter().rev().find(|m| m.role == Role::Assistant);
                    return Ok(json!({
                        "reply": m.map(|m| m.text.clone()),
                        "unsteered_reply": m.and_then(|m| m.baseline.clone()),
                        "steering": m.map(|m| m.steering.clone()),
                        "seconds": (start.elapsed().as_secs_f32() * 10.0).round() / 10.0,
                    }));
                }
                drop(c);
                if start.elapsed() > Duration::from_secs(900) {
                    shared.request_stop();
                    anyhow::bail!("generation timed out");
                }
            }
        }
        "workspace" => {
            let n = shared.trace.read().frames.len();
            anyhow::ensure!(n > 0, "no conversation yet — use chat first");
            let t = arg_usize(a, "token").unwrap_or(n - 1).min(n - 1);
            let k = arg_usize(a, "k").unwrap_or(5).clamp(1, 20);
            let ws = crate::analyst::workspace_rows(shared, &eng, t, k).ok_or_else(|| anyhow::anyhow!("no residuals for token {t}"))?;
            Ok(json!({
                "token": t, "token_text": shared.trace.read().tokens.get(t).map(|x| x.text.clone()),
                "layers": (0..eng.n_layers()).map(|l| {
                    let mut o = serde_json::Map::new();
                    o.insert("layer".into(), json!(l));
                    for (lens, rows) in &ws.rows {
                        o.insert(lens.clone(), json!(tok_list(&rows[l], k)));
                    }
                    Value::Object(o)
                }).collect::<Vec<_>>()
            }))
        }
        "focus_neuron" => {
            let (l, n) = (arg_usize(a, "layer").unwrap_or(usize::MAX), arg_usize(a, "neuron").unwrap_or(usize::MAX));
            check(l, n)?;
            *shared.focus.lock() = Some((l, n));
            shared.repaint();
            Ok(json!({"focused": format!("L{l} N{n}")}))
        }
        other => anyhow::bail!("unknown tool {other}"),
    }
}

/// Bridges an MCP stdio client to a running evie's HTTP endpoint.
pub fn stdio_bridge(url: &str) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let resp = ureq::post(url).header("Content-Type", "application/json").header("Accept", "application/json, text/event-stream").send(line.as_bytes());
        match resp {
            Ok(r) if r.status() == 200 => {
                let body = r.into_body().read_to_string()?;
                writeln!(out, "{}", body.trim())?;
                out.flush()?;
            }
            Ok(_) => {}
            Err(e) => {
                let id = serde_json::from_str::<Value>(&line).ok().and_then(|v| v.get("id").cloned());
                if let Some(id) = id {
                    writeln!(out, "{}", rpc_error(id, -32000, &format!("evie is not reachable at {url}: {e}")))?;
                    out.flush()?;
                }
            }
        }
    }
    Ok(())
}
