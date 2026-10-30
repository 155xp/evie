//! evie — a neuron microscope for Qwen3.5: read every MLP neuron through the logit / J / R lenses,
//! steer them live while chatting, and let agents do the same over MCP.

mod analyst;
mod mcp;
mod state;
mod ui;
mod worker;

use clap::{Parser, Subcommand};
use state::{Job, Shared};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Model to load (e.g. Qwen3.5-4B). Defaults to the largest installed one.
    #[arg(long, global = true)]
    model: Option<String>,
    /// Port for the MCP server.
    #[arg(long, default_value_t = 8765, global = true)]
    port: u16,
    /// Don't start the MCP server.
    #[arg(long)]
    no_mcp: bool,
    /// Compute threads (default: half the logical CPUs, at most 8).
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Max tokens per reply.
    #[arg(long, default_value_t = 320)]
    max_tokens: usize,
    /// Also generate the unsteered reply next to each steered one.
    #[arg(long)]
    compare: bool,
    /// UI scale on top of the display's own (e.g. 1.5 for crisp screenshots).
    #[arg(long)]
    zoom: Option<f32>,
    /// Window size.
    #[arg(long, default_value_t = 1680.0)]
    width: f32,
    #[arg(long, default_value_t = 1020.0)]
    height: f32,
    /// Save a screenshot here once the scripted steps below are done.
    #[arg(long)]
    shot: Option<PathBuf>,
    /// Scripted: send this chat message first.
    #[arg(long)]
    shot_prompt: Option<String>,
    /// Scripted: load this steering preset first.
    #[arg(long)]
    shot_preset: Option<String>,
    /// Scripted: select neuron "L,N".
    #[arg(long)]
    shot_select: Option<String>,
    /// Scripted: run this neuron search.
    #[arg(long)]
    shot_search: Option<String>,
    /// Scripted: steer neurons, e.g. "22:1093:+0.4,13:7956:-1".
    #[arg(long)]
    shot_steer: Option<String>,
    /// Scripted: open this inspector tab (neuron, workspace, agent).
    #[arg(long)]
    shot_tab: Option<String>,
    /// Seconds to wait before the screenshot.
    #[arg(long, default_value_t = 2.0)]
    shot_delay: f32,
    /// Quit after the screenshot.
    #[arg(long)]
    shot_exit: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download a model and its lenses (Qwen3.5-0.8B or Qwen3.5-4B).
    Download { id: String },
    /// Run headless: load the model and serve MCP only.
    Serve,
    /// Bridge a stdio MCP client to a running evie.
    McpStdio {
        #[arg(long, default_value = "http://127.0.0.1:8765/mcp")]
        url: String,
    },
    /// List known models and what is installed.
    Models,
}

fn default_model() -> Option<String> {
    let installed = evie_core::registry::installed();
    installed.iter().rev().find(|m| m.id == "Qwen3.5-4B").or(installed.last()).map(|m| m.id.to_string())
}

/// "L:N:dose,L:N:dose" → add-mode interventions.
fn parse_steer(s: &str) -> Vec<evie_core::Intervention> {
    s.split(',')
        .filter_map(|part| {
            let mut it = part.split(':').map(str::trim);
            let l = it.next()?.trim_start_matches(['L', 'l']).parse().ok()?;
            let n = it.next()?.trim_start_matches(['N', 'n']).parse().ok()?;
            let d = it.next().unwrap_or("1").parse().ok()?;
            Some(evie_core::Intervention::add(l, n, d))
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn,eframe=warn,egui_wgpu=warn")).init();
    let cli = Cli::parse();
    let threads = cli.threads.unwrap_or_else(|| (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) / 2).clamp(1, 8));
    rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("evie-compute-{i}")).build_global()?;

    match &cli.cmd {
        Some(Cmd::Download { id }) => {
            let last = std::cell::Cell::new(0u64);
            evie_core::registry::download(id, &|p| {
                if p.done / 50_000_000 != last.get() / 50_000_000 || p.done == p.total {
                    eprintln!("{} {:>7.1} / {:.1} MB", p.file, p.done as f64 / 1e6, p.total as f64 / 1e6);
                }
                last.set(p.done);
            })?;
            println!("installed {id} into {}", evie_core::registry::model_dir(id).display());
            return Ok(());
        }
        Some(Cmd::McpStdio { url }) => return mcp::stdio_bridge(url),
        Some(Cmd::Models) => {
            for m in evie_core::registry::MODELS {
                let have = evie_core::registry::is_installed(m.id);
                println!("{} {:<14} {:>5.1} GB  {}", if have { "●" } else { "○" }, m.id, m.approx_gb, m.about);
            }
            println!("data dir: {}", evie_core::registry::data_dir().display());
            return Ok(());
        }
        _ => {}
    }

    let (jtx, jrx) = crossbeam_channel::unbounded();
    let (atx, arx) = crossbeam_channel::unbounded();
    let shared = Arc::new(Shared::new(jtx, atx));
    {
        let mut c = shared.chat.lock();
        c.settings.max_new = cli.max_tokens;
        c.settings.compare = cli.compare;
    }
    worker::spawn(shared.clone(), jrx);
    analyst::spawn(shared.clone(), arx);
    if let Some(id) = cli.model.clone().or_else(default_model) {
        shared.send(Job::Load(id));
    }
    if !cli.no_mcp {
        match mcp::serve(shared.clone(), cli.port) {
            Ok(url) => log::info!("MCP server at {url}  (claude mcp add --transport http evie {url})"),
            Err(e) => log::warn!("{e:#}"),
        }
    }
    log::info!("compute threads: {threads}");

    if matches!(cli.cmd, Some(Cmd::Serve)) {
        loop {
            std::thread::park();
        }
    }

    let script = ui::Script {
        shot: cli.shot.clone(),
        prompt: cli.shot_prompt.clone(),
        preset: cli.shot_preset.clone(),
        select: cli.shot_select.as_deref().and_then(|s| {
            let mut it = s.split([',', ' ', ':']).filter_map(|x| x.trim_start_matches(['L', 'N', 'l', 'n']).parse().ok());
            Some((it.next()?, it.next()?))
        }),
        search: cli.shot_search.clone(),
        steer: cli.shot_steer.as_deref().map(parse_steer).unwrap_or_default(),
        tab: cli.shot_tab.clone(),
        delay: cli.shot_delay,
        exit: cli.shot_exit,
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([cli.width, cli.height]).with_min_inner_size([960.0, 600.0]).with_title("evie").with_app_id("evie"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    let zoom = cli.zoom;
    eframe::run_native(
        "evie",
        options,
        Box::new(move |cc| {
            if let Some(z) = zoom {
                cc.egui_ctx.set_zoom_factor(z);
            }
            Ok(Box::new(ui::App::new(cc, shared, script)))
        }),
    )
        .map_err(|e| anyhow::anyhow!("{e}"))
}
