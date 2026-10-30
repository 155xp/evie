//! GPU backend: the same Qwen3.5 forward pass as [`crate::model`], as wgpu compute shaders
//! (Vulkan / Metal / DX12). Weights are uploaded as Q8 (int8 + one f32 scale per 64 inputs) so the
//! 4B model fits in 8 GB; activations stay f32. One forward pass is one command buffer: every layer's
//! kernels, then a single readback of logits, every neuron's activation (companded i8) and the
//! residual stream (f16). Steering is a per-sequence `[layer][neuron] → (mode, value)` map applied
//! inside the SwiGLU kernel, exactly where the CPU hooks apply it.

use crate::config::LayerKind;
use crate::kernels::Mat16;
use crate::model::{Mixer, Model};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

const BLOCK: usize = 64;
const CONST_STRIDE: u64 = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum B {
    U,
    R,
    W,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct C {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
    f: u32,
    fa: f32,
    fb: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RowInfo {
    seq: u32,
    pos: u32,
    token: u32,
    seg: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SegInfo {
    seq: u32,
    start: u32,
    count: u32,
    pos0: u32,
}

struct Q8 {
    w: wgpu::Buffer,
    s: wgpu::Buffer,
    rows: u32,
    cols: u32,
}

#[derive(Clone, Copy)]
enum Dims {
    PerRow(u32),
    PerSeg(u32),
    Rows,
    SegRows,
    Mat { out: u32, seg: bool },
    Scores,
    RowHeads(u32),
}

struct Step {
    kernel: K,
    bind: wgpu::BindGroup,
    dims: Dims,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum K {
    Embed,
    RmsNorm,
    MatSmall,
    MatTiled,
    SwiGlu,
    Residual,
    Gather,
    Conv,
    L2Norm,
    DeltaNet,
    AttnPrep,
    AttnScores,
    AttnOut,
}

impl K {
    fn entry(self) -> &'static str {
        match self {
            K::Embed => "embed",
            K::RmsNorm => "rmsnorm",
            K::MatSmall => "matmul_small",
            K::MatTiled => "matmul_tiled",
            K::SwiGlu => "swiglu",
            K::Residual => "residual",
            K::Gather => "gather_last",
            K::Conv => "conv",
            K::L2Norm => "l2norm",
            K::DeltaNet => "deltanet",
            K::AttnPrep => "attn_prep",
            K::AttnScores => "attn_scores",
            K::AttnOut => "attn_out",
        }
    }
    fn layout(self) -> &'static [B] {
        use B::*;
        match self {
            K::Embed | K::RmsNorm => &[U, R, R, R, R, R, W],
            K::MatSmall | K::MatTiled => &[U, R, R, R, R, R, R, W],
            K::SwiGlu => &[U, R, R, R, W, R, R, R, W, W],
            K::Residual | K::Conv => &[U, R, R, R, W, R, W],
            K::Gather => &[U, R, R, R, R, W],
            K::L2Norm => &[U, R, R, R, W],
            K::DeltaNet => &[U, R, R, R, R, R, R, R, W, W],
            K::AttnPrep => &[U, R, R, R, R, R, R, R, W, W, W],
            K::AttnScores => &[U, R, R, R, R, R, W],
            K::AttnOut => &[U, R, R, R, R, R, W, W],
        }
    }
    const ALL: [K; 13] = [
        K::Embed,
        K::RmsNorm,
        K::MatSmall,
        K::MatTiled,
        K::SwiGlu,
        K::Residual,
        K::Gather,
        K::Conv,
        K::L2Norm,
        K::DeltaNet,
        K::AttnPrep,
        K::AttnScores,
        K::AttnOut,
    ];
}

/// One sequence in a GPU forward pass.
pub struct GpuSeq<'a> {
    /// Cache slot (0..slots).
    pub slot: usize,
    /// Position of the first new token.
    pub pos0: usize,
    pub tokens: &'a [u32],
}

pub struct GpuOut {
    /// Next-token logits for the last new token of each sequence.
    pub logits: Vec<Vec<f32>>,
    /// Per row: `layers × neurons` companded activations (same encoding as the CPU capture).
    pub frames: Vec<Vec<i8>>,
    /// Per row: `layers × hidden` residual stream leaving each layer.
    pub resid: Vec<Vec<half::f16>>,
}

pub struct GpuModel {
    dev: wgpu::Device,
    queue: wgpu::Queue,
    pub adapter_name: String,
    pipes: Vec<(K, wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    steps: Vec<Step>,
    rows_buf: wgpu::Buffer,
    info_buf: wgpu::Buffer,
    segs_buf: wgpu::Buffer,
    ops: wgpu::Buffer,
    scale: wgpu::Buffer,
    maxbuf: wgpu::Buffer,
    logits: wgpu::Buffer,
    cap_frames: wgpu::Buffer,
    cap_resid: wgpu::Buffer,
    conv_state: wgpu::Buffer,
    rec_state: wgpu::Buffer,
    kc: wgpu::Buffer,
    vc: wgpu::Buffer,
    consts: wgpu::Buffer,
    consts_host: Vec<C>,
    swiglu_consts: Vec<usize>,
    pub max_rows: usize,
    pub max_ctx: usize,
    pub slots: usize,
    layers: usize,
    neurons: usize,
    hidden: usize,
    vocab: usize,
    n_heads: u32,
    pub weight_bytes: u64,
}

fn quantize(m: &Mat16) -> (Vec<u32>, Vec<f32>) {
    assert!(m.cols % BLOCK == 0, "Q8 needs widths divisible by {BLOCK}");
    let (rows, cols) = (m.rows, m.cols);
    let mut words = vec![0u32; rows * cols / 4];
    let mut scales = vec![0f32; rows * cols / BLOCK];
    words.par_chunks_mut(cols / 4).zip(scales.par_chunks_mut(cols / BLOCK)).enumerate().for_each(|(r, (w, s))| {
        let mut row = vec![0f32; cols];
        m.row_to_f32(r, &mut row);
        for (b, blk) in row.chunks(BLOCK).enumerate() {
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            s[b] = sc;
            for (q, quad) in blk.chunks(4).enumerate() {
                let mut word = 0u32;
                for (j, &v) in quad.iter().enumerate() {
                    let qv = (v / sc).round().clamp(-127.0, 127.0) as i32;
                    word |= ((qv as u32) & 0xff) << (8 * j);
                }
                w[b * (BLOCK / 4) + q] = word;
            }
        }
    });
    (words, scales)
}

impl GpuModel {
    /// Creates a device on the best GPU and uploads `model`. Returns an error if no suitable GPU
    /// exists or the weights don't fit.
    pub fn new(model: &Model, max_ctx: usize, status: &dyn Fn(&str)) -> Result<Self> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter")?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            bail!("only a software rasterizer ({}) is available", info.name);
        }
        let al = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage.min(16);
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        anyhow::ensure!(limits.max_storage_buffers_per_shader_stage >= 10, "GPU allows too few storage buffers per stage");
        let (dev, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("evie compute"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .context("creating GPU device")?;
        let oom = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let invalid = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut me = Self::build(dev.clone(), queue, info.name.clone(), model, max_ctx, &limits, status)?;
        if let Some(e) = pollster::block_on(invalid.pop()) {
            bail!("GPU validation error: {e}");
        }
        if let Some(e) = pollster::block_on(oom.pop()) {
            bail!("GPU out of memory: {e}");
        }
        me.adapter_name = info.name;
        Ok(me)
    }

    #[allow(clippy::too_many_lines)]
    fn build(dev: wgpu::Device, queue: wgpu::Queue, name: String, model: &Model, max_ctx: usize, limits: &wgpu::Limits, status: &dyn Fn(&str)) -> Result<Self> {
        let cfg = &model.cfg;
        anyhow::ensure!(cfg.lin_k_dim == 128 && cfg.lin_v_dim == 128, "GPU DeltaNet kernel assumes 128-wide heads");
        anyhow::ensure!(cfg.head_dim <= 256 && cfg.rotary_dim == 64, "GPU attention kernel assumes head_dim ≤ 256 and 64 rotary dims");
        anyhow::ensure!(cfg.conv_kernel <= 9, "conv kernel too wide");
        let (d, n, layers, vocab) = (cfg.hidden, cfg.intermediate, cfg.n_layers, model.lm_head.rows);
        let max_rows = 128usize;
        let slots = 2usize;
        let n_lin = cfg.layer_kinds.iter().filter(|k| **k == LayerKind::Linear).count();
        let n_full = layers - n_lin;
        let cdim = cfg.conv_dim();
        let vd = cfg.lin_v_heads * cfg.lin_v_dim;
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);

        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("evie kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels.wgsl").into()),
        });
        // Kernels with the same binding signature share one layout object, so a bind group made for
        // `matmul_small` also works with `matmul_tiled`.
        let mut layouts: Vec<(&'static [B], wgpu::BindGroupLayout)> = Vec::new();
        let pipes: Vec<_> = K::ALL
            .iter()
            .map(|&k| {
                let entries: Vec<wgpu::BindGroupLayoutEntry> = k
                    .layout()
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupLayoutEntry {
                        binding: i as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: match b {
                            B::U => wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                            B::R => wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                            B::W => wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                        },
                        count: None,
                    })
                    .collect();
                let bgl = match layouts.iter().find(|(sig, _)| *sig == k.layout()) {
                    Some((_, l)) => l.clone(),
                    None => {
                        let l = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(k.entry()), entries: &entries });
                        layouts.push((k.layout(), l.clone()));
                        l
                    }
                };
                let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(k.entry()), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
                let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(k.entry()),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(k.entry()),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (k, p, bgl)
            })
            .collect();

        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let mut weight_bytes = 0u64;
        let buf = |label: &str, size: usize| dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(4) as u64, usage: st, mapped_at_creation: false });
        let init_f32 = |label: &str, data: &[f32]| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytemuck::cast_slice(data), usage: st });
        let mut upload = |label: &str, m: &[&Mat16]| -> Q8 {
            let mut words = Vec::new();
            let mut scales = Vec::new();
            let mut rows = 0;
            for mm in m {
                let (w, s) = quantize(mm);
                words.extend(w);
                scales.extend(s);
                rows += mm.rows;
            }
            weight_bytes += (words.len() * 4 + scales.len() * 4) as u64;
            let w = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytemuck::cast_slice(&words), usage: st });
            let s = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytemuck::cast_slice(&scales), usage: st });
            Q8 { w, s, rows: rows as u32, cols: m[0].cols as u32 }
        };

        // Activations.
        let x = buf("x", max_rows * d * 4);
        let h = buf("h", max_rows * d * 4);
        let mixed = buf("mixed", max_rows * d * 4);
        let g = buf("gate", max_rows * n * 4);
        let u = buf("up", max_rows * n * 4);
        let qkv = buf("qkv", max_rows * cdim * 4);
        let z = buf("z", max_rows * vd * 4);
        let ba = buf("ba", max_rows * 2 * cfg.lin_v_heads * 4);
        let core = buf("core", max_rows * vd * 4);
        let qraw = buf("qraw", max_rows * nh * 2 * hd * 4);
        let kk = buf("k", max_rows * nkv * hd * 4);
        let vv = buf("v", max_rows * nkv * hd * 4);
        let qout = buf("qout", max_rows * nh * hd * 4);
        let attn_o = buf("attn_o", max_rows * nh * hd * 4);
        let scores = buf("scores", max_rows * nh * max_ctx * 4);
        let last = buf("last", slots * d * 4);
        let last_n = buf("last_n", slots * d * 4);
        let logits = buf("logits", slots * vocab * 4);
        let cap_frames = buf("frames", max_rows * layers * n);
        let cap_resid = buf("resid", max_rows * layers * d * 2);
        let ops = buf("ops", slots * layers * n * 8);
        let scale = init_f32("scale", &vec![1.0f32; layers * n]);
        let maxbuf = buf("max", layers * n * 4);
        let conv_state = buf("conv state", slots * n_lin.max(1) * (cfg.conv_kernel - 1) * cdim * 4);
        let rec_state = buf("rec state", slots * n_lin.max(1) * cfg.lin_v_heads * 128 * 128 * 4);
        let kc = buf("k cache", slots * n_full.max(1) * max_ctx * nkv * hd * 4);
        let vc = buf("v cache", slots * n_full.max(1) * max_ctx * nkv * hd * 4);
        let rows_buf = buf("rows", 16);
        let info_buf = buf("info", max_rows * 16);
        let segs_buf = buf("segs", slots * 16);

        status("uploading embeddings (Q8)");
        let embed = upload("embed", &[&model.embed]);
        let final_norm = init_f32("final norm", &model.final_norm);

        let mut consts_host: Vec<C> = Vec::new();
        let mut steps_spec: Vec<(K, C, Vec<wgpu::Buffer>, Dims)> = Vec::new();
        let mut push = |k: K, c: C, bufs: Vec<wgpu::Buffer>, dims: Dims| steps_spec.push((k, c, bufs, dims));
        let eps = cfg.eps;
        let mat = |push: &mut dyn FnMut(K, C, Vec<wgpu::Buffer>, Dims), q: &Q8, xin: &wgpu::Buffer, y: &wgpu::Buffer, seg: bool| {
            let wg = q.rows.div_ceil(4);
            push(
                K::MatSmall,
                C { a: q.rows, b: q.cols, c: seg as u32, d: wg.min(65535), ..Default::default() },
                vec![q.w.clone(), q.s.clone(), xin.clone(), y.clone()],
                Dims::Mat { out: q.rows, seg },
            );
        };

        push(K::Embed, C { a: d as u32, ..Default::default() }, vec![embed.w.clone(), embed.s.clone(), x.clone()], Dims::PerRow(((d / 4) as u32).div_ceil(64)));
        let (mut li, mut fi) = (0u32, 0u32);
        for (l, layer) in model.layers.iter().enumerate() {
            status(&format!("uploading layer {l}/{layers} (Q8)"));
            let ln1 = init_f32("ln1", &layer.ln1);
            let ln2 = init_f32("ln2", &layer.ln2);
            push(K::RmsNorm, C { a: d as u32, fa: eps, ..Default::default() }, vec![x.clone(), ln1, h.clone()], Dims::Rows);
            match &layer.mixer {
                Mixer::Linear(dn) => {
                    let wqkv = upload("qkv", &[&dn.qkv]);
                    let wz = upload("z", &[&dn.z]);
                    let wba = upload("ba", &[&dn.b, &dn.a]);
                    let wout = upload("out", &[&dn.out]);
                    let convw = init_f32("conv", &dn.conv);
                    let mut par = dn.neg_a.clone();
                    par.extend(&dn.dt_bias);
                    par.extend(&dn.norm);
                    let par = init_f32("dn params", &par);
                    mat(&mut push, &wqkv, &h, &qkv, false);
                    mat(&mut push, &wz, &h, &z, false);
                    mat(&mut push, &wba, &h, &ba, false);
                    push(K::Conv, C { a: cdim as u32, b: cfg.conv_kernel as u32, c: li, d: n_lin as u32, ..Default::default() }, vec![qkv.clone(), convw, conv_state.clone()], Dims::PerSeg((cdim as u32).div_ceil(64)));
                    push(
                        K::L2Norm,
                        C { a: cdim as u32, b: cfg.lin_k_heads as u32, c: cfg.lin_k_dim as u32, fa: 1.0 / (cfg.lin_k_dim as f32).sqrt(), ..Default::default() },
                        vec![qkv.clone()],
                        Dims::PerRow(2 * cfg.lin_k_heads as u32),
                    );
                    push(
                        K::DeltaNet,
                        C {
                            a: cdim as u32,
                            b: cfg.lin_k_heads as u32,
                            c: cfg.lin_v_heads as u32,
                            d: li,
                            e: n_lin as u32,
                            f: (cfg.lin_k_heads * cfg.lin_k_dim) as u32,
                            fa: eps,
                            fb: 0.0,
                        },
                        vec![qkv.clone(), z.clone(), ba.clone(), par, rec_state.clone(), core.clone()],
                        Dims::PerSeg(cfg.lin_v_heads as u32),
                    );
                    mat(&mut push, &wout, &core, &mixed, false);
                    li += 1;
                }
                Mixer::Full(at) => {
                    let wq = upload("q", &[&at.q]);
                    let wk = upload("k", &[&at.k]);
                    let wv = upload("v", &[&at.v]);
                    let wo = upload("o", &[&at.o]);
                    let mut nrm = at.q_norm.clone();
                    nrm.extend(&at.k_norm);
                    let nrm = init_f32("qk norm", &nrm);
                    mat(&mut push, &wq, &h, &qraw, false);
                    mat(&mut push, &wk, &h, &kk, false);
                    mat(&mut push, &wv, &h, &vv, false);
                    let ac = C { a: nh as u32, b: nkv as u32, c: hd as u32, d: fi, e: n_full as u32, f: max_ctx as u32, fa: eps, fb: cfg.rope_theta as f32 };
                    push(K::AttnPrep, ac, vec![qraw.clone(), kk.clone(), vv.clone(), nrm, qout.clone(), kc.clone(), vc.clone()], Dims::PerRow((nh + nkv) as u32));
                    push(K::AttnScores, ac, vec![qout.clone(), kc.clone(), scores.clone()], Dims::Scores);
                    push(K::AttnOut, ac, vec![qraw.clone(), vc.clone(), scores.clone(), attn_o.clone()], Dims::RowHeads(nh as u32));
                    mat(&mut push, &wo, &attn_o, &mixed, false);
                    fi += 1;
                }
            }
            push(K::Residual, C { a: d as u32, b: l as u32, c: layers as u32, d: 0, ..Default::default() }, vec![x.clone(), mixed.clone(), cap_resid.clone()], Dims::PerRow(((d / 2) as u32).div_ceil(64)));
            push(K::RmsNorm, C { a: d as u32, fa: eps, ..Default::default() }, vec![x.clone(), ln2, h.clone()], Dims::Rows);
            let wg = upload("gate", &[&layer.mlp.gate]);
            let wu = upload("up", &[&layer.mlp.up]);
            let wd = upload("down", &[&layer.mlp.down]);
            mat(&mut push, &wg, &h, &g, false);
            mat(&mut push, &wu, &h, &u, false);
            push(
                K::SwiGlu,
                C { a: n as u32, b: l as u32, c: layers as u32, d: 1, e: 0, ..Default::default() },
                vec![g.clone(), u.clone(), ops.clone(), scale.clone(), cap_frames.clone(), maxbuf.clone()],
                Dims::PerRow(((n / 4) as u32).div_ceil(64)),
            );
            mat(&mut push, &wd, &g, &mixed, false);
            push(K::Residual, C { a: d as u32, b: l as u32, c: layers as u32, d: 1, ..Default::default() }, vec![x.clone(), mixed.clone(), cap_resid.clone()], Dims::PerRow(((d / 2) as u32).div_ceil(64)));
            queue.submit([]);
        }
        push(K::Gather, C { a: d as u32, ..Default::default() }, vec![x.clone(), last.clone()], Dims::PerSeg((d as u32).div_ceil(64)));
        push(K::RmsNorm, C { a: d as u32, b: 1, fa: eps, ..Default::default() }, vec![last.clone(), final_norm, last_n.clone()], Dims::SegRows);
        mat(&mut push, &embed, &last_n, &logits, true);

        // Constants live in one uniform buffer, one 256-byte slot per step.
        let consts = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("consts"),
            size: (steps_spec.len() as u64 * CONST_STRIDE).max(CONST_STRIDE),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut swiglu_consts = Vec::new();
        let mut steps = Vec::new();
        for (i, (k, c, bufs, dims)) in steps_spec.into_iter().enumerate() {
            queue.write_buffer(&consts, i as u64 * CONST_STRIDE, bytemuck::bytes_of(&c));
            consts_host.push(c);
            if k == K::SwiGlu {
                swiglu_consts.push(i);
            }
            let bgl = &pipes.iter().find(|p| p.0 == k).unwrap().2;
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &consts, offset: i as u64 * CONST_STRIDE, size: wgpu::BufferSize::new(32) }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: rows_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: info_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: segs_buf.as_entire_binding() },
            ];
            for (j, b) in bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry { binding: 4 + j as u32, resource: b.as_entire_binding() });
            }
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(k.entry()), layout: bgl, entries: &entries });
            steps.push(Step { kernel: k, bind, dims });
        }
        let _ = limits;
        status("GPU ready");
        Ok(Self {
            dev,
            queue,
            adapter_name: name,
            pipes,
            steps,
            rows_buf,
            info_buf,
            segs_buf,
            ops,
            scale,
            maxbuf,
            logits,
            cap_frames,
            cap_resid,
            conv_state,
            rec_state,
            kc,
            vc,
            consts,
            consts_host,
            swiglu_consts,
            max_rows,
            max_ctx,
            slots,
            layers,
            neurons: n,
            hidden: d,
            vocab,
            n_heads: nh as u32,
            weight_bytes,
        })
    }

    /// Per-neuron activation scales used to normalise captured frames.
    pub fn set_scale(&self, scale: &[f32]) {
        if scale.len() == self.layers * self.neurons {
            self.queue.write_buffer(&self.scale, 0, bytemuck::cast_slice(scale));
        }
    }

    /// Replaces a slot's steering map: `(layer, neuron, mode 1=add 2=mul 3=set, value)`.
    pub fn set_ops(&self, slot: usize, ops: &[(usize, usize, u32, f32)]) {
        let mut dense = vec![[0f32; 2]; self.layers * self.neurons];
        for &(l, n, m, v) in ops {
            if l < self.layers && n < self.neurons {
                dense[l * self.neurons + n] = [m as f32, v];
            }
        }
        let off = (slot * self.layers * self.neurons * 8) as u64;
        self.queue.write_buffer(&self.ops, off, bytemuck::cast_slice(&dense));
    }

    /// Zeroes a slot's recurrent and conv state (KV entries are simply overwritten by position).
    pub fn reset_slot(&self, slot: usize) {
        let mut enc = self.dev.create_command_encoder(&Default::default());
        let lin_conv = self.conv_state.size() / self.slots as u64;
        let lin_rec = self.rec_state.size() / self.slots as u64;
        enc.clear_buffer(&self.conv_state, slot as u64 * lin_conv, Some(lin_conv));
        enc.clear_buffer(&self.rec_state, slot as u64 * lin_rec, Some(lin_rec));
        self.queue.submit([enc.finish()]);
    }

    /// Track the max |activation| of every neuron (positions > 0) across subsequent passes.
    pub fn track_max(&mut self, on: bool) {
        for &i in &self.swiglu_consts {
            self.consts_host[i].e = on as u32;
            self.queue.write_buffer(&self.consts, i as u64 * CONST_STRIDE, bytemuck::bytes_of(&self.consts_host[i]));
        }
        if on {
            let mut enc = self.dev.create_command_encoder(&Default::default());
            enc.clear_buffer(&self.maxbuf, 0, None);
            self.queue.submit([enc.finish()]);
        }
    }

    pub fn read_max(&self) -> Vec<f32> {
        let bytes = self.read_buffer(&self.maxbuf, (self.layers * self.neurons * 4) as u64);
        bytemuck::cast_slice::<u8, u32>(&bytes).iter().map(|&b| f32::from_bits(b)).collect()
    }

    fn read_buffer(&self, src: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.dev.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.dev.poll(wgpu::PollType::wait_indefinitely());
        let out = slice.get_mapped_range().map(|v| v.to_vec()).unwrap_or_default();
        staging.unmap();
        out
    }

    /// Runs the new tokens of up to `slots` sequences (total ≤ `max_rows` rows per call; longer
    /// prompts are split into consecutive calls by the caller).
    pub fn forward(&self, seqs: &[GpuSeq], want_logits: bool, capture: bool) -> Result<GpuOut> {
        let total: usize = seqs.iter().map(|s| s.tokens.len()).sum();
        anyhow::ensure!(total > 0 && total <= self.max_rows, "GPU forward needs 1..={} rows, got {total}", self.max_rows);
        anyhow::ensure!(seqs.len() <= self.slots, "at most {} sequences per pass", self.slots);
        let mut info = Vec::with_capacity(total);
        let mut segs = Vec::with_capacity(seqs.len());
        let mut max_pos = 0;
        for (si, s) in seqs.iter().enumerate() {
            anyhow::ensure!(s.slot < self.slots, "bad slot");
            anyhow::ensure!(s.pos0 + s.tokens.len() <= self.max_ctx, "context longer than {} tokens", self.max_ctx);
            segs.push(SegInfo { seq: s.slot as u32, start: info.len() as u32, count: s.tokens.len() as u32, pos0: s.pos0 as u32 });
            for (i, &t) in s.tokens.iter().enumerate() {
                info.push(RowInfo { seq: s.slot as u32, pos: (s.pos0 + i) as u32, token: t, seg: si as u32 });
                max_pos = max_pos.max(s.pos0 + i);
            }
        }
        let (n, n_seg) = (total as u32, seqs.len() as u32);
        self.queue.write_buffer(&self.rows_buf, 0, bytemuck::cast_slice(&[n, n_seg, 0, 0]));
        self.queue.write_buffer(&self.info_buf, 0, bytemuck::cast_slice(&info));
        self.queue.write_buffer(&self.segs_buf, 0, bytemuck::cast_slice(&segs));

        let mut enc = self.dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("forward"), timestamp_writes: None });
            for st in &self.steps {
                let (kernel, grid) = match st.dims {
                    Dims::PerRow(x) => (st.kernel, [x, n, 1]),
                    Dims::PerSeg(x) => (st.kernel, [x, n_seg, 1]),
                    Dims::Rows => (st.kernel, [n, 1, 1]),
                    Dims::SegRows => (st.kernel, [n_seg, 1, 1]),
                    Dims::Mat { out, seg } => {
                        let rows = if seg { n_seg } else { n };
                        if rows <= 8 {
                            let wg = out.div_ceil(4);
                            (K::MatSmall, [wg.min(65535), wg.div_ceil(65535), 1])
                        } else {
                            (K::MatTiled, [out.div_ceil(64), rows.div_ceil(64), 1])
                        }
                    }
                    Dims::Scores => (st.kernel, [((max_pos + 1) as u32).div_ceil(64), n * self.n_heads, 1]),
                    Dims::RowHeads(h) => (st.kernel, [n * h, 1, 1]),
                };
                if !want_logits && matches!(st.kernel, K::Gather) {
                    break;
                }
                let pipe = &self.pipes.iter().find(|p| p.0 == kernel).unwrap().1;
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &st.bind, &[]);
                pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
            }
        }
        // One readback for everything this pass produced.
        let logit_bytes = if want_logits { (n_seg as usize * self.vocab * 4) as u64 } else { 0 };
        let frame_bytes = if capture { (total * self.layers * self.neurons) as u64 } else { 0 };
        let resid_bytes = if capture { (total * self.layers * self.hidden * 2) as u64 } else { 0 };
        let size = logit_bytes + frame_bytes + resid_bytes;
        let staging = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if logit_bytes > 0 {
            enc.copy_buffer_to_buffer(&self.logits, 0, &staging, 0, logit_bytes);
        }
        if capture {
            enc.copy_buffer_to_buffer(&self.cap_frames, 0, &staging, logit_bytes, frame_bytes);
            enc.copy_buffer_to_buffer(&self.cap_resid, 0, &staging, logit_bytes + frame_bytes, resid_bytes);
        }
        self.queue.submit([enc.finish()]);
        let mut out = GpuOut { logits: Vec::new(), frames: Vec::new(), resid: Vec::new() };
        if size == 0 {
            let _ = self.dev.poll(wgpu::PollType::wait_indefinitely());
            return Ok(out);
        }
        let slice = staging.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.dev.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow::anyhow!("GPU poll: {e:?}"))?;
        {
            let data = slice.get_mapped_range().map_err(|e| anyhow::anyhow!("mapping readback: {e:?}"))?;
            let lg: &[f32] = bytemuck::cast_slice(&data[..logit_bytes as usize]);
            out.logits = lg.chunks(self.vocab).map(<[f32]>::to_vec).collect();
            if capture {
                let fr = &data[logit_bytes as usize..(logit_bytes + frame_bytes) as usize];
                out.frames = fr.chunks(self.layers * self.neurons).map(|c| c.iter().map(|&b| b as i8).collect()).collect();
                let rs: &[u16] = bytemuck::cast_slice(&data[(logit_bytes + frame_bytes) as usize..size as usize]);
                out.resid = rs.chunks(self.layers * self.hidden).map(|c| c.iter().map(|&b| half::f16::from_bits(b)).collect()).collect();
            }
        }
        staging.unmap();
        Ok(out)
    }
}
