//! GPU-rendered neuron atlas: one cell per MLP neuron, one tile per layer, drawn by a fragment
//! shader from a single R8 activation texture (cost scales with pixels, not neurons).

use eframe::egui_wgpu::{self, CallbackTrait, wgpu};
use egui::{Pos2, Rect, Sense, Ui, Vec2};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    pub layers: usize,
    pub neurons: usize,
    pub tile_w: usize,
    pub tile_h: usize,
    pub tiles_x: usize,
    pub tiles_y: usize,
    pub gap_x: f32,
    pub gap_y: f32,
}

impl Geometry {
    pub fn new(layers: usize, neurons: usize) -> Self {
        // Most-square exact factorisation (9216 = 96 × 96, 3584 = 64 × 56), else a padded square.
        let root = (neurons as f64).sqrt();
        let (mut tile_w, mut tile_h) = (root.ceil() as usize, (neurons as f64 / root.ceil()).ceil() as usize);
        for h in (1..=root.floor() as usize).rev() {
            if neurons % h == 0 && neurons / h <= 2 * h {
                tile_w = neurons / h;
                tile_h = h;
                break;
            }
        }
        // Pick a divisor of the layer count giving a wide (≈2.2:1) atlas.
        let ideal = (layers as f64 * 2.2 * tile_h as f64 / tile_w as f64).sqrt();
        let tiles_x = (1..=layers)
            .filter(|d| layers % d == 0)
            .min_by(|a, b| ((*a as f64) - ideal).abs().total_cmp(&((*b as f64) - ideal).abs()))
            .unwrap_or(layers);
        let tiles_y = layers.div_ceil(tiles_x);
        Self { layers, neurons, tile_w, tile_h, tiles_x, tiles_y, gap_x: (tile_w as f32 * 0.09).round(), gap_y: (tile_h as f32 * 0.24).round() }
    }

    pub fn pitch(&self) -> Vec2 {
        Vec2::new(self.tile_w as f32 + self.gap_x, self.tile_h as f32 + self.gap_y)
    }

    /// Atlas size in cells.
    pub fn size(&self) -> Vec2 {
        let p = self.pitch();
        Vec2::new(self.tiles_x as f32 * p.x - self.gap_x, self.tiles_y as f32 * p.y - self.gap_y)
    }

    pub fn tile_origin(&self, layer: usize) -> Vec2 {
        let p = self.pitch();
        Vec2::new((layer % self.tiles_x) as f32 * p.x, (layer / self.tiles_x) as f32 * p.y)
    }

    pub fn cell_center(&self, layer: usize, neuron: usize) -> Vec2 {
        self.tile_origin(layer) + Vec2::new((neuron % self.tile_w) as f32 + 0.5, (neuron / self.tile_w) as f32 + 0.5)
    }

    pub fn locate(&self, p: Vec2) -> Option<(usize, usize)> {
        if p.x < 0.0 || p.y < 0.0 {
            return None;
        }
        let pitch = self.pitch();
        let (tx, ty) = ((p.x / pitch.x).floor() as usize, (p.y / pitch.y).floor() as usize);
        let (lx, ly) = (p.x - tx as f32 * pitch.x, p.y - ty as f32 * pitch.y);
        if tx >= self.tiles_x || lx >= self.tile_w as f32 || ly >= self.tile_h as f32 {
            return None;
        }
        let layer = ty * self.tiles_x + tx;
        let neuron = ly as usize * self.tile_w + lx as usize;
        (layer < self.layers && neuron < self.neurons).then_some((layer, neuron))
    }

    pub fn tex_size(&self) -> (u32, u32) {
        ((self.tiles_x * self.tile_w) as u32, (self.tiles_y * self.tile_h) as u32)
    }

    /// Byte index of (layer, neuron) in the texture image.
    pub fn texel_index(&self, layer: usize, neuron: usize) -> usize {
        let x = (layer % self.tiles_x) * self.tile_w + neuron % self.tile_w;
        let y = (layer / self.tiles_x) * self.tile_h + neuron / self.tile_w;
        y * self.tiles_x * self.tile_w + x
    }

    /// Scatters a layer-major `layers × neurons` buffer into texture layout.
    pub fn to_texture(&self, src: &[i8], dst: &mut Vec<u8>) {
        let (w, h) = self.tex_size();
        dst.clear();
        dst.resize((w * h) as usize, 0);
        for layer in 0..self.layers {
            let row = &src[layer * self.neurons..(layer + 1) * self.neurons];
            for (n, &v) in row.iter().enumerate() {
                dst[self.texel_index(layer, n)] = v as u8;
            }
        }
    }
}

/// Per-neuron marker bits in the marks texture.
pub mod mark {
    pub const STEER_UP: u8 = 1;
    pub const STEER_DOWN: u8 = 2;
    pub const HIT: u8 = 4;
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    origin: [f32; 4],
    view: [f32; 4],
    grid: [f32; 4],
    counts: [f32; 4],
    sel: [f32; 4],
}

struct Gpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    tex: Option<(wgpu::Texture, wgpu::Texture, wgpu::BindGroup, (u32, u32))>,
    uploaded: (u64, u64),
    srgb: bool,
}

pub fn install(rs: &egui_wgpu::RenderState) {
    let device = &rs.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("atlas"),
        source: wgpu::ShaderSource::Wgsl(include_str!("atlas.wgsl").into()),
    });
    let tex_entry = |binding, sample_type| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("atlas"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            tex_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
            tex_entry(2, wgpu::TextureSampleType::Uint),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("atlas"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("atlas"),
        layout: Some(&pl),
        vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(rs.target_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atlas uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    rs.renderer.write().callback_resources.insert(Gpu {
        pipeline,
        layout,
        uniforms,
        tex: None,
        uploaded: (u64::MAX, u64::MAX),
        srgb: rs.target_format.is_srgb(),
    });
}

/// Texture images for the current atlas contents (already in texture layout).
#[derive(Clone)]
pub struct Images {
    pub values: Arc<Vec<u8>>,
    pub values_gen: u64,
    pub marks: Arc<Vec<u8>>,
    pub marks_gen: u64,
}

struct Paint {
    geo: Geometry,
    images: Images,
    uniforms: Uniforms,
}

impl CallbackTrait for Paint {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _sd: &egui_wgpu::ScreenDescriptor,
        _enc: &mut wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(gpu) = res.get_mut::<Gpu>() else { return Vec::new() };
        let size = self.geo.tex_size();
        if gpu.tex.as_ref().map(|t| t.3) != Some(size) {
            let make = |label, format| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            };
            let vals = make("atlas values", wgpu::TextureFormat::R8Snorm);
            let marks = make("atlas marks", wgpu::TextureFormat::R8Uint);
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("atlas"),
                layout: &gpu.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gpu.uniforms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&vals.create_view(&Default::default())) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&marks.create_view(&Default::default())) },
                ],
            });
            gpu.tex = Some((vals, marks, bind, size));
            gpu.uploaded = (u64::MAX, u64::MAX);
        }
        let (vals, marks, _, _) = gpu.tex.as_ref().unwrap();
        let n = (size.0 * size.1) as usize;
        let write = |tex: &wgpu::Texture, data: &[u8]| {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size.0), rows_per_image: Some(size.1) },
                wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            );
        };
        if gpu.uploaded.0 != self.images.values_gen && self.images.values.len() == n {
            write(vals, &self.images.values);
            gpu.uploaded.0 = self.images.values_gen;
        }
        if gpu.uploaded.1 != self.images.marks_gen && self.images.marks.len() == n {
            write(marks, &self.images.marks);
            gpu.uploaded.1 = self.images.marks_gen;
        }
        let mut u = self.uniforms;
        u.origin[2] = if gpu.srgb { 1.0 } else { 0.0 };
        queue.write_buffer(&gpu.uniforms, 0, bytemuck::bytes_of(&u));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let Some(gpu) = res.get::<Gpu>() else { return };
        let Some((_, _, bind, _)) = gpu.tex.as_ref() else { return };
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Pan/zoom state and interaction for the atlas widget.
pub struct AtlasView {
    /// Atlas cell at the widget's top-left corner.
    pub pan: Vec2,
    /// Points per cell.
    pub zoom: f32,
    /// Geometry and widget size of the last automatic fit.
    fitted_for: Option<(Geometry, Vec2)>,
    /// Once the user zooms or pans, resizes stop re-fitting the view.
    user_moved: bool,
    pub hover: Option<(usize, usize)>,
    start: std::time::Instant,
}

impl Default for AtlasView {
    fn default() -> Self {
        Self { pan: Vec2::ZERO, zoom: 1.0, fitted_for: None, user_moved: false, hover: None, start: std::time::Instant::now() }
    }
}

pub struct AtlasResponse {
    pub response: egui::Response,
    pub clicked: Option<(usize, usize)>,
    pub rect: Rect,
}

impl AtlasView {
    pub fn fit(&mut self, geo: &Geometry, rect: Rect) {
        let size = geo.size();
        let margin = 18.0;
        let avail = rect.size() - Vec2::splat(2.0 * margin);
        self.zoom = (avail.x / size.x).min(avail.y / (size.y + geo.gap_y * 0.6)).max(0.05);
        let used = size * self.zoom;
        self.pan = -((rect.size() - used) * 0.5 + Vec2::new(0.0, geo.gap_y * 0.3 * self.zoom)) / self.zoom;
        self.fitted_for = Some((*geo, rect.size()));
        self.user_moved = false;
    }

    pub fn to_screen(&self, rect: Rect, cell: Vec2) -> Pos2 {
        rect.min + (cell - self.pan) * self.zoom
    }

    pub fn to_cell(&self, rect: Rect, p: Pos2) -> Vec2 {
        self.pan + (p - rect.min) / self.zoom
    }

    /// Zooms so the given neuron is centred and comfortably large.
    pub fn focus(&mut self, rect: Rect, geo: &Geometry, layer: usize, neuron: usize) {
        self.zoom = self.zoom.max(7.0);
        self.user_moved = true;
        let c = geo.cell_center(layer, neuron);
        self.pan = c - rect.size() * 0.5 / self.zoom;
    }

    pub fn show(&mut self, ui: &mut Ui, geo: &Geometry, images: Images, selected: Option<(usize, usize)>, glow: f32) -> AtlasResponse {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let stale = match self.fitted_for {
            None => true,
            Some((g, s)) => g != *geo || (!self.user_moved && (s - rect.size()).length() > 1.0),
        };
        if stale {
            self.fit(geo, rect);
        }
        // Zoom around the pointer; drag to pan; double-click to fit.
        if response.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = pinch * (scroll * 0.0035).exp();
            if (factor - 1.0).abs() > 1e-4
                && let Some(p) = response.hover_pos()
            {
                let before = self.to_cell(rect, p);
                self.zoom = (self.zoom * factor).clamp(0.05, 120.0);
                self.pan = before - (p - rect.min) / self.zoom;
                self.user_moved = true;
            }
        }
        if response.dragged() {
            self.pan -= response.drag_delta() / self.zoom;
            self.user_moved = true;
        }
        if response.double_clicked() {
            self.fit(geo, rect);
        }
        self.hover = response.hover_pos().and_then(|p| geo.locate(self.to_cell(rect, p)));
        let clicked = if response.clicked() { self.hover } else { None };

        let ppp = ui.ctx().pixels_per_point();
        let sel = selected.map(|(l, n)| [l as f32, n as f32]).unwrap_or([-1.0, -1.0]);
        let hov = self.hover.map(|(l, n)| [l as f32, n as f32]).unwrap_or([-1.0, -1.0]);
        let uniforms = Uniforms {
            origin: [rect.min.x * ppp, rect.min.y * ppp, 0.0, self.start.elapsed().as_secs_f32()],
            view: [self.pan.x, self.pan.y, self.zoom * ppp, 0.0],
            grid: [geo.tile_w as f32, geo.tile_h as f32, geo.tiles_x as f32, geo.gap_x],
            counts: [geo.layers as f32, geo.neurons as f32, geo.gap_y, glow],
            sel: [sel[0], sel[1], hov[0], hov[1]],
        };
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, Paint { geo: *geo, images, uniforms }));
        AtlasResponse { response, clicked, rect }
    }
}
