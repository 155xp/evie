// Neuron atlas: every MLP neuron of every layer as one cell. Layers are tiles of tile_w × tile_h
// cells laid out tiles_x per row; the activation texture is the same grid without the gaps.

struct Uniforms {
    // physical-pixel origin of the widget, srgb-target flag, time
    origin: vec4<f32>,
    // pan (atlas cells at the widget's top-left), zoom (physical px per cell), unused
    view: vec4<f32>,
    // tile_w, tile_h, tiles_x, gap_x
    grid: vec4<f32>,
    // layers, neurons, gap_y, glow
    counts: vec4<f32>,
    // selected layer, selected neuron, hovered layer, hovered neuron (-1 = none)
    sel: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var vals: texture_2d<f32>;
@group(0) @binding(2) var marks: texture_2d<u32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

const BG: vec3<f32> = vec3<f32>(0.035, 0.043, 0.055);
const CELL: vec3<f32> = vec3<f32>(0.075, 0.090, 0.112);

// egui-wgpu may render into an sRGB or a linear-encoded target; origin.z says which.
fn out(c: vec3<f32>) -> vec4<f32> {
    if (u.origin.z > 0.5) {
        return vec4<f32>(pow(c, vec3<f32>(2.2)), 1.0);
    }
    return vec4<f32>(c, 1.0);
}

// Fire ramp for positive activations, orchid ramp for negative ones.
fn ramp(t: f32, neg: bool) -> vec3<f32> {
    var a: vec3<f32>;
    var b: vec3<f32>;
    var c: vec3<f32>;
    if (neg) {
        a = vec3<f32>(0.30, 0.05, 0.32);
        b = vec3<f32>(0.86, 0.15, 0.47);
        c = vec3<f32>(0.99, 0.80, 0.93);
    } else {
        a = vec3<f32>(0.47, 0.12, 0.04);
        b = vec3<f32>(0.96, 0.62, 0.04);
        c = vec3<f32>(1.00, 0.96, 0.78);
    }
    let x = clamp(t, 0.0, 1.0);
    if (x < 0.55) {
        return mix(mix(CELL, a, smoothstep(0.02, 0.3, x)), b, smoothstep(0.22, 0.55, x));
    }
    return mix(b, c, smoothstep(0.55, 1.0, x));
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let zoom = u.view.z;
    let p = u.view.xy + (pos.xy - u.origin.xy) / zoom;
    let tile_w = u.grid.x;
    let tile_h = u.grid.y;
    let tiles_x = u.grid.z;
    let pitch = vec2<f32>(tile_w + u.grid.w, tile_h + u.counts.z);
    let t = floor(p / pitch);
    let l = p - t * pitch;
    if (p.x < 0.0 || p.y < 0.0 || t.x >= tiles_x || l.x >= tile_w || l.y >= tile_h) {
        return out(BG);
    }
    let layer = t.y * tiles_x + t.x;
    let cell = floor(l);
    let neuron = cell.y * tile_w + cell.x;
    if (layer >= u.counts.x) {
        return out(BG);
    }
    if (neuron >= u.counts.y) {
        return out(BG * 1.4);
    }
    let texel = vec2<i32>(i32(t.x * tile_w + cell.x), i32(t.y * tile_h + cell.y));
    // Stored companded (q = sign·sqrt|v|); decode to the linear activation relative to the
    // neuron's usual range so the map is mostly dark with genuinely firing neurons standing out.
    let q = textureLoad(vals, texel, 0).r;
    let v = q * abs(q);
    let m = textureLoad(marks, texel, 0).r;
    var col = ramp(abs(v) * u.counts.w, v < 0.0);

    // Selection crosshair inside the selected layer's tile.
    let sel_l = u.sel.x;
    let sel_n = u.sel.y;
    if (sel_l == layer && sel_n >= 0.0) {
        let sc = vec2<f32>(sel_n % tile_w, floor(sel_n / tile_w));
        if (cell.x == sc.x || cell.y == sc.y) {
            col = mix(col, vec3<f32>(0.18, 0.83, 0.75), 0.18);
        }
    }

    // Cell borders and markers once cells are big enough to see.
    let f = l - cell;
    let edge = min(min(f.x, f.y), min(1.0 - f.x, 1.0 - f.y)) * zoom;
    if (zoom >= 5.0) {
        col = mix(BG, col, smoothstep(0.35, 1.2, edge));
    }
    if (m != 0u && zoom >= 6.0) {
        var mc = vec3<f32>(0.18, 0.83, 0.75);
        if ((m & 1u) != 0u) { mc = vec3<f32>(1.0, 0.75, 0.25); }
        if ((m & 2u) != 0u) { mc = vec3<f32>(0.98, 0.45, 0.72); }
        let r = length(f - vec2<f32>(0.5)) * zoom;
        let ring = 1.0 - smoothstep(0.0, 1.4, abs(r - zoom * 0.36));
        col = mix(col, mc, ring);
    }
    if (u.sel.z == layer && u.sel.w == neuron) {
        col = mix(col, vec3<f32>(1.0), 0.25);
    }
    return out(col);
}
