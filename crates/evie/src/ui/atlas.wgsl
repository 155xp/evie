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

const BG: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const CELL: vec3<f32> = vec3<f32>(0.075, 0.075, 0.075);

// egui-wgpu may render into an sRGB or a linear-encoded target; origin.z says which.
fn out(c: vec3<f32>) -> vec4<f32> {
    if (u.origin.z > 0.5) {
        return vec4<f32>(pow(c, vec3<f32>(2.2)), 1.0);
    }
    return vec4<f32>(c, 1.0);
}

// Monochrome: brightness is activation strength relative to the neuron's usual range.
// Weak activity stays near black so the neurons that are really firing stand out.
fn shade(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    return vec3<f32>(mix(CELL.x, 1.0, pow(x, 1.6)));
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
        return out(BG);
    }
    let texel = vec2<i32>(i32(t.x * tile_w + cell.x), i32(t.y * tile_h + cell.y));
    // Stored companded (q = sign·sqrt|v|); decode to the linear activation relative to the
    // neuron's usual range so the map is mostly dark with genuinely firing neurons standing out.
    let q = textureLoad(vals, texel, 0).r;
    let v = q * abs(q);
    let m = textureLoad(marks, texel, 0).r;
    var col = shade(abs(v) * u.counts.w);

    // Selection crosshair inside the selected layer's tile.
    let sel_l = u.sel.x;
    let sel_n = u.sel.y;
    if (sel_l == layer && sel_n >= 0.0) {
        let sc = vec2<f32>(sel_n % tile_w, floor(sel_n / tile_w));
        if (cell.x == sc.x || cell.y == sc.y) {
            col = mix(col, vec3<f32>(1.0), 0.07);
        }
    }

    // Cell borders and markers once cells are big enough to see.
    let f = l - cell;
    let edge = min(min(f.x, f.y), min(1.0 - f.x, 1.0 - f.y)) * zoom;
    if (zoom >= 5.0) {
        col = mix(BG, col, smoothstep(0.35, 1.2, edge));
    }
    if (m != 0u && zoom >= 6.0) {
        var mc = vec3<f32>(0.55);
        if ((m & 3u) != 0u) { mc = vec3<f32>(1.0); }
        let r = length(f - vec2<f32>(0.5)) * zoom;
        let ring = 1.0 - smoothstep(0.0, 1.4, abs(r - zoom * 0.36));
        col = mix(col, mc, ring);
    }
    if (u.sel.z == layer && u.sel.w == neuron) {
        col = mix(col, vec3<f32>(1.0), 0.25);
    }
    return out(col);
}
