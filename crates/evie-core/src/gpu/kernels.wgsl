// evie GPU kernels (WGSL). Every pipeline shares this module; each entry point only touches the
// bindings it declares, so wgpu derives a small bind-group layout per entry point.
//
// Conventions: activations are f32, row-major `rows × width`. Quantised weights are Q8: int8 packed
// four per u32 (row-major) with one f32 scale per 64 consecutive inputs (16 words).

struct Rows {
    n: u32,        // rows in this forward pass
    n_seg: u32,    // sequences (segments) in this forward pass
    _a: u32,
    _b: u32,
}

struct RowInfo {
    seq: u32,      // which cache slot the row's sequence uses
    pos: u32,      // absolute position
    token: u32,
    seg: u32,
}

struct Seg {
    seq: u32,
    start: u32,    // first row
    count: u32,    // rows
    pos0: u32,
}

// Per call-site constants (fixed at load time).
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

@group(0) @binding(0) var<uniform> k: C;
@group(0) @binding(1) var<storage, read> rows: Rows;
@group(0) @binding(2) var<storage, read> info: array<RowInfo>;
@group(0) @binding(3) var<storage, read> segs: array<Seg>;

fn i8f(w: u32, sh: u32) -> f32 {
    return f32(bitcast<i32>(w << sh) >> 24u);
}

fn sigm(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// ---------------------------------------------------------------------------------------------
// embed: x[r, :] = dequant(E[token_r, :]).   k.a = d (hidden)

@group(0) @binding(4) var<storage, read> e_w: array<u32>;
@group(0) @binding(5) var<storage, read> e_s: array<f32>;
@group(0) @binding(6) var<storage, read_write> e_out: array<f32>;

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) g: vec3<u32>) {
    let d = k.a;
    let word = g.x;
    let r = g.y;
    if (r >= rows.n || word * 4u >= d) { return; }
    let t = info[r].token;
    let w = e_w[t * (d / 4u) + word];
    let s = e_s[t * (d / 64u) + word / 16u];
    let o = r * d + word * 4u;
    e_out[o] = s * i8f(w, 24u);
    e_out[o + 1u] = s * i8f(w, 16u);
    e_out[o + 2u] = s * i8f(w, 8u);
    e_out[o + 3u] = s * i8f(w, 0u);
}

// ---------------------------------------------------------------------------------------------
// rmsnorm: out[r] = x[r] / rms(x[r]) * w.  One workgroup per row.
// k.a = width, k.b = 1 if the rows are the per-segment gathered rows, k.fa = eps

@group(0) @binding(4) var<storage, read> n_x: array<f32>;
@group(0) @binding(5) var<storage, read> n_w: array<f32>;
@group(0) @binding(6) var<storage, read_write> n_out: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let r = wg.x;
    let nrows = select(rows.n, rows.n_seg, k.b == 1u);
    if (r >= nrows) { return; }
    let d = k.a;
    var ss = 0.0;
    for (var i = l.x; i < d; i += 256u) {
        let v = n_x[r * d + i];
        ss += v * v;
    }
    red[l.x] = ss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (l.x < s) { red[l.x] += red[l.x + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(d) + k.fa);
    for (var i = l.x; i < d; i += 256u) {
        n_out[r * d + i] = n_x[r * d + i] * inv * n_w[i];
    }
}

// ---------------------------------------------------------------------------------------------
// Q8 matmul, few rows (n ≤ 8): Y[t, o] = Σ_i W[o, i] · X[t, i]
// 4 outputs per workgroup, 64 threads each. k.a = out rows, k.b = in cols, k.c = 1 → segment rows,
// k.d = workgroups per grid row (2-D dispatch for big outputs)

@group(0) @binding(4) var<storage, read> m_w: array<u32>;
@group(0) @binding(5) var<storage, read> m_s: array<f32>;
@group(0) @binding(6) var<storage, read> m_x: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> m_y: array<f32>;
var<workgroup> mred: array<f32, 2048>;

@compute @workgroup_size(256)
fn matmul_small(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let n = select(rows.n, rows.n_seg, k.c == 1u);
    let sub = l.x / 64u;
    let lane = l.x % 64u;
    let o = (wg.y * k.d + wg.x) * 4u + sub;
    let words = k.b / 4u;
    var acc: array<f32, 8>;
    for (var t = 0u; t < 8u; t++) { acc[t] = 0.0; }
    if (o < k.a) {
        let base = o * words;
        let sbase = o * (k.b / 64u);
        for (var i = lane; i < words; i += 64u) {
            let w = m_w[base + i];
            let s = m_s[sbase + i / 16u];
            let wv = s * vec4<f32>(i8f(w, 24u), i8f(w, 16u), i8f(w, 8u), i8f(w, 0u));
            for (var t = 0u; t < n; t++) {
                acc[t] += dot(wv, m_x[t * words + i]);
            }
        }
    }
    for (var t = 0u; t < 8u; t++) {
        mred[(sub * 8u + t) * 64u + lane] = acc[t];
    }
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (lane < s) {
            for (var t = 0u; t < n; t++) {
                let b = (sub * 8u + t) * 64u;
                mred[b + lane] += mred[b + lane + s];
            }
        }
        workgroupBarrier();
    }
    if (lane < n && o < k.a) {
        m_y[lane * k.a + o] = mred[(sub * 8u + lane) * 64u];
    }
}

// ---------------------------------------------------------------------------------------------
// Q8 matmul, many rows: 64 outputs × 64 rows per workgroup, 4×4 per thread, K step 32.

var<workgroup> tw: array<f32, 2048>;   // [32][64] weights tile (k-major)
var<workgroup> tx: array<f32, 2048>;   // [32][64] activations tile (k-major)

@compute @workgroup_size(256)
fn matmul_tiled(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let n = rows.n;
    let o0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    let tid = l.x;
    let ty = tid / 16u;   // which 4 rows of tokens
    let txo = tid % 16u;  // which 4 outputs
    let words = k.b / 4u;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
    for (var kb = 0u; kb < k.b; kb += 32u) {
        // weights: 64 rows × 8 words; thread loads 2 words
        for (var j = 0u; j < 2u; j++) {
            let idx = tid * 2u + j;           // 0..512
            let row = idx / 8u;
            let wd = idx % 8u;
            let o = o0 + row;
            var v = vec4<f32>(0.0);
            if (o < k.a) {
                let w = m_w[o * words + kb / 4u + wd];
                let s = m_s[o * (k.b / 64u) + kb / 64u];
                v = s * vec4<f32>(i8f(w, 24u), i8f(w, 16u), i8f(w, 8u), i8f(w, 0u));
            }
            tw[(wd * 4u) * 64u + row] = v.x;
            tw[(wd * 4u + 1u) * 64u + row] = v.y;
            tw[(wd * 4u + 2u) * 64u + row] = v.z;
            tw[(wd * 4u + 3u) * 64u + row] = v.w;
        }
        // activations: 64 rows × 8 vec4; thread loads 2 vec4
        for (var j = 0u; j < 2u; j++) {
            let idx = tid * 2u + j;
            let row = idx / 8u;
            let q = idx % 8u;
            let t = t0 + row;
            var v = vec4<f32>(0.0);
            if (t < n) {
                v = m_x[t * words + kb / 4u + q];
            }
            tx[(q * 4u) * 64u + row] = v.x;
            tx[(q * 4u + 1u) * 64u + row] = v.y;
            tx[(q * 4u + 2u) * 64u + row] = v.z;
            tx[(q * 4u + 3u) * 64u + row] = v.w;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk++) {
            let wv = vec4<f32>(tw[kk * 64u + txo * 4u], tw[kk * 64u + txo * 4u + 1u], tw[kk * 64u + txo * 4u + 2u], tw[kk * 64u + txo * 4u + 3u]);
            for (var a = 0u; a < 4u; a++) {
                let xv = tx[kk * 64u + ty * 4u + a];
                acc[a * 4u] += xv * wv.x;
                acc[a * 4u + 1u] += xv * wv.y;
                acc[a * 4u + 2u] += xv * wv.z;
                acc[a * 4u + 3u] += xv * wv.w;
            }
        }
        workgroupBarrier();
    }
    for (var a = 0u; a < 4u; a++) {
        let t = t0 + ty * 4u + a;
        if (t >= n) { continue; }
        for (var b = 0u; b < 4u; b++) {
            let o = o0 + txo * 4u + b;
            if (o < k.a) {
                m_y[t * k.a + o] = acc[a * 4u + b];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// SwiGLU + steering + activation capture, 4 neurons per thread.
// k.a = neurons per layer, k.b = layer, k.c = layers, k.d = capture (0/1), k.e = track max |a| (0/1)

@group(0) @binding(4) var<storage, read_write> s_g: array<f32>;
@group(0) @binding(5) var<storage, read> s_u: array<f32>;
@group(0) @binding(6) var<storage, read> s_ops: array<vec2<f32>>;   // [slot][layer][neuron] (mode, value)
@group(0) @binding(7) var<storage, read> s_scale: array<f32>;       // per neuron range
@group(0) @binding(8) var<storage, read_write> s_cap: array<u32>;   // [row][layers*neurons/4]
@group(0) @binding(9) var<storage, read_write> s_max: array<atomic<u32>>;  // [layer][neuron] max |a| bits

fn q8c(v: f32) -> u32 {
    let a = round(sqrt(min(abs(v), 1.0)) * 127.0);
    let q = select(i32(a), -i32(a), v < 0.0);
    return u32(q) & 0xffu;
}

@compute @workgroup_size(64)
fn swiglu(@builtin(global_invocation_id) g: vec3<u32>) {
    let nn = k.a;
    let quad = g.x;
    let r = g.y;
    if (r >= rows.n || quad * 4u >= nn) { return; }
    var packed = 0u;
    for (var j = 0u; j < 4u; j++) {
        let i = quad * 4u + j;
        let idx = r * nn + i;
        var a = silu(s_g[idx]) * s_u[idx];
        let op = s_ops[(info[r].seq * k.c + k.b) * nn + i];
        let mode = u32(op.x);
        if (mode == 1u) { a += op.y; }
        else if (mode == 2u) { a *= op.y; }
        else if (mode == 3u) { a = op.y; }
        s_g[idx] = a;
        packed |= q8c(a / s_scale[k.b * nn + i]) << (8u * j);
        if (k.e == 1u && info[r].pos > 0u) {
            atomicMax(&s_max[k.b * nn + i], bitcast<u32>(abs(a)));
        }
    }
    if (k.d == 1u) {
        s_cap[r * (k.c * nn / 4u) + k.b * nn / 4u + quad] = packed;
    }
}

// ---------------------------------------------------------------------------------------------
// residual add (+ optional capture of the stream leaving the layer as f16)
// k.a = width, k.b = layer, k.c = layers, k.d = capture

@group(0) @binding(4) var<storage, read_write> r_x: array<f32>;
@group(0) @binding(5) var<storage, read> r_y: array<f32>;
@group(0) @binding(6) var<storage, read_write> r_cap: array<u32>;   // [row][layers][width/2]

@compute @workgroup_size(64)
fn residual(@builtin(global_invocation_id) g: vec3<u32>) {
    let d = k.a;
    let pair = g.x;
    let r = g.y;
    if (r >= rows.n || pair * 2u >= d) { return; }
    let i = r * d + pair * 2u;
    let a = r_x[i] + r_y[i];
    let b = r_x[i + 1u] + r_y[i + 1u];
    r_x[i] = a;
    r_x[i + 1u] = b;
    if (k.d == 1u) {
        r_cap[(r * k.c + k.b) * (d / 2u) + pair] = pack2x16float(vec2<f32>(a, b));
    }
}

// ---------------------------------------------------------------------------------------------
// gather the last row of each segment: out[s] = x[segs[s].start + count - 1]   k.a = width

@group(0) @binding(4) var<storage, read> gl_x: array<f32>;
@group(0) @binding(5) var<storage, read_write> gl_out: array<f32>;

@compute @workgroup_size(64)
fn gather_last(@builtin(global_invocation_id) g: vec3<u32>) {
    let s = g.y;
    let i = g.x;
    if (s >= rows.n_seg || i >= k.a) { return; }
    let r = segs[s].start + segs[s].count - 1u;
    gl_out[s * k.a + i] = gl_x[r * k.a + i];
}

// ---------------------------------------------------------------------------------------------
// DeltaNet causal conv + SiLU over [state ; rows] per channel and segment.
// k.a = channels, k.b = kernel, k.c = linear-layer index, k.d = linear layers

@group(0) @binding(4) var<storage, read_write> c_x: array<f32>;      // rows × channels (in place)
@group(0) @binding(5) var<storage, read> c_w: array<f32>;            // channels × kernel
@group(0) @binding(6) var<storage, read_write> c_state: array<f32>;  // [seq][lin layer][kernel-1][channels]

@compute @workgroup_size(64)
fn conv(@builtin(global_invocation_id) g: vec3<u32>) {
    let ch = g.x;
    let s = g.y;
    let cc = k.a;
    let kn = k.b;
    if (ch >= cc || s >= rows.n_seg) { return; }
    let sg = segs[s];
    let base = (sg.seq * k.d + k.c) * (kn - 1u) * cc;
    var win: array<f32, 8>;
    for (var j = 0u; j < kn - 1u; j++) { win[j] = c_state[base + j * cc + ch]; }
    for (var t = 0u; t < sg.count; t++) {
        let r = sg.start + t;
        let x = c_x[r * cc + ch];
        var acc = c_w[ch * kn + kn - 1u] * x;
        for (var j = 0u; j < kn - 1u; j++) { acc += c_w[ch * kn + j] * win[j]; }
        for (var j = 0u; j + 1u < kn - 1u; j++) { win[j] = win[j + 1u]; }
        win[kn - 2u] = x;
        c_x[r * cc + ch] = silu(acc);
    }
    for (var j = 0u; j < kn - 1u; j++) { c_state[base + j * cc + ch] = win[j]; }
}

// ---------------------------------------------------------------------------------------------
// L2-normalise q and k per key head (q also × 1/sqrt(dk)). Workgroup = (head slot, row).
// k.a = channels (row stride), k.b = key heads, k.c = dk, k.fa = q scale

@group(0) @binding(4) var<storage, read_write> l_x: array<f32>;
var<workgroup> lred: array<f32, 128>;

@compute @workgroup_size(128)
fn l2norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let slot = wg.x;          // 0..2*heads (q heads then k heads)
    let r = wg.y;
    if (r >= rows.n) { return; }
    let dk = k.c;
    let off = r * k.a + slot * dk;
    var v = 0.0;
    if (l.x < dk) { v = l_x[off + l.x]; }
    lred[l.x] = v * v;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s >>= 1u) {
        if (l.x < s) { lred[l.x] += lred[l.x + s]; }
        workgroupBarrier();
    }
    let scale = select(1.0, k.fa, slot < k.b);
    if (l.x < dk) {
        l_x[off + l.x] = v * inverseSqrt(lred[0] + 1e-6) * scale;
    }
}

// ---------------------------------------------------------------------------------------------
// Gated delta rule recurrence + gated RMSNorm. Workgroup = (value head, segment); thread j owns
// column j of the head's state S[dk][dv].
// k.a = channels (qkv stride), k.b = key heads, k.c = value heads, k.d = linear-layer index,
// k.e = linear layers, k.f = key_dim (q/k block width), k.fa = eps

@group(0) @binding(4) var<storage, read> d_qkv: array<f32>;
@group(0) @binding(5) var<storage, read> d_z: array<f32>;
@group(0) @binding(6) var<storage, read> d_ba: array<f32>;      // rows × (2*hv): b then a
@group(0) @binding(7) var<storage, read> d_par: array<f32>;     // [neg_a(hv) | dt_bias(hv) | norm(dv)]
@group(0) @binding(8) var<storage, read_write> d_state: array<f32>;
@group(0) @binding(9) var<storage, read_write> d_out: array<f32>;  // rows × (hv*dv)
var<workgroup> dk_k: array<f32, 128>;
var<workgroup> dk_q: array<f32, 128>;
var<workgroup> dred: array<f32, 128>;

@compute @workgroup_size(128)
fn deltanet(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let h = wg.x;
    let s = wg.y;
    if (s >= rows.n_seg) { return; }
    let hk = k.b;
    let hv = k.c;
    let dk = 128u;
    let dv = 128u;
    let kh = h / (hv / hk);
    let j = l.x;
    let sg = segs[s];
    let sbase = (((sg.seq * k.e + k.d) * hv) + h) * dk * dv;
    let vd = hv * dv;
    for (var t = 0u; t < sg.count; t++) {
        let r = sg.start + t;
        let row = r * k.a;
        dk_q[j] = d_qkv[row + kh * dk + j];
        dk_k[j] = d_qkv[row + k.f + kh * dk + j];
        let v = d_qkv[row + 2u * k.f + h * dv + j];
        let bb = d_ba[r * 2u * hv + h];
        let aa = d_ba[r * 2u * hv + hv + h];
        workgroupBarrier();
        let sp = select(log(1.0 + exp(aa + d_par[hv + h])), aa + d_par[hv + h], aa + d_par[hv + h] > 20.0);
        let decay = exp(d_par[h] * sp);
        let beta = sigm(bb);
        var kv = 0.0;
        for (var i = 0u; i < dk; i++) { kv += d_state[sbase + i * dv + j] * dk_k[i]; }
        let delta = (v - decay * kv) * beta;
        var o = 0.0;
        for (var i = 0u; i < dk; i++) {
            let idx = sbase + i * dv + j;
            let sv = d_state[idx] * decay + dk_k[i] * delta;
            d_state[idx] = sv;
            o += sv * dk_q[i];
        }
        dred[j] = o * o;
        workgroupBarrier();
        for (var st = 64u; st > 0u; st >>= 1u) {
            if (j < st) { dred[j] += dred[j + st]; }
            workgroupBarrier();
        }
        let inv = inverseSqrt(dred[0] / f32(dv) + k.fa);
        let zz = d_z[r * vd + h * dv + j];
        d_out[r * vd + h * dv + j] = o * inv * d_par[2u * hv + j] * silu(zz);
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------------------------------------
// Full attention prep: per (head slot, row): RMSNorm (1+w folded in) + partial RoPE for q heads
// (slots 0..nh) and k heads (slots nh..nh+nkv); k/v rows are appended to the cache.
// k.a = n_heads, k.b = kv heads, k.c = head_dim, k.d = full-layer index, k.e = full layers,
// k.f = max context, k.fa = eps, k.fb = rope theta.  Rotary dims: 64 (half 32).

@group(0) @binding(4) var<storage, read> p_q: array<f32>;      // rows × nh × (2·hd): [q | gate]
@group(0) @binding(5) var<storage, read> p_k: array<f32>;      // rows × nkv × hd
@group(0) @binding(6) var<storage, read> p_v: array<f32>;
@group(0) @binding(7) var<storage, read> p_norm: array<f32>;   // [q_norm(hd) | k_norm(hd)]
@group(0) @binding(8) var<storage, read_write> p_qout: array<f32>;  // rows × nh × hd
@group(0) @binding(9) var<storage, read_write> p_kc: array<f32>;    // [seq][layer][ctx][nkv][hd]
@group(0) @binding(10) var<storage, read_write> p_vc: array<f32>;
var<workgroup> pv: array<f32, 256>;
var<workgroup> pred: array<f32, 256>;

@compute @workgroup_size(256)
fn attn_prep(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let slot = wg.x;
    let r = wg.y;
    if (r >= rows.n) { return; }
    let nh = k.a;
    let nkv = k.b;
    let hd = k.c;
    let i = l.x;
    let is_q = slot < nh;
    var x = 0.0;
    if (i < hd) {
        if (is_q) { x = p_q[(r * nh + slot) * 2u * hd + i]; } else { x = p_k[(r * nkv + (slot - nh)) * hd + i]; }
    }
    pred[i] = x * x;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (i < s) { pred[i] += pred[i + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(pred[0] / f32(hd) + k.fa);
    var y = 0.0;
    if (i < hd) {
        y = x * inv * p_norm[select(hd, 0u, is_q) + i];
    }
    pv[i] = y;
    workgroupBarrier();
    let half = 32u;
    let pos = info[r].pos;
    if (i < 2u * half) {
        let fi = i % half;
        let ang = f32(pos) * pow(k.fb, -f32(2u * fi) / f32(2u * half));
        let c = cos(ang);
        let sn = sin(ang);
        if (i < half) { y = pv[i] * c - pv[i + half] * sn; } else { y = pv[i] * c + pv[i - half] * sn; }
    }
    if (i >= hd) { return; }
    if (is_q) {
        p_qout[(r * nh + slot) * hd + i] = y;
    } else {
        let kvh = slot - nh;
        let seq = info[r].seq;
        let at = (((seq * k.e + k.d) * k.f + pos) * nkv + kvh) * hd + i;
        p_kc[at] = y;
        p_vc[at] = p_v[(r * nkv + kvh) * hd + i];
    }
}

// ---------------------------------------------------------------------------------------------
// Attention scores: sc[r][h][j] = q·k_j / sqrt(hd) for j ≤ pos. Workgroup = (position chunk, row·head).
// k.a = n_heads, k.b = kv heads, k.c = head_dim, k.d = full-layer index, k.e = full layers, k.f = max ctx

@group(0) @binding(4) var<storage, read> a_q: array<f32>;
@group(0) @binding(5) var<storage, read> a_kc: array<f32>;
@group(0) @binding(6) var<storage, read_write> a_sc: array<f32>;  // rows × nh × max ctx

@compute @workgroup_size(64)
fn attn_scores(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let rh = wg.y;
    let nh = k.a;
    let r = rh / nh;
    let h = rh % nh;
    if (r >= rows.n) { return; }
    let j = wg.x * 64u + l.x;
    let pos = info[r].pos;
    if (j > pos) { return; }
    let hd = k.c;
    let kvh = h / (nh / k.b);
    let seq = info[r].seq;
    let kb = (((seq * k.e + k.d) * k.f + j) * k.b + kvh) * hd;
    let qb = (r * nh + h) * hd;
    var acc = 0.0;
    for (var i = 0u; i < hd; i++) { acc += a_q[qb + i] * a_kc[kb + i]; }
    a_sc[(r * nh + h) * k.f + j] = acc * inverseSqrt(f32(hd));
}

// Softmax over scores then the weighted sum of values, gated: o = σ(gate) ⊙ Σ p_j v_j.
// Workgroup = (row·head), thread = output dim.

@group(0) @binding(4) var<storage, read> o_q: array<f32>;       // raw q_proj output (for the gate)
@group(0) @binding(5) var<storage, read> o_vc: array<f32>;
@group(0) @binding(6) var<storage, read_write> o_sc: array<f32>;
@group(0) @binding(7) var<storage, read_write> o_out: array<f32>;  // rows × nh × hd
var<workgroup> ored: array<f32, 256>;

@compute @workgroup_size(256)
fn attn_out(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let rh = wg.x;
    let nh = k.a;
    let r = rh / nh;
    let h = rh % nh;
    if (r >= rows.n) { return; }
    let hd = k.c;
    let pos = info[r].pos;
    let n = pos + 1u;
    let sb = (r * nh + h) * k.f;
    var m = -1e30;
    for (var j = l.x; j < n; j += 256u) { m = max(m, o_sc[sb + j]); }
    ored[l.x] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (l.x < s) { ored[l.x] = max(ored[l.x], ored[l.x + s]); }
        workgroupBarrier();
    }
    m = ored[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = l.x; j < n; j += 256u) {
        let e = exp(o_sc[sb + j] - m);
        o_sc[sb + j] = e;
        sum += e;
    }
    ored[l.x] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (l.x < s) { ored[l.x] += ored[l.x + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / ored[0];
    let i = l.x;
    if (i >= hd) { return; }
    let kvh = h / (nh / k.b);
    let seq = info[r].seq;
    var acc = 0.0;
    for (var j = 0u; j < n; j++) {
        acc += o_sc[sb + j] * o_vc[(((seq * k.e + k.d) * k.f + j) * k.b + kvh) * hd + i];
    }
    let gate = o_q[(r * nh + h) * 2u * hd + hd + i];
    o_out[(r * nh + h) * hd + i] = acc * inv * sigm(gate);
}
