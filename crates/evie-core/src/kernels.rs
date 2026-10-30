//! Compute kernels: 16-bit-weight matvec / GEMM (AVX2+FMA+F16C, scalar fallback) and small f32 ops.
//!
//! Weights stay in their checkpoint format (bf16, or f16 for lens maps) and are widened to f32 inside
//! the kernels; activations are always f32.

use rayon::prelude::*;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind16 {
    Bf16,
    F16,
}

/// A row-major `rows × cols` matrix of 16-bit floats, usually a view into an mmapped checkpoint.
#[derive(Clone)]
pub struct Mat16 {
    pub rows: usize,
    pub cols: usize,
    pub kind: Kind16,
    ptr: *const u16,
    _owner: Arc<dyn std::any::Any + Send + Sync>,
}

// SAFETY: the data behind `ptr` is immutable and kept alive by `_owner`.
unsafe impl Send for Mat16 {}
unsafe impl Sync for Mat16 {}

impl Mat16 {
    pub fn from_vec(rows: usize, cols: usize, kind: Kind16, v: Vec<u16>) -> Self {
        assert_eq!(v.len(), rows * cols);
        let owner = Arc::new(v);
        let ptr = owner.as_ptr();
        Self { rows, cols, kind, ptr, _owner: owner }
    }

    pub(crate) fn from_mapped(rows: usize, cols: usize, kind: Kind16, ptr: *const u16, owner: Arc<memmap2::Mmap>) -> Self {
        Self { rows, cols, kind, ptr, _owner: owner }
    }

    pub fn data(&self) -> &[u16] {
        // SAFETY: constructed from a buffer of exactly rows*cols u16s that outlives self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.rows * self.cols) }
    }

    pub fn row(&self, r: usize) -> &[u16] {
        &self.data()[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        to_f32(self.kind, self.data()[r * self.cols + c])
    }

    pub fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        let k = self.kind;
        for (o, &x) in out.iter_mut().zip(self.row(r)) {
            *o = to_f32(k, x);
        }
    }

    pub fn col_to_f32(&self, c: usize, out: &mut [f32]) {
        let d = self.data();
        for (r, o) in out.iter_mut().enumerate().take(self.rows) {
            *o = to_f32(self.kind, d[r * self.cols + c]);
        }
    }

    /// Selected columns, transposed into rows: `out[i] = column cols[i]` (each `self.rows` long).
    pub fn cols_to_f32(&self, cols: &[usize]) -> Vec<f32> {
        let rows = self.rows;
        let mut out = vec![0f32; cols.len() * rows];
        // Walk the matrix row by row (sequential memory) and scatter.
        out.par_chunks_mut(rows).zip(cols.par_iter()).for_each(|(o, &c)| self.col_to_f32(c, o));
        out
    }

    /// Euclidean norm of every column.
    pub fn col_norms(&self) -> Vec<f32> {
        let (rows, cols) = (self.rows, self.cols);
        let d = self.data();
        let partial: Vec<Vec<f64>> = (0..rows)
            .into_par_iter()
            .fold(
                || vec![0f64; cols],
                |mut acc, r| {
                    let row = &d[r * cols..(r + 1) * cols];
                    for (a, &x) in acc.iter_mut().zip(row) {
                        let v = to_f32(self.kind, x) as f64;
                        *a += v * v;
                    }
                    acc
                },
            )
            .collect();
        let mut tot = vec![0f64; cols];
        for p in partial {
            for (t, v) in tot.iter_mut().zip(p) {
                *t += v;
            }
        }
        tot.into_iter().map(|v| v.sqrt() as f32).collect()
    }
}

#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

#[inline]
pub fn to_f32(kind: Kind16, x: u16) -> f32 {
    match kind {
        Kind16::Bf16 => bf16_to_f32(x),
        Kind16::F16 => half::f16::from_bits(x).to_f32(),
    }
}

#[cfg(target_arch = "x86_64")]
pub fn simd() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c")
    })
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd() -> bool {
    false
}

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// `y = W x`.
pub fn matvec(w: &Mat16, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols);
    assert_eq!(y.len(), w.rows);
    let chunk = (w.rows / (rayon::current_num_threads() * 6)).clamp(4, 512) & !3;
    y.par_chunks_mut(chunk).enumerate().for_each(|(ci, ys)| dot_rows(w, ci * chunk, x, ys));
}

fn dot_rows(w: &Mat16, r0: usize, x: &[f32], ys: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: features checked; indices bounded by the asserts in the callers.
        unsafe {
            match w.kind {
                Kind16::Bf16 => avx::dot_rows::<false>(w.ptr, w.cols, r0, x.as_ptr(), ys),
                Kind16::F16 => avx::dot_rows::<true>(w.ptr, w.cols, r0, x.as_ptr(), ys),
            }
        }
        return;
    }
    for (i, y) in ys.iter_mut().enumerate() {
        *y = w.row(r0 + i).iter().zip(x).map(|(&a, &b)| to_f32(w.kind, a) * b).sum();
    }
}

/// `Y[t, r] = Σ_k W[r, k] · X[t, k]` for `t < n`: X is `n × cols`, Y is `n × rows`, both row-major.
pub fn matmul(w: &Mat16, x: &[f32], n: usize, y: &mut [f32]) {
    assert_eq!(x.len(), n * w.cols, "matmul input shape");
    assert_eq!(y.len(), n * w.rows, "matmul output shape");
    if n == 0 {
        return;
    }
    if n == 1 {
        return matvec(w, x, y);
    }
    #[cfg(target_arch = "x86_64")]
    if simd() {
        if n < 8 {
            small_batch(w, x, n, y);
        } else {
            gemm(w, x, n, y);
        }
        return;
    }
    for t in 0..n {
        matvec(w, &x[t * w.cols..(t + 1) * w.cols], &mut y[t * w.rows..(t + 1) * w.rows]);
    }
}

#[cfg(target_arch = "x86_64")]
fn small_batch(w: &Mat16, x: &[f32], n: usize, y: &mut [f32]) {
    let rows = w.rows;
    let chunk = (rows / (rayon::current_num_threads() * 6)).clamp(4, 512);
    let yp = SendPtr(y.as_mut_ptr());
    let (wp, xp, cols, f16) = (w.ptr as usize, x.as_ptr() as usize, w.cols, w.kind == Kind16::F16);
    (0..rows.div_ceil(chunk)).into_par_iter().for_each(move |ci| {
        let r0 = ci * chunk;
        let nr = chunk.min(rows - r0);
        let yp = yp;
        // SAFETY: each task writes a disjoint set of output rows.
        unsafe {
            macro_rules! go {
                ($t:literal) => {
                    if f16 {
                        avx::multi_dot::<true, $t>(wp as *const u16, cols, r0, nr, xp as *const f32, yp.0, rows)
                    } else {
                        avx::multi_dot::<false, $t>(wp as *const u16, cols, r0, nr, xp as *const f32, yp.0, rows)
                    }
                };
            }
            match n {
                2 => go!(2),
                3 => go!(3),
                4 => go!(4),
                5 => go!(5),
                6 => go!(6),
                _ => go!(7),
            }
        }
    });
}

#[cfg(target_arch = "x86_64")]
fn gemm(w: &Mat16, x: &[f32], n: usize, y: &mut [f32]) {
    use avx::{KC, NR};
    let (rows, cols) = (w.rows, w.cols);
    let panels = rows.div_ceil(NR);
    let yp = SendPtr(y.as_mut_ptr());
    let (wp, xp, f16) = (w.ptr as usize, x.as_ptr() as usize, w.kind == Kind16::F16);
    // Token blocks keep the activation block cache-resident while every panel streams past it.
    let tb_size = (((2 << 20) / (cols * 4)).max(6) / 6 * 6).min(480);
    for tb0 in (0..n).step_by(tb_size) {
        let tb = tb_size.min(n - tb0);
        (0..panels).into_par_iter().for_each_init(
            || vec![0f32; KC * NR],
            move |pack, p| {
                let yp = yp;
                let r0 = p * NR;
                let nr = NR.min(rows - r0);
                let mut k0 = 0;
                while k0 < cols {
                    let kc = KC.min(cols - k0);
                    // SAFETY: panel p owns output columns r0..r0+nr of every row; reads are in bounds.
                    unsafe {
                        if f16 {
                            avx::pack_panel::<true>(wp as *const u16, cols, rows, r0, k0, kc, pack.as_mut_ptr());
                        } else {
                            avx::pack_panel::<false>(wp as *const u16, cols, rows, r0, k0, kc, pack.as_mut_ptr());
                        }
                        let mut t = 0;
                        while t < tb {
                            let m = 6.min(tb - t);
                            let xr = (xp as *const f32).add((tb0 + t) * cols + k0);
                            let yr = yp.0.add((tb0 + t) * rows + r0);
                            let acc = k0 > 0;
                            match m {
                                6 => avx::micro::<6>(pack.as_ptr(), kc, xr, cols, yr, rows, nr, acc),
                                5 => avx::micro::<5>(pack.as_ptr(), kc, xr, cols, yr, rows, nr, acc),
                                4 => avx::micro::<4>(pack.as_ptr(), kc, xr, cols, yr, rows, nr, acc),
                                3 => avx::micro::<3>(pack.as_ptr(), kc, xr, cols, yr, rows, nr, acc),
                                2 => avx::micro::<2>(pack.as_ptr(), kc, xr, cols, yr, rows, nr, acc),
                                _ => avx::micro::<1>(pack.as_ptr(), kc, xr, cols, yr, rows, nr, acc),
                            }
                            t += m;
                        }
                    }
                    k0 += kc;
                }
            },
        );
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    pub const KC: usize = 256;
    pub const NR: usize = 16;

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn load8<const F16: bool>(p: *const u16) -> __m256 {
        unsafe {
            let v = _mm_loadu_si128(p as *const __m128i);
            if F16 { _mm256_cvtph_ps(v) } else { _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(v))) }
        }
    }

    #[inline]
    fn cvt<const F16: bool>(x: u16) -> f32 {
        if F16 { half::f16::from_bits(x).to_f32() } else { super::bf16_to_f32(x) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps::<1>(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    /// `ys[i] = W[r0 + i] · x`, four rows at a time so each x load feeds four FMAs.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_rows<const F16: bool>(w: *const u16, cols: usize, r0: usize, x: *const f32, ys: &mut [f32]) {
        unsafe {
            let n = ys.len();
            let mut i = 0;
            while i + 4 <= n {
                let p0 = w.add((r0 + i) * cols);
                let (p1, p2, p3) = (p0.add(cols), p0.add(2 * cols), p0.add(3 * cols));
                let z = _mm256_setzero_ps();
                let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
                let (mut b0, mut b1, mut b2, mut b3) = (z, z, z, z);
                let mut k = 0;
                while k + 16 <= cols {
                    let x0 = _mm256_loadu_ps(x.add(k));
                    let x1 = _mm256_loadu_ps(x.add(k + 8));
                    a0 = _mm256_fmadd_ps(load8::<F16>(p0.add(k)), x0, a0);
                    b0 = _mm256_fmadd_ps(load8::<F16>(p0.add(k + 8)), x1, b0);
                    a1 = _mm256_fmadd_ps(load8::<F16>(p1.add(k)), x0, a1);
                    b1 = _mm256_fmadd_ps(load8::<F16>(p1.add(k + 8)), x1, b1);
                    a2 = _mm256_fmadd_ps(load8::<F16>(p2.add(k)), x0, a2);
                    b2 = _mm256_fmadd_ps(load8::<F16>(p2.add(k + 8)), x1, b2);
                    a3 = _mm256_fmadd_ps(load8::<F16>(p3.add(k)), x0, a3);
                    b3 = _mm256_fmadd_ps(load8::<F16>(p3.add(k + 8)), x1, b3);
                    k += 16;
                }
                let mut s = [
                    hsum(_mm256_add_ps(a0, b0)),
                    hsum(_mm256_add_ps(a1, b1)),
                    hsum(_mm256_add_ps(a2, b2)),
                    hsum(_mm256_add_ps(a3, b3)),
                ];
                while k < cols {
                    let xv = *x.add(k);
                    s[0] += cvt::<F16>(*p0.add(k)) * xv;
                    s[1] += cvt::<F16>(*p1.add(k)) * xv;
                    s[2] += cvt::<F16>(*p2.add(k)) * xv;
                    s[3] += cvt::<F16>(*p3.add(k)) * xv;
                    k += 1;
                }
                ys[i..i + 4].copy_from_slice(&s);
                i += 4;
            }
            while i < n {
                let p = w.add((r0 + i) * cols);
                let mut a = _mm256_setzero_ps();
                let mut k = 0;
                while k + 8 <= cols {
                    a = _mm256_fmadd_ps(load8::<F16>(p.add(k)), _mm256_loadu_ps(x.add(k)), a);
                    k += 8;
                }
                let mut s = hsum(a);
                while k < cols {
                    s += cvt::<F16>(*p.add(k)) * *x.add(k);
                    k += 1;
                }
                ys[i] = s;
                i += 1;
            }
        }
    }

    /// Few-token multiply: each weight row is widened once and dotted with all `T` inputs.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn multi_dot<const F16: bool, const T: usize>(
        w: *const u16,
        cols: usize,
        r0: usize,
        nr: usize,
        x: *const f32,
        y: *mut f32,
        ldy: usize,
    ) {
        unsafe {
            for r in r0..r0 + nr {
                let p = w.add(r * cols);
                let mut acc = [_mm256_setzero_ps(); T];
                let mut k = 0;
                while k + 8 <= cols {
                    let wv = load8::<F16>(p.add(k));
                    for (t, a) in acc.iter_mut().enumerate() {
                        *a = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x.add(t * cols + k)), *a);
                    }
                    k += 8;
                }
                for (t, a) in acc.iter().enumerate() {
                    let mut s = hsum(*a);
                    for kk in k..cols {
                        s += cvt::<F16>(*p.add(kk)) * *x.add(t * cols + kk);
                    }
                    *y.add(t * ldy + r) = s;
                }
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn transpose8(r: &mut [__m256; 8]) {
        let t0 = _mm256_unpacklo_ps(r[0], r[1]);
        let t1 = _mm256_unpackhi_ps(r[0], r[1]);
        let t2 = _mm256_unpacklo_ps(r[2], r[3]);
        let t3 = _mm256_unpackhi_ps(r[2], r[3]);
        let t4 = _mm256_unpacklo_ps(r[4], r[5]);
        let t5 = _mm256_unpackhi_ps(r[4], r[5]);
        let t6 = _mm256_unpacklo_ps(r[6], r[7]);
        let t7 = _mm256_unpackhi_ps(r[6], r[7]);
        let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
        let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
        let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
        let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
        let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
        let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
        let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
        let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
        r[0] = _mm256_permute2f128_ps::<0x20>(s0, s4);
        r[1] = _mm256_permute2f128_ps::<0x20>(s1, s5);
        r[2] = _mm256_permute2f128_ps::<0x20>(s2, s6);
        r[3] = _mm256_permute2f128_ps::<0x20>(s3, s7);
        r[4] = _mm256_permute2f128_ps::<0x31>(s0, s4);
        r[5] = _mm256_permute2f128_ps::<0x31>(s1, s5);
        r[6] = _mm256_permute2f128_ps::<0x31>(s2, s6);
        r[7] = _mm256_permute2f128_ps::<0x31>(s3, s7);
    }

    /// Packs `W[r0..r0+16, k0..k0+kc]` k-major (`pack[k*16 + i]`) as f32, zero-padding missing rows.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn pack_panel<const F16: bool>(
        w: *const u16,
        cols: usize,
        rows: usize,
        r0: usize,
        k0: usize,
        kc: usize,
        pack: *mut f32,
    ) {
        unsafe {
            let nr = NR.min(rows - r0);
            if nr == NR && kc % 8 == 0 {
                for half in 0..2 {
                    let base = w.add((r0 + half * 8) * cols + k0);
                    let mut k = 0;
                    while k < kc {
                        let mut m = [_mm256_setzero_ps(); 8];
                        for (i, mi) in m.iter_mut().enumerate() {
                            *mi = load8::<F16>(base.add(i * cols + k));
                        }
                        transpose8(&mut m);
                        for (j, mj) in m.iter().enumerate() {
                            _mm256_storeu_ps(pack.add((k + j) * NR + half * 8), *mj);
                        }
                        k += 8;
                    }
                }
            } else {
                for k in 0..kc {
                    for i in 0..NR {
                        *pack.add(k * NR + i) = if i < nr { cvt::<F16>(*w.add((r0 + i) * cols + k0 + k)) } else { 0.0 };
                    }
                }
            }
        }
    }

    /// `M`-token × 16-row register tile: broadcast one activation, FMA against two packed weight vectors.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn micro<const M: usize>(
        pack: *const f32,
        kc: usize,
        x: *const f32,
        ldx: usize,
        y: *mut f32,
        ldy: usize,
        nr: usize,
        accumulate: bool,
    ) {
        unsafe {
            let mut acc = [[_mm256_setzero_ps(); 2]; M];
            if accumulate {
                for (i, a) in acc.iter_mut().enumerate() {
                    let yr = y.add(i * ldy);
                    if nr == NR {
                        a[0] = _mm256_loadu_ps(yr);
                        a[1] = _mm256_loadu_ps(yr.add(8));
                    } else {
                        let mut tmp = [0f32; NR];
                        for (j, t) in tmp.iter_mut().enumerate().take(nr) {
                            *t = *yr.add(j);
                        }
                        a[0] = _mm256_loadu_ps(tmp.as_ptr());
                        a[1] = _mm256_loadu_ps(tmp.as_ptr().add(8));
                    }
                }
            }
            for k in 0..kc {
                let w0 = _mm256_loadu_ps(pack.add(k * NR));
                let w1 = _mm256_loadu_ps(pack.add(k * NR + 8));
                for (i, a) in acc.iter_mut().enumerate() {
                    let xb = _mm256_broadcast_ss(&*x.add(i * ldx + k));
                    a[0] = _mm256_fmadd_ps(xb, w0, a[0]);
                    a[1] = _mm256_fmadd_ps(xb, w1, a[1]);
                }
            }
            for (i, a) in acc.iter().enumerate() {
                let yr = y.add(i * ldy);
                if nr == NR {
                    _mm256_storeu_ps(yr, a[0]);
                    _mm256_storeu_ps(yr.add(8), a[1]);
                } else {
                    let mut tmp = [0f32; NR];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), a[0]);
                    _mm256_storeu_ps(tmp.as_mut_ptr().add(8), a[1]);
                    for (j, t) in tmp.iter().enumerate().take(nr) {
                        *yr.add(j) = *t;
                    }
                }
            }
        }
    }
}

/// `Wᵀ x` for several vectors at once (one pass over W): each `x` has `rows` entries, results `cols`.
pub fn matvec_t(w: &Mat16, xs: &[&[f32]]) -> Vec<Vec<f32>> {
    let (rows, cols) = (w.rows, w.cols);
    const CHUNK: usize = 512;
    let parts: Vec<Vec<Vec<f32>>> = (0..cols.div_ceil(CHUNK))
        .into_par_iter()
        .map(|ci| {
            let (c0, c1) = (ci * CHUNK, ((ci + 1) * CHUNK).min(cols));
            let mut acc = vec![vec![0f32; c1 - c0]; xs.len()];
            let mut buf = vec![0f32; c1 - c0];
            for r in 0..rows {
                let row = &w.row(r)[c0..c1];
                match w.kind {
                    Kind16::Bf16 => buf.iter_mut().zip(row).for_each(|(b, &v)| *b = bf16_to_f32(v)),
                    Kind16::F16 => buf.iter_mut().zip(row).for_each(|(b, &v)| *b = half::f16::from_bits(v).to_f32()),
                }
                for (a, x) in acc.iter_mut().zip(xs) {
                    let xr = x[r];
                    if xr != 0.0 {
                        a.iter_mut().zip(&buf).for_each(|(ai, &bi)| *ai += xr * bi);
                    }
                }
            }
            acc
        })
        .collect();
    (0..xs.len()).map(|k| parts.iter().flat_map(|p| p[k].iter().copied()).collect()).collect()
}

/// The whole matrix transposed into f32 (`cols × rows`), cache-blocked.
pub fn transpose_to_f32(m: &Mat16) -> Vec<f32> {
    let (r, c) = (m.rows, m.cols);
    let mut out = vec![0f32; r * c];
    const B: usize = 64;
    out.par_chunks_mut(B * r).enumerate().for_each(|(cb, chunk)| {
        let (c0, c1) = (cb * B, ((cb + 1) * B).min(c));
        for i in 0..r {
            let row = &m.row(i)[c0..c1];
            for (j, &v) in row.iter().enumerate() {
                chunk[j * r + i] = to_f32(m.kind, v);
            }
        }
    });
    out
}

// ---------- small f32 ops ----------

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch.nn.functional.softplus with beta=1, threshold=20.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `out = x / rms(x) * w` (w already includes any `1 +` offset).
#[inline]
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = dot(x, x) / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * g;
    }
}

/// Row-wise RMSNorm over a `rows × d` matrix.
pub fn rmsnorm_rows(x: &[f32], w: &[f32], eps: f32, out: &mut [f32], d: usize) {
    out.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, r)| rmsnorm(r, w, eps, o));
}

pub fn add_inplace(x: &mut [f32], y: &[f32]) {
    x.par_chunks_mut(1 << 14).zip(y.par_chunks(1 << 14)).for_each(|(a, b)| {
        for (p, q) in a.iter_mut().zip(b) {
            *p += q;
        }
    });
}

pub fn softmax_inplace(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Indices and values of the `k` largest entries, descending.
pub fn topk(values: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(values.len());
    if k == 0 {
        return Vec::new();
    }
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    let mut floor = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if best.len() < k || v > floor {
            let pos = best.partition_point(|&(_, b)| b >= v);
            best.insert(pos, (i as u32, v));
            if best.len() > k {
                best.pop();
            }
            if best.len() == k {
                floor = best[k - 1].1;
            }
        }
    }
    best
}

/// Rank (0 = top) of `values[idx]` among all values.
pub fn rank_of(values: &[f32], idx: usize) -> usize {
    let v = values[idx];
    values.iter().filter(|&&x| x > v).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_mat(rows: usize, cols: usize, seed: u64) -> (Mat16, Vec<f32>) {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        let v: Vec<u16> = (0..rows * cols).map(|_| (f32::to_bits(next()) >> 16) as u16).collect();
        let f: Vec<f32> = v.iter().map(|&b| bf16_to_f32(b)).collect();
        (Mat16::from_vec(rows, cols, Kind16::Bf16, v), f)
    }

    fn reference(w: &[f32], rows: usize, cols: usize, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = vec![0f32; n * rows];
        for t in 0..n {
            for r in 0..rows {
                y[t * rows + r] = (0..cols).map(|k| w[r * cols + k] as f64 * x[t * cols + k] as f64).sum::<f64>() as f32;
            }
        }
        y
    }

    #[test]
    fn matmul_matches_reference() {
        for &(rows, cols) in &[(37, 29), (64, 256), (48, 520), (33, 1000)] {
            let (m, wf) = rand_mat(rows, cols, rows as u64 * 31 + cols as u64);
            for n in [1, 2, 3, 5, 7, 8, 13, 50] {
                let (_, x) = rand_mat(n, cols, 7 + n as u64);
                let mut y = vec![0f32; n * rows];
                matmul(&m, &x, n, &mut y);
                let r = reference(&wf, rows, cols, &x, n);
                for (a, b) in y.iter().zip(&r) {
                    assert!((a - b).abs() < 1e-3 * (1.0 + b.abs()), "rows={rows} cols={cols} n={n}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn topk_and_rank() {
        let v = [0.5, 3.0, -1.0, 2.0, 2.5];
        assert_eq!(topk(&v, 3).iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 4, 3]);
        assert_eq!(rank_of(&v, 3), 2);
    }
}
