//! Neuron interventions applied at the input of `down_proj` (where `silu(gate) * up` lives).
//!
//! `Add` doses follow the community recipe (Lyra, neuron-explorer): a dose of 1.0 adds
//! `median(‖resid leaving layer L‖) / ‖W_down[:, n]‖` to the neuron's activation, i.e. it moves the
//! residual stream by one typical residual norm along the neuron's write direction, on every token.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Add `value` doses (units of the layer's median residual norm along the write direction).
    #[default]
    Add,
    /// Multiply the activation by `value` (0 ablates).
    Mul,
    /// Clamp the activation to `value`.
    Set,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detector {
    pub layer: usize,
    pub neuron: usize,
    #[serde(default = "one")]
    pub weight: f32,
}

/// Opens (and latches) once the weighted sum of detector activations on some token exceeds `threshold`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gate {
    pub detectors: Vec<Detector>,
    pub threshold: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Intervention {
    pub layer: usize,
    pub neuron: usize,
    #[serde(default)]
    pub mode: Mode,
    pub value: f32,
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<Gate>,
}

fn one() -> f32 {
    1.0
}
fn yes() -> bool {
    true
}

impl Intervention {
    pub fn add(layer: usize, neuron: usize, dose: f32) -> Self {
        Self { layer, neuron, mode: Mode::Add, value: dose, enabled: true, label: String::new(), gate: None }
    }

    pub fn id(&self) -> String {
        format!("L{} N{}", self.layer, self.neuron)
    }

    /// e.g. `L14 N8949 +1.00` / `L3 N7 ×0.00` / `L3 N7 =4.00`.
    pub fn describe(&self) -> String {
        match self.mode {
            Mode::Add => format!("{} {:+.2}", self.id(), self.value),
            Mode::Mul => format!("{} ×{:.2}", self.id(), self.value),
            Mode::Set => format!("{} ={:.2}", self.id(), self.value),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum OpKind {
    Add(f32),
    Mul(f32),
    Set(f32),
}

#[derive(Debug, Clone, Copy)]
struct Op {
    neuron: usize,
    kind: OpKind,
    gate: Option<usize>,
}

#[derive(Debug, Clone)]
struct GateSpec {
    threshold: f32,
    last_layer: usize,
}

/// Interventions compiled for one model: per-layer ops with concrete activation deltas.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    ops: Vec<Vec<Op>>,
    gates: Vec<GateSpec>,
    /// Per layer: (gate, neuron, weight).
    detectors: Vec<Vec<(usize, usize, f32)>>,
}

/// Per-sequence gate state across forward calls.
#[derive(Debug, Clone, Default)]
pub struct GateState {
    pub open: Vec<bool>,
    /// Latest detector score per gate (for display).
    pub score: Vec<f32>,
    partial: Vec<Vec<f32>>,
    opened_at: Vec<Option<usize>>,
}

impl Plan {
    /// `unit(layer, neuron)` = activation change for a dose of 1.0.
    pub fn compile(ivs: &[Intervention], n_layers: usize, n_neurons: usize, unit: impl Fn(usize, usize) -> f32) -> Self {
        let mut plan = Plan { ops: vec![Vec::new(); n_layers], gates: Vec::new(), detectors: vec![Vec::new(); n_layers] };
        for iv in ivs.iter().filter(|i| i.enabled && i.layer < n_layers && i.neuron < n_neurons) {
            let kind = match iv.mode {
                Mode::Add => OpKind::Add(iv.value * unit(iv.layer, iv.neuron)),
                Mode::Mul => OpKind::Mul(iv.value),
                Mode::Set => OpKind::Set(iv.value),
            };
            let gate = iv.gate.as_ref().filter(|g| !g.detectors.is_empty()).map(|g| {
                let gi = plan.gates.len();
                let mut last = 0;
                for d in g.detectors.iter().filter(|d| d.layer < n_layers && d.neuron < n_neurons) {
                    plan.detectors[d.layer].push((gi, d.neuron, d.weight));
                    last = last.max(d.layer);
                }
                plan.gates.push(GateSpec { threshold: g.threshold, last_layer: last });
                gi
            });
            plan.ops[iv.layer].push(Op { neuron: iv.neuron, kind, gate });
        }
        plan
    }

    pub fn is_empty(&self) -> bool {
        self.ops.iter().all(Vec::is_empty)
    }

    pub fn new_state(&self) -> GateState {
        let g = self.gates.len();
        GateState { open: vec![false; g], score: vec![0.0; g], partial: vec![Vec::new(); g], opened_at: vec![None; g] }
    }

    /// Start of a forward pass covering `rows` new positions.
    pub fn begin(&self, st: &mut GateState, rows: usize) {
        if st.open.len() != self.gates.len() {
            *st = self.new_state();
        }
        for p in st.partial.iter_mut() {
            p.clear();
            p.resize(rows, 0.0);
        }
        st.opened_at.iter_mut().for_each(|o| *o = None);
    }

    /// Applies this layer's detectors and interventions to `acts` (`rows × n`).
    pub fn apply(&self, layer: usize, acts: &mut [f32], n: usize, st: &mut GateState) {
        let rows = acts.len() / n;
        if let Some(dets) = self.detectors.get(layer) {
            for &(g, neuron, w) in dets {
                for r in 0..rows {
                    if let Some(p) = st.partial[g].get_mut(r) {
                        *p += w * acts[r * n + neuron];
                    }
                }
            }
            for (g, spec) in self.gates.iter().enumerate() {
                if spec.last_layer != layer || st.partial[g].is_empty() {
                    continue;
                }
                if let Some(&last) = st.partial[g].last() {
                    st.score[g] = last;
                }
                if !st.open[g]
                    && let Some(r) = st.partial[g].iter().position(|&s| s > spec.threshold)
                {
                    st.opened_at[g] = Some(r);
                    st.open[g] = true;
                }
            }
        }
        let Some(ops) = self.ops.get(layer) else { return };
        for op in ops {
            let from = match op.gate {
                None => 0,
                Some(g) if !st.open[g] => continue,
                // Opened during this pass: only rows at/after the trigger, and only if the
                // detectors sit at or below this layer (otherwise it takes effect next pass).
                Some(g) => match st.opened_at[g] {
                    Some(r) if self.gates[g].last_layer <= layer => r,
                    Some(_) => continue,
                    None => 0,
                },
            };
            for r in from..rows {
                let a = &mut acts[r * n + op.neuron];
                match op.kind {
                    OpKind::Add(d) => *a += d,
                    OpKind::Mul(m) => *a *= m,
                    OpKind::Set(v) => *a = v,
                }
            }
        }
    }
}

/// Community recipes for Qwen3.5-4B (layers/neurons zero-based, doses in Add units).
pub fn presets_4b() -> Vec<(&'static str, &'static str, Vec<Intervention>)> {
    let add = |l, n, d| Intervention::add(l, n, d);
    let lab = |mut i: Intervention, s: &str| {
        i.label = s.to_string();
        i
    };
    let gaming_gate = Gate {
        detectors: [(22, 493), (23, 5661), (24, 3667), (25, 8067)]
            .into_iter()
            .map(|(layer, neuron)| Detector { layer, neuron, weight: 1.0 })
            .collect(),
        threshold: 2.0,
    };
    let mut pulse = lab(add(14, 8949, 1.0), "sinister pulse");
    pulse.gate = Some(gaming_gate);
    vec![
        ("Evil Qwen", "@_lyraaaa_ · two neurons flip the assistant hostile", vec![lab(add(14, 8949, 1.0), "hostile"), lab(add(13, 7956, -1.0), "partner")]),
        ("Hostile (sweep #1)", "strongest hostile neuron in a 180-cell sweep", vec![lab(add(16, 8927, 1.0), "hostile")]),
        (
            "Catgirl Meecot",
            "@aryabhuptani · mascot + cat words + kawaii + female",
            vec![
                lab(add(9, 9009, 0.7), "mascot"),
                lab(add(26, 6173, 0.3), "cat-word bias"),
                lab(add(26, 8930, 0.7), "kawaii"),
                lab(add(22, 8295, -0.4), "female"),
            ],
        ),
        ("Anti-gaming Qwen", "@0xabundance_ · sinister only when a video-game detector fires", vec![pulse]),
        ("Ion Channel Qwen", "@JacobMolBio", vec![lab(add(6, 1609, -0.6), "ion channel")]),
        ("Biomechanics Qwen", "@JacobMolBio", vec![lab(add(5, 2817, -0.6), "biomechanics")]),
        ("Chromosome Qwen", "@JacobMolBio", vec![lab(add(6, 3474, 0.6), "chromosome")]),
        ("DNA / RNA Qwen", "@JacobMolBio", vec![lab(add(4, 8842, -0.6), "DNA/RNA")]),
        ("Maglev Qwen", "@JacobMolBio", vec![lab(add(7, 6549, -0.6), "maglev")]),
        ("Nuclear Qwen", "@JacobMolBio", vec![lab(add(7, 1772, 0.6), "nuclear")]),
        ("Plasmid Qwen", "@JacobMolBio", vec![lab(add(7, 7933, -0.6), "plasmid")]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_mul_set() {
        let ivs = vec![
            Intervention::add(0, 1, 2.0),
            Intervention { mode: Mode::Mul, value: 0.0, ..Intervention::add(0, 2, 0.0) },
            Intervention { mode: Mode::Set, value: 7.0, ..Intervention::add(1, 0, 0.0) },
        ];
        let plan = Plan::compile(&ivs, 2, 3, |_, _| 0.5);
        let mut st = plan.new_state();
        plan.begin(&mut st, 2);
        let mut a = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        plan.apply(0, &mut a, 3, &mut st);
        assert_eq!(a, vec![1.0, 2.0, 0.0, 2.0, 3.0, 0.0]);
        let mut b = vec![1.0, 1.0, 1.0];
        plan.apply(1, &mut b, 3, &mut st);
        assert_eq!(b, vec![7.0, 1.0, 1.0]);
    }

    #[test]
    fn gate_latches_and_is_causal() {
        let mut iv = Intervention::add(0, 0, 1.0);
        iv.gate = Some(Gate { detectors: vec![Detector { layer: 1, neuron: 0, weight: 1.0 }], threshold: 5.0 });
        let plan = Plan::compile(&[iv], 2, 1, |_, _| 1.0);
        let mut st = plan.new_state();
        // Pass 1: detector (layer 1) fires on row 1, but the pulse (layer 0) already ran this pass.
        plan.begin(&mut st, 2);
        let mut l0 = vec![0.0, 0.0];
        plan.apply(0, &mut l0, 1, &mut st);
        assert_eq!(l0, vec![0.0, 0.0]);
        let mut l1 = vec![1.0, 9.0];
        plan.apply(1, &mut l1, 1, &mut st);
        assert!(st.open[0]);
        // Pass 2: gate is latched open from the previous pass.
        plan.begin(&mut st, 1);
        let mut l0 = vec![0.0];
        plan.apply(0, &mut l0, 1, &mut st);
        assert_eq!(l0, vec![1.0]);
    }
}
