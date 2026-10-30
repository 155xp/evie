//! evie-core: a hookable Qwen3.5 inference engine for reading and steering individual MLP neurons.

pub mod analysis;
pub mod calib;
pub mod config;
pub mod engine;
pub mod gpu;
pub mod kernels;
pub mod lens;
pub mod model;
pub mod pickle;
pub mod registry;
pub mod sampling;
pub mod steering;
pub mod tokenizer;
pub mod weights;

pub use config::Config;
pub use engine::{Engine, LensSel};
pub use model::{Cache, Hooks, Model, NoHooks, Seq};
pub use sampling::{SampleParams, Sampler};
pub use steering::{Intervention, Mode, Plan};
pub use tokenizer::{ChatMsg, Role, Tok};
