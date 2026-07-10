//! Orquestación del pipeline de dictado: conecta
//! `CaptureSource → Transcriber → Filtro (dedup + frase) → Injector`
//! vía `crossbeam::channel`. Sin estado mutable compartido entre etapas.
//!
//! Arquitectura detallada: [`ARCHITECTURE.md`](https://github.com/Santirrini/oido-rs/blob/main/ARCHITECTURE.md).

pub mod dedup;
pub mod phrase_filter;
pub mod pipeline;
pub mod streaming_pipeline;

pub use oido_platform::{AudioFrame, AudioRx, AudioTx, InjectedText, TextRx, TextTx};
pub use pipeline::{Pipeline, PipelineConfig, PipelineEvent, PipelineState};
pub use streaming_pipeline::{StreamingPipeline, StreamingPipelineConfig};
