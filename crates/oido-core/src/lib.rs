//! Orquestación del pipeline de dictado: conecta
//! `CaptureSource → Transcriber → Filtro (dedup + frase) → Injector`
//! vía `crossbeam::channel`. Sin estado mutable compartido entre etapas.
//!
//! Tras el refactor modular profundo, este crate depende de los
//! crates granulares (`oido-audio`, `oido-hotkey`, `oido-input`) en
//! lugar del antiguo `oido-platform` monolítico.
//!
//! Arquitectura detallada: [`ARCHITECTURE.md`](https://github.com/Santirrini/oido-rs/blob/main/ARCHITECTURE.md).

pub mod chunked_pipeline;
pub mod dedup;
pub mod phrase_filter;
pub mod pipeline;
pub mod streaming_pipeline;

// Re-exports para conveniencia del bin (consume estos tipos sin
// importar cada crate granular individualmente).
pub use chunked_pipeline::{ChunkedPipeline, ChunkedPipelineConfig};
pub use oido_audio::{AudioFrame, AudioRx, AudioTx, CpalCapture, Resampler};
pub use oido_hotkey::{GatedHotkey, GatedReadyHandle, Hotkey, RdevHotkey};
pub use oido_input::{ArboardInjector, Injector};
pub use pipeline::{Pipeline, PipelineConfig, PipelineEvent, PipelineState, STT_WORKERS};
pub use streaming_pipeline::{StreamingPipeline, StreamingPipelineConfig};
