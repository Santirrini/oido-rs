//! Orquestación del pipeline de dictado: conecta
//! `CaptureSource → Transcriber → Filtro (dedup + frase) → Injector`
//! vía `crossbeam::channel`. Sin estado mutable compartido entre etapas.

#![doc = include_str!("../../../ARCHITECTURE.md")]

pub mod dedup;
pub mod phrase_filter;
pub mod pipeline;

pub use oido_platform::{AudioFrame, AudioRx, AudioTx, InjectedText, TextRx, TextTx};
pub use pipeline::{Pipeline, PipelineConfig, PipelineEvent, PipelineState};
